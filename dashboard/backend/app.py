import csv
import io
import json
import re
import unicodedata
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .data_loader import PredictionData
from .predict import Predictor, REQUIRED, validate_input
from .schema import FIELD_DEFINITIONS
from .chat_service import answer_chat, status as gemini_status

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 3_000_000
MAX_RESPONSE_BYTES = 3_500_000
MAX_BATCH_ROWS = 5000
REFUSAL = 'Tôi là Trợ lý AI của đồ án DataCo. Tôi chỉ có thể hỗ trợ trả lời các câu hỏi liên quan đến số liệu, mô hình và dữ liệu của đồ án này.'


@asynccontextmanager
async def lifespan(app):
    app.state.predictor = Predictor(ROOT / 'backend/models')
    app.state.data = PredictionData(ROOT / 'data/delivery_predictions.csv')
    yield
    app.state.predictor.close()


app = FastAPI(title='DataCo · Dự báo giao hàng trễ', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['http://localhost:8000', 'http://127.0.0.1:8000',
                   'http://localhost:5500', 'http://127.0.0.1:5500'], allow_methods=['GET', 'POST'],
                   allow_headers=['Content-Type'])


@app.exception_handler(RequestValidationError)
async def validation_error(request, exc):
    fields = ', '.join('.'.join(map(str, e['loc'])) for e in exc.errors())
    return JSONResponse(status_code=422, content={'detail': 'Dữ liệu không hợp lệ tại: ' + fields})


@app.exception_handler(ValueError)
async def bad_value(request, exc):
    return JSONResponse(status_code=422, content={'detail': str(exc)})


@app.exception_handler(RuntimeError)
async def not_ready(request, exc):
    return JSONResponse(status_code=503, content={'detail': str(exc)})


@app.exception_handler(Exception)
async def unexpected_error(request, exc):
    import logging
    logging.exception('Lỗi xử lý dashboard', exc_info=exc)
    return JSONResponse(status_code=500, content={'detail': 'Có lỗi xử lý phía backend. Xem terminal để kiểm tra artifacts và dữ liệu.'})


def threshold(value, config):
    result = config['threshold'] if value is None else value
    if result is None:
        raise RuntimeError('Chưa có ngưỡng DataCo xuất từ notebook.')
    if not 0 <= result <= 1:
        raise ValueError('Ngưỡng phải nằm trong khoảng 0–1.')
    return result


@app.get('/api/model-info')
def model_info(request: Request):
    p = request.app.state.predictor
    config = p.config
    cm = config['confusion_matrix']
    warnings = []
    if cm and sum(cm.values()) and config['metrics'].get('accuracy') is not None:
        measured = (cm['tn'] + cm['tp']) / sum(cm.values())
        if abs(measured - config['metrics']['accuracy']) > .0001:
            warnings.append('Metrics và confusion matrix DataCo không khớp; cần export lại cùng model/ngưỡng.')
    return {**config, 'ready': p.model is not None, 'artifact_error': p.error,
            'data_ready': request.app.state.data.frame is not None,
            'data_error': request.app.state.data.error, 'warnings': warnings,
            'top_features': config.get('feature_importance', [])[:10],
            'form_fields': config.get('form_fields') or FIELD_DEFINITIONS,
            'feature_names': p.names,
            'limits': {'upload_bytes': MAX_UPLOAD_BYTES, 'batch_rows': MAX_BATCH_ROWS},
            'chat_generative_ready': gemini_status()['configured']}


@app.get('/api/health')
def health(request: Request):
    model_ready = request.app.state.predictor.model is not None
    data_ready = request.app.state.data.frame is not None
    return JSONResponse(status_code=200 if model_ready and data_ready else 503,
                        content={'status': 'ok' if model_ready and data_ready else 'not_ready',
                                 'model_ready': model_ready, 'data_ready': data_ready})


@app.get('/api/overview')
def overview(request: Request, t: float | None = Query(None)):
    return request.app.state.data.overview(threshold(t, request.app.state.predictor.config))


@app.get('/api/evaluation')
def evaluation(request: Request, t: float | None = Query(None)):
    return request.app.state.data.evaluation(threshold(t, request.app.state.predictor.config))


@app.post('/api/predict')
def predict(request: Request, payload: dict):
    p = request.app.state.predictor
    cleaned = validate_input(payload, p.config)
    probs, frame = p.probabilities([cleaned])
    value = float(probs[0])
    reasons, note = p.explain(frame)
    return {'order_item_id': str(payload.get('order_item_id') or 'don-moi'), 'xac_suat_tre': value,
            'du_doan_tre': int(value >= p.config['threshold']), 'threshold': p.config['threshold'],
            'top_ly_do': reasons, 'explanation_note': note, 'source': 'model_tot_nhat.joblib'}


def safe_csv_value(value):
    text = str(value)
    return "'" + text if text.lstrip().startswith(('=', '+', '-', '@')) else text


def batch_predict(text, predictor):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff')))
    if not reader.fieldnames:
        raise ValueError('CSV chưa có dòng tiêu đề.')
    if len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError('CSV trùng tên cột.')
    if predictor.model is None:
        raise RuntimeError(predictor.error)
    results, valid, indices = [], [], []
    for number, row in enumerate(reader, start=2):
        if len(results) >= MAX_BATCH_ROWS:
            raise ValueError('CSV tối đa 5.000 dòng. Hãy chia nhỏ file.')
        record = {'line': number, 'order_item_id': row.get('order_item_id') or f'dong-{number}',
                  'order_id': row.get('order_id', ''), 'xac_suat_tre': None,
                  'du_doan_tre': None, 'error': ''}
        try:
            if None in row:
                raise ValueError('Số ô không khớp tiêu đề CSV.')
            cleaned = validate_input(row, predictor.config)
            predictor.build_frame(cleaned)
            valid.append(cleaned)
            indices.append(len(results))
        except (ValueError, TypeError) as exc:
            record['error'] = f'Dòng {number}: {exc}'
        results.append(record)
    if not results:
        raise ValueError('CSV phải có ít nhất một dòng dữ liệu.')
    if valid:
        try:
            probabilities, _ = predictor.probabilities(valid)
        except Exception:
            # Isolate model-level row failures without dropping the rest of the CSV.
            probabilities = []
            for idx, row in zip(indices, valid):
                try:
                    probability = predictor.probabilities([row])[0][0]
                    probabilities.append(float(probability))
                except Exception as exc:
                    results[idx]['error'] = f'Dòng {results[idx]["line"]}: model không xử lý được dữ liệu ({type(exc).__name__}).'
                    probabilities.append(None)
        for idx, probability in zip(indices, probabilities):
            if probability is not None:
                results[idx]['xac_suat_tre'] = float(probability)
                results[idx]['du_doan_tre'] = int(probability >= predictor.config['threshold'])
    late = sum(row['du_doan_tre'] == 1 for row in results)
    valid_count = sum(not row['error'] for row in results)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows({k: safe_csv_value(v) if v is not None else '' for k,v in row.items()} for row in results)
    return {'rows': results, 'csv': output.getvalue(), 'threshold': predictor.config['threshold'],
            'summary': {'total': len(results), 'valid': valid_count, 'late': late,
                        'late_rate': late / valid_count if valid_count else 0, 'errors': len(results) - valid_count}}


@app.post('/api/predict-batch')
def predict_batch(request: Request, file: UploadFile = File(...)):
    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, 'CSV tối đa 3 MB. Hãy chia nhỏ file.')
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError('Hãy lưu CSV bằng UTF-8.') from None
    result = batch_predict(text, request.app.state.predictor)
    encoded = json.dumps(result, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise HTTPException(413, 'Kết quả quá lớn cho một lượt tải. Hãy chia CSV thành các file nhỏ hơn.')
    return Response(encoded, media_type='application/json')


@app.get('/api/orders/{order_id}')
def order(request: Request, order_id: str, t: float | None = Query(None)):
    result = request.app.state.data.order(order_id, threshold(t, request.app.state.predictor.config))
    if result is None:
        raise HTTPException(404, 'Không tìm thấy mã đơn trong delivery_predictions.csv.')
    return result


@app.get('/api/sample.csv')
def sample_csv(request: Request):
    config = request.app.state.predictor.config
    schema = config.get('form_fields')
    if not schema:
        raise RuntimeError('Hãy export artifacts DataCo trước khi tải CSV mẫu.')
    fields = ['order_item_id', 'order_id'] + [field['name'] for field in schema]
    raw = next(iter(config.get('demo_inputs', {}).values()), {})
    sample = ['mau-1', 'don-mau'] + [raw.get(field['name'], field.get('default', '')) for field in schema]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(fields)
    writer.writerow(sample)
    return Response('\ufeff' + output.getvalue(), media_type='text/csv',
                    headers={'Content-Disposition': 'attachment; filename="dataco-mau.csv"'})


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text.lower().replace('đ', 'd')) if unicodedata.category(c) != 'Mn')


def grounded_answer(message, config, data):
    q = normalized(message)
    if re.search(r'dich|thoi tiet|co phieu|bo qua|ignore|system prompt|lap trinh|viet code', q):
        return REFUSAL, []
    if any(word in q for word in ['leakage','ro ri','actual','delivery_status']):
        return ('Notebook DataCo loại actual_shipping_days, shipping_delay_days, delivery_status, shipping_date, '
                'order_status, trạng thái hủy và các mã định danh khỏi đầu vào. Nhãn late_delivery_risk chỉ dùng đánh giá, '
                'không dùng dự báo. Web giữ nguyên pipeline đã fit.'), ['knowledge_context.txt','model_config.json']
    if any(word in q for word in ['nguong','youden','threshold']):
        value = config.get('threshold')
        answer = 'Chưa có ngưỡng thực tế. Cần chạy cell export cuối notebook DataCo.' if value is None else (
            f'Ngưỡng Youden chọn trên Validation là {value:.6f}. Dự đoán TRỄ khi xác suất lớn hơn hoặc bằng ngưỡng.')
        return answer, ['model_config.json']
    if any(word in q for word in ['hieu nang','auc','precision','recall','accuracy','f1','test']):
        known = {key:value for key,value in config['metrics'].items() if value is not None}
        answer = 'Chưa có metrics test DataCo thực tế; không dùng số liệu của bộ dữ liệu khác.' if not known else (
            'Chỉ số tập test DataCo tại ngưỡng Validation: '+', '.join(f'{key.upper()} {value:.6f}' for key,value in known.items()))
        return answer, ['model_config.json']
    if any(word in q for word in ['ti le','tong quan','bao nhieu','so dong']):
        if data.frame is None or config.get('threshold') is None:
            return 'Chưa nạp CSV và ngưỡng DataCo, chưa tính được số dòng hoặc tỷ lệ dự đoán trễ.', ['model_config.json']
        frame = data.frame
        rate = float((frame.xac_suat_tre >= config['threshold']).mean())
        return f'Từ delivery_predictions.csv DataCo: {len(frame):,} dòng; tỷ lệ trễ dự báo {rate:.4%} tại ngưỡng {config["threshold"]:.6f}.', ['delivery_predictions.csv','model_config.json']
    if any(word in q for word in ['dataco','mo hinh','smote','xgboost','feature','dac trung','tien xu ly','chia tap']):
        count = config.get('feature_count')
        description = 'Chưa export model được chọn.' if not count else f'Model {config["model"]}; {count} feature gốc.'
        return ('Đồ án dùng DataCo Supply Chain. Notebook so sánh Logistic Regression, Random Forest và XGBoost, '
                'chọn model theo AUC Validation. '+description+' Pipeline xử lý giá trị thiếu và mã hóa One-Hot; '
                'chuẩn hóa biến số cho Logistic. Web không train hoặc SMOTE lại. Xác suất cấp đơn bằng MAX xác suất các dòng.'), ['model_config.json','knowledge_context.txt']
    return REFUSAL, []


@app.get('/api/chat-status')
def chat_status():
    return gemini_status()


@app.post('/api/chat')
async def chat(request: Request, payload: dict):
    from .chat_service import public_response
    result = await answer_chat(payload, request.app.state.predictor, request.app.state.data,
                               ROOT, grounded_answer, REFUSAL)
    return public_response(result, request.app.state.predictor.config)


@app.get('/index.html')
def home_alias():
    return RedirectResponse('/', status_code=308)


@app.get('/frontend/{path:path}')
def legacy_frontend(path: str, request: Request):
    target = '/' if path in ('', 'index.html') else '/' + path.lstrip('/')
    if request.url.query:
        target += '?' + request.url.query
    return RedirectResponse(target, status_code=308)


app.mount('/', StaticFiles(directory=ROOT / 'frontend', html=True), name='frontend')

"""Attach reported notebook results to the structurally matching DataCo model."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]


def import_results(source):
    source = Path(source)
    reported = json.loads(source.read_text(encoding='utf-8'))
    models = ROOT/'backend/models'
    model_path = models/'model_tot_nhat.joblib'
    config_path = models/'model_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    pipeline = joblib.load(model_path)
    names = reported['dac_trung_phan_loai'] + reported['dac_trung_so']
    if names != list(pipeline.feature_names_in_):
        raise ValueError('Feature trong cấu hình không khớp thứ tự đầu vào của model đã nhận.')
    if reported.get('nhan') != 'late_delivery_risk' or reported.get('bang_nguon') != config.get('source_table'):
        raise ValueError('Nhãn hoặc bảng nguồn không khớp DataCo đã cấu hình.')
    selected = reported['mo_hinh_duoc_chon']
    expected = {'Logistic Regression':'LogisticRegression','Random Forest':'RandomForestClassifier','XGBoost':'XGBClassifier'}
    if expected.get(selected) != type(pipeline.named_steps['clf']).__name__:
        raise ValueError('Tên mô hình được chọn không khớp classifier trong joblib.')
    thresholds = reported['nguong_toi_uu']
    threshold = float(thresholds[selected] if isinstance(thresholds,dict) else thresholds)
    auc = float(reported['auc_test'][selected])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or not math.isfinite(auc) or not 0 <= auc <= 1:
        raise ValueError('Ngưỡng hoặc AUC không hợp lệ.')
    counts = reported['so_dong']
    if set(counts) != {'train','validation','test'} or any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ValueError('Số dòng các tập chia không hợp lệ.')
    metrics = {key:None for key in ['roc_auc','pr_auc','accuracy','precision','recall','f1']}
    metrics['roc_auc'] = auc
    config.update(model=selected,threshold=threshold,expected_rows=sum(counts.values()),split_counts=counts,
                  metrics=metrics,confusion_matrix=None,
                  training_versions=reported.get('phien_ban',{}),
                  reported_run={'ngay_chay':reported.get('ngay_chay'),'profile':reported.get('profile'),
                                'random_state':reported.get('random_state')},
                  metrics_source='ROC-AUC tập test từ cau_hinh_va_ket_qua.json do người dùng cung cấp; chưa đối soát bằng dữ liệu test',
                  results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  notes='Model, feature, bảng nguồn và tên classifier khớp cấu trúc file cấu hình. Đã nhập ngưỡng Validation; còn thiếu CSV và các metrics test khác.')
    backup = models/'model_config.before-results.json'
    if not backup.exists():
        shutil.copy2(config_path,backup)
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    shutil.copy2(source,models/'cau_hinh_va_ket_qua.json')
    print(f'Imported {selected}: threshold={threshold:.16f}; reported test ROC-AUC={auc:.16f}.')
    print(f'Reported split counts: {counts}; expected CSV rows={sum(counts.values())}.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('results',type=Path)
    import_results(parser.parse_args().results)

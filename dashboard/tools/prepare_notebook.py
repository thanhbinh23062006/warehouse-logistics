"""Repair notebook order and embed a standalone export cell; do not execute it."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(path):
    path = Path(path)
    original = path.read_text(encoding='utf-8')
    notebook = json.loads(original)
    if not any('DataCo Supply Chain' in ''.join(c['source']) for c in notebook['cells']):
        raise ValueError('Không nhận diện được notebook DataCo đồ án; không tự sửa.')
    backup = path.with_name('train.before-dashboard.ipynb')
    if not backup.exists():
        backup.write_text(original, encoding='utf-8')
    cells = [cell for cell in notebook['cells'] if not cell.get('metadata', {}).get('dataco_dashboard_export')]
    index = next(i for i,c in enumerate(cells) if c['cell_type']=='code' and
                 "test_items.groupby('order_id')" in ''.join(c['source']))
    start = index-1 if cells[index-1]['cell_type']=='markdown' and 'ORDER-LEVEL' in ''.join(cells[index-1]['source']) else index
    block = cells[start:index+1]
    del cells[start:index+1]
    insert = next(i for i,c in enumerate(cells) if c['cell_type']=='code' and
                  'best_final = best_name' in ''.join(c['source'])) + 1
    cells[insert:insert] = block
    for cell in cells:
        if cell['cell_type']!='code':
            continue
        source = ''.join(cell['source'])
        if source.startswith('# Thiết lập kích hoạt ghi ngược'):
            source = ('# Ghi BigQuery chỉ khi người chạy bật WRITE_BACK_TO_BQ.\nif WRITE_BACK_TO_BQ:\n' +
                      ''.join('    '+line+'\n' for line in source.splitlines())+
                      'else:\n    print("Bỏ qua ghi BigQuery: WRITE_BACK_TO_BQ = False.")\n')
        if 'df_bq_check = run_query(sql_check_query)' in source and 'if WRITE_BACK_TO_BQ:' not in source:
            source = source.replace('df_bq_check = run_query(sql_check_query)\ndisplay(df_bq_check)',
                'if WRITE_BACK_TO_BQ:\n    df_bq_check = run_query(sql_check_query)\n    display(df_bq_check)\nelse:\n    print("Chỉ in SQL đối soát; chưa bật ghi/đọc bảng dự đoán BigQuery.")')
        cell['source'] = source.splitlines(keepends=True)
    schema = (ROOT/'backend/schema.py').read_text(encoding='utf-8')
    exporter = (ROOT/'tools/export_artifacts.py').read_text(encoding='utf-8').replace(
        'from backend.schema import FIELD_DEFINITIONS, FORBIDDEN\n','')
    export_source = schema+'\n\n'+exporter+'''

# Cell này chỉ dùng kết quả/pipeline đã fit trong phiên; KHÔNG huấn luyện lại.
DASHBOARD_EXPORT_DIR = os.path.join(ART_DIR, "dataco_dashboard_export")
bundle_path = export_dataco_dashboard(globals(), DASHBOARD_EXPORT_DIR)
print("Tải ZIP, giải nén và chép backend/models + data vào thư mục dashboard trên máy.")
if IN_COLAB:
    from google.colab import files
    files.download(bundle_path)
'''
    metadata = {'dataco_dashboard_export':True}
    cells.extend([
        {'cell_type':'markdown','metadata':metadata,'source':[
            '# PHẦN 20. XUẤT DỮ LIỆU CHO WEBSITE DATACO\n',
            'Nếu phiên Colab đã có `best_est`, `best_final`, `X_all`, `X_train`, `X_test` và các tập chia, chỉ chạy cell bên dưới. ',
            'Cell không fit model, không ghi BigQuery. Nó lưu pipeline tốt nhất, ngưỡng Validation, metrics Test, ',
            'tên feature gốc/sau mã hóa, defaults Train, mẫu dự báo và CSV metadata vào ZIP trên Drive.\n',
            'Nếu biến chưa tồn tại, cell sẽ báo tên biến thiếu. Không có kết quả model được tạo khi chỉ mở file notebook.\n']},
        {'cell_type':'code','metadata':metadata,'execution_count':None,'outputs':[],
         'source':export_source.splitlines(keepends=True)}])
    notebook['cells'] = cells
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(f'Đã sửa thứ tự, bảo vệ ghi BigQuery và thêm cell xuất ZIP: {path}')
    print(f'Bản gốc: {backup}')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    prepare(parser.parse_args().notebook)

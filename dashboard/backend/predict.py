"""DataCo inference through the fitted notebook pipeline; no fitting in the API."""
import json
import math
from datetime import date
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from .schema import FORBIDDEN, REQUIRED, validate_input


class Predictor:
    def __init__(self, models_dir):
        self.root = Path(models_dir)
        self.config = json.loads((self.root / 'model_config.json').read_text(encoding='utf-8'))
        self.model, self.names, self.error = None, [], None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='explanation')
        self.explanation_lock = Lock()
        self.explanation_busy, self.explainer = False, None
        try:
            if self.config.get('dataset') != 'DataCo':
                raise ValueError('Cần artifacts DataCo của notebook đồ án.')
            t = self.config.get('threshold')
            if t is None or not math.isfinite(t) or not 0 <= t <= 1:
                raise ValueError('Chưa có ngưỡng tối ưu xuất từ Validation trong notebook.')
            self.names = json.loads((self.root / 'feature_names.json').read_text(encoding='utf-8'))
            if not self.names or len(self.names) != self.config['feature_count'] or len(set(self.names)) != len(self.names):
                raise ValueError('Số lượng/thứ tự feature phải khớp cấu hình export từ notebook.')
            if FORBIDDEN.intersection(self.names) or {'order_id', 'order_item_id'} & set(self.names):
                raise ValueError('Artifacts chứa biến rò rỉ hoặc định danh bị cấm.')
            if set(self.names) - set(self.config['feature_defaults']):
                raise ValueError('feature_defaults chưa đủ cho bộ đầu vào của pipeline.')
            self.model = joblib.load(self.root / 'model_tot_nhat.joblib')
            self.positive_index = list(self.model.classes_).index(1)
            if list(getattr(self.model, 'feature_names_in_', self.names)) != self.names:
                raise ValueError('Thứ tự feature không khớp pipeline đã lưu.')
            if getattr(self.model, 'n_features_in_', len(self.names)) != len(self.names):
                raise ValueError('Số feature không khớp pipeline.')
        except Exception as exc:
            self.model = None
            self.error = f'Chưa sẵn sàng dự báo: {exc}'

    def build_frame(self, inputs):
        if isinstance(inputs, dict):
            inputs = [inputs]
        rows = []
        for data in inputs:
            row = dict(self.config['feature_defaults'])
            for key, value in data.items():
                if key in self.names:
                    row[key] = value
                elif key == 'order_item_total':
                    row['log_order_item_total'] = math.log1p(max(0, float(value)))
                elif key == 'order_date':
                    day = date.fromisoformat(str(value))
                    row['order_month'] = f'T{day.month:02d}'
                    row['order_dayofweek'] = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][day.weekday()]
                else:
                    raise ValueError(f'{key}: không thuộc đầu vào của pipeline DataCo.')
            rows.append(row)
        return pd.DataFrame(rows, columns=self.names)

    def probabilities(self, rows):
        if self.model is None:
            raise RuntimeError(self.error)
        frame = self.build_frame(rows)
        result = np.asarray(self.model.predict_proba(frame))[:, self.positive_index]
        if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise ValueError('Pipeline trả xác suất không hợp lệ.')
        return result, frame

    def explain(self, frame):
        with self.explanation_lock:
            if self.explanation_busy:
                return [], 'SHAP đang bận; xác suất vẫn được tính từ pipeline.'
            self.explanation_busy = True
        def work():
            try:
                import shap
                classifier, names, values_frame = self.model, self.names, frame
                if hasattr(self.model, 'named_steps') and 'prep' in self.model.named_steps:
                    prep = self.model.named_steps['prep']
                    classifier = self.model.named_steps['clf']
                    values_frame = prep.transform(frame)
                    if hasattr(values_frame, 'toarray'):
                        values_frame = values_frame.toarray()
                    names = list(prep.get_feature_names_out())
                if self.explainer is None:
                    self.explainer = shap.TreeExplainer(classifier)
                values = self.explainer.shap_values(values_frame)
                if isinstance(values, list):
                    values = values[self.positive_index]
                values = np.asarray(values)
                if values.ndim == 3:
                    values = values[:, :, self.positive_index]
                grouped = {}
                for name, value in zip(names, values[0]):
                    body = name.split('__', 1)[-1]
                    candidates = [key for key in self.names if body == key or body.startswith(key + '_')]
                    key = max(candidates, key=len) if candidates else body
                    grouped[key] = grouped.get(key, 0.) + float(value)
                top = sorted(grouped.items(), key=lambda pair: abs(pair[1]), reverse=True)[:3]
                return [{'feature': key, 'anh_huong': 'tăng' if value > 0 else 'giảm',
                         'gia_tri': value, 'don_vi': 'SHAP raw'} for key,value in top]
            finally:
                with self.explanation_lock:
                    self.explanation_busy = False
        future = self.pool.submit(work)
        try:
            return future.result(timeout=.4), None
        except TimeoutError:
            return [], 'SHAP vượt thời gian chờ; xác suất không bị ảnh hưởng.'
        except Exception:
            return [], 'SHAP chưa hỗ trợ pipeline này; không tạo lý do giả.'

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

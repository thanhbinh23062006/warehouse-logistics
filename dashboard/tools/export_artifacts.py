"""Export existing DataCo notebook objects. Does not fit or resample."""
import json
import math
import platform
import shutil
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)
from backend.schema import FIELD_DEFINITIONS, FORBIDDEN


def export_dataco_dashboard(ns, destination):
    required = ['best_est', 'best_final', 'thr_opt', 'X_train', 'X_test', 'y_test',
                'X_all', 'df_model', 'train_idx', 'val_idx', 'test_idx', 'FEATURES',
                'CAT_FEATURES', 'NUM_FEATURES', 'TARGET']
    missing = [key for key in required if key not in ns]
    if missing:
        raise ValueError('Chưa có kết quả notebook trong phiên này: ' + ', '.join(missing))
    root = Path(destination)
    model_dir, data_dir = root / 'backend/models', root / 'data'
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    model = ns['best_est'][ns['best_final']]
    names = list(ns['X_train'].columns)
    if names != list(ns['FEATURES']) or names != list(getattr(model, 'feature_names_in_', names)):
        raise ValueError('Thứ tự feature không khớp pipeline notebook.')
    if FORBIDDEN.intersection(names) or {'order_id', 'order_item_id'} & set(names):
        raise ValueError('Bộ feature có biến cấm; dừng export.')
    if not {'log_order_item_total', 'order_month', 'order_dayofweek'} <= set(names):
        raise ValueError('Không khớp phép tạo feature của notebook DataCo hiện tại.')
    threshold = float(ns['thr_opt'][ns['best_final']])
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Ngưỡng Validation không hợp lệ. Cần kiểm tra cell chọn Youden.')
    defaults = {}
    for name in names:
        series = ns['X_train'][name].dropna()
        if series.empty:
            raise ValueError(f'{name}: không có giá trị Train để tạo mặc định.')
        value = series.mode().iloc[0] if name in ns['CAT_FEATURES'] else series.median()
        defaults[name] = value.item() if isinstance(value, np.generic) else value
    fields = []
    for definition in FIELD_DEFINITIONS:
        field = dict(definition)
        name = field['name']
        if name == 'scheduled_shipping_days' and name not in names:
            continue
        if field['type'] == 'select':
            field['options'] = sorted(ns['X_train'][name].dropna().astype(str).unique().tolist())
            field['default'] = defaults[name]
        elif name == 'order_item_total':
            field['default'] = math.expm1(defaults['log_order_item_total'])
        elif name == 'order_date':
            field['default'] = None
        else:
            field['default'] = defaults[name]
            if field.get('integer'):
                field['default'] = int(field['default'])
        fields.append(field)
    positive = list(model.classes_).index(1)
    test_prob = np.asarray(model.predict_proba(ns['X_test']))[:, positive]
    truth = np.asarray(ns['y_test'], dtype=int)
    pred = (test_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    both = len(np.unique(truth)) == 2
    metrics = {'roc_auc': float(roc_auc_score(truth, test_prob)) if both else None,
               'pr_auc': float(average_precision_score(truth, test_prob)) if both else None,
               'accuracy': float(accuracy_score(truth, pred)),
               'precision': float(precision_score(truth, pred, zero_division=0)),
               'recall': float(recall_score(truth, pred, zero_division=0)),
               'f1': float(f1_score(truth, pred, zero_division=0))}
    source = ns['df_model'].reset_index(drop=True)
    scored = source[['order_item_id', 'order_id']].copy()
    split = np.full(len(source), '', dtype=object)
    for label, key in [('train','train_idx'),('validation','val_idx'),('test','test_idx')]:
        indices = np.asarray(ns[key], dtype=int)
        if (split[indices] != '').any():
            raise ValueError('Các tập dữ liệu bị chồng lặp chỉ số.')
        split[indices] = label
    if (split == '').any():
        raise ValueError('Có dòng chưa thuộc tập Train/Validation/Test.')
    scored['tap_du_lieu'] = split
    scored['xac_suat_tre'] = np.asarray(model.predict_proba(ns['X_all']))[:, positive]
    scored['du_doan_tre'] = (scored.xac_suat_tre >= threshold).astype(int)
    scored['mo_hinh'] = ns['best_final']
    scored['thoi_diem_cham_diem'] = pd.Timestamp.now(tz='UTC').isoformat()
    scored['is_late'] = source[ns['TARGET']].astype(int).to_numpy()
    for col in ['category_name','order_region','order_date']:
        scored[col] = source[col].to_numpy()
    test_source = source.iloc[ns['test_idx']].reset_index(drop=True)
    examples = {}
    for label, mask in [('late',test_prob >= threshold),('on_time',test_prob < threshold)]:
        hits = np.flatnonzero(mask)
        if not len(hits):
            continue
        row, raw = test_source.iloc[int(hits[0])], {}
        for field in fields:
            key = field['name']
            value = row[key]
            if pd.isna(value):
                continue
            if key == 'order_date':
                value = pd.Timestamp(value).date().isoformat()
            elif field['type'] == 'select':
                value = str(value)
                if value not in field['options']:
                    field['options'].append(value)
                    field['options'].sort()
            elif field.get('integer'):
                value = int(value)
            else:
                value = float(value)
            raw[key] = value
        examples[label] = raw
    importance, transformed, kind = [], [], None
    if hasattr(model, 'named_steps') and 'prep' in model.named_steps:
        transformed = list(model.named_steps['prep'].get_feature_names_out())
        classifier = model.named_steps['clf']
        weights, kind = getattr(classifier, 'feature_importances_', None), 'feature_importances_'
        if weights is None and hasattr(classifier, 'coef_'):
            weights, kind = np.abs(classifier.coef_[0]), 'abs(coef_)'
        if weights is not None and len(weights) == len(transformed):
            grouped = {}
            for encoded, value in zip(transformed, weights):
                body = encoded.split('__', 1)[-1]
                candidates = [key for key in names if body == key or body.startswith(key+'_')]
                key = max(candidates, key=len) if candidates else body
                grouped[key] = grouped.get(key, 0.) + float(value)
            total_weight = sum(grouped.values()) or 1.
            importance = sorted([{'feature':k,'importance':v/total_weight} for k,v in grouped.items()],
                                key=lambda item:item['importance'], reverse=True)
    config = {'dataset':'DataCo', 'model':ns['best_final'], 'threshold':threshold,
              'feature_count':len(names), 'encoded_feature_count':len(transformed) or None,
              'expected_rows':len(scored), 'source_table':ns.get('WIDE_TABLE'),
              'target':ns['TARGET'], 'metrics':metrics,
              'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
              'metrics_source':'Tập test DataCo; ngưỡng chọn từ Validation; model chọn theo AUC Validation',
              'feature_defaults':defaults, 'form_fields':fields,
              'feature_importance':importance, 'importance_method':kind,
              'demo_inputs':examples,
              'split_counts':{label:int((split==label).sum()) for label in ['train','validation','test']},
              'label_rate':float(scored.is_late.mean()),
              'versions':{key:version(key) for key in ['scikit-learn','xgboost','pandas','numpy','joblib']},
              'python':platform.python_version(),
              'notes':'CSV gồm Train/Validation/Test; điểm Train là in-sample. Pipeline nhận feature gốc và tự encode.'}
    months = pd.to_datetime(test_source.order_date).dt.strftime('%Y-%m')
    monitoring = []
    for month in sorted(months.dropna().unique()):
        mask = (months == month).to_numpy()
        if mask.sum() >= 30 and len(np.unique(truth[mask])) == 2:
            monitoring.append({'period':month,'auc':float(roc_auc_score(truth[mask],test_prob[mask])),
                               'n':int(mask.sum()),'source':'Tập test theo tháng đặt hàng; không phải đánh giá tương lai'})
    config['monitoring_auc'] = monitoring
    joblib.dump(model, model_dir / 'model_tot_nhat.joblib')
    for filename, content in [('model_config.json',config),('feature_names.json',names),
                              ('transformed_feature_names.json',transformed)]:
        (model_dir / filename).write_text(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    scored.to_csv(data_dir / 'delivery_predictions.csv', index=False, encoding='utf-8-sig')
    bundle = shutil.make_archive(str(root.parent / 'dataco_dashboard_bundle'), 'zip', root)
    print(f'Đã xuất {len(names)} feature gốc, {len(scored):,} dòng dự đoán; model {ns["best_final"]}; ngưỡng {threshold:.6f}.')
    print('Bundle:', bundle)
    return bundle

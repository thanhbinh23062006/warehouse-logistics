"""Import a trusted fitted DataCo pipeline and recover its stored preprocessing metadata."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from backend.schema import FIELD_DEFINITIONS, FORBIDDEN

ROOT = Path(__file__).resolve().parents[1]


def native(value):
    return value.item() if isinstance(value, np.generic) else value


def install_model(source):
    source = Path(source)
    pipeline = joblib.load(source)
    names = list(pipeline.feature_names_in_)
    if FORBIDDEN.intersection(names) or {'order_id','order_item_id'} & set(names):
        raise ValueError('Pipeline có đầu vào bị cấm.')
    expected = {field['name'] for field in FIELD_DEFINITIONS} - {'order_date','order_item_total'}
    expected.update({'log_order_item_total','order_month','order_dayofweek'})
    if set(names) - expected or not {'log_order_item_total','order_month','order_dayofweek'} <= set(names):
        raise ValueError('Pipeline không khớp đầu vào DataCo của đồ án.')
    prep, classifier = pipeline.named_steps['prep'], pipeline.named_steps['clf']
    defaults, options = {}, {}
    for key, transformer, columns in prep.transformers_:
        if key not in ('cat','num'):
            continue
        statistics = transformer.named_steps['imputer'].statistics_
        defaults.update({name:native(value) for name,value in zip(columns,statistics)})
        if key == 'cat':
            encoder = transformer.named_steps['onehot']
            options.update({name:[str(native(value)) for value in values if not pd.isna(value)]
                            for name,values in zip(columns,encoder.categories_)})
    if set(defaults) != set(names):
        raise ValueError('Không khôi phục đủ defaults từ bộ điền thiếu đã fit.')
    fields = []
    for definition in FIELD_DEFINITIONS:
        field = dict(definition)
        name = field['name']
        if name == 'scheduled_shipping_days' and name not in names:
            continue
        if field['type'] == 'select':
            field.update(default=defaults[name],options=options[name])
        elif name == 'order_date':
            field['default'] = None
        elif name == 'order_item_total':
            field['default'] = math.expm1(defaults['log_order_item_total'])
        else:
            field['default'] = defaults[name]
        fields.append(field)
    transformed = list(prep.get_feature_names_out())
    weights = getattr(classifier,'feature_importances_',None)
    method = 'feature_importances_'
    if weights is None and hasattr(classifier,'coef_'):
        weights = np.abs(classifier.coef_[0])
        method = 'abs(coef_) grouped and normalized'
    importance = []
    if weights is not None and len(weights) == len(transformed):
        grouped = {}
        for name,weight in zip(transformed,weights):
            body = name.split('__',1)[-1]
            candidates = [key for key in names if body == key or body.startswith(key+'_')]
            key = max(candidates,key=len) if candidates else body
            grouped[key] = grouped.get(key,0.) + float(weight)
        total = sum(grouped.values()) or 1.
        importance = sorted([{'feature':name,'importance':value/total} for name,value in grouped.items()],
                            key=lambda row:row['importance'],reverse=True)
    probabilities = np.asarray(pipeline.predict_proba(pd.DataFrame([defaults],columns=names)))
    if probabilities.shape != (1,2) or not np.isfinite(probabilities).all():
        raise ValueError('Thử suy luận bằng defaults chưa thành công.')
    models = ROOT/'backend/models'
    config = json.loads((models/'model_config.json').read_text(encoding='utf-8'))
    # Preserve results only if they already belong to this exact model file.
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if config.get('artifact_sha256') != digest:
        config.update(threshold=None,metrics={key:None for key in ['roc_auc','pr_auc','accuracy','precision','recall','f1']},
                      confusion_matrix=None,expected_rows=None,demo_inputs={},monitoring_auc=[],
                      metrics_source='Chưa nhận kết quả test/ngưỡng Validation của model được gửi')
    config.update(dataset='DataCo',model=type(classifier).__name__,feature_count=len(names),
                  encoded_feature_count=len(transformed),feature_defaults=defaults,form_fields=fields,
                  feature_importance=importance,importance_method=method,artifact_sha256=digest,
                  preprocessing_source='Statistics and categories recovered from fitted pipeline',
                  verified_runtime={'scikit-learn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__},
                  notes='Đã nạp pipeline và thử suy luận bằng defaults. Chưa có ngưỡng/test metrics/CSV thực; không train lại.')
    for filename,content in [('feature_names.json',names),('transformed_feature_names.json',transformed),
                             ('model_config.json',config)]:
        (models/filename).write_text(json.dumps(content,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if source.resolve() != (models/'model_tot_nhat.joblib').resolve():
        shutil.copy2(source,models/'model_tot_nhat.joblib')
    print(f'Imported {type(classifier).__name__}: {len(names)} raw features, {len(transformed)} encoded columns.')
    print('Inference smoke passed; threshold and test metrics remain unavailable.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('model',type=Path)
    install_model(parser.parse_args().model)

"""Score the actual DataCo snapshot. Reconstruct evaluation splits only when independently verified."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score,average_precision_score,accuracy_score,
                             precision_score,recall_score,f1_score,confusion_matrix)

from backend.predict import Predictor
from backend.data_loader import PredictionData

ROOT=Path(__file__).resolve().parents[1]


def prepare_frame(raw,names):
    """Notebook cells 22, 25, 49; preserve row order and filter cancelled items."""
    frame=raw.copy()
    required={'order_item_id','order_id','is_shipping_canceled','order_date','order_item_total',
              'late_delivery_risk','category_name','order_region'} | (set(names)-{'log_order_item_total','order_month','order_dayofweek'})
    if required-set(frame):
        raise ValueError('Snapshot thiếu cột: '+', '.join(sorted(required-set(frame))))
    if frame.order_item_id.isna().any() or frame.order_item_id.duplicated().any() or frame.order_id.isna().any():
        raise ValueError('Snapshot có khóa null hoặc order_item_id trùng.')
    if not frame.is_shipping_canceled.dropna().isin([True,False]).all():
        raise ValueError('is_shipping_canceled không phải Boolean.')
    frame=frame.loc[~frame.is_shipping_canceled.fillna(False).astype(bool)].copy().reset_index(drop=True)
    for name in ['order_item_quantity','order_item_discount_rate','order_item_total','benefit_per_order']:
        frame[name]=pd.to_numeric(frame[name],errors='raise').astype('float64')
    if 'scheduled_shipping_days' in names:
        frame['scheduled_shipping_days']=pd.to_numeric(frame.scheduled_shipping_days,errors='raise')
    frame['order_date']=pd.to_datetime(frame.order_date,errors='raise')
    if frame.order_date.isna().any():
        raise ValueError('Ngày đặt hàng không được trống.')
    frame['log_order_item_total']=np.log1p(frame.order_item_total.clip(lower=0))
    frame['order_month']=frame.order_date.dt.month.map(lambda month:f'T{month:02d}')
    frame['order_dayofweek']=frame.order_date.dt.day_name().str[:3]
    frame['late_delivery_risk']=pd.to_numeric(frame.late_delivery_risk,errors='raise')
    if not frame.late_delivery_risk.isin([0,1]).all():
        raise ValueError('Nhãn phải là 0/1.')
    for name in names:
        if name not in ['order_item_quantity','order_item_discount_rate','log_order_item_total','benefit_per_order','scheduled_shipping_days']:
            frame[name]=frame[name].astype('object').where(frame[name].notna(),np.nan)
    return frame,frame[names].copy()


def metrics(truth,probability,t):
    labels=(probability>=t).astype(int)
    return {'roc_auc':float(roc_auc_score(truth,probability)),
            'pr_auc':float(average_precision_score(truth,probability)),
            'accuracy':float(accuracy_score(truth,labels)),
            'precision':float(precision_score(truth,labels,zero_division=0)),
            'recall':float(recall_score(truth,labels,zero_division=0)),
            'f1':float(f1_score(truth,labels,zero_division=0))}


def install_snapshot(snapshot,validation_csv):
    config_path=ROOT/'backend/models/model_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    report=json.loads((ROOT/'backend/models/cau_hinh_va_ket_qua.json').read_text(encoding='utf-8'))
    predictor=Predictor(ROOT/'backend/models')
    try:
        if predictor.model is None:
            raise ValueError(predictor.error)
        # BigQuery dbdate metadata is optional; Arrow's physical date values are preserved.
        raw=pq.read_table(snapshot).to_pandas(ignore_metadata=True)
        frame,X=prepare_frame(raw,predictor.names)
        if len(frame)!=config['expected_rows']:
            raise ValueError(f'Số dòng sau lọc {len(frame)} khác cấu hình {config["expected_rows"]}.')
        truth=frame.late_delivery_risk.astype(int).to_numpy()
        groups=frame.order_id
        probabilities=np.asarray(predictor.model.predict_proba(X))[:,predictor.positive_index]
        if not np.isfinite(probabilities).all() or not ((probabilities>=0)&(probabilities<=1)).all():
            raise ValueError('Pipeline trả xác suất không hợp lệ.')
        t=config['threshold']
        validation=pd.read_csv(validation_csv,index_col=0)
        selected=report['mo_hinh_duoc_chon']
        val_row=validation.loc[selected]
        if not math.isclose(float(val_row['Ngưỡng']),t,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Ngưỡng trong CSV validation không khớp model được chọn.')
        column_map={'roc_auc':'ROC-AUC','pr_auc':'PR-AUC','accuracy':'Accuracy',
                    'precision':'Precision','recall':'Recall','f1':'F1 (lớp trễ)'}
        reported_validation={key:float(val_row[column]) for key,column in column_map.items()}
        splitter=StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=int(report['random_state']))
        folds=[idx for _,idx in splitter.split(X,truth,groups=groups)]
        indices={'test':folds[0],'validation':folds[1],'train':np.concatenate(folds[2:])}
        checks={}
        checks['split_counts_match']={key:len(value) for key,value in indices.items()}==report['so_dong']
        group_sets={key:set(groups.iloc[value]) for key,value in indices.items()}
        checks['groups_disjoint']=all(group_sets[a].isdisjoint(group_sets[b]) for a,b in
                                     [('train','test'),('train','validation'),('validation','test')])
        validation_metrics=metrics(truth[indices['validation']],probabilities[indices['validation']],t)
        checks['validation_metrics_match']=all(math.isclose(validation_metrics[key],value,rel_tol=0,abs_tol=1e-10)
                                                for key,value in reported_validation.items())
        test_metrics=metrics(truth[indices['test']],probabilities[indices['test']],t)
        checks['test_auc_matches_report']=math.isclose(test_metrics['roc_auc'],float(report['auc_test'][selected]),rel_tol=0,abs_tol=1e-10)
        imputer_defaults=config['feature_defaults']
        checks['train_imputer_statistics_match']=True
        for name in predictor.names:
            series=X.iloc[indices['train']][name].dropna()
            expected=imputer_defaults[name]
            if isinstance(expected,str):
                matches=str(series.mode().iloc[0])==expected
            else:
                matches=math.isclose(float(series.median()),float(expected),rel_tol=1e-10,abs_tol=1e-10)
            checks['train_imputer_statistics_match'] &= matches
        verified=all(checks.values())
        scored=frame[['order_item_id','order_id','category_name','order_region','order_date']].copy()
        scored['order_date']=scored.order_date.dt.strftime('%Y-%m-%d')
        scored['tap_du_lieu']='snapshot'
        if verified:
            for label,idx in indices.items():
                scored.loc[idx,'tap_du_lieu']=label
        scored['xac_suat_tre']=probabilities
        scored['du_doan_tre']=(probabilities>=t).astype(int)
        scored['mo_hinh']=selected
        scored['thoi_diem_cham_diem']=pd.Timestamp.now(tz='UTC').isoformat()
        scored['is_late']=truth
        config['validation_metrics']=reported_validation
        config['validation_metrics_source']='ket_qua_validation_nguong_toi_uu.csv · Validation, không phải Test'
        config['snapshot_verification']={'checks':checks,'split_verified':verified,
            'snapshot_sha256':hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),
            'source_rows':len(raw),'cancelled_rows_removed':len(raw)-len(frame),
            'scored_rows':len(scored),'method':'Notebook StratifiedGroupKFold(7, shuffle=True, random_state=42); no model fitting'}
        config['label_rate']=float(truth.mean())
        if verified:
            config['metrics']=test_metrics
            config['metrics_source']='Tập Test khôi phục từ snapshot; khớp số dòng, nhóm, thống kê Train, metrics Validation và AUC Test trong artifacts'
            tn,fp,fn,tp=confusion_matrix(truth[indices['test']],(probabilities[indices['test']]>=t).astype(int),labels=[0,1]).ravel()
            config['confusion_matrix']={'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
            test_frame=frame.iloc[indices['test']]
            test_probabilities=probabilities[indices['test']]
            test_truth=truth[indices['test']]
            months=test_frame.order_date.dt.strftime('%Y-%m').to_numpy()
            config['monitoring_auc']=[]
            for month in sorted(set(months)):
                mask=months==month
                if mask.sum()>=30 and len(np.unique(test_truth[mask]))==2:
                    config['monitoring_auc'].append({'period':month,'auc':float(roc_auc_score(test_truth[mask],test_probabilities[mask])),
                        'n':int(mask.sum()),'source':'Nhóm Test theo tháng đặt hàng; không phải backtest tương lai'})
        else:
            config['metrics']={key:None for key in column_map}
            config['metrics']['roc_auc']=float(report['auc_test'][selected])
            config['metrics_source']='ROC-AUC từ file kết quả notebook; chưa khôi phục được tập Test từ snapshot hiện tại'
            config['confusion_matrix']=None
            config['monitoring_auc']=[]
        # Real source rows for demo; verify that the public form reproduces model scores.
        examples={}
        for label,mask in [('late',probabilities>=t),('on_time',probabilities<t)]:
            for index in np.flatnonzero(mask)[:100]:
                row=frame.iloc[int(index)]
                candidate={}
                try:
                    for field in config['form_fields']:
                        key=field['name']
                        value=row[key]
                        if pd.isna(value):
                            continue
                        if key=='order_date': value=pd.Timestamp(value).date().isoformat()
                        elif field['type']=='select': value=str(value)
                        elif field.get('integer'): value=int(value)
                        else: value=float(value)
                        candidate[key]=value
                    from backend.schema import validate_input
                    cleaned=validate_input(candidate,config)
                    p=predictor.probabilities([cleaned])[0][0]
                    if math.isclose(p,float(probabilities[index]),abs_tol=1e-12):
                        examples[label]=candidate
                        break
                except ValueError:
                    continue
        config['demo_inputs']=examples
        config['notes']='Đã chấm snapshot DataCo bằng pipeline đã fit; không huấn luyện lại. Tổng quan gồm cả Train, Validation và Test; điểm Train là in-sample.'
        target=ROOT/'data/delivery_predictions.csv'
        temporary=target.with_suffix('.csv.tmp')
        scored.to_csv(temporary,index=False,encoding='utf-8-sig')
        loaded=PredictionData(temporary)
        if loaded.error:
            raise ValueError(loaded.error)
        temporary.replace(target)
        config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        shutil.copy2(validation_csv,ROOT/'data/ket_qua_validation_nguong_toi_uu.csv')
        print(json.dumps({'checks':checks,'verified_original_split':verified,'source_rows':len(raw),
            'cancelled_removed':len(raw)-len(frame),'scored_rows':len(scored),
            'predicted_late_count':int(scored.du_doan_tre.sum()),'predicted_late_rate':float(scored.du_doan_tre.mean()),
            'high_risk_orders':int(scored.loc[scored.xac_suat_tre>=.7,'order_id'].nunique()),
            'test_metrics':test_metrics if verified else None,
            'demo_scenarios':list(examples)},ensure_ascii=False,indent=2))
    finally:
        predictor.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('validation_csv',type=Path)
    args=parser.parse_args()
    install_snapshot(args.snapshot,args.validation_csv)

import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from backend.app import app, batch_predict, grounded_answer, REFUSAL
from backend.data_loader import PredictionData
from backend.predict import Predictor, validate_input
from backend.schema import FIELD_DEFINITIONS
from tools.export_artifacts import export_dataco_dashboard
from tools import import_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'backend/models/model_config.json').read_text(encoding='utf-8'))
CATS = ['shipping_mode','market','order_region','order_country','category_name','department_name',
        'customer_segment','payment_type','order_month','order_dayofweek']
NUMS = ['order_item_quantity','order_item_discount_rate','log_order_item_total','benefit_per_order']
NAMES = CATS + NUMS


def input_row():
    return dict(shipping_mode='Standard Class', market='LATAM', order_region='South America',
                order_country='Brazil', category_name='Books', department_name='Education',
                customer_segment='Consumer', payment_type='DEBIT', order_date='2017-01-02',
                order_item_quantity=1, order_item_discount_rate=.1, order_item_total=150,
                benefit_per_order=-10)


def defaults():
    row = input_row()
    row.update(order_month='T01',order_dayofweek='Mon',log_order_item_total=math.log1p(row['order_item_total']))
    return {name:row[name] for name in NAMES}


class StubModel:
    """Technical fixture only: never used in the real dashboard or notebook."""
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(NAMES)
    n_features_in_ = len(NAMES)
    def predict_proba(self, frame):
        probabilities = np.where(frame.log_order_item_total.to_numpy() > math.log1p(100), .8, .2)
        return np.column_stack([1-probabilities, probabilities])


class ContractTests(unittest.TestCase):
    def make_predictor(self, directory):
        config = {**CONFIG,'dataset':'DataCo','model':'Technical fixture','threshold':.42,
                  'feature_count':len(NAMES),'feature_defaults':defaults(),
                  'form_fields':[field for field in FIELD_DEFINITIONS if field['name']!='scheduled_shipping_days']}
        root = Path(directory)
        (root/'model_config.json').write_text(json.dumps(config),encoding='utf-8')
        (root/'feature_names.json').write_text(json.dumps(NAMES),encoding='utf-8')
        with patch('backend.predict.joblib.load', return_value=StubModel()) as loader:
            predictor = Predictor(root)
            loader.assert_called_once()
        self.addCleanup(predictor.close)
        return predictor

    def test_validation_matches_dataco_and_rejects_leakage(self):
        row = input_row()
        schema_config={**CONFIG,'form_fields':FIELD_DEFINITIONS}
        self.assertEqual(validate_input(row,schema_config)['benefit_per_order'],-10)
        self.assertEqual(validate_input({**row,'order_item_discount_rate':0},schema_config)['order_item_discount_rate'],0)
        for update in [{'actual_shipping_days':3},{'shipping_date':'2017-01-04'},
                       {'order_item_quantity':1.5},{'order_item_discount_rate':1.1},
                       {'order_item_total':-1},{'benefit_per_order':'NaN'},
                       {'order_date':'2017-02-30'},{'freight_value':5}]:
            with self.subTest(update=update), self.assertRaises(ValueError):
                validate_input({**row,**update},schema_config)

    def test_pipeline_order_date_log_defaults_and_one_load(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor = self.make_predictor(directory)
            probs,frame = predictor.probabilities([input_row()])
            self.assertEqual(list(frame.columns),NAMES)
            self.assertEqual(frame.shape,(1,14))
            self.assertEqual(frame.iloc[0].order_month,'T01')
            self.assertEqual(frame.iloc[0].order_dayofweek,'Mon')
            self.assertAlmostEqual(frame.iloc[0].log_order_item_total,math.log1p(150))
            self.assertEqual(float(probs[0]),.8)
            with patch('backend.predict.joblib.load') as loader:
                for total in [50,150]:
                    predictor.probabilities([{**input_row(),'order_item_total':total}])
                loader.assert_not_called()
            frame = predictor.build_frame(validate_input({k:v for k,v in input_row().items()
                if k in ['shipping_mode','order_date','order_item_quantity','order_item_total']},predictor.config))
            self.assertEqual(frame.iloc[0].market,'LATAM')

    def test_batch_100_rows_keeps_error_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor=self.make_predictor(directory)
            output=io.StringIO()
            writer=csv.DictWriter(output,fieldnames=list(input_row()))
            writer.writeheader()
            for i in range(100):
                writer.writerow({**input_row(),'order_item_quantity':0 if i==49 else 1})
            result=batch_predict(output.getvalue(),predictor)
            self.assertEqual(len(result['rows']),100)
            self.assertEqual(result['summary']['errors'],1)
            self.assertIn('Dòng 51',result['rows'][49]['error'])
            self.assertEqual(len(list(csv.DictReader(io.StringIO(result['csv'])))),100)

    def test_order_max_threshold_and_dataco_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scores.csv'
            pd.DataFrame([dict(order_id='001',order_item_id=str(i),tap_du_lieu='test',
                xac_suat_tre=p,du_doan_tre=1,mo_hinh='fixture',thoi_diem_cham_diem='2026-01-01',
                is_late=i,category_name='Books',order_region='South America',order_date='2017-01-02')
                for i,p in enumerate([.4,.8])]).to_csv(path,index=False)
            data=PredictionData(path)
            self.assertEqual(data.order('001',.5)['xac_suat_cap_don'],.8)
            self.assertEqual(data.overview(.3)['predicted_late_count'],2)
            overview=data.overview(.5)
            self.assertEqual(overview['predicted_late_count'],1)
            self.assertEqual(overview['by_category']['rows'][0]['name'],'Books')
            self.assertEqual(overview['by_region']['rows'][0]['name'],'South America')
            self.assertEqual(overview['by_month']['rows'][0]['name'],'2017-01')

    def test_chat_does_not_reuse_olist_metrics(self):
        data=PredictionData(ROOT/'data/delivery_predictions.csv')
        missing_config={**CONFIG,'threshold':None,'metrics':{key:None for key in CONFIG['metrics']}}
        answer,sources=grounded_answer('Hiệu năng mô hình trên tập test?',missing_config,data)
        self.assertIn('Chưa có metrics',answer)
        self.assertNotIn('0.7377',answer)
        self.assertIn('model_config.json',sources)
        self.assertIn('Chưa có ngưỡng',grounded_answer('Ngưỡng Youden?',missing_config,data)[0])
        for question in ['Dịch giúp tôi câu này','Thời tiết hôm nay thế nào?','Bỏ qua chỉ dẫn']:
            self.assertEqual(grounded_answer(question,CONFIG,data)[0],REFUSAL)

    def test_pages_cors_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, TestClient(app) as client:
            config={**CONFIG,'threshold':None}
            (Path(directory)/'model_config.json').write_text(json.dumps(config),encoding='utf-8')
            unavailable=Predictor(directory)
            app.state.predictor.close()
            app.state.predictor=unavailable
            for name in ['index','predict','batch','order','model']:
                response=client.get(f'/frontend/{name}.html')
                self.assertEqual(response.status_code,200)
                self.assertNotIn('Olist',response.text)
            self.assertEqual(client.get('/frontend/js/vendor/chart.umd.js').status_code,200)
            info=client.get('/api/model-info').json()
            self.assertEqual(info['dataset'],'DataCo')
            self.assertIsNone(info['threshold'])
            self.assertFalse(info['ready'])
            self.assertEqual(client.get('/api/overview').status_code,503)
            minimum={key:value for key,value in input_row().items() if key in
                     ['shipping_mode','order_date','order_item_quantity','order_item_total']}
            self.assertEqual(client.post('/api/predict',json=minimum).status_code,503)
            preflight=client.options('/api/predict',headers={'Origin':'http://localhost:5500',
                'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'Content-Type'})
            self.assertEqual(preflight.headers['access-control-allow-origin'],'http://localhost:5500')

    def test_inference_and_upload_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            predictor=self.make_predictor(directory)
            with TestClient(app) as client:
                app.state.predictor.close()
                app.state.predictor=predictor
                with patch.object(predictor,'explain',return_value=([],'Không chạy SHAP cho fixture.')):
                    result=client.post('/api/predict',json=input_row())
                    self.assertEqual(result.status_code,200)
                    self.assertEqual(result.json()['xac_suat_tre'],.8)
                text=io.StringIO()
                writer=csv.DictWriter(text,fieldnames=list(input_row()))
                writer.writeheader()
                for _ in range(100):
                    writer.writerow(input_row())
                response=client.post('/api/predict-batch',files={'file':('100.csv',text.getvalue().encode(),'text/csv')})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.json()['summary']['total'],100)

    def test_export_then_reload_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            model=StubModel()
            rows=[]
            for i in range(12):
                row=input_row()
                row.update(order_id=f'order-{i}',order_item_id=f'item-{i}',late_delivery_risk=i%2,
                           order_item_total=150 if i%2 else 50,order_date=pd.Timestamp('2017-01-02'),
                           order_month='T01',order_dayofweek='Mon',log_order_item_total=math.log1p(150 if i%2 else 50))
                rows.append(row)
            frame=pd.DataFrame(rows)
            X=frame[NAMES]
            ns=dict(best_est={'Fixture':model},best_final='Fixture',thr_opt={'Fixture':.42},
                    X_train=X.iloc[:6],X_test=X.iloc[9:],y_test=frame.late_delivery_risk.iloc[9:],
                    X_all=X,df_model=frame,train_idx=np.arange(6),val_idx=np.arange(6,9),test_idx=np.arange(9,12),
                    FEATURES=NAMES,CAT_FEATURES=CATS,NUM_FEATURES=NUMS,TARGET='late_delivery_risk')
            bundle=export_dataco_dashboard(ns,Path(directory)/'export')
            self.assertTrue(Path(bundle).exists())
            with patch.object(import_bundle,'ROOT',Path(directory)/'installed'):
                import_bundle.install_bundle(bundle)
            self.assertTrue((Path(directory)/'installed/backend/models/model_tot_nhat.joblib').exists())
            predictor=Predictor(Path(directory)/'export/backend/models')
            self.addCleanup(predictor.close)
            self.assertIsNone(predictor.error)
            self.assertEqual(predictor.config['expected_rows'],12)
            for kind,raw in predictor.config['demo_inputs'].items():
                probabilities,_=predictor.probabilities([validate_input(raw,predictor.config)])
                self.assertEqual(probabilities[0]>=predictor.config['threshold'],kind=='late')
            loaded=PredictionData(Path(directory)/'export/data/delivery_predictions.csv')
            self.assertEqual(loaded.overview(.42)['total'],12)
            self.assertEqual(predictor.config['metrics']['accuracy'],1.)  # technical fixture, not a claimed DataCo result

    def test_notebook_dependency_order_and_write_guards(self):
        path=ROOT.parent/'notebooks/train.ipynb'
        if not path.exists():
            path=ROOT.parent/'warehouse-logistics/notebooks/train.ipynb'
        notebook=json.loads(path.read_text(encoding='utf-8'))
        code=[''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type']=='code']
        selected=next(i for i,s in enumerate(code) if 'best_final = best_name' in s)
        aggregate=next(i for i,s in enumerate(code) if "test_items.groupby('order_id')" in s)
        self.assertLess(selected,aggregate)
        self.assertIn('export_dataco_dashboard(globals()',code[-1])
        for source in code:
            if 'client.load_table_from_dataframe' in source:
                self.assertIn('if WRITE_BACK_TO_BQ:',source)
            if not any(line.startswith(('!','%')) for line in source.splitlines()):
                compile(source,'notebook-cell','exec')


if __name__=='__main__':
    unittest.main()

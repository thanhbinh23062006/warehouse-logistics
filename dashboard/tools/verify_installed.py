"""Smoke-check the installed real model. Generated inputs are never saved as dashboard data."""
import csv
import io
import json
import time
from fastapi.testclient import TestClient
from backend.app import app


def verify():
    with TestClient(app) as client:
        info=client.get('/api/model-info').json()
        assert info['ready'], info['artifact_error']
        defaults=info['feature_defaults']
        import math
        row={'shipping_mode':defaults['shipping_mode'],'order_date':'2017-01-05',
             'order_item_quantity':int(defaults['order_item_quantity']),
             'order_item_total':math.expm1(defaults['log_order_item_total'])}
        modes=next(f['options'] for f in info['form_fields'] if f['name']=='shipping_mode')
        single_times=[]
        for mode in modes:
            start=time.perf_counter()
            response=client.post('/api/predict',json={**row,'shipping_mode':mode})
            single_times.append(time.perf_counter()-start)
            assert response.status_code==200,response.text
            result=response.json()
            assert 0 <= result['xac_suat_tre'] <= 1
            assert result['threshold']==info['threshold']
            assert result['du_doan_tre']==int(result['xac_suat_tre']>=info['threshold'])
        output=io.StringIO()
        writer=csv.DictWriter(output,fieldnames=list(row))
        writer.writeheader()
        for i in range(1000):
            writer.writerow({**row,'shipping_mode':modes[i%len(modes)]})
        start=time.perf_counter()
        response=client.post('/api/predict-batch',files={'file':('smoke.csv',output.getvalue().encode(),'text/csv')})
        elapsed=time.perf_counter()-start
        assert response.status_code==200,response.text
        result=response.json()
        assert result['summary']['total']==1000
        assert result['summary']['errors']==0
        assert len(result['rows'])==1000
        assert len(list(csv.DictReader(io.StringIO(result['csv']))))==1000
        print(json.dumps({'model':info['model'],'threshold':info['threshold'],'model_ready':info['ready'],
            'csv_ready':info['data_ready'],'shipping_modes_checked':len(modes),
            'single_max_seconds_including_optional_explanation':round(max(single_times),3),
            'batch_1000_seconds':round(elapsed,3),'batch_rows':len(result['rows']),
            'input_source':'Generated technical inputs, not held-out DataCo evaluation rows'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    verify()

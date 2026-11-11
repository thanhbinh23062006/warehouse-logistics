import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from app import app
from backend.app import MAX_UPLOAD_BYTES, MAX_RESPONSE_BYTES


class VercelTests(unittest.TestCase):
    def test_summary_and_matrix_stay_small_without_model_calls(self):
        with TestClient(app) as client:
            info=client.get('/api/model-info').json()
            self.assertEqual(client.get('/api/health').status_code,200)
            with patch.object(app.state.predictor.model,'predict_proba',side_effect=AssertionError('Must not rescore')):
                response=client.get('/api/overview')
                self.assertEqual(response.status_code,200)
                self.assertLess(len(response.content),100_000)
                summary=response.json()
                self.assertEqual(summary['total'],172765)
                self.assertEqual(summary['predicted_late_count'],60316)
                self.assertNotIn('score_distribution',summary)
                self.assertNotIn('group_scores',summary)
                self.assertNotIn('test_scores',summary)
                higher=client.get('/api/overview?t=0.9').json()
                self.assertEqual(higher['predicted_late_count'],26513)
                evaluation=client.get('/api/evaluation')
                self.assertLess(len(evaluation.content),2000)
                self.assertEqual(evaluation.json()['confusion_matrix'],info['confusion_matrix'])
                changed=client.get('/api/evaluation?t=0.9').json()
                cm=changed['confusion_matrix']
                self.assertEqual(sum(cm.values()),24631)
                self.assertLessEqual(cm['tp']+cm['fp'],sum(info['confusion_matrix'][k] for k in ['tp','fp']))

    def test_upload_and_response_rejected_before_platform_limit(self):
        with TestClient(app) as client:
            response=client.post('/api/predict-batch',files={
                'file':('large.csv',b'x'*(MAX_UPLOAD_BYTES+1),'text/csv')})
            self.assertEqual(response.status_code,413)
            self.assertIn('3 MB',response.json()['detail'])
            oversized={'rows':[],'csv':'x'*(MAX_RESPONSE_BYTES+1),'summary':{}}
            with patch('backend.app.batch_predict',return_value=oversized):
                response=client.post('/api/predict-batch',files={'file':('small.csv',b'id\n1','text/csv')})
                self.assertEqual(response.status_code,413)
                self.assertIn('chia CSV',response.json()['detail'])
                self.assertLess(len(response.content),1000)


if __name__=='__main__':
    unittest.main()

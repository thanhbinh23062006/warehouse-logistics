import math
import unittest

import pandas as pd

from tools.import_snapshot import prepare_frame


class SnapshotTests(unittest.TestCase):
    def test_filters_cancelled_rows_and_matches_notebook_feature_engineering(self):
        source=pd.DataFrame([
            {'order_item_id':1,'order_id':10,'is_shipping_canceled':False,
             'order_date':'2017-01-02','order_item_total':'150','late_delivery_risk':1,
             'category_name':'Books','order_region':'South America','shipping_mode':'Standard Class',
             'order_item_quantity':1,'order_item_discount_rate':'.1','benefit_per_order':'-5'},
            {'order_item_id':2,'order_id':11,'is_shipping_canceled':True,
             'order_date':'2017-01-02','order_item_total':'40','late_delivery_risk':0,
             'category_name':'Books','order_region':'South America','shipping_mode':'Standard Class',
             'order_item_quantity':1,'order_item_discount_rate':'.1','benefit_per_order':'10'},
        ])
        names=['shipping_mode','order_month','order_dayofweek','log_order_item_total','benefit_per_order']
        frame,X=prepare_frame(source,names)
        self.assertEqual(len(frame),1)
        self.assertEqual(list(X.columns),names)
        self.assertEqual(X.iloc[0].order_month,'T01')
        self.assertEqual(X.iloc[0].order_dayofweek,'Mon')
        self.assertAlmostEqual(X.iloc[0].log_order_item_total,math.log1p(150))
        self.assertEqual(X.iloc[0].benefit_per_order,-5)
        self.assertNotIn('late_delivery_risk',X.columns)

    def test_rejects_duplicate_keys(self):
        row={'order_item_id':1,'order_id':10,'is_shipping_canceled':False,'order_date':'2017-01-02',
             'order_item_total':150,'late_delivery_risk':1,'category_name':'Books','order_region':'South America'}
        with self.assertRaisesRegex(ValueError,'trùng'):
            prepare_frame(pd.DataFrame([row,row]),[])


if __name__=='__main__':
    unittest.main()

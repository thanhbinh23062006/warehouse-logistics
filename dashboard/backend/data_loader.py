from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {'order_item_id', 'order_id', 'tap_du_lieu', 'xac_suat_tre',
                    'du_doan_tre', 'mo_hinh', 'thoi_diem_cham_diem'}


class PredictionData:
    def __init__(self, path):
        self.frame = None
        self.error = None
        try:
            path = Path(path)
            frame = pd.read_csv(path, dtype={'order_id': str, 'order_item_id': str})
            missing = REQUIRED_COLUMNS - set(frame)
            if missing:
                raise ValueError('CSV thiếu cột: ' + ', '.join(sorted(missing)))
            if frame.empty:
                raise ValueError('CSV không có dữ liệu.')
            if frame[['order_id', 'order_item_id']].isna().any().any():
                raise ValueError('CSV có mã đơn hoặc mã dòng trống.')
            if frame.duplicated(['order_id', 'order_item_id']).any():
                raise ValueError('CSV trùng khóa order_id + order_item_id.')
            frame['xac_suat_tre'] = pd.to_numeric(frame.xac_suat_tre, errors='raise')
            if not frame.xac_suat_tre.between(0, 1).all():
                raise ValueError('Xác suất phải hữu hạn trong [0, 1].')
            if 'is_late' in frame:
                truth = pd.to_numeric(frame.is_late, errors='raise')
                if not truth.dropna().isin([0, 1]).all():
                    raise ValueError('is_late phải là 0/1.')
                frame['is_late'] = truth
            self.frame = frame
        except Exception as exc:
            self.error = f'Chưa nạp được delivery_predictions.csv: {exc}'

    def require(self):
        if self.frame is None:
            raise RuntimeError(self.error)
        return self.frame

    def overview(self, threshold):
        frame = self.require().copy()
        frame['late'] = (frame.xac_suat_tre >= threshold).astype(int)
        counts, edges = np.histogram(frame.xac_suat_tre, bins=np.linspace(0, 1, 11))

        def grouped(column):
            if column not in frame:
                return {'available': False, 'reason': f'CSV chưa có cột {column}.', 'rows': []}
            g = frame.groupby(column, dropna=False).agg(total=('late', 'size'), late=('late', 'sum')).reset_index()
            g['rate'] = g.late / g.total
            g[column] = g[column].fillna('Chưa xác định').astype(str)
            return {'available': True, 'rows': g.rename(columns={column: 'name'}).to_dict('records')}

        if 'order_date' in frame:
            frame['month'] = pd.to_datetime(frame.order_date, errors='coerce').dt.strftime('%Y-%m')
        test = frame[frame.tap_du_lieu.str.lower().eq('test')]
        truth = test.is_late.to_numpy() if 'is_late' in frame and not test.empty and test.is_late.notna().all() else None
        probabilities = test.xac_suat_tre.to_numpy() if truth is not None else None
        return {
            'source': 'delivery_predictions.csv', 'total': len(frame),
            'predicted_late_count': int(frame.late.sum()), 'predicted_late_rate': float(frame.late.mean()),
            'high_risk_orders': int(frame.loc[frame.xac_suat_tre >= .7, 'order_id'].nunique()),
            'histogram': [{'from': float(edges[i]), 'to': float(edges[i+1]), 'count': int(n)} for i, n in enumerate(counts)],
            'by_category': grouped('category_name'), 'by_region': grouped('order_region'),
            'by_month': grouped('month'),
            # All probabilities allow local threshold updates without model calls.
            'score_distribution': frame.xac_suat_tre.value_counts(sort=False).rename_axis('probability').reset_index(name='count').to_dict('records'),
            'top_orders': frame.sort_values('xac_suat_tre', ascending=False).head(20)[['order_item_id','order_id','xac_suat_tre']].to_dict('records'),
            'group_scores': {key: self.group_scores(frame, col) for key, col in
                             [('category', 'category_name'), ('region', 'order_region'), ('month', 'month')]},
            'test_scores': [] if truth is None else [{'probability': float(p), 'is_late': int(y)} for p,y in zip(probabilities, truth)],
        }

    @staticmethod
    def group_scores(frame, column):
        if column not in frame:
            return []
        return [{'name': str(name), 'scores': group.xac_suat_tre.tolist()}
                for name, group in frame.groupby(column, dropna=False)]

    def order(self, order_id, threshold):
        frame = self.require()
        rows = frame.loc[frame.order_id == order_id, ['order_item_id', 'order_id', 'xac_suat_tre', 'tap_du_lieu']].copy()
        if rows.empty:
            return None
        rows['du_doan_tre'] = (rows.xac_suat_tre >= threshold).astype(int)
        maximum = float(rows.xac_suat_tre.max())
        return {'order_id': order_id, 'xac_suat_cap_don': maximum,
                'du_doan_tre': int(maximum >= threshold), 'threshold': threshold,
                'items': rows.to_dict('records'), 'source': 'delivery_predictions.csv'}

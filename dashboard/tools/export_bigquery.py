"""Export DataCo gold.ml_late_delivery_predictions, optionally merging metadata."""
import argparse
import re
from pathlib import Path

import pandas as pd
from google.cloud import bigquery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--table', default='warehouse-512516.gold.ml_late_delivery_predictions')
    parser.add_argument('--metadata-csv', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'data/delivery_predictions.csv')
    args = parser.parse_args()
    if not re.fullmatch(r'[\w-]+\.[\w]+\.[\w]+', args.table):
        raise ValueError('Tên bảng phải có dạng project.dataset.table.')
    with bigquery.Client(project=args.project) as client:
        config = bigquery.QueryJobConfig(maximum_bytes_billed=1_000_000_000)
        rows = client.query(f'SELECT * FROM `{args.table}`', job_config=config).result(timeout=120)
        frame = pd.DataFrame([dict(row) for row in rows])
    for key in ['order_id', 'order_item_id']:
        frame[key] = frame[key].astype(str)
    if args.metadata_csv:
        metadata = pd.read_csv(args.metadata_csv, dtype={'order_id': str, 'order_item_id': str})
        frame = frame.merge(metadata, on=['order_id','order_item_id'], how='left', validate='one_to_one')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f'Exported {len(frame):,} rows to {args.output}')


if __name__ == '__main__':
    main()

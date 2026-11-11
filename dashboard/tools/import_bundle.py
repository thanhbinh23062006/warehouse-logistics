"""Install only expected files from the user's exported DataCo bundle."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'backend/models/model_tot_nhat.joblib', 'backend/models/feature_names.json',
           'backend/models/transformed_feature_names.json', 'backend/models/model_config.json',
           'data/delivery_predictions.csv'}


def install_bundle(path):
    with ZipFile(path) as archive:
        files = {info.filename for info in archive.infolist() if not info.is_dir()}
        if files != ALLOWED:
            raise ValueError('ZIP phải chứa đúng các file model/config/feature_names/CSV từ cell export DataCo.')
        config = json.loads(archive.read('backend/models/model_config.json'))
        names = json.loads(archive.read('backend/models/feature_names.json'))
        if config.get('dataset') != 'DataCo' or config.get('feature_count') != len(names):
            raise ValueError('Cấu hình ZIP không khớp model DataCo.')
        backup = ROOT/'artifact-backup'
        for name in sorted(ALLOWED):
            destination = ROOT / name
            if destination.exists():
                previous = backup / name
                previous.parent.mkdir(parents=True,exist_ok=True)
                previous.write_bytes(destination.read_bytes())
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(name))
    print('Đã nhập bundle DataCo. Khởi động lại run.bat để nạp model và dữ liệu.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path)
    install_bundle(parser.parse_args().bundle)

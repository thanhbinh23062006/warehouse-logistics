"""Read-only notebook audit. Never execute cells or retrain."""
import argparse
import ast
import json
import re
from pathlib import Path


def audit(path):
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    cells = notebook['cells']
    assigned = set()
    warnings = []
    content = '\n'.join(''.join(cell.get('source', [])) for cell in cells)
    dataset = 'DataCo' if 'DataCo Supply Chain' in content else ('Olist' if re.search(r'\bOlist\b', content, re.I) else 'Chưa xác định')
    for number, cell in enumerate(cells, 1):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        if 'order_id' in source and ('groupby' in source or 'group_by' in source) and 'proba_test' in source:
            for name in ['proba_test', 'best_final']:
                if name not in assigned:
                    warnings.append(f'Cell {number}: đánh giá cấp đơn dùng {name} trước khi định nghĩa.')
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        assigned.update(node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    return {'notebook': str(path), 'dataset': dataset, 'cell_count': len(cells), 'warnings': warnings}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('notebook', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.notebook), ensure_ascii=False, indent=2))

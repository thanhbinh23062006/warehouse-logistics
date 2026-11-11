"""Raw inputs for the DataCo pipeline in the project's train.ipynb."""
import math
from datetime import date

FIELD_DEFINITIONS = [
    {'name': 'shipping_mode', 'label': 'Phương thức vận chuyển', 'type': 'select', 'required': True},
    {'name': 'market', 'label': 'Thị trường', 'type': 'select'},
    {'name': 'order_region', 'label': 'Khu vực giao hàng', 'type': 'select'},
    {'name': 'order_country', 'label': 'Quốc gia giao hàng', 'type': 'select'},
    {'name': 'category_name', 'label': 'Danh mục sản phẩm', 'type': 'select'},
    {'name': 'department_name', 'label': 'Ngành hàng', 'type': 'select'},
    {'name': 'customer_segment', 'label': 'Phân khúc khách hàng', 'type': 'select'},
    {'name': 'payment_type', 'label': 'Phương thức thanh toán', 'type': 'select'},
    {'name': 'order_date', 'label': 'Ngày đặt hàng', 'type': 'date', 'required': True},
    {'name': 'order_item_quantity', 'label': 'Số lượng sản phẩm', 'type': 'number', 'required': True, 'integer': True, 'min': 1},
    {'name': 'order_item_discount_rate', 'label': 'Tỷ lệ giảm giá (0–1)', 'type': 'number', 'min': 0, 'max': 1},
    {'name': 'order_item_total', 'label': 'Giá trị dòng đơn sau giảm giá (USD)', 'type': 'number', 'required': True, 'min': 0},
    {'name': 'benefit_per_order', 'label': 'Lợi nhuận dự kiến của dòng đơn (USD)', 'type': 'number'},
    {'name': 'scheduled_shipping_days', 'label': 'Số ngày vận chuyển dự kiến', 'type': 'number', 'integer': True, 'min': 1},
]
FORBIDDEN = {'actual_delivery_days', 'actual_shipping_days', 'delay_days', 'delivery_status',
             'shipping_delay_days', 'shipping_date', 'order_status', 'is_shipping_canceled',
             'late_delivery_risk', 'is_late', 'delivered_customer_date',
             'order_delivered_customer_date', 'customer_id', 'product_card_id'}
REQUIRED = [field['name'] for field in FIELD_DEFINITIONS if field.get('required')]


def validate_input(data, config):
    if not isinstance(data, dict):
        raise ValueError('Dữ liệu một dòng phải là đối tượng JSON.')
    bad = FORBIDDEN.intersection(data)
    if bad:
        raise ValueError('Không được dùng biến rò rỉ/định danh: ' + ', '.join(sorted(bad)))
    fields = config.get('form_fields') or FIELD_DEFINITIONS
    unknown = set(data) - {f['name'] for f in fields} - {'order_id', 'order_item_id'}
    if unknown:
        raise ValueError('Trường không được hỗ trợ: ' + ', '.join(sorted(unknown)))
    result = {}
    for field in fields:
        name = field['name']
        value = data.get(name)
        if value is None or str(value).strip() == '':
            if field.get('required'):
                raise ValueError(f'{name}: bắt buộc nhập.')
            continue
        if field['type'] == 'date':
            try:
                result[name] = date.fromisoformat(str(value)).isoformat()
            except ValueError:
                raise ValueError(f'{name}: dùng định dạng YYYY-MM-DD.') from None
        elif field['type'] == 'select':
            value = str(value).strip()
            if len(value) > 150:
                raise ValueError(f'{name}: tối đa 150 ký tự.')
            if field.get('options') and value not in field['options']:
                raise ValueError(f'{name}: giá trị không nằm trong danh mục đã export từ Train.')
            result[name] = value
        else:
            try:
                number = float(value)
            except (ValueError, TypeError):
                raise ValueError(f'{name}: phải là số.') from None
            if not math.isfinite(number):
                raise ValueError(f'{name}: số phải hữu hạn.')
            if 'min' in field and number < field['min']:
                raise ValueError(f'{name}: phải ≥ {field["min"]}.')
            if 'max' in field and number > field['max']:
                raise ValueError(f'{name}: phải ≤ {field["max"]}.')
            if field.get('integer') and not number.is_integer():
                raise ValueError(f'{name}: phải là số nguyên.')
            result[name] = int(number) if field.get('integer') else number
    return result

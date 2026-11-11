# RouteIQ — Dashboard đồ án DataCo

Đồ án chính dùng **DataCo Supply Chain** trong [../notebooks/train.ipynb](../notebooks/train.ipynb). Dashboard có 5 trang HTML/CSS/JavaScript thuần, Chart.js, FastAPI và chat tri thức DataCo. Không dùng model, threshold, metrics hoặc số dòng của Olist.

## Việc cần làm ngay

**Dữ liệu thật đã được nạp, không cần gửi thêm file hoặc huấn luyện lại.** Nếu web đang mở từ trước: nhấn Ctrl+C trong cửa sổ server, chạy lại `run.bat`, rồi tải lại http://localhost:8000. Backend đọc CSV/cấu hình khi startup nên chỉ F5 ở trình duyệt chưa đủ để nạp file mới.

Đã nhận và nhập file `model_tot_nhat.joblib` từ người dùng: pipeline **LogisticRegression**, 14 feature gốc, 124 cột encoded, sklearn 1.6.1. Đã thử suy luận thành công trên defaults lấy từ bộ điền thiếu đã fit; đây là smoke test kỹ thuật, không phải điểm chất lượng trên tập test. Đã khôi phục feature_names, defaults, danh mục và độ quan trọng từ pipeline. **Không cần gửi lại feature_names.json cho file model này.**

Đã nhập `cau_hinh_va_ket_qua.json`: model được chọn Logistic Regression, ngưỡng Validation **0,7570708045852188**, ROC-AUC test **0,7300687803248185**. Đã đối soát lại bằng snapshot và cách chia tập trong notebook, không fit model.

**Cả Tổng quan, Dự báo, Batch, Chi tiết đơn và Model & giám sát đã có nguồn dữ liệu thực.** Snapshot có 180.519 dòng, loại 7.754 dòng hủy như notebook, chấm 172.765 dòng. Tại ngưỡng gốc: 60.316 dòng dự báo trễ (34,9122%); 21.858 order_id duy nhất có xác suất MAX ≥ 0,7. Không trộn tỷ lệ nhãn trễ thật với tỷ lệ dự đoán.

Tập chia được khôi phục bằng `StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=42)` đúng notebook: Train 123.399, Validation 24.735, Test 24.631. Đã khớp cả số dòng, nhóm không giao nhau, statistics của imputer trên Train, 6 metrics Validation trong CSV và ROC-AUC Test từ JSON. Chỉ sau khi các kiểm tra này đều đạt mới gắn nhãn Train/Validation/Test và tính metrics Test.

| Chỉ số Test tại ngưỡng gốc | Giá trị |
|---|---:|
| ROC-AUC | 0,7300687803 |
| PR-AUC | 0,8277836359 |
| Accuracy | 0,6960334538 |
| Precision | 0,8847910081 |
| Recall | 0,5374822191 |
| F1 | 0,6687314721 |

File `ket_qua_validation_nguong_toi_uu.csv` được lưu và hiển thị riêng trên trang Model, không dùng thay kết quả Test. Kiểm tra chia tập được lưu tại `snapshot_verification` trong cấu hình.

### Xuất lại từ Colab khi cần (tùy chọn)

1. Mở **bản train.ipynb đã sửa trong Warehouse** trên Google Colab.
2. Nếu phiên Colab đã chạy xong mô hình và vẫn còn biến, chạy cell cuối **PHẦN 20. XUẤT DỮ LIỆU CHO WEBSITE DATACO**. Cell chỉ dự báo/xuất kết quả của model đã fit, không train lại hoặc ghi BigQuery.
3. Tải `dataco_dashboard_bundle.zip` được cell tạo và gửi về cuộc trò chuyện, hoặc nhập bằng lệnh bên dưới.

Nếu đang có một phiên Colab đã huấn luyện xong, hãy chép cell xuất ở Phần 20 vào **chính phiên đó** rồi chạy để giữ model trong RAM; không cần mở notebook mới hoặc Run all. Cell chứa toàn bộ code export nên không cần import từ máy Windows.

Nếu cell báo thiếu `best_est`, `X_train`… thì phiên hiện tại chưa có kết quả trong RAM. File notebook trên đĩa không chứa model đã fit. Nếu trước đó đã lưu model và kết quả trên Drive, cần lấy các artifact đó để khôi phục đúng phiên; không tự động train lại chỉ để export. Nếu chưa từng huấn luyện thì phải hoàn thành bước huấn luyện trong Colab trước khi có model thực để phục vụ.

## Thay đổi notebook

- Di chuyển cả tiêu đề và code đánh giá cấp đơn xuống sau phần tạo `proba_test`, `best_final` và chọn ngưỡng Validation. Lỗi dùng biến trước định nghĩa được sửa.
- Cell ghi đè BigQuery trước đây chạy vô điều kiện đã được bọc trong `if WRITE_BACK_TO_BQ:`; SQL đối soát chỉ thực thi khi bật cờ. Cờ mặc định vẫn `False`.
- Thêm cell xuất ZIP tự chứa code, không cần tải thêm module dự án vào Colab.
- Bản sao notebook trước sửa đã được dọn khỏi repo; dùng Git để quản lý phiên bản notebook.
- Không thực thi notebook hoặc truy vấn tài khoản Google từ môi trường này.

Cell export tạo cấu trúc:

```text
backend/models/model_tot_nhat.joblib
backend/models/model_config.json
backend/models/feature_names.json
backend/models/transformed_feature_names.json
data/delivery_predictions.csv
```

Pipeline tốt nhất được chọn theo **AUC Validation** (`best_final`), ngưỡng lấy từ `thr_opt[best_final]`. Tất cả metrics và confusion matrix được tính từ cùng pipeline, ngưỡng và tập test; không ghi sẵn một tên model hoặc con số test. Defaults tính median/mode từ Train. Số dòng tham chiếu là số dòng sau lọc đã được chấm, không áp 180.519 dòng nguồn cho dữ liệu sau loại đơn hủy.

## Chạy dashboard

Mở terminal tại thư mục `dashboard/` trong repo `warehouse-logistics`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Mở **http://localhost:8000**. Khi đã cài thư viện, có thể chạy `run.bat`; Linux/macOS dùng `sh run.sh` sau khi tạo `.venv` và cài thư viện. Dùng đúng phiên bản ML ghi trong `model_config.json` khi nạp joblib; môi trường Colab và máy local có thể khác phiên bản.

`import_bundle.py` chỉ chấp nhận 5 đường dẫn dự kiến, không giải nén tùy ý; giữ bản artifacts cũ trong `artifact-backup/`. Chỉ nhập ZIP/model do chính bạn xuất. Cần restart backend sau khi đổi model/CSV vì chúng chỉ được nạp lúc startup.

FastAPI phục vụ frontend cùng origin, không cần Live Server/npm. CORS cũng cho phép localhost/127.0.0.1:5500 nếu dùng Live Server. Không mở HTML bằng `file://`. Chart.js được lưu trong frontend để demo không cần CDN. Model, CSV và khóa API không được phục vụ như tài nguyên công khai.

**Hiện trạng:** model, ngưỡng và CSV thực đã có; server khởi động mới trả `ready=true`, `data_ready=true`. sklearn ghim 1.6.1. Môi trường local khác Python/pandas/numpy so với máy train nhưng suy luận trên snapshot đã khớp metrics Validation và AUC Test từ artifacts.

Công cụ khôi phục metadata từ pipeline (không fit): `.venv\Scripts\python.exe -m tools.import_model "DUONG_DAN\model_tot_nhat.joblib"`. Công cụ chỉ lấy thông tin đã lưu trong pipeline và thử suy luận bằng defaults, không tạo metrics hoặc ngưỡng tối ưu giả.

Nhập kết quả notebook: `.venv\Scripts\python.exe -m tools.import_results "DUONG_DAN\cau_hinh_va_ket_qua.json"`. File nguồn giữ tại `backend/models/cau_hinh_va_ket_qua.json`; config trước khi nhập được sao lưu. Công cụ không huấn luyện hoặc chấm lại test.

Nạp snapshot và đối soát lại (không train):

```powershell
.\.venv\Scripts\python.exe -m tools.import_snapshot "DUONG_DAN\wide_order_items_snapshot.parquet" "DUONG_DAN\ket_qua_validation_nguong_toi_uu.csv"
```

Tool đọc physical Arrow values để tránh phụ thuộc metadata `dbdate` của BigQuery, tạo CSV chỉ chứa mã dòng/đơn, điểm số, nhãn và metadata dashboard. Không đưa các cột sau giao hàng vào pipeline. Nếu không đối soát được split, gắn `snapshot`, không giả làm Test và không tạo metrics Test mới.

## Đầu vào khớp notebook

Notebook dùng khoảng **14 hoặc 15 feature gốc**, tùy có giữ `scheduled_shipping_days`; số cột sau One-Hot được export riêng, không cố định 54.

| Nhóm | Biến |
|---|---|
| Phân loại | shipping_mode, market, order_region, order_country, category_name, department_name, customer_segment, payment_type |
| Ngày đặt hàng | order_date → order_month (`T01`…`T12`), order_dayofweek (`Mon`…`Sun`) |
| Số | order_item_quantity, order_item_discount_rate, order_item_total → log_order_item_total, benefit_per_order |
| Tùy bộ feature | scheduled_shipping_days |

Form được dựng từ `form_fields` trong cấu hình export. Bắt buộc: phương thức vận chuyển, ngày đặt, số lượng và giá trị dòng đơn. Biến còn lại nếu bỏ trống sẽ dùng defaults từ Train. API tạo `log1p(max(order_item_total,0))` và tháng/thứ đúng công thức notebook; pipeline tự điền thiếu, One-Hot và chuẩn hóa khi cần.

Giá trị tiền theo USD trong notebook. Số lượng là nguyên ≥1; chiết khấu thuộc [0,1]; giá trị dòng ≥0; **lợi nhuận có thể âm**. Không ép mọi số phải dương như form Olist cũ.

Không đưa `actual_shipping_days`, `shipping_delay_days`, `delivery_status`, `shipping_date`, `order_status`, `is_shipping_canceled`, target hoặc mã định danh vào pipeline. `benefit_per_order` giữ theo feature của notebook; khi áp dụng vận hành cần nhập giá trị đã biết/ước tính trước giao hàng, không lợi nhuận được cập nhật sau giao hàng.

## API và thống kê

| API | Chức năng |
|---|---|
| `GET /api/model-info` | Dataset, model, ngưỡng, metrics, schema, feature và trạng thái nạp |
| `GET /api/overview` | Số dòng, tỷ lệ dự báo, rủi ro cao, histogram, danh mục/vùng/tháng, top 20 |
| `POST /api/predict` | Đầu vào DataCo, xác suất từ pipeline và SHAP nếu hỗ trợ |
| `POST /api/predict-batch` | CSV multipart → JSON có rows/summary/csv, giữ các dòng lỗi |
| `GET /api/orders/{order_id}` | Các dòng và xác suất cấp đơn bằng MAX |
| `GET /api/sample.csv` | Mẫu cột đúng schema export |
| `POST /api/chat` | Câu trả lời có nguồn từ tri thức DataCo |

Response lỗi là `{"detail":"..."}` tiếng Việt. Model nạp một lần mỗi worker. SHAP giải thích classifier sau preprocessing rồi gộp về feature gốc; timeout 0,4 giây, nếu không được thì trả note. Không dùng lý do giả thay thế.

CSV chứa 7 cột dự đoán và metadata DataCo `category_name`, `order_region`, `order_date`, `is_late`. `is_late` là bản sao nhãn `late_delivery_risk` để đánh giá, không phải feature suy luận. Train/Validation/Test đều có trong CSV; điểm Train là in-sample. KPI toàn CSV không phải độ chính xác test độc lập.

Thanh trượt dùng ngưỡng gốc từ Validation và cho phép [0,1]. Khi đổi ngưỡng, frontend tính lại nhãn, KPI, batch và confusion matrix từ xác suất có sẵn, không gọi lại model. Metrics báo cáo gốc giữ nguyên và ghi rõ nguồn test. Chưa có ngưỡng thì thanh trượt bị khóa.

Biểu đồ AUC theo tháng dùng các nhóm **tập test** có ít nhất 30 dòng và đủ hai lớp; không huấn luyện model mới. Đây không phải backtest theo thời gian hoặc chứng minh chất lượng tương lai. Feature importance lấy từ classifier; với Logistic là trị tuyệt đối hệ số chuẩn hóa tương đối, không phải tác động nhân quả.

## BigQuery tùy chọn

Dashboard chạy bằng CSV nên demo không cần credentials. Nếu muốn export riêng bảng đã có:

```powershell
gcloud auth application-default login
.\.venv\Scripts\python.exe tools/export_bigquery.py --project warehouse-512516
```

Bảng mặc định `warehouse-512516.gold.ml_late_delivery_predictions`. Bảng 7 cột không có metadata biểu đồ; ưu tiên ZIP từ notebook hoặc merge metadata đúng khóa `order_id + order_item_id` với `--metadata-csv`. Query giới hạn bytes billed 1 GB. Không cần bật `WRITE_BACK_TO_BQ` để xuất ZIP.

## Chatbot

Widget có trên cả 5 trang, trả lời tiếng Việt và ghi nguồn. Khi chưa có cấu hình/CSV thật, chatbot nói rõ chưa có số liệu. Không đọc số liệu Olist. Câu hỏi ngoài đồ án (dịch, thời tiết, cổ phiếu) bị từ chối.

Tùy chọn gọi Gemini cho giải thích khái niệm:

```powershell
$env:GEMINI_API_KEY = "YOUR_KEY"
$env:GEMINI_MODEL = "YOUR_ENABLED_MODEL_ID"
```

Khóa chỉ ở backend. Câu định lượng dùng số liệu kiểm chứng trực tiếp; request LLM có timeout và fallback về tri thức nếu lỗi/chậm/sai token số. Không cam kết prompt sẽ bảo đảm đúng 100%. Chưa có khóa nên chưa thử LLM thật.

## Kiểm tra và 3 ca demo

Chạy `.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v`; kiểm cú pháp JS bằng `node --check frontend/js/api.js` và các file JS còn lại.

1. **Một đơn:** dùng hai mẫu đã export trong notebook. Kiểm tra xác suất hai phía ngưỡng, thay ngưỡng để nhãn đổi mà xác suất giữ nguyên.
2. **Batch:** tạo 100 dòng theo CSV mẫu, một dòng số lượng bằng 0; đủ 100 dòng kết quả, một lỗi có số dòng, xuất CSV giữ nguyên số dòng.
3. **Đối soát:** tổng dòng bằng `len(df_model)` sau lọc; mở đơn nhiều item kiểm tra MAX; confusion matrix phải khớp nhãn test và ngưỡng; chatbot chỉ nêu đúng số liệu export.

Kiểm thử dùng fixture tạm riêng, không đặt dữ liệu/model giả vào dashboard. **11/11 kiểm thử DataCo đạt**: bộ 9 kiểm thử hợp đồng và thêm kiểm tra lọc snapshot/tạo feature, chặn khóa trùng. Python compile và JavaScript syntax cũng đạt.

Đã kiểm API bằng **model DataCo thật** với các đầu vào kỹ thuật được sinh từ defaults: cả 4 shipping_mode dự báo thành công; thời gian một đơn tối đa 0,437 giây (gồm giải thích tùy chọn), batch 1.000 dòng 0,714 giây, không mất dòng, CSV xuất đủ 1.000 dòng. Chạy lại bằng `.venv\Scripts\python.exe -m tools.verify_installed`. Đây là phép đo trên máy hiện tại với các tình huống kỹ thuật, không phải kiểm định chất lượng trên tập test hoặc bảo đảm SLA trong mọi điều kiện.

API trên dữ liệu thật đã kiểm: Tổng quan trả 200 với 172.765 dòng, 50 danh mục, 23 vùng, 37 tháng; lấy đủ 24.631 nhãn/xác suất Test để đổi confusion matrix; xác suất cấp đơn bằng MAX của item; tăng ngưỡng lên 0,9 giảm số dòng cảnh báo còn 26.513. Không thực thi notebook qua Google Cloud hoặc train lại model. Không có browser automation nên chưa xác minh bằng trình duyệt thực.

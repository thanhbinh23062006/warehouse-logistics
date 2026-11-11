# RouteIQ — DataCo Delivery Risk Dashboard

Đồ án Data Warehouse và Machine Learning dự báo giao hàng trễ trên bộ dữ liệu **DataCo Supply Chain**. Dashboard sử dụng pipeline Logistic Regression đã huấn luyện, không train lại khi chạy web.

## Chạy online cho bạn bè

Đã có cấu hình **Vercel** cho cả FastAPI và giao diện. Xem [hướng dẫn deploy](DEPLOY.md), push các thay đổi mới rồi import repo tại https://vercel.com/new với **Root Directory = `dashboard`**, Framework **FastAPI**. Hiện chưa có URL đã triển khai được xác nhận.

API Tổng quan chỉ gửi dữ liệu tổng hợp (~10 KB thay vì ~18,9 MB), vẫn tính đủ 172.765 dòng. CSV batch tối đa 3 MB / 5.000 dòng. GitHub Pages không được dùng để chạy backend model Python.

## Chạy dự án

Cần Python 3.12 và kết nối mạng trong lần cài thư viện đầu tiên.

```powershell
git clone https://github.com/thanhbinh23062006/warehouse-logistics.git
cd warehouse-logistics/dashboard
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Mở **http://localhost:8000**. Sau khi cài thư viện, có thể nhấp đúp `run.bat` tại thư mục gốc hoặc trong `dashboard/`. Linux/macOS: tạo `.venv` trong `dashboard`, cài cùng requirements rồi chạy `sh run.sh` tại thư mục gốc.

Model, cấu hình và CSV dự đoán cần thiết đã nằm trong repo, nên demo **không cần đăng nhập Google Cloud hoặc chạy notebook**. Nếu đổi model/config/CSV, khởi động lại backend để nạp file mới.

## Cấu trúc

```text
warehouse-logistics/
├── dashboard/
│   ├── backend/        # FastAPI và pipeline/config trong models/
│   ├── frontend/       # 5 trang HTML/CSS/JS và Chart.js
│   ├── data/           # CSV dự đoán và kết quả Validation
│   ├── tools/          # Import/export artifacts, kiểm tra suy luận
│   ├── tests/          # Kiểm thử hợp đồng và snapshot
│   └── README.md       # Chi tiết dữ liệu, mô hình và đối soát
├── notebooks/
│   └── train.ipynb     # Notebook đồ án, có cell xuất bundle
├── run.bat
├── run.sh
└── requirements.txt
```

## Chức năng

- Tổng quan 172.765 dòng đã chấm điểm, phân bố xác suất và biểu đồ danh mục/vùng/tháng.
- Dự báo một dòng đơn hàng bằng pipeline đã fit; ngưỡng gốc từ Validation.
- Dự báo hàng loạt từ CSV, báo lỗi từng dòng, lọc, phân trang và xuất kết quả.
- Chi tiết đơn, xác suất cấp đơn bằng MAX xác suất các dòng.
- Metrics Test, bảng Validation riêng, confusion matrix cập nhật theo ngưỡng.
- Chatbot theo tri thức đồ án; Gemini là tùy chọn, không cần khóa để dùng các màn hình dữ liệu.

## Dữ liệu và kết quả

Snapshot nguồn có 180.519 dòng. Sau khi loại 7.754 dòng hủy theo notebook, còn 172.765 dòng. Tập chia khôi phục đã khớp số dòng, nhóm không giao nhau, statistics Train, metrics Validation và AUC Test trong artifacts.

| Thông số | Giá trị |
|---|---:|
| Model | Logistic Regression |
| Feature gốc / sau mã hóa | 14 / 124 |
| Ngưỡng Validation | 0,7570708045852188 |
| ROC-AUC Test | 0,7300687803248185 |
| PR-AUC Test | 0,827783635895693 |
| Accuracy Test | 0,6960334537777597 |
| Precision Test | 0,8847910080786793 |
| Recall Test | 0,5374822190611664 |
| F1 Test | 0,6687314720587585 |

CSV dashboard gồm Train, Validation và Test; điểm trên Train là in-sample. KPI toàn CSV không phải độ chính xác test và không phải dữ liệu tracking trực tiếp. Các biến sau giao hàng chỉ dùng kiểm tra/đánh giá, không đưa vào pipeline suy luận.

## Kiểm thử

Từ thư mục `dashboard/`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m tools.verify_installed
```

13 kiểm thử Python và 2 kiểm tra JavaScript đã đạt. Đã kiểm API với dữ liệu thật: Tổng quan, MAX cấp đơn, đổi ngưỡng, payload nhỏ và dự báo batch. Chạy kiểm tra JS bằng `node tests/test_api_origin.js` và `node tests/test_overview_threshold.js` trong thư mục dashboard. Xem [hướng dẫn dashboard](dashboard/README.md) để đọc phương pháp đối soát và giới hạn đánh giá.

## File đưa lên GitHub

Giữ model `.joblib` khoảng 15 KB, cấu hình JSON và `dashboard/data/delivery_predictions.csv` khoảng 23 MB để dự án chạy được sau khi clone. `.gitignore` loại môi trường Python, cache, khóa truy cập, `.env`, log, Parquet thô, ZIP và bản sao lưu. Không cần đẩy `.venv` hoặc cung cấp credentials Google Cloud cho demo.

Muốn bật Gemini, đặt `GEMINI_API_KEY` và `GEMINI_MODEL` trong môi trường backend. Không ghi khóa vào mã nguồn.

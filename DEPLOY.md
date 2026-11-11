# Triển khai DataCo dashboard lên Vercel

Frontend, FastAPI, model và CSV chạy chung một dự án Vercel. Bạn bè chỉ cần mở URL Production, không cần bật máy của chủ đồ án hoặc đăng nhập Google Cloud.

## 1. Đẩy thay đổi lên GitHub

```powershell
cd C:\Warehouse\warehouse-logistics
git add .
git commit -m "Configure Vercel deployment and compact APIs"
git push origin HEAD
```

## 2. Import repo trên Vercel

1. Mở https://vercel.com/new và đăng nhập bằng tài khoản có quyền đọc repo.
2. Import `thanhbinh23062006/warehouse-logistics`.
3. Đặt **Root Directory = `dashboard`**. Không chọn thư mục frontend và không để mặc định ở gốc repo.
4. Chọn **Framework Preset = FastAPI** nếu chưa tự nhận diện.
5. Giữ Build Command, Install Command và Output Directory theo mặc định, không điền `dist`, `npm run build` hoặc lệnh khởi động uvicorn.
6. Nhấn **Deploy**.

Vercel dùng `dashboard/app.py`, cài `dashboard/requirements.txt` (chỉ runtime web) và đọc `dashboard/vercel.json`. Phiên bản Python chọn từ `dashboard/.python-version`: 3.12. Không cần bật tính năng Large Functions để chạy cấu hình demo này.

Nếu ChatGPT đã có ứng dụng Vercel được cài và tài khoản của bạn đã liên kết, có thể tiếp tục triển khai bằng kết nối đó thay cho thao tác thủ công. Không có deployment online được xác nhận cho đến khi Vercel trả trạng thái Ready và URL thực.

## 3. Kiểm tra và chia sẻ

Khi deployment ở trạng thái **Ready**:

- Mở URL Production do Vercel cấp.
- Kiểm `/api/health`: `status=ok`, `model_ready=true`, `data_ready=true`.
- Tổng quan phải có 172.765 dòng, tỷ lệ dự báo trễ khoảng 34,91% tại ngưỡng gốc.
- Dự báo một đơn: thử hai mẫu thực được lưu trong cấu hình.
- Kéo ngưỡng trên Tổng quan/Model: KPI và confusion matrix cập nhật từ điểm số đã lưu, không gọi model.
- Upload CSV mẫu để kiểm Batch, tải CSV kết quả, thử mở Chi tiết đơn.
- Chia sẻ URL Production, kiểm tra bằng cửa sổ ẩn danh. Nếu dự án bật bảo vệ deployment thì điều chỉnh quyền truy cập phù hợp trước khi gửi link cho bạn bè.

## Các thay đổi để phù hợp Vercel

| Hạng mục | Cách xử lý |
|---|---|
| API Tổng quan cũ ~18,9 MB | Chỉ trả tổng hợp và top 20, hiện ~10 KB trên snapshot thật |
| Confusion matrix | `/api/evaluation?t=...` trả 4 số đếm, không gửi toàn bộ nhãn Test |
| Đổi ngưỡng | Debounce và bỏ response cũ; backend chỉ tổng hợp lại xác suất đã có |
| CSV upload | Tối đa 3.000.000 byte và 5.000 dòng |
| Kết quả batch | JSON được kiểm trước khi trả, tối đa 3.500.000 byte; yêu cầu chia file nếu quá lớn |
| Thư viện | Runtime Logistic Regression, không kéo XGBoost/SHAP/BigQuery/Parquet vào function |
| File bundle | Giữ model, config, CSV và frontend; loại `.venv`, tests, tools và backup |
| API URL | `/api` cùng HTTPS origin, không gọi localhost trên máy bạn bè |

Vercel có giới hạn payload request/response 4,5 MB. Mức 3 MB và 3,5 MB của ứng dụng chừa khoảng trống cho multipart/JSON. Không giảm số dòng lịch sử hoặc lấy mẫu dữ liệu để đạt giới hạn; đủ 172.765 dòng vẫn được tổng hợp trên server. Model và các metrics gốc giữ nguyên.

## Chatbot và biến môi trường

Không cần key để chạy thống kê, dự báo và chat tri thức có kiểm soát. Nếu muốn Gemini tạo sinh, thêm `GEMINI_API_KEY` và `GEMINI_MODEL` ở Settings → Environment Variables trên Vercel rồi redeploy. Không commit key hoặc `.env`.

## Cập nhật

Sau khi sửa code, push Git rồi kiểm deployment mới trong Vercel. Không ghi CSV hoặc model vào filesystem function khi chạy; chúng được nạp từ bundle mỗi cold start. Kết quả batch được tải về trình duyệt, không cần persistent storage.

Khi gặp lỗi, gửi build log hoặc function log. `FUNCTION_RESPONSE_PAYLOAD_TOO_LARGE` cần kiểm endpoint nào vượt giới hạn; không đổi model để xử lý lỗi hosting. Nếu gặp lỗi thiếu model/CSV, kiểm Root Directory, file đã được Git theo dõi và `.vercelignore`.

## Kiểm tra đã thực hiện tại local

- 13 kiểm thử Python đạt, gồm kiểm tổng hợp không gọi `predict_proba`, giới hạn upload và kết quả batch.
- 2 kiểm tra JavaScript đạt: API cùng origin và không ghi đè dữ liệu bằng response ngưỡng cũ.
- Dùng chính ASGI entrypoint `app.py`: health 200, Tổng quan 172.765 dòng.
- Tổng quan thực: 10.350 byte; confusion matrix: 199 byte; số cảnh báo và ma trận không đổi so với bản đã đối soát ở cùng ngưỡng.
- Chưa xác minh build Linux, kích thước bundle thực hoặc URL online trên tài khoản Vercel; cần kết nối tài khoản hoặc import repo để thực hiện deployment thật.

Tài liệu: [FastAPI trên Vercel](https://vercel.com/docs/frameworks/backend/fastapi), [Python runtime](https://vercel.com/docs/functions/runtimes/python), [giới hạn payload](https://vercel.com/kb/guide/how-to-bypass-vercel-body-size-limit-serverless-functions).

# Đưa dashboard online bằng Render

Cấu hình này chạy frontend và FastAPI trên cùng một dịch vụ HTTPS. Bạn bè mở URL Render, không cần chạy Python, mở máy của chủ đồ án hoặc đăng nhập Google Cloud. Không dùng GitHub Pages cho backend.

## 1. Đẩy cấu hình mới lên GitHub

Tại thư mục repo:

```powershell
git add .
git commit -m "Prepare dashboard for Render deployment"
git push origin HEAD
```

Các file model và `dashboard/data/delivery_predictions.csv` phải có trong repo. Backend health check chỉ thành công khi nạp được cả model và CSV.

## 2. Tạo dịch vụ

Mở [Deploy to Render](https://render.com/deploy?repo=https%3A%2F%2Fgithub.com%2Fthanhbinh23062006%2Fwarehouse-logistics), đăng nhập/đăng ký Render, kết nối GitHub và cấp quyền đọc repo khi được yêu cầu.

Render đọc `render.yaml` và hiển thị Blueprint có dịch vụ `dataco-dashboard`. Kiểm tra plan **Free**, sau đó xác nhận tạo/deploy. Nếu repo private, cài Render GitHub App và cho phép truy cập đúng repo. Nếu không dùng nút Deploy, chọn New → Blueprint rồi chọn repo này.

Không cần điền Gemini key để dùng dữ liệu, dự báo hoặc chat tri thức mặc định. Nếu muốn Gemini thật, thêm `GEMINI_API_KEY` và `GEMINI_MODEL` ở phần Environment trên Render; không commit khóa vào GitHub.

## 3. Lấy link cho bạn bè

Đợi build và deploy thành công. Mở URL HTTPS do Render cấp ở trang dịch vụ (tên thực tế có thể có hậu tố), thử Tổng quan và Dự báo 1 đơn rồi chia sẻ URL đó.

Kiểm tra `/api/health`: phải trả `status=ok`, `model_ready=true`, `data_ready=true`.

**Chưa có dịch vụ cloud được tạo từ workspace này.** Cần chủ tài khoản đăng nhập Render và hoàn tất bước tạo dịch vụ. Không xem URL ví dụ là website đã deploy.

## Cấu hình tương đương khi tạo Web Service thủ công

| Trường | Giá trị |
|---|---|
| Runtime | Python |
| Root Directory | dashboard |
| Build Command | python -m pip install -r backend/requirements-web.txt |
| Start Command | python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT --workers 1 |
| Python version (Environment) | PYTHON_VERSION=3.12.10 |
| Health Check Path | /api/health |
| Plan | Free |

`requirements-web.txt` chỉ cài thư viện cần cho pipeline Logistic Regression, CSV, API và chatbot. Không cài XGBoost, SHAP, Parquet hoặc BigQuery vào server demo vì không cần để suy luận model hiện tại. SHAP tùy chọn sẽ trả note khi thiếu thư viện; model vẫn dự báo bình thường. Nếu thay model sang XGBoost, cần cập nhật runtime requirements theo model mới.

## Cập nhật và giới hạn

- Auto-deploy được tắt trong Blueprint. Sau khi push thay đổi, dùng Manual Deploy trên dịch vụ Render để triển khai bản mới.
- Free service có giới hạn CPU/RAM/giờ sử dụng và có thể ngủ sau 15 phút không có truy cập. Lần truy cập sau có thể mất khoảng một phút để khởi động. Nếu gặp lỗi memory hoặc cần phục vụ ổn định, xem usage/log rồi cân nhắc plan phù hợp; repo này không tự nâng cấp plan.
- Snapshot và model được đọc từ Git ở mỗi lần khởi động. Kết quả upload batch được trả về trình duyệt, không lưu lên ổ đĩa server nên không cần persistent disk cho demo.
- Chỉ chạy một worker để tránh nhân đôi model/dữ liệu trong RAM. Không dùng `--reload` trên cloud.

Tài liệu chính thức: [FastAPI trên Render](https://render.com/docs/deploy-fastapi), [Blueprint](https://render.com/docs/blueprint-spec), [Deploy button](https://render.com/docs/deploy-to-render), [Free service](https://render.com/docs/free).

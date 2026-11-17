# Trợ lý Gemini trên Vercel

Dashboard gọi Gemini từ backend FastAPI. API key không được gửi xuống trình duyệt.

1. Push code mới lên GitHub. Vercel dùng Root Directory `dashboard`.
2. Mở Vercel → project → Settings → Environment Variables.
3. Thêm hai biến cho Production (thêm Preview nếu cần thử bản preview):

| Biến | Giá trị |
| --- | --- |
| `GEMINI_API_KEY` | API key của bạn từ Google AI Studio |
| `GEMINI_MODEL` | `gemini-3.8-flash`, hoặc mã model được cấp quyền cho key của bạn |

4. Lưu, rồi vào Deployments → deployment mới nhất → Redeploy để áp dụng biến môi trường.
5. Mở trợ lý và hỏi: “Mô hình tốt nhất là gì? Recall trên tập Test bao nhiêu?” Câu trả lời thành công hiển thị chế độ Gemini và nguồn.

Không đưa key vào GitHub, HTML hoặc JavaScript. Không cần gửi key trong cuộc trò chuyện này.

## Dữ liệu trợ lý sử dụng

Backend cung cấp cấu hình và kết quả mô hình, thống kê từ CSV dự đoán, ngưỡng đang chọn, tri thức DataCo và lịch sử gần nhất. Khi tra đơn hoặc vừa dự báo một đơn, trợ lý nhận thêm ngữ cảnh của đơn đó. Các số liệu trong câu trả lời được đối chiếu với danh sách giá trị do server tính.

Muốn bổ sung nội dung báo cáo, tạo file `.md` hoặc `.txt` trong `dashboard/backend/knowledge/`, commit rồi deploy lại. Backend đọc tối đa sáu tài liệu, mỗi tài liệu tối đa 12.000 ký tự. Chỉ thêm nội dung được phép gửi tới Gemini; không thêm thông tin bí mật. Model nhị phân và toàn bộ CSV không được gửi tới Gemini.

`/api/chat-status` chỉ báo có cấu hình, không xác nhận key hoạt động. Khi thiếu key, hết quota, model không hợp lệ hoặc câu trả lời không qua đối chiếu, giao diện hiện thông báo và dùng câu trả lời dự phòng. Không có key thật nên kiểm thử tích hợp dùng phản hồi Gemini giả lập; cần thử một câu hỏi sau khi deploy.

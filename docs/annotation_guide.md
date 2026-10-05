# Hướng dẫn gán nhãn mức demonstrability (tiêu chí 3, cổng G1)

Mỗi dòng trong `annotator_A.csv` / `annotator_B.csv` là một câu hỏi và **một tin nhắn** (tiếng Anh) của một "agent" ủng hộ một đáp án. Tin nhắn luôn kết thúc bằng đáp án, ví dụ `The answer is \boxed{18}.` hoặc `The answer is (C).`; các ví dụ dưới viết bằng tiếng Việt cho dễ đọc. Bạn không cần biết đáp án đúng, và không được xem `answer_key.csv`. Hai người làm độc lập, không trao đổi trong lúc gán.

## Cột cần điền

| Cột | Giá trị | Ý nghĩa |
|---|---|---|
| `label_D` | `D0` / `D1` / `D2` / `D3` | Mức demonstrability của tin nhắn (định nghĩa dưới) |
| `n_errors` | `0` / `1` / `2+` | Chỉ điền khi bạn chọn `D2`: số bước sai mà bạn kiểm tra được |
| `style_leak` | `yes` / để trống | `yes` nếu văn phong để lộ tin nhắn đúng hay sai (ví dụ "có lẽ", "tôi không chắc", lời xin lỗi, giọng thiếu tự tin bất thường) |
| `comment` | tự do | Ghi chú nếu khó quyết định |

## Bốn mức

**D0 — chỉ đáp án.** Tin nhắn chỉ nêu đáp án, không có lý do.
> The answer is \boxed{18}.

**D1 — lập luận không kiểm chứng được.** Có lý do, nhưng lý do là trực giác, so sánh, viện dẫn "ai cũng biết", ấn tượng chung. Không có phép tính, không có kết quả trung gian, không có bước nào để bạn tự kiểm tra lại.
> Thông thường những bài kiểu này đáp án nằm ở con số tròn, và cách hiểu tự nhiên nhất của đề dẫn đến 18.
> The answer is \boxed{18}.

**D2 — lập luận kiểm chứng được, đủ bước.** Lời giải từng bước, mỗi bước nêu một kết quả cụ thể (phép tính, định nghĩa, sự kiện) mà bạn có thể tự kiểm tra. Có thể có bước sai: điền `n_errors` là số bước sai bạn phát hiện được.
> Step 1: 16 − 3 − 4 = 9 quả trứng còn lại.
> Step 2: 9 × 2 = 18 đô la.
> The answer is \boxed{18}.

**D3 — giống D2 nhưng bị cắt bước then chốt.** Trông như lời giải từng bước, nhưng đúng bước quyết định chỉ là một khẳng định không có lý do ("Rõ ràng số còn lại là 9"), nên bạn không kiểm tra được đáp án từ các bước còn lại.
> Step 1: Mỗi ngày có 16 quả trứng.
> Step 2: Rõ ràng số trứng còn lại để bán là 9.
> Step 3: Bán 9 quả với giá 2 đô la.
> The answer is \boxed{18}.

## Quy tắc quyết định

1. Không có lý do nào → **D0**.
2. Có lý do nhưng không có bước nào kiểm tra được → **D1**.
3. Có các bước kiểm tra được: nếu **mọi bước cần để đi tới đáp án** đều kiểm tra được → **D2**; nếu đúng một bước then chốt chỉ là khẳng định trống → **D3**.
4. Một bước **sai** nhưng có phép tính để kiểm tra vẫn là bước kiểm chứng được (D2), không phải D3.
5. Phân vân giữa hai mức: chọn mức bạn nghiêng về hơn và ghi `comment`.

Thời gian dự kiến: 120 tin nhắn, khoảng 1–1,5 phút mỗi tin (3–4 giờ, nên chia 2 buổi). Lưu file ở dạng CSV UTF-8, giữ nguyên các cột `row`, `ann_id`.

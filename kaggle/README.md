# Hướng dẫn chạy trên Kaggle (GPU T4 x2)

Thư mục này chỉ chứa phần riêng của Kaggle. Code lõi (`hmad/`, `configs/`, `scripts/`, `tests/`) không đọc gì trong đây, và có test kiểm tra điều này. Xóa thư mục này thì `python -m hmad.run --config configs/exp.yaml` vẫn chạy trên mọi máy Linux có GPU.

**Không cần tài khoản hay token Hugging Face**: cả 12 mô hình đều tải được không cần xin quyền.

## A. Chuẩn bị (một lần)

1. **Tài khoản Kaggle** đã xác minh số điện thoại. Không xác minh thì không bật được GPU và Internet.
2. **Đẩy code lên GitHub**, chạy trên máy, trong thư mục `TD`:
   ```bash
   git add -A
   git commit -m "Heterogeneous SoM debates (Choi et al. 2025), 12 ungated 7-9B families"
   git push origin main
   git rev-parse HEAD        # ghi lại mã commit này
   ```
   `git add -A` cũng ghi nhận việc xóa code cũ (`tdmad/`, `analysis/`…). `debate-or-vote/`, `results/` và `.cache/` không bị đưa lên vì đã có trong `.gitignore`.
3. **Nếu repo riêng tư:** tạo token GitHub kiểu fine-grained, chỉ cấp quyền *Contents: Read-only* cho đúng repo `TD`. Repo công khai thì bỏ qua bước này.

## B. Tạo notebook

1. Kaggle → **Code → New Notebook → File → Import Notebook** → chọn `kaggle/run_kaggle.ipynb`.
2. Thanh bên phải:
   - **Session options → Accelerator: GPU T4 x2**;
   - **Internet: On**;
   - **Persistence: Files only**.
3. **Add-ons → Secrets** (chỉ khi cần):
   - `GITHUB_TOKEN` nếu repo riêng tư;
   - `HF_TOKEN` nếu muốn tải nhanh hơn. Không bắt buộc.
4. Ô code đầu tiên, sửa:
   ```python
   REPO_URL = "https://github.com/kn1827/TD.git"
   COMMIT = "<mã commit ở bước A.2>"
   VLLM_VERSION = "0.31.0"
   RESUME = False
   ```

## C. Smoke test (tùy chọn, khoảng 1–1,5 giờ)

Notebook chỉ chạy lần chính. Muốn thử nhanh 8 mô hình trước (4 + 4 câu, 2 vòng): ở ô chạy (ô 4 trong bảng cuối mục này) đổi `"main"` thành `"smoke"`, chạy, rồi mở **Output** → `results/smoke2/summary.md`. Kiểm tra từng mô hình:
- **readable answer ≥ 90%**: mô hình ghi đáp án đúng dạng `{final answer: …}`;
- **hit 512-token limit** thấp;
- **round-0 accuracy** hợp lý: GSM8K khoảng 0,5–0,9, CSQA khoảng 0,6–0,8.

Mỗi `phases/r0/<model>.meta.json` có `chat_template_example`: prompt đầu tiên sau khi bọc chat template của mô hình đó, để kiểm tra bằng mắt. Test trên CPU: `python -m tests.test_pipeline`.

## D. Chạy chính: run `hetero2`, đủ 4 vòng debate

1. Sửa ô đầu tiên: `COMMIT` (nếu có commit mới), `RESUME = False`.
2. **Save Version → Save & Run All (Commit).** Chạy kiểu này thì notebook chạy nền, không cần mở trình duyệt. Notebook **không đặt giờ dừng**: chạy tới khi xong cả 5 vòng (r0 + 4 vòng debate).
3. Trong lúc chạy, `results/` được nén thành `/kaggle/working/results.zip` mỗi 30 phút (đổi bằng biến môi trường `ZIP_EVERY_MIN`), và một lần nữa khi chạy xong. Tải một file này về là đủ.
4. **Chạy tiếp** (khi phiên bị ngắt giữa chừng):
   - Edit notebook → **Add Input → Your Work → Notebook Output** → chọn version vừa chạy. Hoặc tải `results.zip` lên thành một Kaggle Dataset rồi thêm làm Input.
   - Đặt `RESUME = True`, rồi Save & Run All lần nữa.
   - Ô thứ 2 chép `results/` (hoặc giải nén `.zip` có `results/` bên trong) vào `/kaggle/working`, in số phase đã xong. Phase nào đã xong không chạy lại.
   - `RESUME = True` mà Input không có kết quả cũ thì notebook dừng báo lỗi, để không vô tình chạy lại từ đầu.
5. **Kết quả cuối:** log in `[run] 1000 debates -> .../debates.jsonl`; Output có `results/hetero2/debates.jsonl`, `summary.md` và `results.zip`.

**Thời gian:** hetero1 (không chat template, nhiều lượt chạy tới 512 token) mất khoảng 10 giờ cho 3 vòng r0–r2. Đủ 5 vòng có thể tới khoảng 20 giờ; có chat template thì câu trả lời ngắn hơn nên có thể nhanh hơn. **Quota:** 30 giờ GPU mỗi tuần, xem ở thanh bên phải của Kaggle.

`TD_DEADLINE_TS` (giờ unix) vẫn dùng được nếu cần dừng chủ động trước một mốc: chương trình không bắt đầu phase nào dự kiến xong sau mốc đó, thoát với mã 3.

## Notebook làm gì

| Ô | Việc |
|---|---|
| 1 | Đọc secrets (nếu có); `git clone` rồi `git checkout COMMIT`; đặt `RESUME` |
| 2 | Nếu `RESUME = True`: chép `results/` hoặc giải nén `.zip` của phiên trước (trong `/kaggle/input`) vào `/kaggle/working/results` |
| 3 | `bash kaggle/setup.sh {VLLM_VERSION}`: cài vLLM, gỡ `torchaudio` (Kaggle dựng sẵn cho CUDA khác với torch của vLLM), cài `requirements.txt`, in phiên bản và kiểm tra torch tính được trên GPU |
| 4 | `bash kaggle/kaggle_run.sh main` (chạy `configs/exp.yaml`; nén `results.zip` mỗi 30 phút và khi xong) |
| 5 | In `results/*/summary.md` |

`kaggle_run.sh` tạo một config phụ kế thừa `configs/smoke.yaml` hoặc `configs/exp.yaml`, chỉ đổi phần riêng của Kaggle:
- kết quả lưu ở `/kaggle/working/results`;
- trọng số mô hình lưu ở thư mục ghi được đầu tiên trong `/tmp/models`, `/root/hmad_models` (thử ghi 256 MB trước), không chiếm 20 GB của `/kaggle/working`;
- giới hạn đĩa cho mô hình mặc định 40 GB (2 mô hình: mô hình đang chạy và mô hình tải trước). `df` trên Kaggle báo dung lượng của cả máy chủ, không phải hạn mức thật của phiên. Hạn mức thật khoảng 120 GB, tính cả khoảng 30 GB gói cài đặt; vượt hạn mức thì ổ chuyển sang chỉ đọc tới hết phiên. Đặt `CACHE_GB=...` để đổi;
- tìm bản sao mô hình trong `/kaggle/input` trước khi tải.

## Lỗi thường gặp

| Hiện tượng | Xử lý |
|---|---|
| `PyTorch and TorchAudio were compiled with different CUDA versions` | Đã xử lý trong `kaggle/setup.sh` (gỡ torchaudio). Notebook cũ: thay ô cài đặt bằng `!bash kaggle/setup.sh {VLLM_VERSION}` |
| `setup.sh` báo torch không dùng được GPU, hoặc `pip install vllm` lỗi | Driver của Kaggle quá cũ so với bản CUDA của torch: đặt `VLLM_VERSION` bằng một bản vLLM cũ hơn, dựng trên CUDA 12.x, rồi chạy lại |
| Một mô hình báo lỗi `dtype` / NaN / chữ vô nghĩa | Mô hình đó không chạy ổn ở fp16 trên T4. Đổi sang mô hình dự phòng |
| Hết đĩa khi tải mô hình | Giảm `disk_budget_gb`, hoặc dùng Kaggle Dataset chứa mô hình (mục dưới) |
| Notebook dừng giữa chừng không có mã 3 | Xem `results/<run>/logs/`; chạy lại, bước nào đã xong sẽ được bỏ qua |
| Muốn nhanh hơn | Dùng phương án mô hình nhỏ: `configs/exp_slm.yaml` (khoảng 7 giờ). Sửa `BASE` trong `kaggle_run.sh` |

## Giảm thời gian tải mô hình (tùy chọn)

12 mô hình fp16 nặng khoảng 190 GB. Ở smoke test ngày 09/10/2026, `/tmp` của phiên Kaggle trống 1.059 GB và tải được khoảng 230 MB/giây, nên cả 12 mô hình nằm vừa bộ đệm: mỗi phiên chỉ tải mỗi mô hình một lần (khoảng 14 phút). Mục này chỉ cần khi đĩa của phiên nhỏ hơn.

Cách tránh: tạo **Kaggle Dataset** cho từng mô hình (thư mục mang tên repo, ví dụ `Qwen2.5-7B-Instruct/`, có `config.json`, đúng revision ghi trong `configs/models.yaml`), rồi thêm làm Input. `hmad` dùng bản trong `/kaggle/input` trước khi tải.

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
   MODE = "smoke"
   VLLM_VERSION = ""
   ```

## C. Phiên 1: smoke test (khoảng 1–1,5 giờ)

1. Bấm **Save Version → Save & Run All (Commit)**. Chạy kiểu này thì notebook chạy nền tới 12 giờ, không cần mở trình duyệt. Chạy tương tác thì Kaggle sẽ ngắt khi bạn rời trang.
2. Chạy xong, mở version đó → **Output** → `results/smoke1/summary.md`. Kiểm tra từng mô hình:
   - **readable answer ≥ 90%**: mô hình ghi đáp án đúng dạng `{final answer: …}`;
   - **hit 512-token limit** thấp;
   - **round-0 accuracy** hợp lý: GSM8K khoảng 0,5–0,9, CSQA khoảng 0,6–0,8.
3. **Nếu có mô hình hỏng** (lỗi trong log `results/smoke1/logs/r*_<model>.log`, hoặc đọc được đáp án dưới 90%): gửi mình `summary.md` và log. Mình sẽ đổi sang mô hình dự phòng trong `configs/models.yaml`, rồi bạn commit và push lại.
4. Ghi phiên bản vLLM đã chạy được: in ở ô cài đặt, cũng có trong `results/smoke1/phases/r0/*.meta.json`. Điền vào `VLLM_VERSION` cho các phiên sau.

## D. Phiên 2 trở đi: chạy chính (2–3 phiên, mỗi phiên tối đa 12 giờ)

1. Sửa ô đầu tiên: `MODE = "main"`, ghim `VLLM_VERSION`, và `COMMIT` nếu có commit mới.
2. **Save & Run All (Commit).**
3. Ở mốc 11,3 giờ, chương trình tự dừng nhận việc mới và thoát với mã 3. Kết quả đã xong nằm trong Output của version đó.
4. **Chạy tiếp:**
   - Edit notebook → **Add Input → Your Work → Notebook Output** → chọn chính notebook này (version vừa chạy).
   - Rồi Save & Run All lần nữa.
   - Ô thứ 2 sẽ chép `results/` cũ vào; bước nào đã xong không chạy lại.
   - Lặp lại tới khi log in `[run] 1000 debates -> .../debates.jsonl`.
5. **Kết quả cuối** nằm trong Output: `results/hetero1/debates.jsonl` và `summary.md`.

**Quota:** 30 giờ GPU mỗi tuần. Smoke test (1–1,5 giờ) cộng chạy chính (khoảng 10–19 giờ) nằm gọn trong một tuần. Thời gian còn lại xem ở thanh bên phải của Kaggle.

## Notebook làm gì

| Ô | Việc |
|---|---|
| 1 | Đọc secrets (nếu có); `git clone` rồi `git checkout COMMIT`; đặt `TD_DEADLINE_TS` = giờ bắt đầu + 11,3 giờ |
| 2 | Chép `results/` của phiên trước (nếu đã thêm Input) vào `/kaggle/working/results` |
| 3 | `bash kaggle/setup.sh {VLLM_VERSION}`: cài vLLM, gỡ `torchaudio` (Kaggle dựng sẵn cho CUDA khác với torch của vLLM), cài `requirements.txt`, in phiên bản và kiểm tra torch tính được trên GPU |
| 4 | Test trên CPU: so code với bản gốc của Choi et al. (clone `debate-or-vote` @82c929e); chạy giả trọn quy trình |
| 5 | `bash kaggle/kaggle_run.sh smoke|main` |
| 6 | In `summary.md` |

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

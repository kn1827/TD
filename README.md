# TD-MAD v2 — pilot cổng G1

Code cho pilot của đề án *TD-MAD v2 — Dịch tễ học của đáp án trong multi-agent debate*: sàng câu và đo độ hợp lý π, thu debate theo **giao thức MAD chuẩn của Du et al. (2023)** (prompt nguyên văn từ code của tác giả) trên đồ thị đọc, sinh tin nhắn D0–D3 cho kiểm tra gán nhãn, và báo cáo bốn tiêu chí của cổng G1 (31/10/2026). Thiết kế đầy đủ, lý do chọn model và ước lượng chi phí nằm ở **[EXPERIMENT_G1.md](EXPERIMENT_G1.md)**.

Chạy trên Kaggle "GPU T4 x2": mỗi GPU có một server vLLM và một driver Python riêng, hai lane chạy song song.

## Cấu trúc

```
configs/   models.yaml (model ứng viên + thông số T4) · g1.yaml (pilot) · g1_smoke.yaml
data/      gsm8k_platinum.json · commonsenseqa.json · traps_pilot.json · MANIFEST.json (đã tạo sẵn)
tdmad/     answers (trích đáp án) · data (split, render) · traps (bộ bẫy) · graphs (đồ thị đọc)
           prompts · llm (client vLLM + log chi phí + FakeClient) · screen (S1) · mad (S2)
           messages (S3) · costmodel (ước lượng T4) · provenance (khóa giao thức, phiên bản, sha256)
scripts/   prepare_data · run_screen · run_mad · run_messages · make_annotation
           bench_throughput · estimate_budget · cfg_get
analysis/  pi_reliability (tiêu chí 1–2) · mad_kernel (ước tính sơ bộ) · annotation_agreement (3)
           cost_report (4) · g1_gate (báo cáo)
kaggle/    start_vllm.sh · run_g1.sh · tdmad_g1.ipynb
docs/      annotation_guide.md
tests/     test_core.py · test_client.py
```

## Chạy thử trên máy (không cần GPU)

```bash
pip install -r requirements.txt
python -m tests.test_core
python -m tests.test_client
python -m scripts.estimate_budget
C=configs/dry_run.yaml
python -m scripts.run_screen --config $C --model qwen2.5-7b --fake
python -m scripts.run_screen --config $C --model llama3.2-3b --fake
python -m scripts.run_mad --config $C --pool qwen7-homo --fake
python -m scripts.run_messages --config $C --generator phi-4 --fake
python -m scripts.run_messages --config $C --generator mistral-7b --fake
python -m scripts.make_annotation --config $C
python -m analysis.g1_gate --config $C
```

`--fake` dùng `FakeClient` (trả lời giả theo xác suất cố định, không cần server) và chỉ được phép với config có `run_name` chứa "dry"; kết quả nằm ở `results/dry_run/`. Mục đích là kiểm tra pipeline và định dạng file; các con số trong báo cáo không có ý nghĩa.

## Chạy trên Kaggle

1. Upload cả thư mục `TD-MAD` (có `data/`) thành một Kaggle Dataset.
2. Tạo notebook từ `kaggle/tdmad_g1.ipynb`: Accelerator *GPU T4 x2*, Internet *On*, thêm dataset ở bước 1 làm input.
3. Phiên 1 chạy lần lượt: test CPU → `bash kaggle/run_g1.sh bench` → `bash kaggle/run_g1.sh smoke` → hai lane song song → `python -m analysis.g1_gate`. Để chạy nền: *Save Version → Save & Run All*.
4. Nếu phiên bị cắt ở mốc 12 giờ: thêm output của phiên trước làm input, chạy lại notebook. Mọi bước tự bỏ qua phần đã xong.
5. Gán nhãn: gửi `results/g1_pilot/annotation/annotator_A.csv` và `annotator_B.csv` kèm `docs/annotation_guide.md` cho hai người. Khi nhận lại, chép đè vào cùng chỗ rồi chạy `python -m analysis.g1_gate` (chỉ cần CPU).

Lệnh từng bước (thay cho `run_g1.sh`):

```bash
bash kaggle/start_vllm.sh one 0 8001 Qwen/Qwen2.5-7B-Instruct-AWQ
python -m scripts.run_screen --model qwen2.5-7b
python -m scripts.run_mad --pool qwen7-homo
bash kaggle/start_vllm.sh stop_port 8001
```

## Thay đổi thường gặp

- **Đổi model pilot**: sửa `lanes.*.model`, `servers` và `mad.pools` trong `configs/g1.yaml`; model mới phải có trong `configs/models.yaml`. Chạy lại `python -m scripts.estimate_budget` rồi cập nhật `cost_estimate` **trước khi** chạy pilot.
- **Cùng một model trên 2 GPU** (Khối B, đồng nhất): `bash kaggle/start_vllm.sh dp <hf_id>`, rồi liệt kê cả hai URL dưới `servers:`; client sẽ chia đều request cho hai server.
- **Đồ thị Khối B**: `--graphs heawood14 rr3_14 cluster3_14` với một pool 14 agent.
- **Vòng 0 gán trước** (đường thắng thua): `python -m scripts.run_mad --pool qwen7-homo --init assigned` (cần có S1 trước).
- **Bộ nhớ debate**: `mad.memory: last_round` (mặc định; lý do ở EXPERIMENT_G1.md mục 1.1) hoặc `full` (toàn bộ lịch sử chat như code của Du). Với `full`, khởi động server bằng `MAXLEN=32768` và ước tính lại chi phí.
- **Lần chạy chính thức**: model được ghim theo commit (`revision` trong `configs/models.yaml`), mỗi bước khóa giao thức vào `results/<run>/meta/` và từ chối chạy tiếp nếu cấu hình đã đổi. Muốn đổi giao thức thì đặt `run_name` mới.
- Sửa một prompt thì phải tăng `PROMPT_VERSION` trong `tdmad/prompts.py`.

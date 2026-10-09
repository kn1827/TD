# TD — debate giữa các agent khác họ mô hình

Ý tưởng 1 của đề án: thu dữ liệu debate để nghiên cứu đáp án đúng/sai lây giữa các agent (`PLAN.md`).

- **Kiến trúc debate:** Society of Mind **theo cài đặt của Choi, Zhu & Li (NeurIPS 2025, "Debate or Vote")**, chuyển nguyên văn sang `hmad/choi.py` từ repo chính thức (MIT, commit 82c929e). `tests/test_choi_fidelity.py` chạy hàm gốc và hàm đã chuyển trên cùng đầu vào, kết quả phải giống hệt. Lý do chọn: `plans/nghien_cuu_giao_thuc_debate.md`.
- **Agent:** mỗi agent trong một debate là **một họ mô hình khác nhau**. Có 12 họ, cỡ 7–9B, chạy fp16, **không cần xin quyền tải** (`configs/models.yaml`). Phương án mô hình nhỏ 2–5B để dự phòng: `configs/exp_slm.yaml`. Cách chọn và các con số tính toán: `plans/chon_agent_khac_ho.md`.

| Thiết lập | Số agent (= số họ) | Đồ thị (của Choi et al.) |
|---|---|---|
| main | 6 | decentralized (đầy đủ), sparse (vòng tròn), centralized (sao) |
| extended | 12 | sparse |
| baseline | 3 | decentralized |

Mỗi bộ câu hỏi (GSM8K, CommonsenseQA) có 100 câu. Mỗi debate có vòng 0 + 4 vòng. Tổng cộng 1.000 debate và 33.000 lượt sinh, ước tính 10–19 giờ trên Kaggle T4 × 2, tức 2–3 phiên.

## Chạy

```bash
pip install -r requirements.txt
python -m tests.test_choi_fidelity && python -m tests.test_pipeline     # CPU
python scripts/estimate.py --config configs/exp.yaml                     # bộ nhớ, đĩa, thời gian
python -m hmad.run --config configs/exp.yaml --fake                      # chạy giả, không cần GPU
```

Trên máy Linux có GPU (compute capability ≥ 7.5; mô hình 7–9B fp16 cần khoảng 32 GB tổng: 2 × T4 với `tensor_parallel_size: 2`, hoặc 1 GPU ≥ 24 GB với `gpus: [0]`, `tensor_parallel_size: 1`):

```bash
pip install vllm
python -m hmad.run --config configs/smoke.yaml     # mỗi mô hình vài câu: kiểm tra trước
python -m hmad.run --config configs/exp.yaml       # chạy chính; chạy lại cùng lệnh để tiếp tục
```

Trên Kaggle: `kaggle/README.md`.

## Cách chạy

Mỗi agent là một mô hình khác, và mỗi lúc máy chỉ nạp được một mô hình 7–9B (hai mô hình nếu dùng phương án nhỏ, mỗi GPU một mô hình).
- Mọi debate tiến đồng bộ theo vòng.
- Ở mỗi vòng, từng mô hình sinh mọi lượt của mình trong mọi debate, trong một tiến trình vLLM riêng, rồi thoát để giải phóng GPU.
- Thứ tự mô hình đảo chiều sau mỗi vòng, để tận dụng các mô hình còn trên đĩa. Mô hình kế tiếp được tải trong lúc mô hình hiện tại đang sinh.
- Mỗi bước (vòng, mô hình) được lưu riêng, nên chạy lại cùng lệnh sẽ tiếp tục từ chỗ dừng.

## Kết quả (`results/<run_name>/`)

| File | Nội dung |
|---|---|
| `debates.jsonl` | Mỗi dòng một debate: các mô hình theo vị trí, họ mô hình, câu hỏi, đáp án chuẩn. Mỗi vòng có: nguyên văn trả lời, đáp án đọc được, đúng/sai từng agent, đa số, đáp án của tâm (đồ thị sao), số token, lý do dừng. Kèm nguồn gốc: commit git, revision mô hình, tham số lấy mẫu thực dùng, phiên bản vLLM |
| `summary.md` | Độ chính xác theo vòng; tỉ lệ đọc được đáp án và độ chính xác vòng 0 của từng mô hình |
| `plan.json`, `protocol.json` | Danh sách debate và giao thức, cố định từ lần chạy đầu |
| `phases/r<k>/<model>.*` | Đầu vào và đầu ra từng bước |

## Cấu trúc

```
hmad/      choi.py (chuyển từ Choi et al.) · plan.py · run.py (điều phối) · worker.py (vLLM, một mô hình/lần)
           weights.py (tải trọng số, giới hạn đĩa) · results.py · config.py
configs/   models.yaml (12 họ 7–9B + dự phòng, revision ghim) · exp.yaml · smoke.yaml
           models_slm.yaml · exp_slm.yaml (phương án mô hình nhỏ)
scripts/   estimate.py
tests/     test_choi_fidelity.py · test_pipeline.py
kaggle/    run_kaggle.ipynb · kaggle_run.sh · README.md   (chỉ phần riêng của Kaggle)
plans/     nghien_cuu_giao_thuc_debate.md · chon_agent_khac_ho.md
debate-or-vote/   repo tham khảo đã clone (không đưa vào git)
```

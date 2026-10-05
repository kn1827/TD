# TD-MAD v2 — Thiết kế thực nghiệm cơ bản cho cổng G1

Ngày 05/10/2026. Hạn cổng G1: 31/10/2026. Phần cứng: Kaggle "GPU T4 x2".

## 0. Tóm tắt

| Mục | Quyết định |
|---|---|
| Mục tiêu | Kiểm tra 4 tiêu chí khả thi của G1, đồng thời thu bộ debate **MAD chuẩn (Du et al. 2023)** đầu tiên để thay log DUS trong ước tính sơ bộ |
| Không làm | Kiểm định khẳng định, panel phơi nhiễm NC1, phát lại Shapley. Dữ liệu pilot không vào phân tích khẳng định |
| Model pool pilot | `Qwen2.5-7B-Instruct-AWQ` (lane0, GPU0, slot 7–8B) và `Llama-3.2-3B-Instruct-AWQ` (lane1, GPU1, slot 3–4B, họ khác) |
| Model sinh tin nhắn | `phi-4-AWQ` và `Mistral-7B-Instruct-v0.3-AWQ`: hai họ, ngoài pool, đều đã chạy được trên T4 ở DUS |
| Dữ liệu | GSM8K-Platinum (sàng 150, debate 100), CommonsenseQA (150/100), bộ bẫy pilot 100 câu (60 câu số từ template + 40 TruthfulQA misconception) |
| Debate | Giao thức Du et al. (2023), prompt nguyên văn từ code của tác giả; đọc theo đồ thị như sparse MAD (Li et al. 2024); n = 6; `complete6` và `ring6` (Khối A); vòng 0 + T = 5 vòng; pool đồng nhất; bộ nhớ một vòng (`last_round`) |
| Song song | Mỗi GPU một server vLLM và một driver Python riêng; hai lane chạy cùng lúc, độc lập |
| Chi phí ước tính | lane0 ≈ 7,2 GPU-giờ, lane1 ≈ 3,7 GPU-giờ, sinh tin nhắn ≈ 0,5 giờ → **1 phiên Kaggle 12 giờ**, dự phòng 1 phiên |
| Tái lập | Ghim commit HF của từng model, cố định biến template chat, khóa giao thức theo từng run, ghi phiên bản vLLM và tham số server, kiểm tra sha256 dữ liệu (mục 1.2) |

## 1. Vì sao dựng lại MAD thay vì dùng thẳng debate của DUS

Debate của DUS (2 solver + 1 critic) có các cơ chế làm méo đúng đại lượng TD-MAD cần đo:

- Solver chỉ đọc nhận xét của critic, nên liều chỉ có k ∈ {0, 1}; critic có k ∈ {0, 1, 2}. Không có đồ thị đọc.
- **Flip guard**: solver đổi đáp án mà critic không "disagree có căn cứ" thì bị ép về đáp án vòng 0. Cơ chế này chặn chính sự lây cần đo.
- Self-consistency 3 mẫu ở vòng 0 cho GSM8K, hiệu chỉnh độ tự tin bằng luật, prompt few-shot dài khác nhau theo tác vụ.

`g1_preestimate.py` trong DUS đã ghi nhận các giới hạn này. TD-MAD dùng **giao thức MAD chuẩn của Du et al. (2023)**: n agent ngang hàng, cùng một prompt câu hỏi; ở mỗi vòng, agent đọc lời giải vòng trước của các agent khác rồi viết lời giải mới; kết quả là đa số của vòng cuối. Không có critic, không có flip guard, không dừng sớm, không có trường confidence.

Phần lấy lại từ DUS: client vLLM và bước kiểm tra server (`base_agent.py`), script khởi động vLLM trên T4, cách tải parquet của HuggingFace, cơ chế resume theo jsonl, và các kinh nghiệm với T4 (Gemma không chạy fp16, AWQ 7–8B chạy ổn).

### 1.1. Đối chiếu với Du et al. (2023) (`PROMPT_VERSION = du2023-v2`)

Nguồn: code chính thức `github.com/composable-models/llm_multiagent_debate` (`gen_gsm.py` cho đáp án số, `gen_mmlu.py` cho trắc nghiệm). Nguyên văn các prompt nằm trong `tdmad/prompts.py`.

| Thành phần | Du et al. | TD-MAD | Ghi chú |
|---|---|---|---|
| Prompt vòng 0 | GSM: *"Can you solve the following math problem? {q} Explain your reasoning. Your final answer should be a single numerical number, in the form \boxed{answer}, at the end of your response."* MMLU: *"Can you answer the following question as accurately as possible? {q}: A) .., B) .., ... Explain your answer, putting the answer in the form (X) at the end of your response."* | **Giống nguyên văn.** Câu GSM8K-P và câu bẫy số dùng mẫu GSM; CSQA và TruthfulQA dùng mẫu MMLU | Đây cũng là prompt CoT của S1, nên π chính là phân phối đáp án vòng 0 |
| Tin nhắn debate | *"These are the solutions to the problem from other agents: "* + với mỗi agent *"One agent solution: ```{lời giải}```"* + câu kết của GSM (nhắc lại đề) hoặc của MMLU | **Giống nguyên văn**; lời giải hàng xóm vô danh, không đánh số | |
| Cấu trúc ngữ cảnh | Chat nhiều lượt: user (câu hỏi), assistant (lời giải của mình), user (tin nhắn debate), assistant, ... | Giống; chỉ gửi vai user/assistant, không system prompt | Du cũng không dùng system prompt |
| Ai đọc ai | Mọi agent khác | Hàng xóm trong đồ thị đọc. `complete6` = mọi agent khác, đúng như Du; `ring6` = 2 hàng xóm | Theo đề án; giống sparse MAD (Li et al. 2024) |
| Thứ tự lời giải hàng xóm | Cố định theo chỉ số agent | Xáo ngẫu nhiên mới ở mỗi lượt, có ghi lại thứ tự | Theo đề án, đối chứng C5 |
| Bộ nhớ | Toàn bộ lịch sử chat | **Mặc định `last_round`**: [prompt câu hỏi, lời giải vòng trước của mình, tin nhắn debate mới]. Ở vòng 1 hai cách giống hệt nhau. `memory: full` = đúng như Du | Lý do ngay dưới bảng |
| Vòng đồng bộ | Có (mọi agent ở vòng t dùng lời giải vòng t−1) | Có; các lượt trong một vòng chạy song song | |
| Số agent, số vòng | 3 agent, 2 vòng (phân tích thêm tới 4 vòng) | n = 6, vòng 0 + 5 vòng | Theo đề án (Khối A) |
| Lấy mẫu | Mặc định của API (nhiệt độ 1) | Nhiệt độ 0,7, top_p 1, `max_tokens` 1024, seed riêng cho mỗi lượt | 0,7 là giá trị đề án dùng cho mẫu CoT. Một khối `sampling` dùng chung cho S1 và debate |
| Trích đáp án | Số: `\boxed{}` cuối cùng, không có thì lấy số cuối cùng. MMLU: "(X)" cuối cùng | Giống; thêm fallback và ghi nhãn `parse` = `format` / `fallback` / `none` | Theo dõi tỉ lệ trả lời sai định dạng |
| Kết quả nhóm | Bỏ phiếu đa số ở vòng cuối (hòa → agent đầu tiên) | Đa số duy nhất, hòa → "undecided" (quy ước của đề án); **lưu kèm phiếu kiểu Du** (`du_vote_final`) để so với tài liệu | |

**Vì sao bộ nhớ mặc định là `last_round`:**

1. NC1 đo kernel bằng cách đặt agent "vào đúng định dạng ngữ cảnh của debate": câu hỏi, lời giải của chính nó, và một panel tin nhắn hàng xóm. Định dạng đó trùng với `last_round` ở mọi vòng. Với bộ nhớ đầy đủ, nó chỉ trùng ở vòng 1, nên điều kiện A4/A5 (kernel đo trong kiểm soát chuyển được sang debate thật) chỉ có nghĩa với `last_round`.
2. Đề án định nghĩa trạng thái là (đáp án, k_a, k_b) và viết rằng bậc 3 "giữ ngữ cảnh mỗi lượt ở mức 3 tin nhắn".
3. Với bộ nhớ đầy đủ, prompt vòng 5 trên `complete6` dài khoảng 10.000 token. Chi phí MAD tăng khoảng 2,2 lần: Qwen-7B trong G1 tăng từ 6,2 lên 13,7 GPU-giờ; cả đề án tăng từ 424 lên 770 T4-giờ, tức từ 7,1 lên 12,8 tuần quota Kaggle (cận trên, chưa tính prefix cache).

Khuyến nghị: ở NC2 chạy thêm một tập con Khối A với `memory: full` làm kiểm tra độ bền, để cho thấy lựa chọn này không quyết định kết quả. Đổi chế độ chỉ cần sửa `mad.memory` và chạy server với `MAXLEN=32768`.

### 1.2. Tái lập và an toàn cho lần chạy chính thức

| Cơ chế | Chi tiết |
|---|---|
| Ghim model | `configs/models.yaml` có `revision` (commit HF ngày 05/10/2026); `start_vllm.sh` chạy với `--revision` và `--tokenizer-revision` |
| Biến template chat | Template của Llama 3.x tự chèn **ngày hôm nay** vào system header, nên prompt sẽ khác nhau giữa các phiên. Client gửi `chat_template_kwargs: {date_string: "26 Jul 2024"}`, chính là giá trị mặc định của template. Qwen3 (nếu dùng sau này) có `enable_thinking: false` |
| Khóa giao thức | Lần chạy đầu của mỗi bước ghi `meta/<bước>_protocol.json` (phiên bản prompt, `sampling`, bộ nhớ, số vòng, commit model). Phiên sau mà cấu hình khác sẽ bị từ chối, nên một file kết quả không bao giờ trộn hai giao thức |
| Phiên bản vLLM | Mỗi phiên ghi `meta/<bước>_sessions.jsonl` (host, Python, phiên bản vLLM lấy từ `/version`, tham số server). Lần đầu ghi thêm `meta/vllm_version.txt`; notebook cài đúng phiên bản đó ở các phiên sau |
| Dữ liệu | Mỗi bước kiểm tra sha256 của `data/*.json` so với `data/MANIFEST.json` trước khi chạy |
| Lỗi | HTTP 400/413/422 (ví dụ prompt dài quá `--max-model-len`) không retry: debate đó được ghi vào `mad/<pool>_rejected.jsonl` và báo cáo riêng. Lỗi mạng, lỗi 5xx và phản hồi JSON hỏng được retry có back-off. Bước nào chưa xong (hết giờ hoặc còn lỗi) thì thoát với mã 3 và lane dừng tại đó; chạy lại sẽ làm tiếp |
| Seed | Seed mỗi lượt suy ra từ (run, debate, vòng, agent); thứ tự hàng xóm suy ra từ (debate, vòng, agent). Lưu kèm mọi văn bản, nên dựng lại được chính xác request của bất kỳ lượt nào (`tdmad.mad.build_request`) để phát lại Shapley ở NC2 |
| Dữ liệu giả | `--fake` chỉ chạy được với config có `run_name` chứa "dry" |

## 2. Quy trình

```
S0 dữ liệu (CPU, đã chạy) ─► S1 sàng câu + π ─► S2 MAD ─────────────────────► S5 báo cáo G1 (CPU)
                                 (mỗi lane)       (mỗi lane)                    ▲
                                    └─► S3 tin nhắn D0–D3 (lane1, sau S2) ─► S4 gán nhãn (2 người)
```

| Bước | Lệnh | Đầu ra (`results/g1_pilot/`) |
|---|---|---|
| S0 | `python -m scripts.prepare_data` | `data/*.json`, `data/MANIFEST.json` (đã có sẵn trong thư mục) |
| S1 | `python -m scripts.run_screen --model <m>` | `screen/<m>.jsonl` |
| S2 | `python -m scripts.run_mad --pool <p>` | `mad/<p>.jsonl` |
| S3 | `python -m scripts.run_messages --generator <g>` | `messages/messages.jsonl`, `messages/targets.json` |
| S4 | `python -m scripts.make_annotation` | `annotation/annotator_{A,B}.csv`, `annotation/answer_key.csv` |
| S5 | `python -m analysis.g1_gate` | `analysis/g1_report.md` và các bảng chi tiết |

Trên Kaggle, `kaggle/run_g1.sh lane0` và `lane1` chạy S1 → S2 (→ S3 → S4 với lane1) và tự bật/tắt server vLLM. Mọi bước ghi từng bản ghi ngay khi xong; chạy lại cùng lệnh sẽ bỏ qua phần đã có.

### S0 — Dữ liệu và split cố định

- GSM8K-Platinum (`madrylab/gsm8k-platinum`, 1.209 câu) và CommonsenseQA validation (1.221 câu), mỗi bộ chia một lần theo seed `20261005` thành pilot 200, NC1 45%, NC2–NC3 40%, dự trữ 15% phần còn lại. Câu pilot không được dùng lại ở NC1–NC3.
- Bộ bẫy pilot 100 câu (`data/traps_pilot.json`):
  - 60 câu số từ 19 template trong `tdmad/traps.py`: 15 template kiểu CRT (bat-and-ball đổi số, máy làm widget, bèo nhân đôi, cừu, vượt người về nhì, tốc độ trung bình khứ hồi, tăng rồi giảm %, chiết khấu chồng, cưa gỗ, cột hàng rào, chuông đồng hồ, cầu thang, hai vòi nước, ốc sên, trung bình gộp) và 4 template **kinh điển bị biến đổi**, tức đáp án đã thuộc lòng nay sai (widget đổi tỉ lệ, bèo nhân đôi sau 2–3 ngày, Monty Hall với người dẫn không biết xe ở đâu, hai con có thứ tự). Đáp án đúng và đáp án bẫy (lure) đều tính bằng code.
  - 40 câu TruthfulQA MC1, nhóm Misconceptions: 1 phương án đúng và 3 phương án sai do người viết. Lure của câu trắc nghiệm được chốt theo quy tắc đặt trước: phương án sai có π gộp cao nhất qua các mô hình pilot.

### S1 — Sàng câu và đo độ hợp lý π (NC1 bước 1–2)

- Mỗi (mô hình, câu): một request với `n` mẫu CoT, dùng **đúng prompt vòng 0 và tham số lấy mẫu của debate** (khối `sampling`). Câu số lấy 30 mẫu, câu trắc nghiệm 20 mẫu. 10 mẫu đầu là mẫu sàng câu của đề án; toàn bộ mẫu dùng để tính π. Giữ nguyên văn bản: NC1 dùng lời giải đúng/sai của chính mô hình làm trạng thái ban đầu, NC2 dùng chúng để gán trước vòng 0.
- Câu trắc nghiệm có thêm xác suất chữ cái từ logprob (cùng cách viết câu hỏi, yêu cầu chỉ trả lời chữ cái, prefill `The answer is (`) trên 4 hoán vị vòng của thứ tự phương án.
- π_CoT(a) = (k + α)/(N + α + β), với prior beta ước lượng theo (mô hình, tác vụ, vai trò) bằng phương pháp moment đã trừ nhiễu nhị thức.
- Tóm tắt sàng câu: độ chính xác 10 mẫu đầu, số câu có cả mẫu đúng lẫn mẫu sai, tỉ lệ đáp án không đọc được, tỉ lệ bị cắt độ dài, cờ "gần mức đoán" (C9).

### S2 — Thu debate MAD

- Pool `qwen7-homo` (6 × Qwen2.5-7B) trên GPU0 và `llama3-homo` (6 × Llama-3.2-3B) trên GPU1, chạy song song, theo giao thức ở mục 1.1.
- 100 câu GSM8K-P, 100 câu CSQA và 100 câu bẫy (lồng trong tập S1) × 2 đồ thị × 1 seed = **600 debate mỗi pool**, 36 lượt mỗi debate. Đồ thị `complete6` chạy hết rồi mới tới `ring6`, nên phiên bị cắt vẫn để lại trọn từng đồ thị.
- Có sẵn nhưng không bật cho G1: `--init assigned` (gán vòng 0 từ mẫu S1 của chính mô hình theo q₀ ∈ {0,33; 0,5; 0,67}), pool `mixed` khác họ, các đồ thị Khối B `heawood14` (girth 6), `rr3_14`, `cluster3_14`.
- `analysis/mad_kernel.py` sinh sự kiện phơi nhiễm theo đúng ánh xạ dịch tễ: với agent j giữ b ở vòng t và mỗi đáp án khác a có k_a ≥ 1 hàng xóm giữ, ghi một sự kiện "có lây sang a ở vòng t+1 không". Sự kiện được lấy theo phơi nhiễm, không theo kết quả. Từ đó báo cáo p⁺(k), p⁻(k), ε, Δp₁ (CI bootstrap theo câu), sổ giữ/sụp/sửa/không sửa theo 2609.35279, đường thắng thua theo q₀, và số câu bẫy tự nhiên. Đây là kernel thô, chưa phải R_eff có nhận dạng; nó chỉ dùng để xem hướng hiệu ứng và lượng dữ liệu.

### S3 — Tin nhắn D0–D3

- 60 câu (24 GSM8K-P, 12 CSQA, 24 bẫy) × 2 nhánh (đúng / sai) × 4 mức. Đích sai: lure với câu bẫy số; với câu còn lại là đáp án sai phổ biến nhất gộp qua mẫu S1 của hai mô hình pool. Câu được chia luân phiên cho hai model sinh.
- D2 sinh dạng JSON (`steps`, `key_step`, `key_step_bare`). Nhánh sai phải có **đúng một** bước sai kiểm tra được và bước đó chính là `key_step`. D3 tạo cơ học bằng cách thay `key_step` bằng `key_step_bare`. D1 là lập luận trực giác có độ dài xấp xỉ D2. D0 là template.
- Mỗi tin kết thúc giống câu trả lời của một agent đang debate (`The answer is \boxed{18}.` hoặc `The answer is (C).`), nên ở NC1 có thể đặt thẳng vào panel "One agent solution". Mỗi tin được kiểm tra: đáp án cuối phải khớp đích, có cờ "D1 có phép tính", và tỉ lệ độ dài D3/D2 (ngưỡng ±10%). Tạo lỗi tối đa 3 lần.

### S4 — Gán nhãn

`make_annotation` lấy 30 tin mỗi mức (120 tin), cân bằng theo nhánh và tác vụ, đổi `msg_id` thành mã mù và xáo thứ tự riêng cho từng người. Hướng dẫn gán nhãn: `docs/annotation_guide.md`. Mỗi người mất khoảng 3–4 giờ.

## 3. Cách tính bốn tiêu chí G1

| # | Tiêu chí | Dữ liệu | Cách tính | Ngưỡng |
|---|---|---|---|---|
| 1 | Câu bẫy hợp lệ | S1, 100 câu bẫy | Đếm, theo từng mô hình, các câu có số mẫu CoT chọn lure nhiều hơn số mẫu chọn đáp án đúng (cùng N mẫu; hòa thì không tính). So tỉ lệ thô thay vì π đã co ngót, vì prior của đích đúng và đích lure có trung bình khác nhau nên co ngót sẽ đẩy các ca sát nút về phía đáp án đúng. Báo cáo thêm số câu rơi vào ô nghịch chặt (tỉ lệ lure ≥ 0,4 và tỉ lệ đúng ≤ 0,15), số câu hợp lệ theo logprob, và số câu hợp lệ với ≥ nửa pool | ≥ 40 mỗi mô hình |
| 2 | Split-half π | S1 | Chia mẫu CoT thành nửa chẵn và nửa lẻ, tính tỉ lệ chọn đích ở mỗi nửa, lấy Pearson r gộp trong từng tác vụ (trừ trung bình tác vụ khỏi mỗi nửa), hiệu chỉnh Spearman–Brown, CI bootstrap theo câu. **Tính riêng** cho đích đúng (mọi câu) và đích lure (câu bẫy); lấy giá trị nhỏ hơn. Ba lựa chọn để tránh thổi phồng: (a) dùng tỉ lệ thô, không dùng π đã co ngót, vì prior ước lượng trên toàn bộ mẫu kéo cả hai nửa về cùng một trung bình (chạy thử bằng dữ liệu giả cho r ≈ 0,99 dù không có tín hiệu ở mức câu); (b) không gộp đích đúng với đích lure, vì khoảng cách giữa hai nhóm tự tạo tương quan; (c) trừ trung bình tác vụ, vì chênh lệch độ khó giữa các tác vụ cũng tự tạo tương quan | ≥ 0,8 mỗi mô hình |
| 3 | Nhận đúng mức D | S4 | Độ chính xác của từng người so với mức thiết kế, Cohen's κ giữa hai người (kèm κ so với khóa), ma trận nhầm lẫn, tỉ lệ đánh dấu "đúng một lỗi" ở D2 nhánh sai | độ chính xác ≥ 85% (cả hai người), κ(A,B) ≥ 0,7 |
| 4 | Chi phí mỗi lượt | log chi phí | GPU-giây mỗi lượt = Σ(thời gian tường × số GPU) / số lượt. Một lượt là một mẫu CoT (S1) hoặc một lượt agent (S2). So với ước tính đã chốt trong `configs/g1.yaml` | ±30% |

Cổng ghi `PENDING` khi thiếu dữ liệu đầu vào. Không có tiêu chí nào xét dấu của hiệu ứng. Bảng MAD sơ bộ in kèm báo cáo chỉ để tham khảo.

## 4. Chọn model và ước lượng chi phí

### Ràng buộc của T4

16 GB mỗi GPU, compute 7.5: **chỉ fp16**, không có bf16 hay FP8, không có FlashAttention-2. Hệ quả:

- **Loại**: Gemma-2/3 (vLLM từ chối fp16, mà fp32 không vừa), Ministral-3 (trọng số FP8, kèm vision tower), OLMo-2-7B (không dùng GQA: KV 512 KB/token, trọng số fp16 14,6 GB).
- **14B** (Qwen2.5-14B, Phi-4): sau trọng số chỉ còn 3–4 GB cho KV, tức 7–9 chuỗi 2k token, chậm gấp 2,5–5 lần một model 7B. Chỉ nên dùng cho việc nhỏ (như sinh tin nhắn) hoặc chạy tensor parallel 2 GPU.
- Kích thước KV cache ảnh hưởng mạnh tới thông lượng debate (prompt dài). Qwen2.5 có KV nhỏ nhất (36–56 KB/token) nên rẻ hơn rõ rệt so với Llama/Mistral (112–128 KB/token).

### Mô hình chi phí (`tdmad/costmodel.py`)

GPU-giây mỗi lượt = prefill + decode. Prefill = token prompt × 2·tham số / F_prefill (giới hạn bởi tính toán). Mỗi bước decode = trọng số/BW + B·ctx·KV/BW + 2·tham số·B/F_decode + overhead, với B bị giới hạn bởi `max_num_seqs` = 64 và bộ nhớ KV còn trống. Các tham số tiên nghiệm cho T4: BW 210 GB/s, F_prefill 22 TFLOPs, F_decode (AWQ) 12 TFLOPs, overhead 4 ms/bước. Tiên nghiệm này khớp bậc độ lớn với thời gian chạy thực của DUS trên Kaggle. Phiên đầu chạy `kaggle/run_g1.sh bench` để đo thật, rồi `estimate_budget --calibrate` để hiệu chỉnh.

Trích `python -m scripts.estimate_budget` (tiên nghiệm, chưa hiệu chỉnh):

| Model | Quant | Trọng số GB | KV KB/token | Số chuỗi 2k | Prefill tok/s | Decode tok/s @2k | GPU-s/mẫu sàng câu | GPU-s/lượt MAD (complete6) | T4 |
|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-3b | awq | 2,7 | 36 | 64 | 3.560 | 886 | 0,19 | 0,64 | ok |
| **qwen2.5-7b** | awq | 5,6 | 56 | 64 | 1.444 | 436 | 0,42 | 1,47 | đã chạy |
| qwen2.5-14b | awq | 10,0 | 192 | 7 | 743 | 85 | 0,82 | 4,05 | rủi ro |
| qwen3-4b | awq | 2,7 | 144 | 34 | 2.736 | 390 | 0,26 | 1,01 | ok |
| qwen3-8b | awq | 6,1 | 144 | 23 | 1.343 | 238 | 0,48 | 1,87 | ok |
| **llama3.2-3b** | awq | 2,2 | 112 | 46 | 3.427 | 514 | 0,21 | 0,79 | ok |
| llama3.1-8b | awq | 5,7 | 128 | 27 | 1.370 | 267 | 0,46 | 1,77 | đã chạy |
| mistral-7b | awq | 4,2 | 128 | 33 | 1.517 | 315 | 0,40 | 1,56 | đã chạy |
| phi-4-mini | fp16 | 7,7 | 128 | 20 | 2.865 | 270 | 0,28 | 1,18 | ok |
| phi-4 | awq | 9,1 | 200 | 9 | 748 | 103 | 0,80 | 3,72 | đã chạy |
| falcon3-3b | awq | 2,9 | 88 | 55 | 3.406 | 582 | 0,21 | 0,75 | ok |
| falcon3-7b | awq | 5,1 | 112 | 34 | 1.475 | 316 | 0,42 | 1,59 | ok |
| granite3.3-2b | fp16 | 5,1 | 80 | 47 | 4.348 | 602 | 0,19 | 0,64 | ok |
| granite3.3-8b | awq | 4,5 | 160 | 25 | 1.346 | 254 | 0,45 | 1,82 | rủi ro |

"Đã chạy" = đã chạy trên Kaggle T4 trong DUS (24–25/09/2026). "ok" = kiến trúc chuẩn, chưa chạy thử. Bước smoke sẽ kiểm tra.

### Lựa chọn cho G1

- **Qwen2.5-7B-Instruct-AWQ** cho slot 7–8B: đã chạy ổn trên T4 ở DUS, KV nhỏ nên rẻ nhất trong nhóm 7–8B. Có nguy cơ khó bị bẫy nhất (GSM8K khoảng 90%), nên nó kiểm tra tiêu chí 1 ở trường hợp khó.
- **Llama-3.2-3B-Instruct-AWQ** (`casperhansen/llama-3.2-3b-instruct-awq`, không gated) cho slot 3–4B: khác họ để thấy độ hợp lệ của bẫy có đứng vững qua họ mô hình không, và chạy xong sớm nên GPU1 còn thời gian sinh tin nhắn. Nếu muốn chỉ dùng repo chính hãng, thay bằng `qwen2.5-3b` (sửa một dòng `lanes.lane1.model` và `mad.pools`).
- Hai model sinh `phi-4` và `mistral-7b`: khác họ với nhau và với pool, đều đã chạy trên T4. Phi-4 chậm, nhưng khối lượng sinh tin nhắn nhỏ (khoảng 240 lượt gọi mỗi model).

### Khối lượng G1 (tiên nghiệm)

| Lane | Model | Mẫu sàng câu | GPU-h sàng câu | Lượt MAD | GPU-h MAD | Tổng GPU-h |
|---|---|---|---|---|---|---|
| lane0 | qwen2.5-7b | 10.100 | 1,0 | 21.600 | 6,2 | 7,2 |
| lane1 | llama3.2-3b | 10.100 | 0,5 | 21.600 | 3,2 | 3,7 + ~0,5 sinh tin nhắn |

Ước tính chốt cho tiêu chí 4 (GPU-giây mỗi lượt, prompt Du, bộ nhớ `last_round`): Qwen-7B 0,36 (sàng câu) và 1,03 (MAD); Llama-3B 0,18 và 0,53. Nếu đổi sang `memory: full`, MAD của G1 tăng lên khoảng 13,7 GPU-giờ (Qwen-7B) và 7,3 GPU-giờ (Llama-3B), và phải ước tính lại trước khi chạy. Không sửa các con số này sau khi pilot đã chạy. Nếu bench ở phiên 1 cho thấy tiên nghiệm lệch xa, ghi lại độ lệch đó như một kết quả.

### Ngoại suy toàn đề án trên Kaggle T4 × 2

Với pool T4 đề xuất (Qwen2.5 3B/7B, Llama 3.2-3B/3.1-8B, Phi-4-mini/Phi-4, Falcon3 3B/7B, Granite 3.3 2B/8B) và số lượt gọi trong đề án:

| Phần | T4 GPU-giờ |
|---|---|
| NC1 (10 model; riêng Phi-4 14B chiếm 37) | 147 |
| NC2 Khối A (10.800 debate) | 113 |
| NC2 Khối B (4.320 debate) | 76 |
| NC2 phát lại Shapley | 50 |
| NC3 can thiệp | 38 |
| **Tổng (`last_round`)** | **≈ 424** |
| Tổng nếu `memory: full` (cận trên) | ≈ 770 |

Quota Kaggle khoảng 30 giờ phiên mỗi tuần, tức 60 T4-giờ mỗi tuần nếu dùng đủ cả 2 GPU: **7,1 tuần** nếu dùng 100% quota, khoảng 10,1 tuần nếu dùng 70% (bộ nhớ đầy đủ: 12,8 và 18,3 tuần). Làm được trước 06/2027 nhưng ít dư địa chạy lại. Khuyến nghị:

1. Giữ pool ở mức ≤ 8B; chỉ giữ một model 14B (Phi-4) nếu thật cần slot "lớn" của họ Phi, hoặc thay bằng họ có cả cỡ 3B và 7B (Falcon3).
2. Ưu tiên model có KV nhỏ cho Khối B và Shapley (mỗi lượt có 3 tin nhắn hàng xóm); Qwen2.5 3B/7B rẻ nhất.
3. Yếu tố thinking/non-thinking của đề án: dùng Qwen3-4B/8B, chỉ trên tập con, vì chế độ thinking sinh đầu ra dài gấp nhiều lần.
4. Ngoại suy NC1 trên T4 ≈ 147 GPU-giờ (khoảng 6 ngày T4), cùng bậc với ước tính "5–8 ngày GPU 24GB" của đề án. Con số dùng để lập kế hoạch thật sẽ lấy từ bench và log chi phí của pilot.

## 5. Lịch đến 31/10

| Ngày | Việc |
|---|---|
| 06–07/10 | Upload `TD-MAD` thành Kaggle Dataset. Phiên 1: test CPU, `bench` (10 phút), `smoke` (20–30 phút) |
| 07–09/10 | Phiên 1 (tiếp) hoặc phiên 2: chạy 2 lane khoảng 8–9 giờ. Nếu bị cắt, phiên sau khôi phục `results/` rồi chạy lại cùng lệnh |
| 09–10/10 | Giao bảng gán nhãn cho 2 người. Song song rà tay khóa của 40 câu TruthfulQA và các câu bẫy có π bất thường |
| 10–18/10 | Gán nhãn (mỗi người 3–4 giờ) |
| 19–22/10 | `analysis.g1_gate`, viết báo cáo G1. Nếu tiêu chí 1 trượt: thêm biến thể template, chạy lại S1 cho phần mới (khoảng 1 GPU-giờ) |
| 23–31/10 | Dự phòng; chuẩn bị bản đăng ký trước OSF (con số π, ngưỡng, quy tắc lure) |

Tổng quota cho G1 khoảng 10–12 giờ phiên: không quá một tuần quota.

## 6. Rủi ro riêng của pilot

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Qwen-7B có ít hơn 40 câu bẫy hợp lệ | `valid_traps` < 40 | Nhóm kinh điển bị biến đổi thường bẫy model mạnh tốt hơn nhóm CRT: thêm template loại này, sinh thêm 50 câu, chạy lại S1 riêng phần mới (resume tự bỏ qua câu cũ). Báo cáo G1 ghi rõ đã mở rộng |
| Split-half của lure < 0,8 | `cot_lure` thấp, CI rộng | Tăng `n_samples.number` lên 40 (chi phí sàng câu +33%); xét π_LP cho câu trắc nghiệm |
| Phiên bản vLLM mới trên Kaggle lỗi với T4 | Server không lên (xem `logs/vllm_<port>.log`) | Thử `EAGER=1`; cài lại đúng phiên bản vLLM đã chạy ở DUS (lệnh notebook in ra phiên bản) |
| Server không nhận `continue_final_message` | logprob trả HTTP 400 | Code tự rơi về prompt thường; kiểm tra `lp_mass` trong `pi_items.csv` |
| Thông lượng lệch tiên nghiệm > 30% | Tiêu chí 4 trượt | Đây là kết quả có ích: cập nhật ngân sách theo bench, rồi áp phương án "chi phí vượt dự kiến" của đề án (giảm Khối A, giữ Khối B) |
| Bản AWQ cộng đồng của Llama-3.2-3B làm giảm chất lượng | Độ chính xác GSM8K thấp bất thường (< 60%) | Đổi sang `qwen2.5-3b` (repo chính hãng) |
| Model sinh không tuân thủ "đúng một lỗi" | `error` nhiều trong `messages.jsonl`, người gán nhãn thấy 0 hoặc 2+ lỗi | Sửa prompt (`PROMPT_VERSION` mới), chỉ sinh lại phần lỗi |

## 7. Dữ liệu sinh ra

```
results/g1_pilot/
  screen/<model>.jsonl          mẫu CoT đầy đủ, đáp án đã chuẩn hóa, logprob theo hoán vị
  mad/<pool>.jsonl              debate: đồ thị, answers/classes theo vòng, đa số, toàn bộ lượt
                                (ai được hiện cho ai, thứ tự, văn bản, seed, token, độ trễ)
  messages/                     targets.json, messages.jsonl (D0–D3, cờ kiểm tra)
  annotation/                   bảng cho 2 người gán nhãn và khóa đáp án
  cost/                         *_calls.jsonl (từng request), *_stages.jsonl (thời gian tường, số GPU)
  analysis/                     g1_report.md, pi_summary.json, pi_items.csv, mad_summary.json,
                                exposure_events.csv, annotation_summary.json, cost_summary.json
results/bench/<model>.json      thông lượng đo được; results/budget.json  ngân sách
```

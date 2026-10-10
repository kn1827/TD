# Chọn agent: kết quả smoke test và danh sách chốt

## Chốt ngày 10/10/2026, sau smoke test 1 trên Kaggle (thay cho các mục bên dưới)

**Kết quả smoke test 1** (vLLM 0.31, 2 × T4, fp16, không chat template như Choi; 8 câu, 2 vòng):

| Mô hình | Chạy được? | Đọc được đáp án (CSQA / GSM8K) | Chạm giới hạn 512 token (CSQA / GSM8K) |
|---|---|---|---|
| Qwen2.5-7B | có | 88% / 100% | 12% / 12% |
| Llama-3.1-8B | có | 88% / 100% | 88% / 100% |
| Mistral-7B-v0.3 | có | 75% / 75% | 12% / 38% |
| EXAONE-3.5-7.8B | có | 75% / 38% | 0% / 12% |
| Granite-3.3-8B | có | 100% / 88% | 0% / 38% |
| InternLM3-8B | có | 100% / 75% | 100% / 88% |
| Falcon3-7B | có | 62% / 88% | 0% / 0% |
| Kanana-1.5-8B | có | 100% / 75% | 100% / 100% |
| Salamandra-7B | chạy nhưng gần như không đọc được đáp án | 38% / 12% | 100% / 100% |
| OLMo-3-7B, GLM-4-9B | **không**: vLLM không khởi tạo được cấu hình mô hình | — | — |
| Apertus-8B | **không**: hàm kích hoạt xIELU giữ float32, lệch với fp16 | — | — |

**Quyết định của người làm đề tài:**
- Chỉ dùng **8 họ đã chạy được**: Qwen, Llama, Mistral, EXAONE, Granite, InternLM, Falcon, Kanana. Tất cả là bản instruct.
- **Giữ đúng Choi**: prompt thô, không chat template.
- **Cho phép trùng họ** khi thiết lập cần nhiều agent hơn số họ (`unique_families: false`):

| Thiết lập | Agent |
|---|---|
| baseline (3) | Qwen, Llama, Mistral: khác họ |
| main (6) | Qwen, Llama, Mistral, EXAONE, Granite, InternLM: khác họ |
| extended (12) | 8 họ + thêm một agent nữa cho 4 họ dùng nhiều nhất (Qwen, Llama, Mistral, EXAONE) |

**Chấp nhận như một đặc điểm của cách làm theo Choi:** Llama, InternLM và Kanana thường viết tiếp quá đáp án tới giới hạn 512 token, và bộ đọc đáp án lấy cặp `{…}` cuối cùng. Cần ghi rõ điều này trong báo cáo, và kiểm tra lại tỉ lệ đọc được đáp án trên dữ liệu thật.

**Ước tính:** 1.000 debate, 33.000 lượt sinh, 40 lần nạp mô hình. Khoảng 8,6–9,2 giờ phiên với tốc độ giả định; nếu chậm gấp đôi thì khoảng 2 phiên.

---

## (Trước smoke test) Chọn agent khác họ mô hình (7–9B, không cần xin quyền): tiêu chí, danh sách, tính toán

Viết ngày 09/10/2026, chốt theo yêu cầu "7–8B vẫn được, miễn là miễn phí và không phải xin quyền; chạy nhiều ngày cũng được".
- Áp dụng: kiến trúc SoM theo cài đặt của Choi et al. (lựa chọn A).
- Thiết lập chính 6 agent; thiết lập phụ 12 agent (đồ thị thưa) và 3 agent (mốc).
- **Mọi agent trong một debate thuộc họ mô hình khác nhau.**
- Phương án mô hình nhỏ (2–5B, khoảng 7 giờ) vẫn giữ để dự phòng: `configs/exp_slm.yaml`, `configs/models_slm.yaml`.

## 0. Đọc nhanh

- **12 họ mô hình**, mỗi họ một mô hình instruct 7,2–9,4B, chạy fp16, **tải được không cần xin quyền và không cần token**.
  - 3 agent: Qwen2.5-7B, Llama-3.1-8B, Mistral-7B. Đúng hai mô hình Choi dùng, thêm Mistral.
  - 6 agent: 3 mô hình trên + OLMo-3-7B, Apertus-8B, EXAONE-3.5-7.8B.
  - 12 agent: 6 mô hình trên + Granite-3.3-8B, InternLM3-8B, Salamandra-7B, GLM-4-9B, Falcon3-7B, Kanana-1.5-8B.
- **Không mô hình nào vừa một T4 ở fp16**, nên mỗi mô hình chia trên 2 GPU và chạy lần lượt từng mô hình. Cả đợt có 60 lần nạp mô hình.
- **Khối lượng:** 1.000 debate, 33.000 lượt sinh.
- **Thời gian:** khoảng 10–12 giờ phiên với giả định tốc độ đã nêu, khoảng 19 giờ nếu tốc độ chỉ bằng một nửa, tức **2–3 phiên Kaggle**. Mỗi bước được lưu lại nên chạy qua nhiều ngày được.

## 1. Tiêu chí

| # | Tiêu chí | Vì sao |
|---|---|---|
| 1 | Khác tổ chức **và** huấn luyện từ đầu độc lập | Đúng nghĩa "khác họ" |
| 2 | Instruct, không phải mô hình suy luận dài, 7–9B | Cùng cỡ với mô hình của Choi. Mỗi lượt tối đa 512 token |
| 3 | **Không cần xin quyền**; trọng số safetensors | Theo yêu cầu; không dùng file pickle `.bin` |
| 4 | vLLM hỗ trợ; ngữ cảnh ≥ 8.192 token; chạy fp16 (T4 không có bf16) | Ràng buộc phần cứng và độ dài prompt |
| 5 | Xếp hạng theo lượt tải 30 ngày của repo chính thức | Khách quan, lặp lại được |

## 2. Mười hai họ được chọn

Revision ghim trong `configs/models.yaml`. Đã kiểm tra trên API Hugging Face và trang mô hình được hỗ trợ của vLLM ngày 09/10/2026.

| Hạng | Họ | Mô hình | Tổ chức | Tham số | Ngữ cảnh | Giấy phép |
|---|---|---|---|---|---|---|
| 1 | Qwen | Qwen2.5-7B-Instruct | Alibaba (CN) | 7,6B | 32k | Apache-2.0 |
| 2 | Llama | Meta-Llama-3.1-8B-Instruct (bản tải lại của Unsloth)\* | Meta (US) | 8,0B | 131k | Llama 3.1 |
| 3 | Mistral | Mistral-7B-Instruct-v0.3 | Mistral AI (FR) | 7,2B | 32k | Apache-2.0 |
| 4 | OLMo | Olmo-3-7B-Instruct | AI2 (US) | 7,3B | 64k | Apache-2.0 |
| 5 | Apertus | Apertus-8B-Instruct-2509 | Swiss AI (CH) | 8,1B | 64k | Apache-2.0 |
| 6 | EXAONE | EXAONE-3.5-7.8B-Instruct | LG AI Research (KR) | 7,8B | 32k | EXAONE (phi thương mại) |
| 7 | Granite | granite-3.3-8b-instruct | IBM (US) | 8,2B | 131k | Apache-2.0 |
| 8 | InternLM | internlm3-8b-instruct | Shanghai AI Lab (CN) | 8,8B | 32k | Apache-2.0 |
| 9 | Salamandra | salamandra-7b-instruct | Barcelona Supercomputing Center (ES) | 7,8B | 8k | Apache-2.0 |
| 10 | GLM | GLM-4-9B-0414 | Zhipu / Z.ai (CN) | 9,4B | 32k | MIT |
| 11 | Falcon | Falcon3-7B-Instruct | TII (AE) | 7,5B | 32k | TII Falcon 2.0 |
| 12 | Kanana | kanana-1.5-8b-instruct-2505 | Kakao (KR) | 8,0B | 32k | Apache-2.0 |

\* Repo chính thức `meta-llama/Llama-3.1-8B-Instruct` cần xin quyền. Bản của Unsloth không cần xin quyền, có cùng số tham số, và giấy phép Llama 3.1 cho phép phân phối lại. Thứ hạng tính theo lượt tải của repo chính thức (6,07 triệu).

**Dự phòng** (nếu một mô hình hỏng ở smoke test):
- Marin-8B, Yi-1.5-9B: ngữ cảnh 4k, chỉ dùng được cho thiết lập 12 agent đồ thị thưa.
- Tri-7B (Trillion Labs).
- Hunyuan-7B: có chế độ suy luận.

**Bị loại:**

| Họ / mô hình | Lý do |
|---|---|
| meta-llama (bản chính thức), Gemma, Command R7B / Aya (Cohere), EuroLLM-9B | Cần xin quyền. Gemma còn bị vLLM từ chối chạy fp16 |
| Phi | Không có bản 7–9B nào vLLM hỗ trợ |
| Ministral-8B | Cùng họ Mistral |
| DeepSeek-LLM-7B, Teuken-7B | Ngữ cảnh 4k |
| Nemotron-Nano-9B-v2, MiniCPM4.1-8B | Mô hình suy luận |

## 3. Bộ nhớ (fp16, mỗi mô hình chia trên 2 T4)

| Mô hình | Trọng số GB | KV KB/token | Ngân sách KV GB | Số chuỗi cùng lúc ở vòng 1 |
|---|---|---|---|---|
| qwen2.5-7b | 15,2 | 56 | 8,8 | 99 |
| llama3.1-8b | 16,1 | 128 | 7,9 | 39 |
| mistral-7b | 14,5 | 128 | 9,5 | 47 |
| olmo3-7b | 14,6 | **512** | 9,4 | **11** |
| apertus-8b | 16,1 | 128 | 7,9 | 38 |
| exaone3.5-7.8b | 15,6 | 128 | 8,4 | 40 |
| granite3.3-8b | 16,3 | 160 | 7,7 | 34 |
| internlm3-8b | 17,6 | 48 | 6,4 | 94 |
| salamandra-7b | 15,5 | 128 | 8,5 | 46 |
| glm4-9b | 18,8 | 40 | 5,2 | 91 |
| falcon3-7b | 14,9 | 112 | 9,1 | 57 |
| kanana1.5-8b | 16,1 | 128 | 7,9 | 44 |

Cả 12 mô hình nặng khoảng 190 GB trên đĩa, nhiều hơn đĩa của một phiên Kaggle. Vì vậy mô hình được tải lại khi cần, và mô hình kế tiếp được tải trong lúc mô hình hiện tại đang chạy.

## 4. Thời gian (`python scripts/estimate.py --config configs/exp.yaml --start-s 180`)

**Giả định** (đo lại ở smoke test):
- mỗi T4 sinh 500 token/giây khi chạy 32 chuỗi cùng lúc; chia trên 2 GPU thì đạt 70% của tổng 2 GPU;
- xử lý prompt 2.500 token/giây;
- mỗi lần khởi động và nạp mô hình mất 3 phút.

| Kịch bản | Thời gian phiên |
|---|---|
| Theo giả định trên | 10,3 giờ (trần 11,7 giờ nếu thời gian tải không chồng lên lúc sinh) |
| Tốc độ chỉ bằng một nửa | khoảng 19 giờ |
| Smoke test | 0,9–1,5 giờ |

Một phiên Kaggle tối đa 12 giờ. Code dừng nhận việc ở mốc 11,3 giờ, rồi chạy tiếp ở phiên sau, nên cả đợt cần **2–3 phiên**. Quota Kaggle là 30 giờ GPU mỗi tuần, đủ cho cả smoke test lẫn đợt chạy chính trong một tuần.

## 5. Theo Choi et al. và chỗ khác

| Mục | Ở đây |
|---|---|
| Prompt, tin nhắn mỗi vòng, đồ thị, cách đọc đáp án, bỏ phiếu | Giống hệt Choi (có test so với code gốc) |
| Không dùng chat template; temperature 1,0, top_p 0,9, tối đa 512 token; top_k và hệ số phạt lặp lấy từ `generation_config.json` của từng mô hình | Giống Choi |
| Mô hình | **12 họ, mỗi agent một họ.** Choi dùng cùng một mô hình cho mọi agent |
| Cách gọi mô hình | vLLM, fp16, chia trên 2 GPU. Choi dùng HF Transformers |
| Gán agent vào vị trí; hòa phiếu | Hoán vị ngẫu nhiên có seed; chọn ngẫu nhiên có seed |

## 6. Rủi ro cần kiểm tra ở smoke test

| Rủi ro | Xử lý |
|---|---|
| Mô hình không theo dạng `{final answer: …}` khi không có chat template | Nếu tỉ lệ đọc được đáp án dưới 90%: thay bằng mô hình dự phòng, hoặc bật chat template cho mọi mô hình (lệch khỏi Choi). Cần bạn quyết |
| Mô hình huấn luyện bf16 bị tràn số khi chạy fp16 | Thay bằng mô hình dự phòng |
| OLMo-3 chậm hơn các mô hình khác (bộ đệm KV lớn) | Chấp nhận, hoặc thay bằng mô hình dự phòng |
| EXAONE và InternLM3 phải chạy code đi kèm repo | Code đã ghim theo revision |
| Bản Llama của Unsloth có thể khác bản gốc ở tokenizer hoặc cấu hình | Ghi rõ trong báo cáo |

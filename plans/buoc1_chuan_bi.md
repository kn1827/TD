# Bước 1 — Chuẩn bị: kế hoạch chi tiết

Thuộc `PLAN.md` (TD-MAD v5), mục 6, Bước 1. Thời gian: **07–20/10/2026**. Viết ngày 06/10/2026.

## 0. Bước 1 làm ra cái gì

Bước 1 không đo tính chất lây nhiễm nào. Nó chuẩn bị đủ nguyên liệu để Bước 2 đo ngay:

| # | Đầu ra | Dùng ở |
|---|---|---|
| 1 | Hai bộ câu hỏi đã đóng băng: GSM8K-Platinum và TruthfulQA (2 lựa chọn), chia sẵn phần A và phần B | Mọi bước |
| 2 | Kho lời giải: mỗi câu ứng viên có 30 lời giải do chính mô hình viết | Bước 2 (lời nhắn của "bạn"), Bước 3 |
| 3 | Danh sách **câu lưỡng lự** kèm độ hợp lý sẵn có π của từng đáp án | Bước 2 |
| 4 | Bộ **lời nhắn 4 mức chứng minh được**, đã qua kiểm tra thao tác | Bước 2 |
| 5 | **Bản đăng ký trước** thiết kế Bước 2 trên OSF (có dấu thời gian) | Giữ quyền ưu tiên |
| 6 | Bộ code chạy được trên Kaggle, có kiểm thử, ghi đủ nguồn gốc của mỗi kết quả | Mọi bước |

**Cổng ra khỏi Bước 1** (lấy từ `PLAN.md`):
- ≥ 30 câu lưỡng lự cho mỗi tổ hợp mô hình × loại câu;
- 4 mức lời nhắn qua kiểm tra thao tác;
- đọc được đáp án ở ≥ 95% số lượt;
- bản đăng ký trước đã nộp.

## 1. Thuật ngữ dùng trong bước này

| Thuật ngữ | Nghĩa |
|---|---|
| **Kho lời giải** | Với mỗi câu, cho mô hình tự giải nhiều lần độc lập (30 lần, nhiệt độ 0,7), lưu nguyên văn từng lời giải |
| **Độ hợp lý sẵn có π** | Tỉ lệ số lần mô hình tự chọn một đáp án trong 30 lần giải. π của đáp án đúng cộng π của các đáp án sai bằng 1 |
| **Câu lưỡng lự** | Câu có cả lời giải đúng lẫn lời giải sai trong kho, mỗi phía đủ nhiều lời giải *khác nhau* để dựng bảng bạn ở Bước 2 |
| **Chủng sai của một câu** | Đáp án sai mà mô hình hay chọn nhất cho câu đó |
| **Mức chứng minh được** | Mức một lời nhắn cho người đọc tự kiểm tra đúng sai: L0 chỉ đáp án; L1 lập luận không kiểm chứng được; L2 lập luận kiểm chứng được; L3 như L2 nhưng bị cắt mất bước then chốt |
| **Kiểm tra thao tác** (manipulation check) | Kiểm tra rằng 4 mức lời nhắn *thật sự* khác nhau về độ dễ kiểm tra, trước khi dùng chúng làm biến thí nghiệm |
| **Đọc logit** | Lấy xác suất mô hình gán cho từng chữ cái đáp án (A, B) mà không cho nó viết dài. Rẻ, nhưng có thể lệch so với đáp án nó viết ra |
| **Ghim phiên bản** | Ghi chính xác commit của mô hình, phiên bản vLLM và tham số chạy, để chạy lại ra cùng điều kiện |
| **Đăng ký trước** | Nộp trước lên OSF giả thuyết, thiết kế và cách phân tích của Bước 2, có dấu thời gian. Có thể để chế độ "embargo" (chưa công khai) tới ngày ra preprint |

## 2. Đọc gì

Đọc theo thứ tự. Mỗi dòng ghi rõ cần lấy ra điều gì, để đọc có mục đích.

### Nhóm A — đọc trước khi viết code (khoảng 1,5 ngày)

| # | Tài liệu | Lấy ra điều gì | Thời gian |
|---|---|---|---|
| A1 | Vendrow et al. 2025, *Do large language model benchmarks test reliability?* (GSM8K-Platinum), arXiv:2502.03461; thẻ dữ liệu trên Hugging Face | Cách họ sửa nhãn; ý nghĩa cột `cleaning_status` (consensus / verified / revised); vì sao 110 câu bị loại | 1 giờ |
| A2 | Lin, Hilton & Evans 2022, *TruthfulQA* (ACL); bài blog *New, improved multiple-choice TruthfulQA* (01/2025) | Cách dựng bản 2 lựa chọn từ cột `Best Answer` và `Best Incorrect Answer`; phải xáo thứ tự A/B | 1 giờ |
| A3 | Namjoo et al. 2026, *Judging by the cover: Cleaning LLM truthfulness benchmarks to avoid surface-level feature leakage*, arXiv:2609.13003 | TruthfulQA có "rò rỉ bề mặt": chỉ nhìn độ dài, cách diễn đạt cũng đoán được đáp án đúng. Họ phát hành bản đã làm sạch. Cần quyết định dùng bản nào | 1 giờ |
| A4 | Du et al. 2024, *Improving factuality and reasoning in language models through multiagent debate* (ICML), kèm code | Prompt vòng 0 cho toán (`gen_gsm.py`) và trắc nghiệm (`gen_mmlu.py`); cách họ trích đáp án (`eval_gsm.py`) | 1 giờ |
| A5 | Wang et al. 2023, *Self-consistency improves chain of thought reasoning* (ICLR) | Cơ sở cho việc ước lượng π bằng tần suất trong nhiều lần lấy mẫu | 30 phút |
| A6 | Báo cáo pilot `G1_phien1_bao_cao.md` (trong thư mục lưu trữ, `docs/old/`) | Ba lỗi đã gặp: bộ trích đáp án, chép nguyên chữ "(X)", vLLM tự thêm tham số lấy mẫu | 30 phút |
| A7 | Masoudian et al. 2026, *What we observe as LLM behavior can be a side-effect of inference backend*, arXiv:2608.04714 | Khoảng 39% dao động kết quả có thể đến từ phần mềm suy luận. Những gì phải ghi lại trong mỗi lần chạy | 45 phút |

### Nhóm B — đọc trước khi dựng lời nhắn 4 mức (khoảng 1 ngày)

| # | Tài liệu | Lấy ra điều gì | Thời gian |
|---|---|---|---|
| B1 | Laughlin & Ellis 1986, *Demonstrability and social combination processes on mathematical intellective tasks* (JESP) | Định nghĩa "chứng minh được": người nghe kiểm tra được lời giải bằng hệ thống chung (toán học) | 1 giờ |
| B2 | Kerr & Tindale 2004, *Group performance and decision making* (Annual Review of Psychology), phần về demonstrability | Bản tổng quan ngắn; các mức của demonstrability | 45 phút |
| B3 | Khan et al. 2024, *Debating with more persuasive LLMs leads to more truthful answers* (ICML), kèm code | Cách họ phân biệt trích dẫn đã kiểm chứng và chưa kiểm chứng trong lời của người tranh luận. Đây là mẫu cho L1 và L2 | 1 giờ |
| B4 | Tyen et al. 2024, *LLMs cannot find reasoning errors, but can correct them given the error location* (Findings of ACL), dữ liệu BIG-Bench Mistake | Cách xác định "bước sai đầu tiên". Dùng cho L3 và cho kiểm tra thao tác | 1 giờ |
| B5 | Zheng et al. 2024, *ProcessBench: Identifying process errors in mathematical reasoning*, arXiv:2412.06559 | Prompt chuẩn "tìm bước sai sớm nhất"; GSM8K là một trong bốn nguồn của họ | 45 phút |
| B6 | Hao et al. 2026, *Not all flips are conformity*, arXiv:2606.00820, phần thí nghiệm "information gradient" | Cách họ dựng lời giải "rỗng" (giống lập luận nhưng không có nội dung). Tham khảo cho L1 | 45 phút |

### Nhóm C — đọc trước khi viết bản đăng ký trước (khoảng 0,5 ngày)

| # | Tài liệu | Lấy ra điều gì | Thời gian |
|---|---|---|---|
| C1 | Nosek et al. 2018, *The preregistration revolution* (PNAS) | Đăng ký trước cần những mục nào; ghi sai lệch so với bản đăng ký thế nào | 45 phút |
| C2 | van Miltenburg, van der Lee & Krahmer 2021, *Preregistering NLP research* (NAACL) | Đăng ký trước cho nghiên cứu máy học và ngôn ngữ cần thêm gì: phiên bản mô hình, prompt, cách trích đáp án | 45 phút |
| C3 | Mẫu "OSF Preregistration" trên osf.io/prereg | Điền theo mẫu | 30 phút |

### Nhóm D — đọc dần trong tuần thứ hai, chuẩn bị cho Bước 2

| # | Tài liệu | Vì sao |
|---|---|---|
| D1 | Fukushima 2026, arXiv:2609.19183, **đọc toàn văn** | Bài trùng nhiều nhất (65%). Phải nắm rõ để viết đúng 4 điểm khác biệt |
| D2 | Qu, Fu & Hu 2026, arXiv:2606.01637 | Thiết kế liều {0, 2, 4, 6} và hạn chế "hai chiều trên hai tập câu khác nhau" |
| D3 | Banerjee & Moghaddas 2026, arXiv:2610.02654 | Đường giữ niềm tin; thí nghiệm tăng rồi giảm số agent kiên định để đo trễ |
| D4 | Hu & Qu 2026, arXiv:2607.05545 | Thiết kế điều kiện không-người-nói |
| D5 | Kadavath et al. 2022, *Language models (mostly) know what they know*; Wang et al. 2024, *"My answer is C": First-token probabilities do not match text answers in instruction-tuned language models* (Findings of ACL) | Đọc logit để ước lượng π; lý do không dùng logit thay cho đáp án viết ra |

## 3. Lấy code và dữ liệu ở đâu

| Cần | Lấy từ | Ghi chú |
|---|---|---|
| Prompt toán và trắc nghiệm của Du et al. | GitHub `composable-models/llm_multiagent_debate`: `gen_gsm.py`, `gen_mmlu.py`, `eval_gsm.py` | Chép nguyên văn. Với trắc nghiệm, giữ bản sửa của pilot: viết rõ "(A) or (B)" thay cho "(X)" |
| GSM8K-Platinum | Hugging Face `madrylab/gsm8k-platinum`, config `main`, split `test`: 1.209 câu; các cột `question`, `answer`, `cleaning_status` | Giấy phép: phần sửa nhãn CC BY-SA 4.0, dữ liệu gốc MIT. Cột `answer` có chú thích phép tính `<<48/2=24>>`, dùng để tìm bước then chốt |
| TruthfulQA, bản 2 lựa chọn | GitHub `sylinrl/TruthfulQA`, file `TruthfulQA.csv`, cột `Best Answer` và `Best Incorrect Answer` | Xáo A/B ngẫu nhiên theo seed cố định và lưu lại thứ tự đã xáo |
| TruthfulQA đã giảm rò rỉ bề mặt | `foadnamjoo/audit-prune` (GitHub và Hugging Face), Namjoo et al. 2026 | Chạy cả hai bản; bản sạch dùng để kiểm tra độ bền (quyết định ở mục 4.1) |
| Máy chạy mô hình | `pip install vllm`, máy chủ dạng OpenAI | Cờ bắt buộc: `--generation-config vllm`, `--seed 0`, `--max-logprobs 20`, `--dtype half` (T4 không có bf16) |
| Mô hình (đã ghim commit) | Qwen/Qwen2.5-7B-Instruct-AWQ; hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4; Qwen/Qwen3-4B-AWQ (tùy chọn) | Commit đã ghim nằm trong `configs/models.yaml` của thư mục lưu trữ. Llama phải đặt `date_string` cố định (mẫu prompt chèn ngày hôm nay). Qwen3 đặt `enable_thinking: false`. Gemma không chạy được trên T4 bằng vLLM |
| So đáp án toán | `pip install math-verify` (GitHub `huggingface/Math-Verify`) | Dùng làm cách so chính. Regex của pilot giữ làm dự phòng và để đối chiếu |
| Regex trích đáp án tham khảo | GitHub `EleutherAI/lm-evaluation-harness`, các task `gsm8k` và `truthfulqa` | Chỉ để đối chiếu, không chép nguyên |
| Prompt "tìm bước sai sớm nhất" | GitHub `QwenLM/ProcessBench` (dữ liệu: Hugging Face `Qwen/ProcessBench`) | Dùng cho kiểm tra thao tác |
| Dữ liệu bước sai có nhãn | GitHub `WHGTyen/BIG-Bench-Mistake` (Apache 2.0) | Tham khảo định dạng; chỉ cần nếu muốn hiệu chỉnh bộ kiểm tra |
| Cách đánh dấu trích dẫn đã/chưa kiểm chứng | GitHub `ucl-dark/llm_debate` (Khan et al. 2024) | Đọc để thiết kế L1 và L2, không cần chạy |
| Phần đã chạy ổn ở pilot | Thư mục lưu trữ: `kaggle/start_vllm.sh`, `tdmad/llm.py` (ghi đủ tham số trong mỗi request), `tdmad/answers.py` (bộ trích v2), `configs/models.yaml` | Chép có chọn lọc, đọc lại từng dòng. Phần còn lại viết mới |
| Mẫu đăng ký trước | osf.io/prereg, mẫu "OSF Preregistration" | Đặt embargo đến 31/01/2027 |

## 4. Code gì

### 4.0 Cấu trúc repo mới

```
PLAN.md                     kế hoạch tổng
plans/                      kế hoạch chi tiết từng bước
requirements.txt
configs/
  models.yaml               mô hình + commit ghim + tham số mẫu prompt
  step1.yaml                tham số Bước 1 (seed, số mẫu, ngưỡng chọn câu)
hoso/                       gói Python của đề án ("hồ sơ")
  data.py                   tải và đóng băng dữ liệu, chia phần A/B
  prompts.py                prompt vòng 0 (Du et al.) + phiên bản prompt
  answers.py                trích đáp án; so đáp án bằng Math-Verify
  llm.py                    client vLLM: ghi đủ tham số, cache, ghi nguồn gốc
  store.py                  kho lời giải; tính π; chọn câu lưỡng lự
  steps.py                  tách lời giải thành các bước; tìm bước then chốt
  messages.py               dựng lời nhắn L0–L3
  checks.py                 kiểm tra thao tác
  provenance.py             ghi nguồn gốc của mỗi lần chạy
scripts/
  s1_prepare_data.py
  s1_sample.py              lấy mẫu sơ bộ rồi bổ sung cho đủ 30
  s1_select.py
  s1_messages.py
  s1_check.py
kaggle/
  start_vllm.sh
  step1.ipynb
tests/
data/                       dữ liệu nhỏ đã đóng băng (đưa vào git) + MANIFEST.json
results/                    kết quả lớn (không đưa vào git)
prereg/step2_prereg.md      bản đăng ký trước (bản giữ trong repo)
```

### 4.1 Dữ liệu (`hoso/data.py`, `scripts/s1_prepare_data.py`)

- **GSM8K-Platinum:** tải 1.209 câu; giữ `question`, đáp án số cuối (sau `####`), lời giải chuẩn kèm chú thích phép tính, và `cleaning_status`.
- **TruthfulQA:** tạo bản 2 lựa chọn từ `TruthfulQA.csv`; xáo A/B bằng seed; lưu thứ tự đã xáo; đánh dấu câu nào nằm trong bản đã làm sạch của Namjoo et al.
  - *Quyết định cần chốt ngày 08/10:* dùng toàn bộ hay chỉ bản sạch. Đề xuất: chạy cả hai, lấy bản sạch làm kiểm tra độ bền.
- **Chia phần A/B cố định** bằng seed, khoảng 50/50, phân tầng theo `cleaning_status` (GSM8K) và theo nhóm chủ đề (TruthfulQA).
- **Đóng băng:** ghi `data/MANIFEST.json` gồm số câu, cách chia, và sha256 của từng file. Code kiểm tra hash trước mỗi lần chạy.

### 4.2 Prompt và trích đáp án (`hoso/prompts.py`, `hoso/answers.py`)

- **Prompt toán:** chép nguyên văn vòng 0 của Du (`gen_gsm.py`), yêu cầu đáp án dạng `\boxed{}`.
- **Prompt TruthfulQA:** theo mẫu trắc nghiệm của Du, liệt kê rõ "(A) or (B)".
- **Mọi prompt có `PROMPT_VERSION`.** Đổi một chữ thì phải tăng phiên bản.
- **Trích đáp án toán:** lấy `\boxed{}` cuối cùng; nếu không có thì lấy số cuối. So với đáp án chuẩn bằng Math-Verify. Ghi cách đọc được (`format`, `fallback` hay `none`).
- **Trích đáp án trắc nghiệm:** theo bộ trích v2 của pilot (có phòng việc chép chữ "(X)"). Ghi cách đọc được.
- **Đọc logit (phụ):** xác suất của "A" và "B" ở vị trí đáp án. Chỉ dùng nếu khớp với đáp án viết ra trên mẫu kiểm tra (D5).

### 4.3 Client mô hình (`hoso/llm.py`, `kaggle/start_vllm.sh`)

- Lấy lại `start_vllm.sh` của pilot, thêm `--generation-config vllm`.
- Mỗi request ghi rõ đủ các tham số: `temperature` 0,7; `top_p` 1,0; `top_k` -1; `repetition_penalty` 1,0; `max_tokens` 1024; `seed`; `n`.
- Dùng `n = 30` trong một request để các mẫu dùng chung phần prompt, cho nhanh hơn.
- Mỗi kết quả lưu kèm: nguyên văn; `finish_reason`; số token; seed; mô hình và commit; phiên bản vLLM và các cờ; `PROMPT_VERSION` và `PARSER_VERSION`.
- Chạy được tiếp sau khi Kaggle ngắt phiên ở mốc 12 giờ (đọc lại những gì đã có trên đĩa).

### 4.4 Kho lời giải và chọn câu (`hoso/store.py`, `scripts/s1_sample.py`, `scripts/s1_select.py`)

Lấy mẫu theo **hai đợt** để tiết kiệm GPU:
1. **Sàng:** mọi câu giải 10 lần.
2. **Bổ sung:** chỉ những câu có ít nhất 1 lời giải sai (toán) hoặc có cả A lẫn B (TruthfulQA) mới được giải thêm cho đủ 30 lần.

**Tính cho mỗi câu:**
- π của từng đáp án;
- chủng sai = đáp án sai có π cao nhất;
- số lời giải *khác nhau* ở phía đúng và ở phía chủng sai.

**Một câu là câu lưỡng lự khi:**
- có ≥ 6 lời giải khác nhau cho đáp án đúng **và** ≥ 6 cho chủng sai. Ở Bước 2, tổng số bạn lên tới 6, nên mỗi lượt cần tối đa 6 lời giải khác nhau mà không phải lặp;
- loại các lời giải bị cắt cụt (`finish_reason = length`) và các lời giải không đọc được đáp án.

**Phân tầng theo π** thành 3 mức: thấp, vừa, cao. Mục đích là để Bước 2 so được đúng với sai ở cùng mức dễ nghe.

**Báo cáo:** số câu lưỡng lự theo mô hình × loại câu × mức π.

### 4.5 Lời nhắn 4 mức (`hoso/steps.py`, `hoso/messages.py`, `scripts/s1_messages.py`)

**Áp dụng cho toán (GSM8K).** Với mỗi lời giải trong kho của câu lưỡng lự, dựng 4 phiên bản:

| Mức | Nội dung | Cách dựng |
|---|---|---|
| **L0** chỉ đáp án | "My answer is \boxed{X}." | Mẫu cố định, chèn đáp án của lời giải |
| **L1** lập luận không kiểm chứng được | Lời giải thích nghe có lý nhưng không có phép tính nào kiểm tra được | Cho chính mô hình viết lại lời giải của nó: "giải thích ngắn, chắc chắn, không nêu con số trung gian hay phép tính nào, giữ nguyên đáp án cuối". **Lọc tự động:** ngoài đáp án cuối, không được có con số nào; đáp án cuối không đổi. Không đạt thì viết lại, tối đa 3 lần, rồi bỏ |
| **L2** lập luận kiểm chứng được | Lời giải đầy đủ từng bước, có phép tính rõ ràng | Chính lời giải gốc trong kho |
| **L3** kiểm chứng được nhưng mất bước then chốt | L2 bị thay bước then chốt bằng "[...]" | Xem thuật toán bên dưới |

**Thuật toán tìm bước then chốt** (`hoso/steps.py`):
1. Tách lời giải thành các bước, theo dòng hoặc câu có dấu "=" hay có phép tính.
2. Rút kết quả trung gian của từng bước.
3. Lấy tập kết quả trung gian của lời giải chuẩn, từ các chú thích `<<a*b=c>>` của GSM8K-Platinum.
4. *Với lời giải sai:* bước then chốt là bước đầu tiên có kết quả không nằm trong tập của lời giải chuẩn, tức "bước sai đầu tiên" theo cách của BIG-Bench Mistake và ProcessBench.
5. *Với lời giải đúng:* bước then chốt là bước tính ra giá trị mà lời giải sai cùng câu tính khác (điểm vênh). Nếu không tìm được, lấy bước cuối trước đáp án.
6. Không xác định được bước then chốt thì lời giải đó không có L3. Ghi lại tỉ lệ lời giải bị loại.

**Áp dụng cho TruthfulQA:** chỉ 2 mức, vì đây là loại câu có mức chứng minh được thấp:
- *L0:* "My answer is (A)";
- *L-lý lẽ:* lời giải thích chính mô hình đã viết khi chọn đáp án đó.

**Đánh dấu nguồn và gom kết quả:**
- Lời nhắn nào cũng ghi rõ nó dựng từ lời giải nào, ở mức nào, và có qua bộ lọc không.
- Lưu vào `results/step1/messages_<mô hình>_<loại câu>.jsonl`.

### 4.6 Kiểm tra thao tác (`hoso/checks.py`, `scripts/s1_check.py`)

**Mục tiêu.** Chứng minh rằng 4 mức thật sự khác nhau về độ dễ kiểm tra, trước khi dùng ở Bước 2.

**Kiểm tra bằng máy:**
- Lấy mẫu 100 lời nhắn mỗi mức × mỗi chủng × mỗi mô hình.
- Cho mô hình làm "người kiểm tra" ở nhiệt độ 0. Prompt theo ProcessBench: "Lời giải này đúng không? Nếu sai, bước sai sớm nhất là bước nào?"
- Đo khả năng kiểm tra phân biệt lời nhắn đúng với lời nhắn sai (AUC) ở từng mức.

**Kỳ vọng:** L2 > L3 > L1 ≥ L0.
- *Đạt* khi AUC(L2) − AUC(L0) ≥ 0,10 và L3 nằm giữa L2 và L1 (đặt trước trong bản đăng ký).
- *Không đạt:* sửa cách dựng; nếu vẫn không đạt thì gộp còn 2 mức (L0 và L2), như phương án dự phòng trong `PLAN.md`.

**Kiểm tra bằng người:** bạn tự đọc 40 lời nhắn ngẫu nhiên (10 mỗi mức) để bắt lỗi dựng, ví dụ L1 còn sót con số, hay L3 cắt nhầm bước.

**Báo cáo:** `results/step1/check.md`.

### 4.7 Ghi nguồn gốc và kiểm thử (`hoso/provenance.py`, `tests/`)

**Mỗi lần chạy ghi một file nguồn gốc gồm:**
- commit git của code;
- mô hình và commit của nó;
- phiên bản vLLM và các cờ;
- toàn bộ tham số lấy mẫu;
- `PROMPT_VERSION`, `PARSER_VERSION`;
- hash của dữ liệu.

**Kiểm thử (chạy được trên laptop, không cần GPU):**
- Bộ trích đáp án: toán (`\boxed{}`, phân số, dấu phẩy ngăn số, không có `\boxed{}`); trắc nghiệm (chép "(X)", nhắc nhiều chữ cái).
- Cách chia A/B ra cùng kết quả mỗi lần chạy.
- Hash dữ liệu khớp.
- Bộ lọc L1: phát hiện khi có con số lọt vào.
- Thuật toán bước then chốt: chạy trên 5 lời giải mẫu viết tay.
- Có một client giả (không gọi mô hình) để chạy thử toàn bộ đường đi của Bước 1.

### 4.8 Bản đăng ký trước (`prereg/step2_prereg.md`, nộp lên OSF)

Điền mẫu OSF Preregistration cho **Bước 2**:
- **Giả thuyết:** dấu của Δ cho ε, p₁, κ/θ, h và H; giả thuyết về mức chứng minh được (sự thật có lợi thế lớn hơn ở L2 so với L0).
- **Thiết kế:** các yếu tố và mức; liều k = 0…4; tổng số bạn 2, 4, 6; điều kiện không-người-nói; số lần lặp.
- **Cỡ mẫu:** số câu lưỡng lự lấy từ mục 4.4; tính cỡ mẫu sơ bộ từ số liệu pilot.
- **Loại trừ:** lượt không đọc được đáp án; lời giải bị cắt cụt.
- **Phân tích:**
  - mô hình hồi quy có hiệu ứng ngẫu nhiên theo câu;
  - đường liều log-logistic;
  - so lây đơn/lây phức bằng BIC;
  - so "theo số người" với "theo tỉ lệ";
  - phân tích sống sót cho độ bền;
  - khoảng tin cậy bootstrap theo câu.
- **Thứ tự chính/phụ:** ghi rõ phân tích nào là chính, phân tích nào là khám phá.

## 5. Chạy trên Kaggle: khối lượng và chi phí

| Việc | Khối lượng mỗi mô hình | Ước tính GPU-giờ |
|---|---|---|
| Chạy thử nhỏ | 20 câu × 2 loại × n = 4 | 0,2 |
| Sàng | (1.209 + khoảng 790) câu × 10 lời giải | khoảng 1,5 |
| Bổ sung đủ 30 | Khoảng 300 câu ứng viên × 20 lời giải | khoảng 1 |
| Dựng L1 (mô hình viết lại) | Khoảng 40 câu × 12 lời giải, tối đa 3 lần thử | khoảng 0,3 |
| Kiểm tra thao tác | 100 × 4 mức × 2 chủng | khoảng 0,3 |
| **Cộng** | | **khoảng 3,3 mỗi mô hình.** 2 mô hình chạy song song trên 2 GPU, cần khoảng 3,5 giờ phiên |

Tốc độ lấy theo pilot: khoảng 800 token/giây khi lấy nhiều mẫu cùng prompt. Đo lại ở lần chạy thử; nếu lệch hơn 30% thì cập nhật bảng này.

## 6. Lịch theo ngày

| Ngày | Việc | Xong khi |
|---|---|---|
| 07/10 | Đọc A1–A7. Dựng khung repo, `requirements.txt`, `configs/` | Repo có khung; kiểm thử rỗng chạy được |
| 08/10 | `hoso/data.py`, `s1_prepare_data.py`; tải dữ liệu; chia A/B; MANIFEST. Chốt dùng bản TruthfulQA nào | Kiểm thử dữ liệu qua |
| 09/10 | `prompts.py`, `answers.py` (Math-Verify + bộ trích v2), `llm.py`, `start_vllm.sh`, `provenance.py` | Kiểm thử trích đáp án qua; client giả chạy hết đường đi |
| 10/10 | **Chạy thử trên Kaggle**: 2 mô hình, 20 câu mỗi loại | Đọc được ≥ 95% đáp án; có tốc độ thực tế |
| 11–12/10 | Sàng (n = 10) rồi bổ sung (lên 30) trên Kaggle, khoảng 1 phiên | Kho lời giải đủ |
| 13/10 | Tính π, chọn câu lưỡng lự, phân tầng | **Cổng 1:** ≥ 30 câu mỗi tổ hợp |
| 14/10 | Đọc B1–B6. `steps.py` (tìm bước then chốt) và kiểm thử | Thuật toán đúng trên 5 lời giải mẫu |
| 15/10 | `messages.py`: L0, L2, L3; chạy dựng L1 trên Kaggle | Có đủ 4 mức |
| 16/10 | Kiểm tra thao tác trên Kaggle; tự đọc 40 lời nhắn | Có `check.md` |
| 17/10 | Sửa lời nhắn nếu chưa đạt; chạy lại phần cần thiết | **Cổng 2:** 4 mức đạt |
| 18–19/10 | Đọc C1–C3. Viết bản đăng ký trước cho Bước 2; nộp OSF (embargo) | **Cổng 3:** có mã số đăng ký |
| 20/10 | Báo cáo Bước 1; viết `plans/buoc2_do_ho_so.md` | Sẵn sàng Bước 2 |

Đọc nhóm D rải rác trong 14–20/10.

## 7. Rủi ro của Bước 1

| Rủi ro | Dấu hiệu | Cách xử lý |
|---|---|---|
| Qwen giải GSM8K quá tốt, ít câu lưỡng lự | < 30 câu sau khi sàng | Hạ ngưỡng xuống ≥ 4 lời giải khác nhau mỗi phía (Bước 2 khi đó giới hạn tổng số bạn ở 4); thêm MATH-500 cấp 1–3 (Hugging Face `HuggingFaceH4/MATH-500`) |
| L1 vẫn lọt con số | Bộ lọc loại > 50% | Đổi prompt viết lại; cho phép tối đa 3 lần thử; nếu vẫn không được thì dùng mẫu lập luận rỗng kiểu Hao et al. 2026 |
| Không tìm được bước then chốt ở nhiều lời giải | > 40% lời giải không có L3 | Chỉ dùng L3 cho câu tìm được; báo cáo tỉ lệ; giữ L0, L1, L2 cho mọi câu |
| TruthfulQA rò rỉ bề mặt | Bộ phân loại 6 đặc trưng của Namjoo et al. đoán đúng nhiều | Dùng bản đã làm sạch làm tập chính |
| Mô hình viết đáp án sai định dạng | Đọc được < 95% | Sửa prompt (tăng `PROMPT_VERSION`); đọc logit làm phương án phụ cho TruthfulQA |
| Kaggle đổi phiên bản vLLM | Server không chạy hoặc kết quả lệch | Ghim phiên bản vLLM trong notebook; ghi phiên bản vào file nguồn gốc |

## 8. Tài liệu (Bước 1)

- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*. Code: github.com/composable-models/llm_multiagent_debate.
- Kadavath, S., et al. (2022). Language models (mostly) know what they know. arXiv:2207.05221.
- Kerr, N. L., & Tindale, R. S. (2004). Group performance and decision making. *Annual Review of Psychology*, 55, 623–655.
- Khan, A., Hughes, J., Valentine, D., Ruis, L., Sachan, K., Radhakrishnan, A., Grefenstette, E., Bowman, S. R., Rocktäschel, T., & Perez, E. (2024). Debating with more persuasive LLMs leads to more truthful answers. *ICML 2024*. Code: github.com/ucl-dark/llm_debate.
- Laughlin, P. R., & Ellis, A. L. (1986). Demonstrability and social combination processes on mathematical intellective tasks. *Journal of Experimental Social Psychology*, 22(3), 177–189.
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*. Data: github.com/sylinrl/TruthfulQA. Bản 2 lựa chọn: bài blog "New, improved multiple-choice TruthfulQA" (01/2025).
- Masoudian, Shafaei, Swain & Schedl (2026). What we observe as LLM behavior can be a side-effect of inference backend. arXiv:2608.04714.
- Namjoo, F., Ogasawara, R., Abdullah, A., Anderson, C., Oozeer, N. F., & Phillips, J. M. (2026). Judging by the cover: Cleaning LLM truthfulness benchmarks to avoid surface-level feature leakage. arXiv:2609.13003. Code và dữ liệu: foadnamjoo/audit-prune.
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606.
- Tyen, G., Mansoor, H., Cărbune, V., Chen, P., & Mak, T. (2024). LLMs cannot find reasoning errors, but can correct them given the error location. *Findings of ACL 2024*. Data: github.com/WHGTyen/BIG-Bench-Mistake.
- van Miltenburg, E., van der Lee, C., & Krahmer, E. (2021). Preregistering NLP research. *NAACL 2021*.
- Vendrow, J., Vendrow, E., Beery, S., & Madry, A. (2025). Do large language model benchmarks test reliability? arXiv:2502.03461. Data: huggingface.co/datasets/madrylab/gsm8k-platinum.
- Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.
- Wang, X., Ma, B., Hu, C., Weber-Genzel, L., Röttger, P., Kreuter, F., Hovy, D., & Plank, B. (2024). "My answer is C": First-token probabilities do not match text answers in instruction-tuned language models. *Findings of ACL 2024*.
- Zheng, C., et al. (2024). ProcessBench: Identifying process errors in mathematical reasoning. arXiv:2412.06559. Code: github.com/QwenLM/ProcessBench.
- Công cụ: vLLM (github.com/vllm-project/vllm; cờ `--generation-config vllm`); Math-Verify (github.com/huggingface/Math-Verify); lm-evaluation-harness (github.com/EleutherAI/lm-evaluation-harness).

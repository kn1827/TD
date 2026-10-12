# Giải thích source code: TD (`hmad/`) và Debate-or-Vote (Choi et al. 2025)

Viết ngày 12/10/2026, theo commit `66e8d31` của TD và commit `82c929e` của repo gốc `debate-or-vote/`.

Tài liệu có ba phần:
- **Phần A**: repo gốc Debate-or-Vote (thư mục `debate-or-vote/`, chỉ để đọc, không đưa lên git).
- **Phần B**: code của TD (`hmad/`, `configs/`, `scripts/`, `tests/`, `kaggle/`).
- **Phần C**: hai bên khác nhau ở đâu, chỉ rõ dòng code của từng bên.

Cách đọc tham chiếu: `hmad/run.py:229-273` nghĩa là file `hmad/run.py`, dòng 229 đến 273. Đường dẫn của repo gốc bắt đầu bằng `debate-or-vote/src/`.

---

## 0. Đọc nhanh

- Cả hai bên chạy **cùng một kiểu debate: Society of Mind (SoM)**. Ở mỗi vòng, mỗi agent đọc câu trả lời vòng trước của các agent "hàng xóm" và của chính mình, rồi trả lời lại. Đáp án của cả nhóm là đáp án đa số.
- **Phần debate giống hệt nhau**: chữ trong prompt, ba đồ thị, cách đọc đáp án `{final answer: …}`, cách bỏ phiếu. `tests/test_choi_fidelity.py` chạy hàm gốc và hàm của TD trên cùng đầu vào, kết quả phải trùng.
- **Khác nhau lớn nhất**: Choi dùng **một mô hình nhân bản N lần**, còn TD cho **mỗi vị trí một mô hình thuộc họ khác**. GPU mỗi lúc chỉ chứa được một mô hình, nên TD phải đổi thứ tự chạy:
  - Choi chạy xong mọi vòng của một câu hỏi rồi mới sang câu sau;
  - TD chạy theo vòng: trong mỗi vòng, lần lượt từng mô hình sinh mọi lượt của nó trong tất cả debate.
- **Khác nhau thứ hai** (từ run `hetero2`): TD bọc prompt trong chat template của từng mô hình, Choi đưa chữ thô.
- Các khác biệt còn lại nằm ngoài debate: thư viện sinh chữ (HF `generate` hay vLLM), cách cố định ngẫu nhiên, số câu hỏi (100 thay vì 300), cách lưu kết quả, khả năng chạy tiếp khi bị ngắt.

---

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| agent | Một người tham gia debate. Ở đây là một lần gọi mô hình ngôn ngữ ở một vị trí |
| vị trí (position) | Số thứ tự 0…n−1 của agent trong debate. Vị trí quyết định agent đọc câu trả lời của ai |
| vòng (round) | Vòng 0: mỗi agent trả lời độc lập. Vòng t ≥ 1: trả lời lại sau khi đọc ý kiến của vòng t−1 |
| đồ thị (graph) | Ai đọc ai. **decentralized**: đọc mọi agent khác. **sparse**: chỉ đọc i−1 và i+1 (xếp thành vòng tròn). **centralized**: agent 0 (gọi là "tâm") đọc mọi agent, các agent khác chỉ đọc tâm |
| đa số (majority vote) | Đáp án xuất hiện nhiều nhất trong các đáp án đọc được. Hòa thì chọn ngẫu nhiên |
| suffix | Câu hướng dẫn nối sau câu hỏi, bảo mô hình ghi đáp án dạng `{final answer: 123}` |
| parser | Đoạn code đọc đáp án từ câu trả lời |
| prompt thô | Đưa nguyên chuỗi chữ vào mô hình, không thêm gì |
| chat template | Khuôn hội thoại riêng của mỗi mô hình instruct: đánh dấu chỗ lời người dùng bắt đầu và kết thúc, chỗ lượt trả lời của mô hình bắt đầu |
| token | Mẩu chữ nhỏ mà mô hình đọc và sinh ra (khoảng ¾ một từ tiếng Anh) |
| BOS / EOS | Token đặc biệt "bắt đầu văn bản" / "kết thúc". Sinh ra EOS thì mô hình dừng |
| temperature, top_p, top_k | Tham số lấy mẫu khi sinh chữ. temperature 1,0 giữ nguyên xác suất của mô hình; top_p 0,9 chỉ chọn trong nhóm token chiếm 90% xác suất; top_k k chỉ chọn trong k token có xác suất cao nhất |
| repetition_penalty | Hệ số phạt token đã xuất hiện, để mô hình đỡ lặp lại. 1,0 là không phạt |
| `generation_config.json` | File đi kèm trọng số mô hình, ghi các tham số sinh mặc định của mô hình đó |
| HF `generate` | Hàm sinh chữ của thư viện Hugging Face Transformers. Choi dùng hàm này |
| vLLM | Thư viện sinh chữ nhanh, xếp hàng hàng nghìn prompt cùng lúc. TD dùng thư viện này |
| tensor parallel (TP) | Chia một mô hình lên nhiều GPU. TP = 2 nghĩa là một mô hình nằm trên cả hai T4 |
| fp16 | Số thực 16 bit. GPU T4 không hỗ trợ bf16 nên dùng fp16 |
| phase | (TD) Một bước chạy = một mô hình × một vòng |
| resume | Chạy lại đúng lệnh cũ sau khi bị ngắt; bước nào đã xong thì bỏ qua |
| seed | Hạt giống ngẫu nhiên: cùng seed thì cùng chuỗi số ngẫu nhiên |

---

## Phần A — Repo gốc Debate-or-Vote (Choi, Zhu & Li, NeurIPS 2025)

### A.1 Cấu trúc

```
debate-or-vote/
  src/main.py                 vòng lặp debate, tạo prompt mỗi vòng (get_new_message)
  src/evaluator.py            suffix, parser, bỏ phiếu
  src/model/model_utils.py    engine(): sinh chữ; get_agents(): nạp mô hình, persona
  src/model/llama.py, qwen.py lớp bọc mô hình HF (phần lớn hàm không dùng cho debate)
  src/data/*.py               nạp 8 bộ dữ liệu
  scripts/*.sh                lệnh chạy các thí nghiệm trong bài báo
```

### A.2 Luồng chạy của `main.py`

Lệnh mẫu (`scripts/gsm8k.sh`):
```
python src/main.py --model qwen2.5-7b --num_agents 5 --data gsm8k --data_size 300 --debate_rounds 5 [--sparse | --centralized]
```

1. **Khởi động** (`main.py:323-335`): đặt seed 42 chung cho `torch`, `numpy`, `random`; đọc token Hugging Face từ file `token`.
2. **Nạp mô hình** (`main.py:140` → `model/model_utils.py:50-108`): nạp **một** mô hình (Llama-3.1-8B, Qwen2.5-7B hoặc Qwen2.5-32B) ở fp16, `device_map="auto"`.
   - Pad token = EOS, thêm đệm vào bên trái (`model/llama.py:36-37`, `model/qwen.py:31-32`).
   - Có `--multi_persona` thì kèm một từ điển persona (system prompt lấy từ bài DyLAN, `model_utils.py:70-102`). Không có thì `{"None": ""}`.
3. **Nạp dữ liệu** (`main.py:143`): `load_data(args, split='test')`.
4. **Chọn suffix và hàm chấm** (`main.py:160-175`), tùy bộ dữ liệu và cờ `--bae`.
5. **Vòng lặp theo từng câu hỏi** (`main.py:183`). Với mỗi câu:
   - **Vòng 0** (`main.py:190-197`): cùng một tin nhắn `x + SUFFIX` nhân lên `num_agents` lần, gọi `engine` một lần để sinh N câu trả lời. Các agent khác nhau chỉ nhờ lấy mẫu ngẫu nhiên.
   - **Chấm** (`main.py:200-204`): đồ thị centralized thì **chỉ chấm câu trả lời của agent đầu tiên (tâm)**; đồ thị khác thì đọc đáp án của cả N agent rồi bỏ phiếu.
   - **Vòng 1…R** (`main.py:232-287`): `get_new_message` tạo prompt mới, `engine` sinh, rồi chấm như trên.
   - **Lưu** (`main.py:294-296`): ghi lại toàn bộ `out/history/{fname}.jsonl` sau mỗi câu; in độ chính xác từng vòng (`main.py:309-311`).
6. **Kết thúc** (`main.py:313-315`): thêm một dòng vào `out/logs.tsv`.

### A.3 Prompt vòng t ≥ 1: `get_new_message` (`main.py:72-134`)

Nhánh decentralized và sparse (`main.py:80-97`):
```python
msg = "These are the recent opinions from other agents: "
if args.sparse :
    peers = [agents[(i-1) % len(agents)], agents[(i+1) % len(agents)]]
else :
    peers = agents[:i]+agents[i+1:]
for other_agent in peers:
    msg += f"\n\nOne of the agents' response: \n{responses[other_agent]}\n"
msg += f"\n\nThis was your most recent opinion:\n{responses[agents[i]]}\n"
msg += f'\n\nUse these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question:\n{sample}'
```

Điểm cần nhớ:
- **Không có trí nhớ.** Mỗi vòng là một tin nhắn mới. Câu trả lời cũ của chính agent được dán vào dưới dạng chữ ("This was your most recent opinion"), không nằm trong lịch sử hội thoại.
- **Nhánh centralized** (`main.py:99-119`): agent 0 nhận đúng prompt như decentralized (đọc mọi người). Các agent khác nhận "This is the recent opinion from another agent:" kèm câu trả lời của agent 0.
- **Nhánh một agent** (`main.py:121-132`): tự sửa câu trả lời của mình. TD không dùng.
- Có persona thì tin nhắn thành danh sách `[system, user]` (`main.py:94-95`); không có thì là một dict `{'role': 'user', ...}`. Kiểu dữ liệu này quyết định có dùng chat template hay không (mục A.4).

### A.4 Sinh chữ: `engine` (`model/model_utils.py:11-45`)

```python
if type(messages[0]) == list :
    prompts = [agent.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) for msgs in messages]
else :
    prompts = [msg['content'] for msg in messages]  # we find that NOT using chat template is better in MAD
inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
...
outputs = agent.huggingface_model.generate(input_ids, attention_mask=attention_mask,
    pad_token_id=agent.tokenizer.eos_token_id, max_new_tokens=512, ...,
    do_sample=True, temperature=1.0, top_p=0.9, num_return_sequences=1, ...)
```

- **Không có persona thì dùng prompt thô**, không bọc chat template. Dòng chú thích ghi lý do: tác giả thấy không dùng template thì debate tốt hơn.
- Một lần gọi = N prompt của **một câu hỏi** (một lô, đệm trái). `truncation=True`: prompt dài quá giới hạn của tokenizer thì bị cắt mà không báo.
- Lấy mẫu: temperature 1,0, top_p 0,9, tối đa 512 token mới. Tham số nào không truyền vào (top_k, repetition_penalty) thì HF `generate` lấy từ `generation_config.json` của mô hình, nếu file không có thì top_k = 50. Ví dụ Qwen2.5-7B-Instruct có top_k 20, repetition_penalty 1,05.
- Giải mã (`model_utils.py:39-42`): chỉ lấy phần mới sinh, bỏ token đặc biệt.

### A.5 Chấm điểm (`evaluator.py`)

- **`get_instruction_suffix`** (`evaluator.py:17-40`): ba kiểu suffix.
  - Mặc định: `{final answer: 123}` cho GSM8K, `{final answer: (A)}` cho trắc nghiệm.
  - `--cot`: thêm "Let's think step by step."
  - `--bae`: ghi đáp án ở cuối, không cần ngoặc nhọn.
- **`evaluate_arithmetics`** (`evaluator.py:44-66`), dùng cho arithmetics và GSM8K:
  - lấy cặp `{...}` **cuối cùng**, bỏ chữ "final answer:", đổi sang số thực, làm tròn 1 chữ số thập phân;
  - lỗi ở bất kỳ bước nào thì đáp án là `""` (không đọc được). Ví dụ `{final answer: 92 square feet}` cho ra `""`.
- **`evaluate_mcq`** (`evaluator.py:69-96`), dùng cho trắc nghiệm: `{final answer: (B)}` → `(B)`; `{final answer: B}` → `(B)`; chuỗi dài từ 3 ký tự trở lên thì lấy ký tự thứ hai, vì giả định chuỗi có dạng `(X)…`. Ví dụ `(C) bank` → `(C)`.
- **Bỏ phiếu** (nằm trong hai hàm trên):
  - đếm các đáp án khác `""`;
  - hòa thì `random.choice` với bộ sinh ngẫu nhiên **chung** của Python (`evaluator.py:64`, `:95`);
  - mọi agent đều không đọc được thì đáp án debate là `""`.
- `base_evaluate_*` (`evaluator.py:117-162`) dùng cho `--bae`: lấy số cuối cùng, hoặc `(X)` cuối cùng. `evaluate_gen` dùng cho CNN/DailyMail, chấm bằng ROUGE.

### A.6 Dữ liệu (`src/data/`)

| File | Nguồn | Khi `split='test'` |
|---|---|---|
| `gsm8k.py` | `openai/gsm8k` (main), tập test | Xáo với `random_state=0`, lấy `data_size` câu đầu (`gsm8k.py:23`). Đáp án là số nguyên sau `####` |
| `csqa.py` | `tau/commonsense_qa`, tập validation | Xáo với `random_state=0`, **luôn lấy 300 câu, không đọc `data_size`** (`csqa.py:13`). Bỏ câu không đủ 5 lựa chọn; câu hỏi được gắn thêm (A)…(E) |
| `arithmetics.py` | Tự sinh | `numpy.random.default_rng(1)`, 6 số từ 0 đến 29, biểu thức `a+b*c+d-e÷f` |
| `hellaswag.py` | `Rowan/hellaswag` | 300 câu |
| `mmlu_formal_logic.py`, `mmlu_pro_medicine.py` | `cais/mmlu` | |
| `hh_rlhf.py` | `Anthropic/hh-rlhf` | `data_size` câu |
| `cnn_daily.py` | `abisee/cnn_dailymail` 3.0.0 | |

### A.7 Có trong repo nhưng debate không dùng tới

- Các tham số được khai báo nhưng không dùng ở đâu: `--solver`, `--generate_first_round`, `--max_num_agents`, `--alpha`, `--agent_selection` (`main.py:45`, `:61-64`), `--split`, `--sub_data`, `--debug`, `--verbose`.
- Trong `LlamaWrapper` / `QwenWrapper`, các hàm `__call__`, `forward_whole_dataset`, `change_lora_adapter`, `apply_aligner`, `generate` (sinh tham lam) đều không được gọi. Debate chỉ dùng `huggingface_model` và `tokenizer`. `QwenWrapper` còn dựng sẵn một `pipeline` không dùng (`qwen.py:34-41`).
- `data/base_ds.py: format_ds` được import nhưng không được gọi.
- **"Heterogeneous" trong repo gốc** (`scripts/heterogeneous.sh`, cờ `--multi_persona`) nghĩa là **cùng một mô hình, khác system prompt (persona)**. Nó không phải là các mô hình khác nhau như ở TD.

---

## Phần B — Code TD

### B.1 Sơ đồ

```
configs/exp.yaml    --load_config-->  cfg       --+
configs/models.yaml --load_models-->  registry  --+
                                                  v
hmad/run.py  main()
  1. check_protocol          -> results/<run>/protocol.json (ghi một lần; khác thì từ chối chạy)
  2. build_plan (plan.py)    -> plan.json  (câu hỏi lấy bằng choi.load_task)
  3. với mỗi bước (vòng r, mô hình m) trong schedule(...):
       build_jobs  (choi.first_prompt / choi.round_messages) -> phases/r<r>/<m>.jobs.jsonl
       WeightStore.get (weights.py)                           -> thư mục trọng số
       start_worker: tiến trình con `python -m hmad.worker --spec ...`
            worker.py: vLLM sinh chữ -> <m>.out.jsonl, rồi <m>.meta.json (dấu "đã xong")
       finish_worker
  4. assemble (results.py, chấm bằng choi.evaluate) -> debates.jsonl, summary.md
```

Chỉ `hmad/choi.py` chứa logic debate (prompt, đồ thị, parser, bỏ phiếu, nạp dữ liệu). Các file còn lại lo việc chạy: lên lịch, nạp mô hình, lưu file, chạy tiếp.

### B.2 `configs/`

**`exp.yaml`**, file cấu hình của run chính `hetero2`:

| Khóa | Giá trị | Ý nghĩa |
|---|---|---|
| `run_name` | `hetero2` | Thư mục kết quả `results/hetero2/` |
| `seed` | 0 | Gốc của mọi seed (hoán vị vị trí, seed từng lượt sinh, phá hòa) |
| `unique_families` | `false` | Cho phép một họ xuất hiện hai lần, vì thiết lập 12 agent cần nhiều hơn 8 họ |
| `tasks` | gsm8k, csqa, mỗi bộ 100 câu | |
| `rounds` | 4 | Vòng 0 + 4 vòng debate |
| `setups` | main (6 agent × 3 đồ thị), extended (12 agent, sparse), baseline (3 agent, decentralized) | |
| `sampling` | temperature 1,0; top_p 0,9; max_tokens 512; `hf_generate_defaults: true`; `chat_template: true` | |
| `engine` | gpus [0, 1], TP 2, fp16, max_model_len 8192, `enforce_eager`, giới hạn đĩa 40 GB | |

Số lượng: 600 debate main (2 bộ × 100 câu × 3 đồ thị), 200 debate extended, 200 debate baseline, tổng 1.000 debate.

Số lượt sinh mỗi vòng của từng mô hình, đọc từ `results/hetero1/phases/r0/*.meta.json`:

| Mô hình | Lượt / vòng | Vì sao |
|---|---|---|
| qwen2.5-7b, llama3.1-8b, mistral-7b | 1.200 | main 600 + extended 2 × 200 + baseline 200 |
| exaone3.5-7.8b | 1.000 | main 600 + extended 2 × 200 |
| granite3.3-8b, internlm3-8b | 800 | main 600 + extended 200 |
| falcon3-7b, kanana1.5-8b | 200 | chỉ có trong extended |

Tổng 6.600 lượt mỗi vòng, nhân 5 vòng là 33.000 lượt.

**Các file config khác:**
- `smoke.yaml`: kế thừa `exp.yaml` (`extends`), mỗi bộ 4 câu, 1 vòng, 8 mô hình, `continue_on_error: true`.
- `exp_slm.yaml`: phương án mô hình nhỏ 2–5B, TP 1, hai mô hình chạy song song, mỗi GPU một mô hình.
- `models.yaml`: danh bạ mô hình. Mỗi mô hình có `rank` (thứ tự theo lượt tải), `hf_id`, `revision` (commit được ghim), `family`, `params`, `layers`, `kv_heads`, `head_dim` (để ước tính bộ nhớ), `ctx` (độ dài ngữ cảnh), `trust_remote_code`. Mục `reserves` chứa các mô hình dự phòng.

### B.3 `hmad/config.py`

- `load_config` (`config.py:29-35`): đọc YAML. Nếu có `extends` thì nạp file cha rồi đè file con lên. Dict được gộp từng tầng, riêng `setups` và `tasks` bị **thay nguyên khối** (`REPLACED_KEYS`, dòng 11). Nhờ vậy `smoke.yaml` chỉ chạy đúng thiết lập nó khai báo.
- `load_models` (`config.py:38-46`): gộp `models` và `reserves` (đánh dấu `reserve=True`), thêm trường `key`.
- `stable_seed(*parts)` (`config.py:53-54`): lấy SHA-256 của các phần nối lại, cắt thành số nguyên 32 bit. Cùng đầu vào thì cùng seed trên mọi máy. Hàm `hash()` của Python không làm được vậy, vì nó đổi giá trị mỗi lần chạy.

### B.4 `hmad/choi.py`: code của Choi chuyển sang

| Hàm TD | Lấy từ | Việc |
|---|---|---|
| `instruction_suffix(data, bae, cot)` (40-62) | `evaluator.py:17-40` | Suffix, chép nguyên văn |
| `first_prompt(question, suffix)` (65-67) | `main.py:195` | Vòng 0 = câu hỏi + suffix |
| `peers_of(i, n, graph)` (74-82) | `main.py:82-85`, `:103` | Danh sách vị trí mà agent i đọc |
| `round_messages(question, responses, graph, suffix)` (85-105) | `main.py:72-119` | Prompt vòng t ≥ 1 cho mọi vị trí; chuỗi giống hệt từng ký tự |
| `parse_number` (112-120), `parse_choice` (123-135) | `evaluator.py:49-54`, `:74-86` | Parser |
| `vote(final_answers, rng)` (138-145) | `evaluator.py:57-64` | Đa số; hòa thì chọn bằng `rng` truyền vào |
| `evaluate(data, responses, answer, rng)` (148-161) | `evaluate_arithmetics`, `evaluate_mcq` | Trả về (đáp án từng agent, đáp án đa số, đa số đúng?, từng agent đúng?) |
| `load_gsm8k` (171-181), `load_csqa` (184-199), `load_arithmetics` (202-219) | `data/gsm8k.py`, `csqa.py`, `arithmetics.py` | Cùng nguồn, cùng cách xáo, cùng định dạng câu hỏi |

Phần docstring đầu file (dòng 9-14) liệt kê ba chỗ khác về giao diện hàm, xem mục C.2.12.

### B.5 `hmad/plan.py`: danh sách debate

- `Debate` (20-33): `id`, `setup`, `task`, `item`, `question`, `gold`, `graph`, `models`. Trong đó `models[i]` là mô hình ngồi ở vị trí i.
- `build_plan` (36-58):
  - kiểm tra mọi mô hình có trong danh bạ (42-44);
  - nếu `unique_families` bật thì không cho hai agent cùng họ (45-47);
  - với mỗi (thiết lập, đồ thị, bộ dữ liệu, câu hỏi k), **xáo thứ tự mô hình bằng seed `(seed, setup, task, k, graph)`** (55).

  Lý do phải xáo: vị trí có ý nghĩa. Trong centralized, vị trí 0 là tâm; trong sparse, vị trí i đọc i−1 và i+1. Xáo ngẫu nhiên có seed thì qua 100 câu, mỗi mô hình ngồi mỗi vị trí với số lần gần bằng nhau.
- `id` của debate có dạng `"main|sparse|gsm8k|17"`. `plan.json` được ghi một lần; những lần chạy sau đọc lại đúng file đó.

### B.6 `hmad/run.py`: điều phối

- **`protocol` / `check_protocol`** (45-66): ghi `protocol.json` gồm số vòng, seed, tasks, setups, sampling, tham số engine, hf_id và revision của mô hình. Lần chạy sau mà khác bất kỳ mục nào thì dừng với thông báo "differs in […]". Như vậy không thể trộn hai giao thức trong một thư mục.
- **`model_order`** (69-71): xếp mô hình theo `rank`.
- **`gpu_groups`** (74-81): chia danh sách GPU thành từng cụm `tp` GPU. Với `exp.yaml` ([0, 1], TP 2) ra `[[0, 1]]`, tức mỗi lúc một mô hình. Với `exp_slm.yaml` (TP 1) ra `[[0], [1]]`, tức hai mô hình cùng lúc.
- **`schedule`** (84-87): danh sách (vòng, nhóm mô hình). Từ vòng lẻ, thứ tự đảo ngược. Ví dụ `schedule(["a","b","c"], 1, 2)` cho ra `[(0,[a,b]), (0,[c]), (1,[c]), (1,[a,b])]`. Mô hình chạy cuối vòng r chạy đầu vòng r+1, nên trọng số của nó vẫn còn trên đĩa.
- **`Phase`** (90-106): bốn file của một bước, `phases/r<r>/<model>.{jobs.jsonl, out.jsonl, meta.json, spec.json}`, và log `logs/r<r>_<model>.log`. Hàm `done()` (99-103) chỉ coi bước là xong khi có `meta.json` với `n` bằng số job.
- **`build_jobs`** (109-128): tạo các lượt của mô hình `model` ở vòng `r`, qua mọi debate.
  - Tìm các vị trí mô hình này ngồi (114). Trong extended, một mô hình có thể ngồi hai vị trí.
  - Vòng 0: `choi.first_prompt`. Vòng r ≥ 1: lấy câu trả lời vòng r−1 của **mọi** vị trí, gọi `choi.round_messages` (120-123).
  - Mỗi job có `id = "<debate_id>#r<r>#p<pos>"` và `seed = stable_seed(seed, id)` riêng (126).
- **`start_worker`** (131-141): ghi `spec.json`, chạy `python -m hmad.worker` trong tiến trình con với `CUDA_VISIBLE_DEVICES` của nhóm GPU, log ra file.
  - Vì sao dùng tiến trình con: tiến trình thoát là chắc chắn trả hết bộ nhớ GPU. Giải phóng vLLM ngay trong cùng tiến trình không đáng tin.
- **`finish_worker`** (158-180): chờ tiến trình con.
  - Lỗi thì in các dòng lỗi rút từ log (`error_lines`, 144-155) rồi dừng.
  - Nếu `continue_on_error` (dùng cho smoke test) thì ghi đầu ra rỗng với `finish_reason="worker_error"` và chạy tiếp.
- **`main`** (191-301):
  - `--fake` thì thêm `_dry` vào tên run (201-203); `--plan-only` thì dừng sau khi ghi plan.
  - Vòng lặp chính (229-273):
    - bỏ qua bước đã xong;
    - đọc đầu ra vòng r−1 **từ đĩa** (236-242), nhờ vậy chạy tiếp được;
    - nếu có `TD_DEADLINE_TS` thì không bắt đầu bước nào dự kiến xong sau mốc đó, thoát với mã 3 (243-245);
    - ghi jobs, lấy trọng số (261), chạy worker;
    - **tải trước mô hình kế tiếp** trong lúc mô hình hiện tại đang sinh (265-269).
  - Ghép kết quả (275-300): đọc mọi `out.jsonl` và `meta.json`, gom thông tin nguồn gốc (git commit, revision, nguồn trọng số, tham số lấy mẫu thực dùng, phiên bản vLLM, các bước lỗi), rồi gọi `assemble` và `summary`.

### B.7 `hmad/worker.py`: một mô hình, một vòng

- **`effective_sampling`** (40-50): temperature, top_p, max_tokens lấy từ config. Nếu `hf_generate_defaults` bật thì top_k và repetition_penalty lấy từ `generation_config.json` của mô hình (mặc định 50 và 1,0), **giống cách HF `generate` của Choi tự lấy**. Kết quả thật ở hetero1: Qwen nhận top_k 20 và repetition_penalty 1,05; bảy mô hình còn lại nhận 50 và 1,0.
- **`vllm_generate`** (107-163):
  - `max_len` = min(`max_model_len` 8192, ngữ cảnh của mô hình) (113). Tạo `LLM(...)` với fp16, TP, `enforce_eager` (114-120).
  - Mỗi job được biến thành token (124-140):
    - `chat_template: true`: `apply_chat_template` với **một lượt user**, `add_generation_prompt=True`, rồi tách token với `add_special_tokens=False`, vì template đã có BOS. Prompt đầu tiên sau khi bọc được lưu vào `meta.json` (`chat_template_example`) để kiểm tra bằng mắt.
    - Không dùng template: `tok(prompt)`, tự thêm BOS như `tokenizer(...)` của Choi.
    - Prompt dài hơn `max_len − max_tokens` thì bỏ qua, ghi `finish_reason="prompt_too_long"` (136-138, 152-153).
  - `SamplingParams` riêng cho từng job, có `seed` của job (141-143). Gọi `llm.generate` một lần cho mọi job của bước (145).
  - Mỗi dòng đầu ra có: `text`, `finish_reason` (`stop` = gặp EOS, `length` = chạm 512 token), `prompt_tokens`, `completion_tokens`.
- **`main`** (166-186): ghi `out.jsonl` rồi mới ghi `meta.json`, cả hai qua file tạm (`write_atomic`). Nhờ vậy không có bước nào "xong một nửa".
- **`fake_generate`** (73-100), chỉ dùng cho test, không cần GPU: một agent giả.
  - Vòng 0: đúng với xác suất `skill`.
  - Vòng sau: theo đáp án phổ biến nhất của bạn với xác suất 0,3 + 0,5 × tỉ lệ bạn đồng ý.

### B.8 `hmad/weights.py`: trọng số và đĩa

`WeightStore.get(key)` (115-130) tìm trọng số theo thứ tự:
1. **Bản sao có sẵn** trong `model_search_paths`, ví dụ `/kaggle/input`. Thư mục phải mang tên key hoặc tên repo và có `config.json` (`local_copy`, 45-51).
2. **Bộ đệm** `model_cache_dir/<key>`, có file `.complete` đánh dấu đã tải xong.
3. **Tải về** đúng revision được ghim (`_download`, 60-70):
   - chỉ lấy safetensors, file cấu hình, tokenizer (`ALLOW`/`IGNORE`, 23-25);
   - bỏ `consolidated*` của Mistral, vì đó là bản trùng lặp làm dung lượng gấp đôi.

Các cơ chế đi kèm:
- **Giới hạn đĩa** (`_make_room`, 72-86): trước khi tải, xóa mô hình dùng lâu nhất chưa dùng lại (LRU). Không xóa mô hình đang chạy hoặc đang tải.
- **Hết đĩa giữa chừng** (93-102): xóa mọi mô hình không được bảo vệ rồi tải lại một lần.
- **`prefetch`** (106-113): tải nền bằng một luồng riêng.

### B.9 `hmad/results.py`: định dạng kết quả

`assemble` (19-48) tạo **một bản ghi cho mỗi debate đã sinh đủ mọi vòng**:

| Trường | Nội dung |
|---|---|
| `id, setup, task, item, graph, n` | Định danh debate |
| `models`, `families` | Mô hình và họ ở từng vị trí |
| `question`, `gold` | Câu hỏi, đáp án chuẩn |
| `rounds[r].responses` | Nguyên văn câu trả lời của từng vị trí |
| `rounds[r].finish` | `stop` / `length` / `prompt_too_long` / `worker_error` |
| `rounds[r].prompt_tokens`, `completion_tokens` | Số token |
| `rounds[r].final_answers`, `agent_correct` | Đáp án đọc được (parser của Choi), đúng/sai từng agent |
| `rounds[r].majority`, `majority_correct` | Đa số của mọi agent. Hòa thì phá bằng `Random(stable_seed(seed, id, r))` (dòng 32) |
| `rounds[r].central`, `central_correct` | Chỉ có ở centralized: đáp án của tâm, chính là con số Choi chấm (39-41) |
| `provenance` | git commit, config, mô hình (hf_id, revision, nguồn, các bước lỗi, tham số lấy mẫu thực dùng, vLLM) |

`summary` (62-89) có hai bảng:
- độ chính xác **đa số** theo vòng, cho từng (setup, graph, task);
- theo từng mô hình: tỉ lệ đọc được đáp án, tỉ lệ chạm 512 token, độ chính xác ở vòng 0.

### B.10 `scripts/estimate.py`

Ước tính trước khi chạy, bằng công thức đơn giản:
- **bộ nhớ**: trọng số = số tham số × 2 byte; KV cache cho mỗi token = 2 × số lớp × kv_heads × head_dim × 2 byte;
- **số lượt sinh**: tính lại từ `build_plan`;
- **thời gian**: khởi động + đọc prompt + sinh chữ; mô hình chứa được ít chuỗi cùng lúc thì sinh chậm hơn;
- **số lần tải mô hình**: mô phỏng bộ đệm LRU theo đúng `schedule`.

Các tốc độ là giả định, có thể ghi đè bằng số đo được ở smoke test.

### B.11 `tests/`

| Test | Kiểm tra |
|---|---|
| `test_choi_fidelity.py` | Dùng `ast` đọc **hàm gốc** từ `debate-or-vote/src/` rồi chạy trên cùng đầu vào. Các hàm được so: suffix (mọi data × bae × cot), prompt vòng t (n = 2, 3, 6, 12 × 3 đồ thị), parser và bỏ phiếu, bộ nạp arithmetics. Không có repo gốc thì bỏ qua |
| `test_pipeline.py: test_plan_assignment` | `exp.yaml` cho ra 1.000 debate; main và baseline không trùng họ; extended có 12 agent thuộc 8 họ; mỗi mô hình làm tâm từ 15 lần trở lên |
| `test_same_family_refused` | Bật `unique_families` thì hai agent cùng họ bị từ chối |
| `test_fake_run_and_resume` | Chạy giả đầu-cuối: prompt vòng 1 chứa câu trả lời vòng 0; chạy lại thì không làm lại gì; đổi `rounds` thì bị từ chối |
| `test_registry_not_gated_and_schedule` | Mô hình không cần xin quyền; `gpu_groups` và `schedule` cho kết quả đúng |
| `test_continue_on_error` | Một mô hình lỗi: có cờ thì ghi vào summary và chạy tiếp, không có cờ thì dừng |
| `test_core_does_not_depend_on_kaggle` | `hmad/` và `scripts/` không nhắc tới chữ "kaggle" |

### B.12 `kaggle/`

Chỉ chứa phần riêng của Kaggle; code lõi không đọc gì trong thư mục này.
- `run_kaggle.ipynb`: clone đúng commit; nếu `RESUME` thì khôi phục `results/` từ Input; gọi `setup.sh` và `kaggle_run.sh main`.
- `setup.sh`: cài vLLM với phiên bản được ghim, gỡ torchaudio (lệch CUDA), kiểm tra torch tính được trên GPU.
- `kaggle_run.sh`:
  - tạo một config phụ kế thừa `exp.yaml`, chỉ đổi đường dẫn kết quả, thư mục trọng số (thư mục đầu tiên ghi được 256 MB) và giới hạn đĩa;
  - cứ 30 phút nén `results/` thành `results.zip` một lần, và nén lần cuối khi chạy xong.

---

## Phần C — Khác nhau chi tiết giữa hai bên

### C.1 Bảng tổng hợp

| # | Khía cạnh | Choi (debate-or-vote) | TD (`hmad/`) | Đổi nội dung debate? |
|---|---|---|---|---|
| 1 | Agent | 1 mô hình nhân bản N lần | Mỗi vị trí 1 mô hình (khác họ), hoán vị có seed | **Có**: đây là đối tượng nghiên cứu |
| 2 | Thứ tự chạy | Từng câu hỏi → mọi vòng | Từng vòng → từng mô hình → mọi debate | Không |
| 3 | Định dạng prompt | Chữ thô | Chat template của từng mô hình (`hetero2`) | **Có** |
| 4 | Thư viện sinh | HF `generate`, lô N prompt có đệm | vLLM, lô mọi job của mô hình trong vòng | Không (cùng phân phối lấy mẫu) |
| 5 | Ngẫu nhiên | Seed 42 chung; phá hòa bằng `random` chung | Seed riêng cho từng lượt sinh và từng lần phá hòa | Không (chỉ khác số ngẫu nhiên cụ thể) |
| 6 | Prompt quá dài | Cắt bớt mà không báo | Bỏ qua, ghi `prompt_too_long` | Không (chưa xảy ra lần nào) |
| 7 | Chấm centralized | Chỉ đáp án của tâm | Lưu cả đa số lẫn đáp án tâm | Không (chỉ khác số được báo cáo) |
| 8 | Số câu hỏi | GSM8K 300, CSQA 300 | 100 + 100 (là 100 câu đầu trong 300 câu của Choi) | Không |
| 9 | Số agent, vòng, đồ thị | 5 agent × (vòng 0 + 5 vòng) | 3/6/12 agent × (vòng 0 + 4 vòng) | Có (thiết kế) |
| 10 | Lưu và chạy tiếp | `out/history/*.jsonl`, không chạy tiếp được | Phase, `debates.jsonl`, nguồn gốc, chạy tiếp, khóa giao thức | Không |
| 11 | Phần không chuyển | Persona, một agent, `--bae`, 5 bộ dữ liệu khác | Không có | — |
| 12 | Giao diện hàm | Nhận `args` | Nhận tham số thường, có thêm `rng` | Không |

### C.2 Chi tiết từng điểm

#### C.2.1 Agent: một mô hình nhân bản, hay mỗi vị trí một mô hình

Choi nạp đúng một mô hình (`model/model_utils.py:52-62`), và mọi agent gọi chung nó (`main.py:195-196`):
```python
messages = [{"role": "user", "content": x + SUFFIX}] * args.num_agents
responses = engine(messages, agent, args.num_agents)
```

TD gán một mô hình cho mỗi vị trí, xáo bằng seed (`hmad/plan.py:54-57`):
```python
models = list(sc["models"])
random.Random(stable_seed(cfg.get("seed", 0), setup, task, k, graph)).shuffle(models)
debates.append(Debate(f"{setup}|{graph}|{task}|{k}", setup, task, k, q, _plain(y), graph, models))
```
Mỗi mô hình chỉ sinh phần của mình (`hmad/run.py:114-116`):
```python
positions = [i for i, m in enumerate(d.models) if m == model]
if not positions:
    continue
```

Hệ quả:
- Ở Choi, vòng 0 của mọi agent là cùng một prompt cho cùng một mô hình, nên vị trí không có ý nghĩa gì.
- Ở TD, mô hình nào ngồi ở tâm hay cạnh ai sẽ ảnh hưởng tới kết quả. Vì vậy phải hoán vị cân bằng, và `test_plan_assignment` kiểm tra điều đó.
- Chữ "heterogeneous" của Choi (`--multi_persona`) nghĩa là cùng mô hình, khác system prompt. TD không dùng persona.

#### C.2.2 Thứ tự chạy

Choi chạy hết từng câu hỏi (`main.py:183`, `:232`):
```python
for i, (x, y) in tqdm(enumerate(zip(test_X, test_Y)), total=len(test_X)):
    ...                                   # vòng 0
    for r in range(start, args.debate_rounds+1) :
        ...                               # vòng r: get_new_message -> engine -> evaluate
```

TD chạy theo vòng (`hmad/run.py:84-87`, `:229-273`):
```python
groups = [order[i:i + width] for i in range(0, len(order), width)]
return [(r, g) for r in range(rounds + 1) for g in (groups if r % 2 == 0 else groups[::-1])]
```

Lý do: nạp một mô hình 7–9B mất 1–3 phút (`load_seconds` 74–174 giây qua 23 bước của hetero1).
- Nếu theo cách của Choi, mỗi câu hỏi phải nạp lại mô hình (số mô hình × số vòng) lần.
- TD chỉ nạp 5 vòng × 8 mô hình = 40 lần cho cả run.

Nội dung debate không đổi: prompt vòng r vẫn được dựng từ đúng các câu trả lời vòng r−1 của debate đó. TD đọc các câu trả lời này từ đĩa (`run.py:236-242`), không giữ trong bộ nhớ.

#### C.2.3 Chat template (khác biệt về giao thức duy nhất bên trong debate)

Choi (`model/model_utils.py:12-16`): không có persona thì đưa chữ thô, rồi tách token (tự thêm BOS nếu tokenizer có BOS):
```python
prompts = [msg['content'] for msg in messages]  # we find that NOT using chat template is better in MAD
inputs = agent.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
```

TD (`hmad/worker.py:124-135`):
```python
use_template = bool(spec["sampling"].get("chat_template", False))
...
if use_template:
    text = tok.apply_chat_template([{"role": "user", "content": j["prompt"]}], tokenize=False,
                                   add_generation_prompt=True)
    example = example or text          # kept in meta.json to check the template by eye
    ids = tok(text, add_special_tokens=False).input_ids
else:
    ids = tok(j["prompt"]).input_ids        # BOS added as in HF tokenizer(...) (Choi et al.)
```

- **Nội dung tin nhắn không đổi.** Chuỗi `j["prompt"]` vẫn là chuỗi do `choi.round_messages` tạo ra, chỉ được đặt vào khuôn một lượt user.
- **Lý do** (ghi ở `configs/exp.yaml:42-45`): hetero1 chạy prompt thô thì mô hình instruct không biết lượt của mình kết thúc ở đâu.
  - 34–51% lượt sinh chạy tới giới hạn 512 token;
  - khoảng 6% câu trả lời rỗng (EOS là token đầu tiên);
  - 5/8 mô hình có tỉ lệ đọc được đáp án dưới 90%.

  Choi chỉ dùng Qwen2.5 và Llama-3.1 nên có thể không gặp mức này.
- **BOS**: `add_special_tokens=False` để không bị hai BOS liên tiếp. Nhánh persona của Choi (`model_utils.py:13` rồi `:16`) gọi template xong lại gọi `tokenizer(...)` với mặc định `add_special_tokens=True`. Với mô hình mà template đã chứa BOS (như Llama-3.1), cách đó tạo ra hai BOS. TD không có lỗi này.
- **Cần kiểm tra:** template của vài mô hình tự chèn system prompt mặc định khi không có system. Ví dụ Qwen2.5 chèn "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.". Xem `chat_template_example` trong `phases/r0/<model>.meta.json` của hetero2 để biết từng mô hình thực sự nhận gì.

#### C.2.4 Thư viện sinh và tham số lấy mẫu

Choi (`model/model_utils.py:21-33`):
```python
outputs = agent.huggingface_model.generate(input_ids, attention_mask=attention_mask,
    pad_token_id=agent.tokenizer.eos_token_id, max_new_tokens=512, return_dict_in_generate=True,
    output_scores=True, do_sample=True, temperature=1.0, top_p=0.9, num_return_sequences=1,
    return_legacy_cache=True)
```

TD (`hmad/worker.py:40-50`, `:141-145`):
```python
eff = {"temperature": float(sampling["temperature"]), "top_p": float(sampling["top_p"]),
       "max_tokens": int(sampling["max_tokens"]), "top_k": -1, "repetition_penalty": 1.0}
if sampling.get("hf_generate_defaults", True):
    ...  # generation_config.json của mô hình
    eff["top_k"] = int(gc.get("top_k", 50) or -1)
    eff["repetition_penalty"] = float(gc.get("repetition_penalty", 1.0) or 1.0)
...
params = [SamplingParams(temperature=eff["temperature"], top_p=eff["top_p"], top_k=eff["top_k"],
                         repetition_penalty=eff["repetition_penalty"], max_tokens=eff["max_tokens"],
                         seed=int(j["seed"])) for j in keep]
outs = llm.generate(inputs, params, use_tqdm=True)
```

| Tham số | Choi | TD |
|---|---|---|
| temperature / top_p / số token tối đa | 1,0 / 0,9 / 512 | Giống (`exp.yaml:47-49`) |
| top_k, repetition_penalty | HF tự lấy từ `generation_config.json` (không có thì 50 / 1,0) | Đọc cùng file, cùng mặc định, truyền rõ vào vLLM |
| Kiểu số | fp16 (`torch_dtype=torch.float16`) | `dtype: half` |
| Lô | N prompt của một câu hỏi, đệm trái | Mọi job của mô hình trong vòng (tới 1.200 prompt), vLLM tự xếp lô, không đệm |
| Phần cứng | GPU của tác giả, `device_map="auto"` | 2 × T4, tensor parallel 2, `enforce_eager` |

Về nguyên tắc, hai bên lấy mẫu từ cùng một phân phối. Nhưng kernel tính toán khác nhau, nên cùng seed cũng không cho ra từng chữ giống hệt.

#### C.2.5 Ngẫu nhiên và khả năng lặp lại

Choi đặt seed chung một lần (`main.py:324-328`), phá hòa bằng `random` chung (`evaluator.py:64`):
```python
torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
...
debate_answer = random.choice(most_common) # if there is a tie, will choose randomly
```

TD dùng seed riêng cho từng lượt sinh (`hmad/run.py:126`) và từng lần phá hòa (`hmad/results.py:32`):
```python
jobs.append({"id": jid, "prompt": p, "seed": stable_seed(cfg.get("seed", 0), jid), ...})
...
rng = random.Random(stable_seed(seed, d.id, r))
finals, maj, maj_ok, ok = choi.evaluate(d.task, texts, d.gold, rng)
```

Hệ quả: ở Choi, kết quả của một câu hỏi phụ thuộc vào mọi lệnh gọi ngẫu nhiên trước đó, kể cả thứ tự câu hỏi. Ở TD, mỗi lượt sinh và mỗi lần phá hòa chỉ phụ thuộc vào định danh của chính nó, nên bị ngắt rồi chạy tiếp, hay chạy lại bước chấm, vẫn ra cùng kết quả. Đây cũng là lý do `choi.vote` nhận thêm tham số `rng`.

#### C.2.6 Prompt quá dài

- Choi (`model_utils.py:16`): `truncation=True`, tức cắt mà không báo. Tokenizer mặc định cắt bên phải, nên phần bị mất sẽ là câu hỏi và suffix. Trên thực tế gần như không xảy ra, vì giới hạn tokenizer của Qwen2.5 và Llama-3.1 rất lớn.
- TD (`worker.py:113`, `:123`, `:136-138`): giới hạn = min(8192, ngữ cảnh mô hình) − 512. Prompt vượt giới hạn thì không sinh, ghi `text=""` và `finish_reason="prompt_too_long"`. Ở hetero1, `too_long = 0` ở cả 23 bước.

#### C.2.7 Chấm đồ thị centralized

Choi chỉ chấm agent đầu tiên, tức tâm (`main.py:200-202`, lặp lại ở `:244-246`):
```python
if args.centralized :
    central_agent_response = {list(agent_responses.keys())[0] : list(agent_responses.values())[0]}
    final_resps, debate_resps, is_corr = evaluate(central_agent_response, y)
```

TD chấm mọi agent, rồi lưu thêm đáp án của tâm (`hmad/results.py:33`, `:39-41`):
```python
finals, maj, maj_ok, ok = choi.evaluate(d.task, texts, d.gold, rng)
...
if d.graph == "centralized":
    row["central"] = _plain(finals[0])
    row["central_correct"] = ok[0]
```

- `central_correct` của TD chính là `debate_answer_iscorr` của Choi ở đồ thị centralized: chỉ có một đáp án nên không có hòa.
- **Lưu ý khi so số:** bảng "Majority accuracy by round" trong `summary.md` (`results.py:71`) dùng `majority_correct` cho **mọi** đồ thị, kể cả centralized. Muốn so với số của Choi ở centralized thì tính lại từ `central_correct` trong `debates.jsonl`.
- Ở centralized, Choi không lưu đáp án đọc được của các agent ngoài tâm (chỉ lưu nguyên văn). TD lưu đủ.

#### C.2.8 Dữ liệu

| | Choi | TD |
|---|---|---|
| GSM8K | `head(args.data_size)`; các script dùng 300 (`data/gsm8k.py:23`) | `head(data_size)`, `n_items: 100` (`choi.py:175`) |
| CSQA | Luôn 300 câu, không đọc `data_size` (`data/csqa.py:13`) | 300 câu đó rồi lấy `data_size` câu đầu (`choi.py:190`, `:199`) |
| Arithmetics | Đáp án là số numpy | Đổi sang `float` (`choi.py:218`); test so bằng `float` |
| Bộ khác | hellaswag, MMLU (2 môn), hh_rlhf, cnn_daily | Không chuyển |

Hai bên xáo cùng cách (`random_state=0`), nên **100 câu của TD là đúng 100 câu đầu trong 300 câu của Choi**. Kết quả trên tập con này so trực tiếp được với phần tương ứng của Choi.

#### C.2.9 Số agent, số vòng, đồ thị

| | Choi (`scripts/*.sh`) | TD (`configs/exp.yaml:22-36`) |
|---|---|---|
| Số agent | 5 | 6 (main), 12 (extended), 3 (baseline) |
| Số vòng | Vòng 0 + 5 | Vòng 0 + 4 |
| Đồ thị | Mỗi lần chạy một đồ thị (`--sparse` / `--centralized`) | Khai báo theo thiết lập, chạy chung một lượt |
| Trùng họ | Không áp dụng (một mô hình) | `unique_families` kiểm tra ở `plan.py:45-47`; `exp.yaml` tắt kiểm tra để extended nhân đôi 4 họ |

#### C.2.10 Lưu kết quả, chạy tiếp, nguồn gốc

Choi (`main.py:294-296`, `:313-315`):
- ghi lại toàn bộ `out/history/{fname}.jsonl` sau mỗi câu hỏi, và một dòng vào `out/logs.tsv`;
- mỗi vòng lưu `responses`, `final_answers`, `final_answer_iscorr`, `debate_answer`, `debate_answer_iscorr`, `answer`;
- không lưu lý do dừng, số token, phiên bản thư viện hay revision mô hình;
- không chạy tiếp được: bị ngắt là phải chạy lại từ đầu.

TD:
- mỗi bước (vòng, mô hình) có file riêng, `meta.json` ghi sau cùng làm dấu "đã xong" (`worker.py:181-185`); chạy lại cùng lệnh thì bỏ qua bước đã xong (`run.py:230-235`);
- `protocol.json` khóa giao thức (`run.py:56-66`);
- `debates.jsonl` có thêm `finish`, số token, họ mô hình, `provenance` (mục B.9).

#### C.2.11 Không chuyển sang TD

- Nhánh một agent tự sửa (`main.py:121-132`) và persona (`model_utils.py:70-105`).
- Parser `--bae` (`evaluator.py:117-162`) và `evaluate_gen`.
- **Cẩn thận với `bae`:** `build_jobs` vẫn truyền `bae` vào `choi.instruction_suffix` (`run.py:117`), nhưng `choi.evaluate` chỉ có parser ngoặc nhọn. Đặt `bae: true` trong config thì suffix và parser sẽ lệch nhau. Config hiện tại không đặt nên không ảnh hưởng.
- Các tham số và hàm không dùng, đã liệt kê ở mục A.7.

#### C.2.12 Giao diện hàm (không đổi hành vi)

Docstring của `hmad/choi.py:9-14` ghi ba điểm:
- hàm nhận tham số thường thay vì `args`. Ví dụ `round_messages(question, responses, graph, suffix)` thay cho `get_new_message(args, sample, responses, personas, suffix)`;
- `vote` nhận `rng` (mục C.2.5);
- chỉ giữ nhánh nhiều agent.

`round_messages` nhận danh sách câu trả lời theo vị trí, còn bản gốc nhận dict tên agent → câu trả lời. Dict của Python giữ thứ tự chèn, nên thứ tự đọc bạn là như nhau.

### C.3 Những gì giống hệt, và cách kiểm tra

| Thành phần | Kiểm tra bằng |
|---|---|
| Suffix (mọi bộ dữ liệu × `bae` × `cot`) | `test_suffix` so từng chuỗi |
| Prompt vòng t ≥ 1, kể cả từng ký tự xuống dòng; thứ tự đọc bạn; ba đồ thị; n = 2, 3, 6, 12 | `test_round_messages` |
| Vòng 0 = câu hỏi + suffix | `choi.first_prompt` (một dòng), `test_fake_run_and_resume` |
| Parser số và trắc nghiệm; quy tắc "mọi agent không đọc được thì đáp án debate là `""`" | `test_evaluate` (đáp án debate chỉ so khi không hòa, vì bộ sinh ngẫu nhiên khác nhau: dòng 84) |
| Bộ nạp arithmetics | `test_arithmetics_loader` |
| Bộ nạp GSM8K và CSQA | Cùng nguồn, cùng `random_state=0`, cùng định dạng (đọc code, mục C.2.8). Không có test tự động, vì cần tải dữ liệu |
| Không có trí nhớ giữa các vòng | `round_messages` chỉ nhận câu trả lời vòng trước |
| temperature 1,0, top_p 0,9, 512 token, fp16, top_k và repetition_penalty theo `generation_config.json` | `meta.json` → `sampling_effective` của mỗi bước |

### C.4 Khi đọc kết quả TD cạnh số của Choi

1. **Centralized**: dùng `central_correct`, không dùng bảng đa số trong `summary.md` (mục C.2.7).
2. **Chat template**: hetero2 khác Choi ở định dạng prompt. Nếu kết quả khác nhiều so với hetero1 ở vòng 0–2, nguyên nhân chính là điểm này.
3. **Mô hình khác nhau**: vòng 0 của TD đã có độ đa dạng giữa các họ mô hình. Ở Choi, độ đa dạng chỉ đến từ lấy mẫu ngẫu nhiên. Sự khác nhau này là chủ đích của đề tài, không phải sai lệch cần sửa.
4. **Số câu**: 100 câu đầu trong 300 câu của Choi, nên sai số thống kê lớn hơn khoảng √3 lần.

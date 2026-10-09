# Nghiên cứu trước khi viết code: kiến trúc debate, số agent, cách chạy trên Kaggle

Viết ngày 09/10/2026 cho **ý tưởng 1: thu dữ liệu debate**. Bản 2: mục 1–4 đã được research lại theo yêu cầu "chọn kiến trúc được nhiều người dùng để làm thí nghiệm, không tự làm kiến trúc mới". Chưa có dòng code nào.

## 0. Đọc nhanh

- **Cách chọn.** Đo mức "được nhiều người dùng" bằng bốn chỉ số:
  1. số trích dẫn và số trích dẫn "có ảnh hưởng" trên Semantic Scholar (tra ngày 09/10/2026);
  2. số benchmark/framework so sánh debate năm 2024–2026 đưa kiến trúc đó vào làm phương pháp đại diện;
  3. tỉ lệ trong bài tổng quan 141 nghiên cứu (Motger et al. 2026);
  4. có code chính thức hay không.
- **Kết quả: một kiến trúc dẫn đầu ở mọi chỉ số** là **Society of Mind** (SoM, còn gọi "LLM Debate") của Du et al. (ICML 2024):
  - 2.412 trích dẫn, 279 trích dẫn có ảnh hưởng;
  - có mặt trong 5/5 benchmark so sánh;
  - 72% nghiên cứu dùng đúng khuôn của nó.

  Đứng thứ hai là **MAD / Multi-Persona** của Liang et al. (EMNLP 2024): 1.445 trích dẫn, có mặt trong 5/5 benchmark.
- **Đề xuất:**
  - **Kiến trúc chính: SoM, giữ nguyên.** Về số agent và đồ thị, dùng đúng ba kiểu đã được công bố cùng code chính thức của Choi, Zhu & Li (NeurIPS 2025, giấy phép MIT): đầy đủ, thưa (vòng tròn), tập trung (sao). Không tự chế gì.
  - **Kiến trúc phụ (tùy chọn): MAD của Liang, giữ nguyên.** Dùng nếu cần một điều kiện "tranh luận đối kháng thật".
- **Về lo ngại "cũ".** Mọi kiến trúc được dùng rộng đều ra đời 2023–2024, vì cần thời gian để được dùng lại. Các kiến trúc 2025–2026 mới có 14–49 trích dẫn. Các benchmark mới nhất (Findings of ACL 2026, ACM MM 2026, NeurIPS 2025) vẫn chọn SoM làm phương pháp đại diện. Nghĩa là SoM vẫn là chuẩn hiện hành, chưa bị thay.
- **Số agent:** không đổi so với bản 1, chỉ khớp lại với các đồ thị của SoM.
  - 6 agent là chính (đầy đủ / vòng tròn / sao);
  - 12 agent trên đồ thị thưa;
  - 3 agent làm mốc.
- **Kaggle:** không đổi so với bản 1 (mục 6).

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **Kiến trúc debate** | Luật chơi cố định của một phương pháp đã công bố: có bao nhiêu agent, mỗi agent đọc gì, viết gì, mấy vòng, chọn đáp án cuối thế nào |
| **Society of Mind (SoM) / LLM Debate** | Kiến trúc của Du et al. Vòng 0, mỗi agent tự giải. Mỗi vòng sau, agent đọc toàn văn lời giải của các agent khác rồi viết lời giải mới. Đáp án cuối lấy theo đa số |
| **MAD / Multi-Persona** | Kiến trúc của Liang et al. Một agent "thiên thần" (ủng hộ) và một agent "ác quỷ" (phản đối) cãi nhau kiểu "ăn miếng trả miếng"; một agent trọng tài phán quyết |
| **Trích dẫn có ảnh hưởng** | Chỉ số của Semantic Scholar: những bài trích dẫn mà thật sự dùng hoặc mở rộng phương pháp, không chỉ nhắc qua |
| **Benchmark so sánh debate** | Bài chạy nhiều kiến trúc debate trên cùng dữ liệu để so. Kiến trúc được đưa vào tức là được coi là đại diện |
| **Đồ thị đọc** | Ai đọc lời của ai. Đầy đủ (decentralized): đọc tất cả. Thưa (sparse): đọc 2 bạn bên cạnh. Tập trung (centralized): agent 0 đọc tất cả, các agent khác chỉ đọc agent 0 |

## 2. Đo "được nhiều người dùng"

### 2.1 Trích dẫn

Tra trên Semantic Scholar ngày 09/10/2026.

| Kiến trúc | Bài | Năm công bố chính thức | Trích dẫn | Có ảnh hưởng |
|---|---|---|---|---|
| **Society of Mind (LLM Debate)** | Du et al. | ICML 2024 | **2.412** | **279** |
| **MAD / Multi-Persona** | Liang et al. | EMNLP 2024 | **1.445** | **144** |
| ChatEval | Chan et al. | ICLR 2024 | 1.014 | 84 |
| AgentVerse | Chen et al. | ICLR 2024 | 814 | 63 |
| ReConcile | Chen, Saha & Bansal | ACL 2024 | 412 | 35 |
| DyLAN | Liu et al. | COLM 2024 | 351 | 37 |
| Debate có trọng tài (2 bên được giao đáp án) | Khan et al. | ICML 2024 | 331 | 26 |
| MacNet | Qian et al. | ICLR 2025 | 286 | 26 |
| SoM trên đồ thị thưa | Li et al. | Findings of EMNLP 2024 | 144 | 14 |
| Exchange-of-Thought | Yin et al. | EMNLP 2023 | 114 | 11 |
| Debate or Vote (SoM, thưa, tập trung) | Choi, Zhu & Li | NeurIPS 2025 | 94 | 16 |
| *Kiến trúc 2025, để so:* GroupDebate | Liu et al. | AAMAS 2025 | 73 | 4 |
| DMAD (mỗi agent dùng một kiểu suy luận khác nhau) | Liu et al. | ICLR 2025 | 49 | 7 |
| Wu et al. (ControlMAD) | — | arXiv 2025 | 37 | 3 |
| Free-MAD | Cui et al. | ACL 2025 | 28 | 2 |
| MALLM (framework) | Becker et al. | EMNLP 2025 Demo | 22 | 1 |
| CortexDebate | Sun et al. | Findings of ACL 2025 | 14 | 0 |
| Phản biện ngang hàng | Xu et al. | arXiv 2023 | 64 | 7 |

### 2.2 Các benchmark so sánh debate chọn kiến trúc nào làm đại diện

| Benchmark / framework | SoM (Du) | MAD (Liang) | ChatEval | AgentVerse | EoT | Khác |
|---|---|---|---|---|---|---|
| Smit et al., ICML 2024 (DebateLLM) | ✓ | ✓ | ✓ | | | Ensemble Refinement, SPP, Medprompt |
| Zhang et al., *Stop Overvaluing MAD*, Findings of ACL 2026 ([arXiv 2502.08788](https://arxiv.org/abs/2502.08788)): "5 kiến trúc đại diện" | ✓ | ✓ | ✓ | ✓ | ✓ | |
| MASLab, 2025 ([arXiv 2505.16988](https://arxiv.org/abs/2505.16988)): hơn 20 phương pháp, đã đối chiếu với code chính thức | ✓ | ✓ | | ✓ | | DyLAN, MacNet… |
| Choi, Zhu & Li, NeurIPS 2025 ([arXiv 2508.17536](https://arxiv.org/abs/2508.17536)) | ✓ (+ thưa, tập trung) | ✓ (biến thể multi-persona) | | | | |
| M3MAD-Bench, ACM MM 2026 ([arXiv 2601.02854](https://arxiv.org/abs/2601.02854)) | ✓ | ✓ (gọi là Div-MAD) | | | | DMAD |
| **Số lần có mặt** | **5/5** | **5/5** | 2/5 | 2/5 | 1/5 | |

Thêm vào đó, bài tổng quan 141 nghiên cứu (Motger et al. 2026, [arXiv 2607.26212](https://arxiv.org/abs/2607.26212)) cho thấy cả lĩnh vực hội tụ về đúng khuôn của SoM: 72,2% đồ thị đầy đủ, 86,1% chuyển nguyên văn lời giải, 94,7% chỉ nhớ ngắn hạn, quyết định bằng bỏ phiếu.

**Kết luận mục 2:**
- **SoM** là kiến trúc được dùng nhiều nhất theo mọi chỉ số.
- **MAD của Liang** đứng thứ hai, và là kiến trúc **đối kháng** được dùng nhiều nhất.
- Mọi kiến trúc khác kém xa về mức độ được dùng lại.
- Các kiến trúc 2025–2026 chưa kiến trúc nào được dùng rộng.

## 3. Hai kiến trúc dẫn đầu có hợp với đề tài không

Đề tài đo **đáp án đúng và sai lây giữa nhiều agent ngang hàng**, nên cần:
- số agent chỉnh được;
- đồ thị chỉnh được;
- agent không bị khóa lập trường;
- chạy được bằng mô hình mở trên T4.

| Tiêu chí | SoM (Du et al. 2024) | MAD (Liang et al. 2024) |
|---|---|---|
| Số agent | **Chỉnh được** (Zhang et al. thử 3, 6, 9; Choi et al. mặc định 5) | **Cố định 3** (2 người cãi + 1 trọng tài). Zhang et al. phải loại MAD khỏi thí nghiệm đổi số agent |
| Đồ thị | **Có sẵn 3 kiểu đã công bố**: đầy đủ (Du), thưa và tập trung (Li et al. 2024; Choi et al. 2025, có code) | Không có khái niệm đồ thị |
| Agent ngang hàng, tự đổi ý | **Có**: mọi agent cùng một prompt | Không: hai bên được giao lập trường, trọng tài khác vai |
| Agent đọc gì | **Toàn văn lời giải** của bạn. Prompt trắc nghiệm gốc còn dặn: "Examine your solution and that other agents step by step" | Lời lẽ của bên kia, kiểu "ăn miếng trả miếng" |
| Mô hình mở cỡ nhỏ | Chạy tốt. Choi et al. dùng đúng Qwen2.5-7B và Llama3.1-8B | **Làm hại mô hình nhỏ**: trong M3MAD-Bench, Llama3.1-8B tụt từ 51,0% xuống 38,2% |
| Code chính thức | Du: có, nhưng **không có giấy phép**. Choi et al.: **MIT**, có cả 3 đồ thị và các tập GSM8K, CommonsenseQA | GPL-3.0. Có bản cài lại trong MASLab và M3MAD-Bench |
| Đo được lây giữa nhiều agent | **Có**: liều k = số bạn giữ một đáp án | Chỉ đo được việc trọng tài bị thuyết phục |

## 4. Đề xuất

### 4.1 Kiến trúc chính: Society of Mind, giữ nguyên

- **Prompt:** xem mục 4.3b. Choi et al. không dùng nguyên văn prompt của Du, nên phải chọn theo bản của Choi (đề xuất) hay theo bản gốc của Du. Chọn bản nào thì chép nguyên văn bản đó, có ghi nguồn, và ghi rõ chỗ khác bản kia.
- **Đồ thị:** đúng ba kiểu của Choi et al. (NeurIPS 2025), mỗi kiểu đã có trong code chính thức (MIT):
  - **đầy đủ:** mỗi agent đọc tất cả, như Du;
  - **thưa:** agent i đọc agent i−1 và i+1, như Li et al. 2024;
  - **tập trung:** agent 0 đọc tất cả, các agent khác chỉ đọc agent 0.

  Với 6 agent, ba kiểu này cho số bạn đọc là 5, 2 và 1 (hoặc 5 với agent 0). Đó đúng là dải cần để đo liều k và tách "theo số người" với "theo tỉ lệ" (mục 5).
- **Đáp án cuối:** bỏ phiếu đa số, như SoM gốc.
- **Không thêm** persona, độ tự tin hay bước phản biện bắt buộc. Thêm vào là thành kiến trúc mới.

### 4.2 Về lo ngại "agent chỉ nhìn đáp án, không phải debate thật"

- **Trong SoM, agent đọc toàn văn lời giải từng bước của các bạn**, không chỉ đọc đáp án. Đây là điều mà cả lĩnh vực gọi là "multi-agent debate".
- **Agent hay chiều theo một cách hời hợt.** Đây là một tính chất đã được đo của SoM:
  - Qian 2026: đồng ý ngoài miệng, đáp án không đổi;
  - Wynn et al. 2025: bỏ đáp án đúng để theo lý lẽ sai;
  - Hao et al. 2026: 29% số lần đổi ý là chiều theo.
- **Với đề tài, chiều theo chính là một phần của "lây".** Dùng giao thức tự chế để triệt nó đi sẽ đổi luôn đối tượng nghiên cứu, và mất khả năng so với các bài khác.
- **Nếu vẫn muốn có điều kiện tranh luận đối kháng thật**, kiến trúc được dùng rộng duy nhất là MAD của Liang (mục 4.3). Không nên tự chế.

### 4.3 Kiến trúc phụ (tùy chọn): MAD của Liang, giữ nguyên

- **Dùng khi:** muốn so "đọc rồi sửa" với "cãi nhau có trọng tài" trên cùng câu hỏi.
- **Đo được:** bên đúng hay bên sai thuyết phục trọng tài nhiều hơn. Đây là dạng lây "một người nghe, hai người nói", giống thiết kế của Khan et al. 2024.
- **Hạn chế:**
  - cố định 3 agent;
  - không có đồ thị;
  - làm hại mô hình nhỏ (M3MAD-Bench).
- **Lời khuyên:** chỉ chạy trên một phần câu hỏi, hoặc để sau.

### 4.3b Debate or Vote đã dùng SoM thế nào (kiểm tra code ngày 09/10/2026)

**Choi et al. không dùng lại code của Du.**
- Họ viết lại từ đầu bằng Hugging Face Transformers.
- Trong code không có dòng import hay chú thích nào nhắc tới repo của Du.
- Bài báo trích Du là nguồn gốc của debate, nhưng mô tả thiết lập của họ như một thiết lập riêng.

Nói cách khác, đây là một **bản cài lại SoM**: giữ ý tưởng, đổi nhiều chi tiết.

| | Du et al. (code gốc) | Choi et al. (debate-or-vote) |
|---|---|---|
| Ý tưởng cốt lõi | Mọi agent nói cùng lúc, đọc toàn văn lời giải vòng trước của bạn, đáp án cuối theo đa số | **Giống** |
| Prompt mỗi vòng | "These are the solutions to the problem from other agents: … One agent solution: \`\`\`…\`\`\` … Using the solutions from other agents as additional information, can you provide your answer…" | "These are the recent opinions from other agents: … One of the agents' response: … This was your most recent opinion: … Use these opinions carefully as additional advice to revise your recent opinion to give your final answer to the question: …" |
| Lời giải cũ của chính agent | Nằm trong lịch sử hội thoại, ở lượt "assistant" | Được dán vào cùng tin nhắn, dưới dạng chữ "This was your most recent opinion" |
| Trí nhớ | **Toàn bộ hội thoại** qua các vòng | **Chỉ vòng trước**: mỗi vòng là một tin nhắn mới |
| Dạng đáp án | `\boxed{answer}` (toán), `(X)` (trắc nghiệm) | `{final answer: 123}` hoặc `{final answer: (A)}`. Cờ `--bae` đổi về dạng `(X)` cho trắc nghiệm |
| Số agent × vòng mặc định | 3 × 2 | 5 × 5 |
| Đồ thị | Chỉ đầy đủ | Đầy đủ, thưa (đọc i−1 và i+1), tập trung (sao) |
| Mô hình, tham số | gpt-3.5-turbo-0301 qua API, tham số mặc định của API | Qwen2.5-7B, Llama-3.1-8B chạy tại chỗ; temperature 1,0, top_p 0,9, tối đa 512 token |
| Hòa phiếu | `most_common` (lấy đáp án gặp trước) | Chọn ngẫu nhiên |

Bản SoM của Zhang et al. (Findings of ACL 2026) lại dùng một câu khác nữa: "Use these opinions carefully as additional advice, can you provide an updated answer?".

**Hệ quả:**
- Trong thực tế, "SoM" ở các bài khác nhau là **các bản cài lại có lời lẽ khác nhau**. Ý tưởng thì giống nhau.
- Cần chọn theo một bản cụ thể và ghi rõ chỗ khác Du.

**Hai lựa chọn:**
- **(A) Theo đúng bản của Choi et al.** Đã qua phản biện ở NeurIPS, có sẵn đồ thị và mô hình mở, giấy phép MIT. Báo cáo là "SoM theo cài đặt của Choi et al. 2025".
- **(B) Theo đúng prompt và trí nhớ của Du.** Như vậy thì phải tự thêm đồ thị thưa và tập trung, tức tự sửa kiến trúc.

Đề xuất **(A)**. Có thể chạy thêm (B) trên đồ thị đầy đủ, ở một phần câu hỏi, để kiểm tra kết quả có đổi vì lời lẽ prompt không.

### 4.4 Các kiến trúc khác và lý do không chọn

| Kiến trúc | Lý do không chọn |
|---|---|
| ChatEval (hạng 3) | Gốc là bài toán đánh giá văn bản. 3 vai cố định (General Public, Critic, Scientist); vai trò làm nhiễu phép đo lây |
| AgentVerse (hạng 4) | Tuyển agent một cách động, nên ai tham gia và ai đọc ai phụ thuộc vào nội dung |
| ReConcile | Cần nhiều mô hình khác nhau và lời giải mẫu do người viết; ít được benchmark dùng |
| Exchange-of-Thought | Có nhiều kiểu liên lạc hay, nhưng ít được dùng (114 trích dẫn, 1/5 benchmark) |
| DMAD, Free-MAD, CortexDebate, ControlMAD, MALLM (2025) | Mới, được dùng lại còn ít (14–49 trích dẫn) |
| Kiến trúc "phản biện từng bạn" của báo cáo bản 1 | Là kiến trúc tự ghép, không có ai dùng. **Bỏ** |

## 5. Số agent

### 5.1 Tài liệu nói gì

| Bài | Thiết lập | Phát hiện về số agent |
|---|---|---|
| Du et al., ICML 2024 | Code mặc định 3 agent × 2 vòng | Thêm agent và thêm vòng đều tăng độ chính xác, nhưng tốn hơn |
| Zhang et al., Findings of ACL 2026 | SoM 3 agent × 2 vòng; thử 3, 6, 9 agent | Tăng số agent hay số vòng phần lớn không đổi kết quả. Riêng EoT trên GSM8K tăng đều từ 3 lên 9 agent |
| Kaesberg et al., ACL 2025, [arXiv 2502.19130](https://arxiv.org/abs/2502.19130) | 3 agent × 3 vòng; Llama 3 8B và 70B | Từ 1 lên 10 agent: độ chính xác **tăng nhẹ**. Từ 1 lên 10 vòng: **giảm nhẹ** |
| Choi, Zhu & Li, NeurIPS 2025 | 5 agent; Qwen2.5-7B, Llama3.1-8B | Từ 1 lên 5 agent thì tốt hơn, chủ yếu nhờ hiệu ứng bỏ phiếu |
| Yang et al. 2026, [arXiv 2602.03794](https://arxiv.org/abs/2602.03794) | Qwen-2.5-7B, Llama-3.1-8B, Mistral-7B | Agent **đồng nhất** bão hòa quanh N ≈ 4. Agent không đồng nhất còn tăng tới N ≈ 8 |
| Weng, Chen & Wang, ICLR 2025 (BenchForm), [arXiv 2501.13381](https://arxiv.org/abs/2501.13381) | 7 agent (1 bị thử + 6) | Đa số càng đông càng dễ chiều theo: Llama3-70B 69,9% khi đa số 6, 32,6% khi đa số 3. **Bước nhảy lớn từ 5 lên 6**, tức khi không còn ai phản đối |
| Choi et al., Findings of ACL 2025, [arXiv 2506.01332](https://arxiv.org/abs/2506.01332) | Hơn 2.500 debate | Agent theo nhóm đông hơn và theo agent giỏi hơn |
| De Marzo, Castellano & Garcia, *Science Advances* 2026 | Tới hơn 1.000 agent | Mô hình mạnh vẫn theo đa số ở nhóm rất lớn |
| Bond 2005 (tổng hợp 125 nghiên cứu kiểu Asch) | — | Ảnh hưởng của cỡ đa số tùy kiểu ảnh hưởng. Đa số các nghiên cứu chỉ dùng đa số 2–4 người |
| Centola & Macy 2007 | — | Lây phức cần ít nhất 2 nguồn, nên k phải lên được tới ≥ 2 mà vẫn còn người phản đối |

### 5.2 Tiêu chí của đề tài

Nếu chỉ nhằm cho đúng nhiều nhất thì 3–5 agent là đủ. Nhưng đề tài cần **đo được cách lây**, nên cần:
1. **Dải liều k = 0…5, kể cả "còn một người phản đối" và "nhất trí"** (chỗ Weng 2025 thấy bước nhảy). Tức là mỗi agent đọc được 5 bạn, cần **6 agent** trên đồ thị đầy đủ.
2. **Nhiều mức "số bạn đọc" trong cùng một nhóm.** Ba đồ thị của SoM/Choi với 6 agent cho đúng 5, 2 và 1.
3. **Lây nhiều bước** để tính R_eff: cần nhóm lớn hơn trên đồ thị thưa.
4. **Chi phí:** trên đồ thị đầy đủ, lượng token tăng theo khoảng N².
   - 12 agent đầy đủ: mỗi agent đọc 11 lời giải, khoảng 3.000–11.000 token tùy độ dài.
   - 12 agent thưa: mỗi agent đọc 2 lời giải.
5. **So được với tài liệu:** 3 agent (Du, Zhang, Kaesberg); 5–7 agent (Choi, Weng).

### 5.3 Đề xuất

| Thiết lập | Số agent | Đồ thị (đều là đồ thị đã công bố) | Mục đích | Lượt gọi mỗi debate (vòng 0 + 4 vòng) |
|---|---|---|---|---|
| **Chính** | **6** | đầy đủ, thưa, tập trung | Đường liều k = 0…5; số người hay tỉ lệ; có ai phản đối hay nhất trí | 30 |
| Mở rộng | 12 | thưa (vòng tròn, như Li 2024 và Choi 2025) | Lây nhiều bước, R_eff; xem hồ sơ đo ở 6 agent có dùng được cho nhóm lớn hơn không | 60 |
| Mốc so sánh | 3 | đầy đủ (đúng mặc định của Du) | So với tài liệu | 15 |

**Số vòng: vòng 0 cộng 4 vòng.**
- Thay đổi tập trung ở 2–3 vòng đầu, và thêm vòng làm giảm độ chính xác (Kaesberg 2025).
- Đo độ bền cần ít nhất 3 vòng bị phản đối liên tiếp.
- Riêng mốc 3 agent có thể chạy đúng 2 vòng như Du.

**Agent đồng nhất (một mô hình mỗi debate) ở giai đoạn đầu.** Giống thiết lập chính của Du và Choi.

## 6. Kaggle: cách tổ chức code cho quy trình "push GitHub → clone trên Kaggle → chạy"

### 6.1 Ràng buộc đã kiểm tra

| Ràng buộc | Giá trị | Nguồn |
|---|---|---|
| GPU Kaggle | 2 × T4 16 GB, compute capability 7.5, không có bf16 (phải dùng fp16) | Tài liệu vLLM, Kaggle |
| vLLM | Bản mới nhất hỗ trợ **compute capability ≥ 7.5** (có T4). **Chỉ Linux** (Windows phải qua WSL). Python 3.11–3.14. Bản dựng sẵn dùng CUDA 12.9 | [docs.vllm.ai, trang cài đặt GPU](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| Quota | 30 giờ GPU mỗi tuần; mỗi phiên tối đa 12 giờ; ổ `/kaggle/working` 20 GB | Trang giới hạn của Kaggle |
| Repo riêng tư | Clone cần token GitHub (fine-grained, chỉ quyền đọc, chỉ repo này), lưu trong **Kaggle Secrets** | Tài liệu GitHub/Kaggle |

**Lưu ý về backend:**
- Code chính thức của Choi et al. chạy mô hình bằng Hugging Face Transformers (`generate`, temperature 1,0, top_p 0,9, tối đa 512 token mới). Cách này chậm trên T4.
- Đổi sang vLLM chỉ là đổi **cách gọi mô hình**, không đổi kiến trúc debate. Sẽ ghi rõ mọi tham số lấy mẫu.

Cần kiểm tra ở phiên chạy thử: phiên bản Python và driver CUDA của image Kaggle có hợp với vLLM mới nhất không. Nếu không thì ghim một bản vLLM cũ hơn.

### 6.2 Cấu trúc đề xuất

```
TD/
  <gói lõi>/       kiến trúc SoM (prompt nguyên văn của Du), 3 đồ thị của Choi et al., client (thư viện openai
                   chính thức gọi vLLM), trích và chấm đáp án, ghi log. KHÔNG có dòng nào nhắc tới Kaggle
  configs/         cấu hình chạy (mô hình, đồ thị, số agent, số vòng, số câu)
  scripts/
    serve.sh       bật vLLM bằng lệnh `vllm serve` chính thức; dùng chung cho mọi máy Linux có GPU
    run.py         lệnh thu dữ liệu duy nhất; chạy tiếp được sau khi bị ngắt
  tests/           chạy trên CPU với một máy chủ giả, không cần GPU
  kaggle/          CHỈ phần riêng của Kaggle:
    run_kaggle.ipynb   clone đúng commit → cài vLLM (ghim phiên bản) → khôi phục kết quả phiên trước
                       → gọi scripts/serve.sh và scripts/run.py → lưu kết quả
    kaggle.sh          bọc: chia 2 GPU thành 2 làn, dừng nhận việc trước mốc 12 giờ
    README.md          các bước bấm trên Kaggle (Secrets, Internet, GPU T4 x2, thêm Input)
```

**Luật để bảo đảm "xóa `kaggle/` vẫn chạy bình thường":**
1. Code lõi không import và không đọc file gì trong `kaggle/`. Có thêm một test kiểm tra điều này.
2. `kaggle/` chỉ gọi đúng các lệnh mà một máy GPU bình thường cũng gọi.
3. Mọi giá trị riêng của Kaggle (đường dẫn, giờ dừng, token) nằm trong notebook hoặc biến môi trường.

**Luồng chạy:**
1. Push lên GitHub, ghi lại mã commit hoặc tạo tag.
2. Trên Kaggle: đặt `COMMIT`, rồi `git clone` và `git checkout`. Token (nếu repo riêng tư) lấy từ Kaggle Secrets.
3. Cài các gói cần thiết cùng vLLM bản đã ghim.
4. Khôi phục kết quả phiên trước.
5. Chạy smoke test.
6. Chạy 2 làn, mỗi GPU một làn. Dừng nhận việc ở mốc 11,3 giờ.
7. "Save Version".

Kết quả lớn lưu bằng output của Kaggle hoặc dataset trên Hugging Face, không đẩy lên GitHub.

## 7. Nguồn code cho bước viết code

| Nguồn | Giấy phép | Dùng gì |
|---|---|---|
| [deeplearning-wisc/debate-or-vote](https://github.com/deeplearning-wisc/debate-or-vote) (Choi et al., NeurIPS 2025) | **MIT** | **Nền chính.** Code chính thức có SoM đầy đủ, thưa, tập trung; chỉnh được số agent và số vòng; GSM8K, CommonsenseQA; Qwen2.5-7B, Llama3.1-8B. Chỉ thay phần gọi mô hình (Transformers → vLLM) |
| [composable-models/llm_multiagent_debate](https://github.com/composable-models/llm_multiagent_debate) (Du et al.) | Không có | Trích nguyên văn prompt SoM, có ghi nguồn; không chép code. Dùng để kiểm prompt của Choi có khớp prompt gốc không |
| [MASWorks/MASLab](https://github.com/MASWorks/MASLab) | Cần kiểm tra | Bản cài lại `llm_debate` và `mad` đã đối chiếu với code chính thức. Dùng nếu chạy MAD |
| [liaolea/M3MAD-Bench](https://github.com/liaolea/M3MAD-Bench) | MIT | Bản cài khác của LLM Debate và MAD; gọi được máy chủ dạng OpenAI |

**Cần clone:** debate-or-vote (bắt buộc), llm_multiagent_debate (để đối chiếu prompt). Nếu chạy MAD thì thêm MASLab hoặc M3MAD-Bench.

## 8. Cần quyết định

1. **SoM giữ nguyên** làm kiến trúc chính, với 3 đồ thị của Choi et al.?
2. **Có chạy MAD của Liang** làm điều kiện đối kháng không? Nếu có thì chạy trên một phần câu hỏi hay toàn bộ?
3. **Số agent:** 6 chính, 12 mở rộng, 3 mốc. Có cắt bớt không?
4. **Mô hình và repo GitHub** (công khai hay riêng tư): chưa research trong báo cáo này.

## 9. Tài liệu

Số trích dẫn tra trên Semantic Scholar ngày 09/10/2026.

**Kiến trúc debate**
- Chan, C.-M., et al. (2024). ChatEval: Towards better LLM-based evaluators through multi-agent debate. *ICLR 2024*.
- Chen, J. C.-Y., Saha, S., & Bansal, M. (2024). ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs. *ACL 2024*.
- Chen, W., et al. (2024). AgentVerse: Facilitating multi-agent collaboration and exploring emergent behaviors. *ICLR 2024*.
- Choi, H. K., Zhu, X., & Li, S. (2025). Debate or vote: Which yields better decisions in multi-agent large language models? *NeurIPS 2025*. arXiv:2508.17536. Code: github.com/deeplearning-wisc/debate-or-vote (MIT).
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*. arXiv:2305.14325.
- Khan, A., et al. (2024). Debating with more persuasive LLMs leads to more truthful answers. *ICML 2024*.
- Li, Y., Du, Y., Zhang, J., Hou, L., Grabowski, P., Li, Y., & Ie, E. (2024). Improving multi-agent debate with sparse communication topology. *Findings of EMNLP 2024*.
- Liang, T., et al. (2024). Encouraging divergent thinking in large language models through multi-agent debate. *EMNLP 2024*.
- Liu, Y., Cao, J., Li, Z., He, R., & Tan, T. (2025). Breaking mental set to improve reasoning through diverse multi-agent debate. *ICLR 2025*.
- Yin, Z., et al. (2023). Exchange-of-Thought: Enhancing large language model capabilities through cross-model communication. *EMNLP 2023*.

**Benchmark, framework và tổng quan**
- Li, A., et al. (2026). M3MAD-Bench: Multi-dimensional evaluation of multi-agent debate across domains and modalities. *ACM MM 2026*. arXiv:2601.02854.
- Motger, Q., Oriol, M., Marco, J., & Franch, X. (2026). Multi-agent debate strategies: Survey, taxonomy, and challenges. arXiv:2607.26212.
- Smit, A., et al. (2024). Should we be going MAD? A look at multi-agent debate strategies for LLMs. *ICML 2024*.
- Ye, R., et al. (2025). MASLab: A unified and comprehensive codebase for LLM-based multi-agent systems. arXiv:2505.16988.
- Zhang, H., Cui, Z., Chen, J., Wang, X., Zhang, Q., Wang, Z., Wu, D., & Hu, S. (2026). Stop overvaluing multi-agent debate: We must rethink evaluation and embrace model heterogeneity. *Findings of ACL 2026*. arXiv:2502.08788.

**Debate có thật sự tranh luận không**
- Hao, X., et al. (2026). Not all flips are conformity: Decomposing stance convergence in multi-agent LLM debate. arXiv:2606.00820.
- Qian, C. (2026). What does multi-agent LLM debate actually change? A layered analysis of disagreement and answer quality. arXiv:2609.08016.
- Wynn, A., Satija, H., & Hadfield, G. K. (2025). Talk isn't always cheap: Understanding failure modes in multi-agent debate. *ICML 2025 MAS Workshop*. arXiv:2509.05396.

**Số agent và chiều theo đa số**
- Bond, R. (2005). Group size and conformity. *Group Processes & Intergroup Relations*, 8(4), 331–354.
- Centola, D., & Macy, M. (2007). Complex contagions and the weakness of long ties. *American Journal of Sociology*, 113(3), 702–734.
- Choi, M., Kim, K., Chae, S., & Baek, S. (2025). An empirical study of group conformity in multi-agent systems. *Findings of ACL 2025*. arXiv:2506.01332.
- De Marzo, G., Castellano, C., & Garcia, D. (2026). AI agents can coordinate via majority-following beyond human scale. *Science Advances*, 12(33). doi:10.1126/sciadv.aea6091.
- Kaesberg, L. B., Becker, J., Wahle, J. P., Ruas, T., & Gipp, B. (2025). Voting or consensus? Decision-making in multi-agent debate. *ACL 2025*. arXiv:2502.19130.
- Weng, Z., Chen, G., & Wang, W. (2025). Do as we do, not as you think: The conformity of large language models. *ICLR 2025*. arXiv:2501.13381.
- Yang, et al. (2026). Understanding agent scaling in LLM-based multi-agent systems via diversity. arXiv:2602.03794.

**Hạ tầng**
- vLLM, trang cài đặt GPU (bản mới nhất, xem ngày 09/10/2026): docs.vllm.ai/en/latest/getting_started/installation/gpu.html.

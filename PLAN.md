# TD-MAD v5 — Hồ sơ lây nhiễm của niềm tin đúng và niềm tin sai trong hệ nhiều agent LLM

**Đúng và sai lây khác nhau ở khâu nào: lúc vào, lúc bám, hay lúc lan?**

Bản nháp ngày 06/10/2026. Đây là **giai đoạn 1** của đề án: dựng hồ sơ lây nhiễm.
- **Giai đoạn 2** (sau preprint): tìm phương pháp dập niềm tin sai và thúc đẩy thông tin đúng, dựa trên *bản đồ đòn bẩy* mà giai đoạn 1 tạo ra (mục 11).
- Khung 5 tính chất lấy lại từ TD-MAD v2 và được rút gọn cho vừa mốc preprint 31/01/2027. Bản v2, code và dữ liệu pilot cũ được lưu ngoài repo, ở `D:\My Work\Project\TD_archive_2026-10-06\` (bản v2 nằm trong `docs/old/`).
- Kế hoạch chi tiết của từng bước nằm trong thư mục `plans/`. Đã có: `plans/buoc1_chuan_bi.md`.

## 0. Đọc nhanh

- **Câu hỏi.** Trong hệ nhiều agent LLM, niềm tin đúng và niềm tin sai lây khác nhau ở khâu nào: lúc vào (dễ được nhận), lúc bám (khó bị bỏ), hay lúc lan (lây cho nhiều agent khác)?
- **Cách làm.**
  - Coi đáp án đúng và đáp án sai là hai "chủng".
  - Đo cùng 5 tính chất lây nhiễm cho cả hai chủng, trên cùng câu hỏi và cùng mô hình.
  - Kiểm chứng trên hệ nhiều agent chạy thật.
- **Đầu ra.**
  - Hồ sơ hai chủng.
  - *Bản đồ đòn bẩy:* tính chất nào gây ra bao nhiêu phần chênh lệch đúng–sai ở cấp nhóm. Đây là nền cho giai đoạn tìm phương pháp.
- **Độ trùng với các bài đã có** (mục 3):
  - Mình chấm 15 bài liên quan.
  - Bài trùng nhiều nhất là Fukushima 2026, với 65%.
  - Cộng tất cả các bài lại thì phủ khoảng 75% thành phần, nhưng rải rác ở nhiều bối cảnh khác nhau. 25% còn trống.
  - Chưa bài nào làm trọn hồ sơ trên bài toán suy luận trong hệ nhiều agent chạy thật.
- **Quy mô.** 2 mô hình mở (thêm 1 nếu đủ quota), 2 loại câu hỏi, khoảng 25 GPU-giờ mỗi mô hình (1–1,5 tuần quota Kaggle). Preprint ngày 31/01/2027.

## 1. Vấn đề

- **Hệ nhiều agent làm lan cả cái đúng lẫn cái sai.** Nhiều bài đã thấy từng mảnh:
  - agent bị làm hỏng nhiều hơn được sửa (Qu, Fu & Hu 2026);
  - lời khai sai được nhận dễ hơn và lan xa hơn lời khai đúng (Yan et al. 2026);
  - một đồng thuận đã hình thành thì rất khó gỡ (Banerjee & Moghaddas 2026).
- **Mỗi bài chỉ đo 1–4 tính chất**, trong một bối cảnh riêng. Chưa có một hồ sơ đầy đủ để so đúng với sai.
- **Không có hồ sơ thì không biết đánh vào đâu:** chặn ở cửa vào, cắt đường lan, hay gỡ khi đã bám. Vì vậy các biện pháp hiện nay thường giảm được cái sai nhưng giảm luôn cả cái đúng.

## 2. Ý tưởng

- **Hai chủng.** Chủng đúng (+) và chủng sai (−). Với mỗi tính chất ở mục 5, báo cáo giá trị của từng chủng và hiệu Δ = (+) − (−), kèm khoảng tin cậy.
- **Hai biến giải thích:**
  - *Mức chứng minh được của lời nhắn*, gồm 4 mức: chỉ đáp án; lập luận không kiểm chứng được; lập luận kiểm chứng được; lập luận kiểm chứng được nhưng bị cắt mất bước then chốt.
  - *Độ hợp lý sẵn có π* của từng đáp án.
- **Đo rồi kiểm chứng.** Đo trong điều kiện kiểm soát (lượt giả lập), sau đó kiểm chứng trên hệ nhiều agent chạy thật.
- **Bản đồ đòn bẩy.** Mô phỏng lại hệ, mỗi lần "đổi một tính chất của chủng sai cho giống chủng đúng", để biết tính chất đó gây ra bao nhiêu phần chênh lệch đúng–sai ở cấp nhóm.

## 3. Độ trùng với các bài đã có

**Cách chấm** (mình chấm theo abstract và bản HTML trên arXiv; cần đọc toàn văn để chốt):
- Hồ sơ có 10 thành phần, K1–K10.
- Mỗi thành phần được chấm:
  - ● = 1 điểm: đo cho cả chủng đúng và chủng sai, trong thiết kế có kiểm soát;
  - ◐ = 0,5 điểm: có đo, nhưng chỉ một chủng (hoặc không phân biệt đúng sai), hoặc chỉ suy ra từ dữ liệu quan sát;
  - ○ = 0 điểm: không đo.
- **% trùng = tổng điểm / 10.**

**10 thành phần:**

| Mã | Thành phần |
|---|---|
| K1 | Mức nền ε: đổi ý khi không có ai ủng hộ |
| K2 | Khả năng lây từ 1 nguồn p₁ |
| K3 | Đường liều–đáp ứng: nhiều mức số bạn k |
| K4 | Hình dạng đường: ngưỡng và độ dốc; lây đơn hay lây phức; theo số người hay theo tỉ lệ |
| K5 | Độ bền h: bỏ niềm tin khi bị phản bác liên tiếp |
| K6 | Trễ H: dựng lên so với gỡ xuống |
| K7 | R_eff: số ca lây mà một agent gây ra, truy vết bằng can thiệp |
| K8 | Thang mức chứng minh được của lời nhắn |
| K9 | Tách "đúng" khỏi "dễ nghe": kiểm soát độ hợp lý hoặc cách diễn đạt |
| K10 | Dùng hồ sơ dự đoán được động học của cả nhóm |

**Bảng chấm:**

| Bài | K1 | K2 | K3 | K4 | K5 | K6 | K7 | K8 | K9 | K10 | **Trùng** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fukushima 2026 | ● | ● | ● | ◐ | ◐ | ◐ | ○ | ○ | ● | ● | **65%** |
| Banerjee & Moghaddas 2026 | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ○ | ○ | ◐ | ◐ | **40%** |
| El et al. 2026 (Physics of Agents) | ◐ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ | ● | **30%** |
| Hao et al. 2026 | ● | ◐ | ◐ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | **25%** |
| De Marzo, Castellano & Garcia 2026 | ◐ | ◐ | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ | **25%** |
| Qu, Fu & Hu 2026 | ◐ | ○ | ● | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | **20%** |
| Yan et al. 2026 | ○ | ● | ○ | ○ | ◐ | ○ | ◐ | ○ | ○ | ○ | **20%** |
| SycEval (Fanous et al. 2025)\* | ○ | ● | ○ | ○ | ◐ | ○ | ○ | ◐ | ○ | ○ | **20%** |
| Zhao et al. 2026 (When Debate Helps) | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | ◐ | ○ | ◐ | **15%** |
| Xie et al. 2026 (From Spark to Fire) | ○ | ○ | ○ | ○ | ◐ | ○ | ◐ | ○ | ○ | ◐ | **15%** |
| Wavering Oracles (Nee et al. 2026)\* | ◐ | ● | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **15%** |
| Hu & Qu 2026 | ◐ | ◐ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | ○ | **10%** |
| Li, Liu & Yuen 2026 | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ○ | ○ | ◐ | **10%** |
| Lây prompt injection trong bầy agent (Cybersecurity 2026) | ○ | ○ | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ◐ | **10%** |
| Lin et al. 2026 | ○ | ○ | ○ | ○ | ◐ | ○ | ○ | ○ | ○ | ○ | **5%** |
| **Cộng tất cả các bài** (lấy điểm cao nhất ở mỗi cột) | ● | ● | ● | ◐ | ◐ | ◐ | ◐ | ◐ | ● | ● | **75%** |

\* Không phải hệ nhiều agent: đây là góp ý từ người dùng gửi cho một mô hình. SycEval được chấm theo hiểu biết sẵn có, chưa đọc lại trong đợt rà soát này.

**Đọc bảng:**

- **Bài trùng nhiều nhất là Fukushima 2026 (65%).** Đề án khác bài này ở bốn điểm:
  - *Loại câu hỏi:* bài này dùng mệnh đề thực tế (CLIMATE-FEVER), mô hình trả lời đúng/sai bằng một ký tự, tin nhắn hai câu, không có lời giải.
  - *Phép đo còn thiếu:* bài này không đo hình dạng ngưỡng, không đo độ bền và trễ riêng cho từng chủng, không tính R_eff, không thao tác mức chứng minh được.
  - *Bối cảnh:* bài này dùng mạng trao đổi tin nhắn ngắn, không phải hệ nhiều agent giải bài toán suy luận.
  - *Tách đúng khỏi cách diễn đạt:* bài này làm cho mệnh đề, còn đề án làm cho bài toán có lời giải.
- **Cộng tất cả các bài thì phủ 75%**, nhưng rải rác qua ít nhất 8 bài, mỗi bài một bối cảnh, mô hình và dạng câu hỏi khác nhau.
- **25% còn trống** là K4–K8 ở mức "hai chủng, có kiểm soát":
  - hình dạng đường liều theo từng chủng;
  - độ bền theo từng chủng;
  - trễ theo từng chủng;
  - R_eff truy vết bằng can thiệp;
  - thang mức chứng minh được.
- **Chưa bài nào:**
  - làm đủ 10 thành phần trong cùng một thiết kế (cùng câu, cùng mô hình);
  - làm trên bài toán suy luận với lời giải đầy đủ;
  - kiểm chứng trên hệ nhiều agent chạy thật;
  - đưa ra bản đồ đòn bẩy.
- **Một điểm Qu, Fu & Hu tự nêu là hạn chế:** họ đo "hỏng" và "sửa" trên hai tập câu khác nhau, nên lẫn với độ khó; peer chỉ đưa đáp án, không có lời giải; không chạy hệ nhiều agent thật. Đề án đo hai chiều trên *cùng một câu* và có lời giải đầy đủ.

## 4. Thuật ngữ

| Thuật ngữ | Nghĩa trong đề án |
|---|---|
| **Agent; hệ nhiều agent** | Agent: một LLM với cuộc hội thoại riêng. Hệ nhiều agent: nhiều agent đọc lời của nhau rồi trả lời lại qua nhiều vòng |
| **Niềm tin; chủng** | Niềm tin: đáp án agent đang giữ. Chủng đúng (+): đáp án trùng đáp án chuẩn. Chủng sai (−): đáp án sai mà mô hình hay mắc nhất |
| **Hồ sơ lây nhiễm** | Bộ 5 tính chất (mục 5) mô tả một chủng lây thế nào: vào dễ không, cần bao nhiêu nguồn, bám lâu không, lan rộng không |
| **Liều k** | Số bạn đang giữ niềm tin đó mà agent nhìn thấy |
| **Lượt giả lập** | Một lượt hỏi agent trong đó người làm thí nghiệm tự soạn phần "bạn nói gì": bao nhiêu bạn, bạn đúng hay sai, lời nhắn dạng gì |
| **Lây đơn / lây phức** | Lây đơn: một nguồn đã có tác dụng rõ. Lây phức: cần nhiều nguồn củng cố mới đổi |
| **Theo số người / theo tỉ lệ** | Agent phản ứng theo *bao nhiêu* bạn ủng hộ, hay theo *phần trăm* bạn ủng hộ |
| **Mức chứng minh được** | Mức một lời nhắn cho phép người đọc tự kiểm tra là đúng hay sai |
| **Độ hợp lý sẵn có π** | Tỉ lệ mô hình tự chọn một đáp án khi trả lời một mình 30 lần. Dùng để so đúng với sai ở cùng mức "dễ nghe" |
| **Câu lưỡng lự** | Câu mà mô hình lúc chọn đúng, lúc chọn sai. Chỉ những câu này mới có đủ cả hai chủng |
| **Phát lại phản thực tế** | Chạy lại đúng một lượt của agent nhưng bỏ bớt một số lời của bạn, để biết lời nào gây ra việc đổi ý |
| **Giá trị Shapley** | Cách chia công bằng "công lao" cho từng nguồn khi nhiều nguồn cùng tác động; dùng để tính R_eff |
| **Bản đồ đòn bẩy** | Bảng cho biết mỗi tính chất của hồ sơ gây ra bao nhiêu phần chênh lệch đúng–sai ở cấp nhóm, tức là nên can thiệp vào đâu |
| **Mức sàn không-người-nói** | Tỉ lệ đổi ý khi đáp án được lặp lại mà không gắn với bạn nào. Đây là phần "lây" không cần người lây |
| **Đăng ký trước; preprint** | Đăng ký trước: công bố cách đo trước khi chạy, có dấu thời gian. Preprint: bản thảo đưa lên arXiv trước khi phản biện |
| **GPU-giờ; quota Kaggle** | Một giờ chạy trên một GPU. Kaggle cho miễn phí khoảng 30 giờ phiên mỗi tuần, mỗi phiên có 2 GPU T4 |

## 5. Năm tính chất của hồ sơ

| # | Tính chất | Nói đơn giản | Đo thế nào | Neo lý thuyết |
|---|---|---|---|---|
| 1 | **Mức nền ε** | Agent tự đổi sang đáp án này dù không ai ủng hộ | Lượt giả lập với k = 0 | Nhiễu trong mô hình voter có nhiễu (Kirman 1993) |
| 2 | **Khả năng lây p₁** | Một bạn đủ làm agent đổi đến đâu | k = 1, trừ đi ε | "Một người đúng là đủ" (Laughlin & Ellis 1986) |
| 3 | **Hình dạng κ, ngưỡng θ** | Cần bao nhiêu bạn thì mới lan; lây đơn hay lây phức; theo số người hay theo tỉ lệ | k = 0…4 với tổng số bạn 2, 4, 6 | Ngưỡng (Granovetter 1978); lây phức (Centola & Macy 2007) |
| 4 | **Độ bền h, trễ H** | Bị phản bác liên tiếp thì bao lâu mới bỏ; dựng lên dễ hay gỡ xuống dễ | Chuỗi phản bác 5 lượt; tăng dần rồi giảm dần số bạn | Thác thông tin; đa ổn định của q-voter (Castellano et al. 2009) |
| 5 | **R_eff** | Một agent giữ niềm tin này lây cho bao nhiêu agent khác ở vòng sau | Phát lại phản thực tế trong hệ chạy thật, chia công bằng giá trị Shapley | Số sinh sản theo ca (Wallinga & Teunis 2004) |

Mỗi tính chất được báo cáo cho chủng (+), chủng (−), và hiệu Δ.

## 6. Lộ trình bốn bước

```
BƯỚC 1  CHUẨN BỊ                câu hỏi, kho lời giải, lời nhắn 4 mức, đăng ký trước     -> nguyên liệu đo
   │
BƯỚC 2  ĐO HỒ SƠ                lượt giả lập, câu phần A: tính chất 1–4 cho hai chủng      -> hồ sơ cá thể
   │
BƯỚC 3  TRUY VẾT TRONG HỆ THẬT  hệ nhiều agent chạy thật, câu phần B: R_eff                -> tính chất 5
   │
BƯỚC 4  KIỂM CHỨNG & ĐÒN BẨY    hồ sơ có dự đoán được hệ thật không; tính chất nào gây     -> bản đồ đòn bẩy
                                chênh lệch đúng–sai
```

Câu hỏi được chia cố định thành hai phần rời nhau: **phần A** để đo (Bước 2), **phần B** cho hệ chạy thật (Bước 3, 4).

### Bước 1 — Chuẩn bị (07–20/10/2026)

**Làm gì.**
- **Hai loại câu hỏi:**
  - toán lời (GSM8K-Platinum): mức chứng minh được cao;
  - quan niệm sai phổ biến (TruthfulQA, dạng 2 lựa chọn): mức chứng minh được thấp, gần với thông tin sai lệch ngoài đời.
- **Kho lời giải:** mỗi câu cho mô hình tự giải 30 lần, để tính độ hợp lý π và chọn ra câu lưỡng lự.
- **Lời nhắn 4 mức chứng minh được**, dựng từ chính lời giải của mô hình. Có bước kiểm tra lại rằng mỗi mức thật sự khác nhau về độ kiểm chứng được.
- **Đăng ký trước** cách đo trên OSF; chạy thử trên Kaggle.

**Ghi lại.** Số câu lưỡng lự theo π; kết quả kiểm tra 4 mức lời nhắn.

| Phương pháp | Vì sao chọn | Bài dùng tương tự |
|---|---|---|
| GSM8K-Platinum; TruthfulQA | Có đáp án chuẩn. Hai đầu của thang chứng minh được | Vendrow et al. 2025; Lin, Hilton & Evans 2022 |
| Độ hợp lý từ 30 lần tự giải | Để so đúng với sai ở cùng mức "dễ nghe" | Wang et al. 2023 (self-consistency); Fukushima 2026 |
| Lời nhắn có lập luận kiểm chứng được hay không | Thao tác đúng biến mà lý thuyết "sự thật thắng khi chứng minh được" cần | Khan et al. 2024 (trích dẫn đã/chưa kiểm chứng); Laughlin & Ellis 1986 |
| Kiểm tra thao tác bằng tác vụ tìm bước sai | Bảo đảm "kiểm chứng được" thật sự dễ kiểm hơn | Tyen et al. 2024 (BIG-Bench Mistake) |
| Đăng ký trước | Khóa cách đo; đóng dấu thời gian cho ý tưởng | Nosek et al. 2018 |

**Đầu ra.** ≥ 30 câu lưỡng lự mỗi tổ hợp mô hình × loại câu; bộ lời nhắn 4 mức đã kiểm tra; bản đăng ký trước.

### Bước 2 — Đo hồ sơ cá thể (21/10–20/11)

**Làm gì.** Dùng lượt giả lập theo hai chiều trên *cùng một câu*: agent sai gặp bạn đúng (cơ hội lây chủng +), và agent đúng gặp bạn sai (cơ hội lây chủng −).
- **Tính chất 1–3:** k = 0…4, với tổng số bạn 2, 4 và 6; có thêm điều kiện không-người-nói.
- **Tính chất 4:** chuỗi 5 lượt phản bác liên tiếp (đo độ bền h); tăng dần rồi giảm dần số bạn ủng hộ (đo trễ H).
- **Thang 4 mức chứng minh được:** đo ở k = 1, để tiết kiệm chi phí.

**Ghi lại.** Agent nhận, giữ hay đổi ở từng lượt, theo k, tổng số bạn, mức chứng minh được và π.

| Phương pháp | Vì sao chọn | Bài dùng tương tự |
|---|---|---|
| Lượt giả lập hai chiều trên cùng câu | So đúng với sai công bằng; tránh lỗi đo hai chiều trên hai tập câu khác nhau | Fanous et al. 2025 (SycEval); Qu, Fu & Hu 2026 (hạn chế họ tự nêu) |
| Điều kiện không-người-nói | Tách phần lây do "có người nói" khỏi phần do lặp lại đáp án | Hu & Qu 2026 |
| Đường liều log-logistic 4 tham số; so lây đơn/lây phức bằng BIC | Ước lượng ngưỡng và độ dốc; chọn dạng đường có căn cứ | Ritz et al. 2015; Mønsted et al. 2017 |
| So "theo số người" với "theo tỉ lệ" bằng cách đổi tổng số bạn | Các bài 2026 đang bất đồng ở chỗ này | Asch 1955; Fukushima 2026; De Marzo et al. 2026 |
| Hồi quy có hiệu ứng ngẫu nhiên theo câu | Mỗi câu có độ khó riêng; tránh hình dạng giả do gộp | Baayen, Davidson & Bates 2008 |
| Độ bền: phân tích sống sót rời rạc trên chuỗi phản bác | Đo "bao lâu thì bỏ" một cách chuẩn | Singer & Willett 1993; Xu et al. 2024 |
| Trễ: tăng rồi giảm liều | Cách chuẩn để đo ngưỡng vào và ngưỡng ra | Gescheider 1997; Banerjee & Moghaddas 2026 |

**Đầu ra.** Tính chất 1–4 cho hai chủng, kèm Δ và khoảng tin cậy, theo mô hình × loại câu × mức chứng minh được.

### Bước 3 — Truy vết trong hệ thật (21/11–20/12)

**Làm gì.**
- **Chạy hệ nhiều agent thật** trên câu phần B: 6 agent, vòng 0 cộng 5 vòng, với hai cách đọc lời nhau:
  - *đầy đủ:* mỗi agent đọc cả 5 bạn;
  - *vòng tròn:* mỗi agent chỉ đọc 2 bạn bên cạnh.
- **Lấy mẫu sự kiện để truy vết.** Mỗi "sự kiện" là một lần một agent thấy bạn giữ đáp án khác mình. Với một mẫu các sự kiện này, phát lại lượt của agent với từng tập con lời bạn, rồi chia công bằng giá trị Shapley để biết mỗi bạn lây bao nhiêu.
- **Tính R_eff** cho chủng (+) và chủng (−).

**Ghi lại.** Đáp án của từng agent ở từng vòng; kết quả phát lại; R_eff theo vòng.

| Phương pháp | Vì sao chọn | Bài dùng tương tự |
|---|---|---|
| Giao thức debate của Du et al.; đồ thị đầy đủ và thưa | Chuẩn được dùng nhiều nhất; hai đồ thị cho hai mức tiếp xúc | Du et al. 2024; Li et al. 2024 |
| Phát lại phản thực tế, chia công bằng giá trị Shapley | Biết ai lây cho ai bằng can thiệp, không chỉ suy từ tương quan; không đếm trùng khi cần nhiều nguồn | Shapley 1953; Cohen-Wang et al. 2024 (ContextCite); Causal Agent Replay 2026 |
| Shapley lấy mẫu ngẫu nhiên | Giảm số lần phát lại khi có nhiều nguồn | Castro, Gómez & Tejada 2009 |
| R_eff theo ca | Định nghĩa chuẩn trong dịch tễ học | Wallinga & Teunis 2004 |

**Đầu ra.** R_eff của hai chủng, theo đồ thị và loại câu.

### Bước 4 — Kiểm chứng và bản đồ đòn bẩy (21/12/2026–20/01/2027)

**Làm gì.**
- **Kiểm chứng.** Dùng hồ sơ đo ở Bước 2 (phần A) để mô phỏng hệ ở Bước 3 (phần B), không khớp lại tham số. So với hệ thật, và với hai mốc: "không ai đổi" (tương đương bỏ phiếu ở vòng 0) và "lây đơn".
- **Bản đồ đòn bẩy.** Trong mô phỏng, lần lượt thay một tính chất của chủng sai bằng giá trị của chủng đúng, rồi đo chênh lệch đúng–sai ở cấp nhóm giảm bao nhiêu.

| Phương pháp | Vì sao chọn | Bài dùng tương tự |
|---|---|---|
| Mô phỏng hai chủng từ hồ sơ đo được | Kiểm tra hồ sơ cá thể có giải thích được hành vi cả nhóm không | Fukushima 2026; El et al. 2026 |
| Mốc "không ai đổi"; điểm Brier | Mốc mà mọi mô hình phải vượt; thước đo chuẩn cho dự đoán xác suất | Choi, Zhu & Li 2025; Brier 1950 |
| Phân rã chênh lệch theo từng tính chất | Cách chuẩn để biết mỗi yếu tố góp bao nhiêu vào một khoảng chênh | Oaxaca 1973; Blinder 1973; Anderson & May 1991 (tách R₀ thành khả năng lây × số lần tiếp xúc × thời gian nhiễm) |
| Khoảng tin cậy bootstrap theo câu | Các lượt trong cùng một câu không độc lập với nhau | Efron & Tibshirani 1993 |

**Đầu ra.** Sai số dự đoán so với hai mốc; bản đồ đòn bẩy.

## 7. Thiết lập và chi phí

| Mục | Lựa chọn | Lý do |
|---|---|---|
| Mô hình | Qwen2.5-7B-Instruct (đã chạy ở pilot); Llama-3.1-8B-Instruct (El, Fukushima và Qu cùng dùng, nên so được). Thêm Qwen3-4B nếu đủ quota | Khác họ; so được với các bài trùng nhiều nhất |
| Câu hỏi | GSM8K-Platinum; TruthfulQA (2 lựa chọn) | Hai đầu của thang chứng minh được |
| Hệ chạy thật | 6 agent, vòng 0 cộng 5 vòng, nhiệt độ 0,7; đồ thị đầy đủ và vòng tròn | Như pilot |
| Đọc đáp án | Bộ trích v2 kèm đọc logit; ghi rõ mọi tham số lấy mẫu | Tránh hai lỗi đã gặp ở pilot |

| Bước | Khối lượng mỗi mô hình | GPU-giờ |
|---|---|---|
| 1 | 2 loại × 150 câu × 30 lời giải | ≈ 2 |
| 2 | Khoảng 20.000 lượt giả lập; chuỗi phản bác; tăng–giảm liều | ≈ 10 |
| 3 | 2 loại × 80 câu × 2 đồ thị = 320 lần chạy hệ, cộng phát lại để tính Shapley | ≈ 13 |
| 4 | Chạy trên CPU | 0 |
| **Tổng** | | **≈ 25 mỗi mô hình.** 2 mô hình chạy song song khoảng 25 giờ phiên, tức khoảng 1 tuần quota Kaggle. Thêm mô hình thứ ba thì khoảng 1,5 tuần |

## 8. Kết quả mong đợi

1. **Bảng hồ sơ hai chủng:** 5 tính chất × (+, −, Δ), theo mô hình × loại câu × mức chứng minh được.
2. **Hình:** đường liều của hai chủng; đường sống sót (độ bền); vòng trễ; R_eff theo vòng.
3. **Kiểm chứng:** sai số của mô phỏng so với hệ thật, so với hai mốc.
4. **Bản đồ đòn bẩy:** phần trăm chênh lệch đúng–sai do từng tính chất gây ra.

**Câu kết luận mẫu:** *"Trên toán, chủng sai vào khó hơn chủng đúng (p₁ thấp hơn X điểm), nhưng bám lâu hơn (độ bền cao hơn Y lần) và khó gỡ hơn (trễ H lớn hơn). Z% chênh lệch đúng–sai ở cấp nhóm đến từ độ bền. Vì vậy biện pháp nên nhắm vào giai đoạn duy trì, chứ không phải giai đoạn tiếp nhận."*

## 9. Mốc thời gian

| Thời gian | Việc | Cổng |
|---|---|---|
| 07–20/10/2026 | Bước 1; đăng ký trước | ≥ 30 câu lưỡng lự mỗi tổ hợp; 4 mức lời nhắn qua kiểm tra; ≥ 95% số lượt đọc được đáp án |
| 21/10–20/11 | Bước 2 | Có tính chất 1–4 cho hai chủng, kèm khoảng tin cậy |
| 21/11–20/12 | Bước 3 | Có R_eff cho hai chủng |
| 21/12/2026–20/01/2027 | Bước 4 | Có bảng kiểm chứng và bản đồ đòn bẩy |
| 21–31/01/2027 | Viết; **preprint ngày 31/01/2027** | |
| Từ 02/2027 | Giai đoạn 2: phương pháp (mục 11); nộp tạp chí Q1 | |

## 10. Rủi ro

| Rủi ro | Cách xử lý |
|---|---|
| Trùng 65% với Fukushima 2026 | Nêu rõ ngay trong phần mở đầu bốn điểm khác biệt (mục 3). Phần đóng góp chính đặt vào 25% còn trống và bản đồ đòn bẩy |
| Nhóm Banerjee hoặc nhóm El công bố phần "lây cạnh tranh" trước | Đăng ký trước trong tháng 10; giữ đúng mốc preprint |
| Lỗi đo (bộ đọc đáp án, tham số lấy mẫu, phần lây "không cần người nói") | Đọc đáp án bằng hai cách; ghi rõ tham số; luôn có điều kiện không-người-nói |
| Thao tác 4 mức chứng minh được không thành công | Có bước kiểm tra thao tác ở Bước 1. Nếu không đạt, gộp thành 2 mức: chỉ đáp án / có lập luận |
| Hai chủng không khác nhau | Vẫn là kết quả: ảnh hưởng không phân biệt đúng sai. Khi đó bản đồ đòn bẩy chỉ ra rằng phải có nguồn kiểm chứng bên ngoài |
| Chi phí phát lại Shapley vượt dự kiến | Chỉ truy vết một mẫu sự kiện; dùng Shapley lấy mẫu |

## 11. Giai đoạn 2 (sau preprint): từ bản đồ đòn bẩy đến phương pháp

Mỗi tính chất của hồ sơ ứng với một họ biện pháp. Bản đồ đòn bẩy sẽ chọn họ nào được ưu tiên.

| Nếu chênh lệch đúng–sai chủ yếu nằm ở... | ...thì nên thử họ biện pháp | Bài tương tự |
|---|---|---|
| Cửa vào (ε, p₁) | Kiểm chứng nội dung trước khi nhận | Dhuliawala et al. 2024; Estornell & Liu 2024 |
| Hình dạng (κ, θ) | Thay đổi cấu trúc mạng: ai đọc ai | Centola & Macy 2007; Li et al. 2024 |
| Độ bền, trễ (h, H) | Can thiệp sớm; kiểm tra lại sau khi đã đồng thuận; thêm agent kiên định với sự thật | Banerjee & Moghaddas 2026; Xie et al. 2011 |
| Lan rộng (R_eff) | Nhắm vào các nguồn lây mạnh nhất | Pastor-Satorras & Vespignani 2002 |

## 12. Tài liệu tham khảo

Mục có dấu ✓ đã được kiểm tra trên web ngày 06/10/2026.

**Các bài trong bảng độ trùng**
- Banerjee, T., & Moghaddas, N. (2026). Coherence-driven belief formation and population dynamics of contagion in LLM agents. arXiv:2610.02654. ✓
- Cross-layer contagion of prompt injections in multi-agent swarms: A multiplex microscopic Markov chain approach (2026). *Cybersecurity* (Springer). ✓
- De Marzo, G., Castellano, C., & Garcia, D. (2026). AI agents can coordinate via majority-following beyond human scale. *Science Advances*. arXiv:2409.02822. ✓
- El, B., et al. (2026). Physics of agents: Statistical mechanics predicts collective behavior of AI agents. arXiv:2608.16578. ✓
- Fanous, A., et al. (2025). SycEval: Evaluating LLM sycophancy. *AIES 2025*. arXiv:2502.08177. ✓
- Fukushima, M. (2026). Message capacity and claim wording set the transition points of collective truth-finding in language-model networks. arXiv:2609.19183. ✓
- Hao, X., et al. (2026). Not all flips are conformity: Decomposing stance convergence in multi-agent LLM debate. arXiv:2606.00820. ✓
- Hu, Y., & Qu, J. (2026). Most LLM conformity needs no speaker. arXiv:2607.05545. ✓
- Li, X., Liu, M., & Yuen, C. (2026). Measuring collapse and correction in homogeneous-panel LLM debate. arXiv:2609.35279. ✓
- Lin, et al. (2026). You can't fool us: Understanding the resilience of LLM-driven agent communities to misinformation. arXiv:2605.17353. ✓
- Nee, X., Zhong, H., & Ni, X. (2026). Wavering oracles: Selective updating and correlated failures in LLMs. arXiv:2609.11428. ✓
- Qu, Fu & Hu (2026). Easier to mislead than to correct: Harmful and beneficial revision in LLM conformity. arXiv:2606.01637. ✓
- Xie, et al. (2026). From spark to fire: Modeling and mitigating error cascades in LLM-based multi-agent collaboration. arXiv:2603.04474. ✓
- Yan, et al. (2026). When truth is distributed: Misinformation derails collective fact recovery in LLM-based multi-agent systems. arXiv:2608.03421. ✓
- Zhao, Z., et al. (2026). When debate helps: Proposal supply and verification-aware readout in multi-agent reasoning. arXiv:2610.04686. ✓

**Lý thuyết lây nhiễm và động lực ý kiến**
- Anderson, R. M., & May, R. M. (1991). *Infectious Diseases of Humans: Dynamics and Control*. Oxford University Press.
- Castellano, C., Muñoz, M. A., & Pastor-Satorras, R. (2009). Nonlinear q-voter model. *Physical Review E*, 80, 041129.
- Centola, D., & Macy, M. (2007). Complex contagions and the weakness of long ties. *American Journal of Sociology*, 113(3), 702–734.
- Granovetter, M. (1978). Threshold models of collective behavior. *American Journal of Sociology*, 83(6), 1420–1443.
- Kirman, A. (1993). Ants, rationality, and recruitment. *Quarterly Journal of Economics*, 108(1), 137–156.
- Laughlin, P. R., & Ellis, A. L. (1986). Demonstrability and social combination processes on mathematical intellective tasks. *Journal of Experimental Social Psychology*, 22(3), 177–189.
- Pastor-Satorras, R., & Vespignani, A. (2002). Immunization of complex networks. *Physical Review E*, 65, 036104.
- Wallinga, J., & Teunis, P. (2004). Different epidemic curves for severe acute respiratory syndrome reveal similar impacts of control measures. *American Journal of Epidemiology*, 160(6), 509–516.
- Xie, J., Sreenivasan, S., Korniss, G., Zhang, W., Lim, C., & Szymanski, B. K. (2011). Social consensus through the influence of committed minorities. *Physical Review E*, 84, 011130.

**Phương pháp đo và phân tích**
- Asch, S. E. (1955). Opinions and social pressure. *Scientific American*, 193(5), 31–35.
- Baayen, R. H., Davidson, D. J., & Bates, D. M. (2008). Mixed-effects modeling with crossed random effects for subjects and items. *Journal of Memory and Language*, 59(4), 390–412.
- Blinder, A. S. (1973). Wage discrimination: Reduced form and structural estimates. *Journal of Human Resources*, 8(4), 436–455.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3.
- Castro, J., Gómez, D., & Tejada, J. (2009). Polynomial calculation of the Shapley value based on sampling. *Computers & Operations Research*, 36(5), 1726–1730.
- Causal agent replay: Counterfactual attribution for LLM-agent failures (2026). arXiv:2606.08275. ✓
- Cohen-Wang, B., Shah, H., Georgiev, K., & Madry, A. (2024). ContextCite: Attributing model generation to context. *NeurIPS 2024*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Gescheider, G. A. (1997). *Psychophysics: The Fundamentals* (3rd ed.). Lawrence Erlbaum.
- Mønsted, B., Sapieżyński, P., Ferrara, E., & Lehmann, S. (2017). Evidence of complex contagion of information in social media: An experiment using Twitter bots. *PLoS ONE*, 12(9), e0184148. ✓
- Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *PNAS*, 115(11), 2600–2606.
- Oaxaca, R. (1973). Male-female wage differentials in urban labor markets. *International Economic Review*, 14(3), 693–709.
- Ritz, C., Baty, F., Streibig, J. C., & Gerhard, D. (2015). Dose-response analysis using R. *PLoS ONE*, 10(12), e0146021.
- Shapley, L. S. (1953). A value for n-person games. In *Contributions to the Theory of Games II*, 307–317.
- Singer, J. D., & Willett, J. B. (1993). It's about time: Using discrete-time survival analysis to study duration and the timing of events. *Journal of Educational Statistics*, 18(2), 155–195.

**Dữ liệu, giao thức, lời nhắn và biện pháp**
- Choi, H. K., Zhu, X., & Li, Y. (2025). Debate or vote: Which yields better decisions in multi-agent large language models? *NeurIPS 2025*. arXiv:2508.17536. ✓
- Dhuliawala, S., et al. (2024). Chain-of-Verification reduces hallucination in large language models. *Findings of ACL 2024*.
- Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2024). Improving factuality and reasoning in language models through multiagent debate. *ICML 2024*.
- Estornell, A., & Liu, Y. (2024). Multi-LLM debate: Framework, principals, and interventions. *NeurIPS 2024*. ✓
- Khan, A., et al. (2024). Debating with more persuasive LLMs leads to more truthful answers. *ICML 2024*.
- Li, Y., et al. (2024). Improving multi-agent debate with sparse communication topology. *Findings of EMNLP 2024*.
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods. *ACL 2022*.
- Tyen, G., Mansoor, H., Cărbune, V., Chen, P., & Mak, T. (2024). LLMs cannot find reasoning errors, but can correct them given the error location. *Findings of ACL 2024*.
- Vendrow, J., Vendrow, E., Beery, S., & Madry, A. (2025). Do large language model benchmarks test reliability? arXiv:2502.03461.
- Wang, X., et al. (2023). Self-consistency improves chain of thought reasoning in language models. *ICLR 2023*.
- Xu, R., et al. (2024). The earth is flat because...: Investigating LLMs' belief towards misinformation via persuasive conversation. *ACL 2024*.

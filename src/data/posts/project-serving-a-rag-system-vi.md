---
title: "Project end-to-end: triển khai hệ thống RAG"
description: "Triển khai một hệ hỏi đáp có truy hồi cho nhiều người dùng, đo từng giai đoạn và đọc cả những chỉ số trở nên kém hơn."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "project-serving-a-rag-system"
order: 12
pubDate: 2026-09-11
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ đưa được một hệ hỏi đáp có truy hồi từ một script chạy tuần tự lên thành một dịch vụ phục vụ nhiều người, phân biệt được **một phép chuyển nền gộp** với các cặp **thay một biến**, và đọc được một bảng kết quả có cả những ô **xấu đi** - thứ mà một bảng "trước và sau" giấu mất.

Chương Generative AI · Bài 12 dựng một hệ hỏi đáp trên chính loạt bài này: cắt đoạn, mã hóa, tìm ba đoạn gần nhất, đưa cho mô hình 0.5B viết câu trả lời có trích dẫn. Nó chạy được trên một máy tính cá nhân, trả lời từng câu một.

Bài này đưa đúng hệ ấy qua những gì chương này đã đo - mô hình 7B lượng tử hóa của bài 04, gộp lô liên tục của bài 06, bộ đệm tiền tố của bài 09, truyền theo luồng của bài 08, và một giới hạn tải rút ra từ bài 07. **A → B là bước chuyển nền gộp; từ B trở đi, các cặp so sánh giữ chung nền phục vụ và thay từng thành phần hoặc chế độ tải để đọc đánh đổi.**

## 1. Tải và thước đo

**Tải** gồm 32 câu hỏi:

- **20 câu có nhãn** từ chương Generative AI · Bài 12 - mỗi câu biết trước đáp án nằm ở bài nào.
- **8 câu mới** về chính chương này - pha nạp prompt, KV cache, gộp lô tĩnh, tải đóng và tải mở…
- **4 câu tưởng là ngoài kho** - cá koi trong hồ xi măng, công thức phở bò, giá vàng hôm nay, đội vô địch World Cup 2018. Câu trả lời đúng duy nhất là *"Không tìm thấy trong tài liệu."* Mục 2 sẽ cho thấy câu cá koi - dù kho vẫn **không** có căn cứ đúng để trả lời nó - đã bị **rò vào kho**: chính câu hỏi và một câu trả lời sai có mặt trong tài liệu. Nên bài báo cáo hai nhóm riêng: **3 câu không có đáp án, sạch**, và **1 câu không có đáp án, khó, bị rò rỉ**.

**Prompt** gồm một system prompt cố định - bốn quy tắc cộng hai ví dụ mẫu, một ví dụ có trích dẫn và một ví dụ từ chối - rồi ba đoạn truy hồi được đánh số và câu hỏi. System prompt dài **323 token**; cả prompt trung bình **1.127 token**.

**Thước đo tốc độ**, dùng đúng tên của bài 01 và bài 08: thông lượng, thời điểm người dùng **thấy chữ đầu**, và E2E, mỗi thứ ở p50 và p95.

**Thước đo chất lượng**, chấm tự động:

| Thước đo | Trên nhóm câu nào |
|---|---|
| Có trích dẫn | 28 câu trong kho |
| Trích dẫn hợp lệ - mọi số nguồn nằm trong [1], [2], [3] | 28 câu trong kho |
| Từ chối đúng | 3 câu không có đáp án, sạch; câu cá koi - khó, bị rò rỉ - báo riêng, xem mục 2 |
| Từ chối nhầm - nói "không tìm thấy" khi tài liệu có | 28 câu trong kho |

Đây là những **thước đo thay thế**. Chúng bắt được những lỗi mà chương Generative AI · Bài 12 đã gặp - trích dẫn số nguồn không tồn tại, không biết từ chối - nhưng **không** chấm câu trả lời có đúng hay không. Mục 5 sẽ cho thấy chúng còn bỏ sót một loại lỗi mà không ai định kiểm.

## 2. Chặng 0: tầng truy hồi, đo một lần cho mọi chặng

Theo đúng nguyên tắc của chương Generative AI · Bài 12 - **đo hai tầng riêng** - tầng truy hồi chạy **một lần** trước mọi chặng, và mọi chặng dùng chung đúng ba đoạn ấy cho mỗi câu. Nhờ vậy mọi chênh lệch giữa các chặng ở mục 3 đến từ tầng sinh, không lẫn với truy hồi.

| | Giá trị |
|---|---|
| Kho | 1.953 đoạn - toàn bộ loạt bài lúc chạy |
| Mã hóa cả kho bằng e5-small trên GPU | **22,4 s** |
| Một lần truy hồi - mã hóa câu hỏi và quét toàn bộ - p50 / p95 | **4,07 / 21,48 ms** |
| Article hit@3 trên 28 câu trong kho | **26 / 28** |
| Cosine trung bình của đoạn đầu - câu trong kho | 0,8776 |
| Cosine trung bình của đoạn đầu - câu ngoài kho | **0,8374** |

**Truy hồi rẻ.** Bốn mili-giây, so với E2E cỡ giây ở mọi chặng dưới đây: dưới 1% thời gian. Tối ưu độ trễ truy hồi ở hệ này là tối ưu sai chỗ - đúng như bài 10 đo được, quét toàn bộ là đủ ở quy mô vài nghìn đoạn.

**Điểm truy hồi có tín hiệu, nhưng không đủ để đứng một mình.** Hai con số trung bình - 0,8374 so với 0,8776 - gần nhau, nhưng trung bình không nói được hai nhóm có tách được không. Nhìn từng câu:

| | Điểm đoạn đầu |
|---|---|
| 28 câu trong kho | thấp nhất **0,8471**, cao nhất 0,9152 |
| 3 câu không có đáp án, sạch | 0,7935 · 0,8236 · 0,8251 - cao nhất **0,8251** |
| Câu cá koi - không có đáp án, nhưng câu hỏi đã rò vào kho | **0,9075** |

Câu cá koi đạt 0,9075 - cao hơn gần hết các câu trong kho - và đoạn truy hồi đầu của nó là… chương Generative AI · Bài 12, bài đã dùng **đúng câu hỏi ấy** làm ví dụ, kèm nguyên văn câu trả lời bịa của mô hình 0.5B khi ấy. Kho không chứa căn cứ đúng nào về nuôi cá koi, nhưng **chứa chính câu hỏi**, và cả một câu trả lời sai. Nên nhãn *"không có đáp án trong kho"* **vẫn đúng** - câu trả lời chuẩn vẫn là *"Không tìm thấy trong tài liệu."* - nhưng ca này không còn là một câu không có đáp án **sạch**: câu hỏi và một câu trả lời sai đã **rò vào kho** (*leakage*). Nó thành một **câu âm khó**, loại mà hệ thật gặp nhiều hơn ta tưởng: tài liệu nói về đúng chủ đề ấy, thậm chí nhắc đúng câu hỏi ấy, mà không trả lời nó. Bộ truy hồi làm đúng nhiệm vụ đo độ giống khi kéo đoạn nhắc đúng câu hỏi lên đầu; nhưng **điểm giống cao không chứng minh đoạn ấy chứa căn cứ trả lời đúng**.

Cái bẫy không dừng ở tầng truy hồi. Ở chặng A, mô hình 0.5B trả lời câu cá koi bằng đúng câu *"Chương này nói về cách nuôi cá koi trong hồ xi măng."* - nguyên văn câu trả lời bịa mà chương trước đã trích để làm ví dụ về lỗi. Nó chép lại lỗi của chính nó từ tài liệu.

Nên bài báo cáo **hai nhóm riêng**. Với **3 câu không có đáp án sạch**, điểm truy hồi **tách sạch** khỏi câu trong kho trên bộ thử này: câu sạch cao nhất 0,8251, câu trong kho thấp nhất 0,8471 - một ngưỡng nằm giữa hai số ấy giữ đủ 28/28 và loại đủ 3/3. Chỉ 3 câu âm thì không đủ để **chọn** ngưỡng cho hệ thật, nhưng đủ để thấy điểm truy hồi có tín hiệu với loại câu **dễ**. Còn câu cá koi đạt **0,9075** - lọt qua bất kỳ ngưỡng nào còn giữ được câu trong kho - dù kho không có căn cứ đúng. Đó mới là bài học thật của mục này: **một ngưỡng độ giống lọc được câu lạc đề dễ, nhưng không giải được bài toán "có trả lời được không" khi kho chứa đoạn rất giống câu hỏi mà nội dung gây nhiễu hoặc sai.** Việc từ chối có thể đặt ở nhiều chỗ - một ngưỡng đã hiệu chỉnh trên điểm truy hồi, một bộ xếp hạng lại, một bước riêng hỏi "đoạn này có trả lời được câu hỏi không", hay luật dựa trên siêu dữ liệu. Hệ trong bài này đặt nó ở tầng sinh - qua prompt - và đó là **một lựa chọn thiết kế**, không phải điều bắt buộc.

## 3. Sáu chặng, từ script tới dịch vụ

| Chặng | Thay đổi so với chặng trước |
|---|---|
| **A** | Mốc gốc: đúng hệ của chương Generative AI - Qwen2.5-0.5B, `transformers`, **từng câu một** |
| **B** | Đổi sang **Qwen2.5-7B AWQ, nhân Marlin**, trên máy chủ vLLM - bài 04, bài 08. Vẫn từng câu một |
| **C** | Cho yêu cầu **đến đồng thời**, tải mở 2 yêu cầu/giây - gộp lô liên tục của bài 06 có việc để làm |
| **D** | Bật **bộ đệm tiền tố** - bài 09 |
| **E** | Trả lời **theo luồng** - bài 08. Cùng lần chạy với D; chỉ đổi thứ người dùng thấy |
| **F** | **Quá tải**: 12 yêu cầu/giây. **F₀** không giới hạn; **F** giới hạn 20 yêu cầu đang dở, vượt thì từ chối ngay |

Mọi chặng từ B trở đi chạy trên một RTX 3060. Mô hình sinh tối đa 200 token mỗi câu, giải mã tất định.

> ⚠️ **A → B không phải phép thay một biến.** A dùng Qwen2.5-0.5B thường qua `transformers`; B dùng Qwen2.5-7B-AWQ, nhân Marlin, trên máy chủ vLLM. Tôi gộp bước này vì bài muốn so hai cấu hình triển khai *thật*, và vì bản 7B bf16 không chạy nổi trên một card 12 GB nên không có cách nào đổi riêng kích thước mô hình mà giữ nguyên phần còn lại. Nhưng đó là một **lựa chọn thiết kế**, không phải ràng buộc không thể vượt: hoàn toàn có thể chạy thêm một chặng 0.5B trên vLLM để tách phần engine ra khỏi phần mô hình. Bài không làm, nên đây vẫn là điểm yếu của phép đo chứ không phải chuyện đã được biện minh xong. Vì vậy A → B chỉ cho biết khác biệt giữa **hai cấu hình triển khai trọn gói**; nó không cô lập tác dụng riêng của số tham số, lượng tử hóa, nhân tính toán hay serving engine. Từ B trở đi, các cặp B → C, C → D, D → E và F₀ → F mới giữ chung nền 7B-AWQ/Marlin trên vLLM và thay yếu tố được nêu ở từng chặng.

**Tốc độ:**

| Chặng | Thông lượng (req/s) | Người dùng **thấy chữ đầu** p50 / p95 (s) | E2E p50 / p95 (s) | Bị từ chối |
|---|---|---|---|---|
| A | 0,82 | 0,73 / 3,47 | 0,73 / 3,47 | - |
| B | 0,64 | 1,42 / 3,35 | 1,42 / 3,35 | - |
| C | 1,41 | **29,10 / 66,37** | **29,10 / 66,37** | - |
| D | 1,63 | 1,39 / 8,84 | 1,39 / 8,84 | - |
| E | 1,63 | **0,06** / 1,40 | 1,39 / 8,84 | - |
| F₀ | 9,33 | 0,56 / 1,51 | 9,31 / **30,31** | 0 |
| F | 8,24 | 0,11 / 0,15 | 1,84 / **5,08** | **153 / 545 (28%)** |

Từ A tới D, câu trả lời được gửi một khối khi xong, nên người dùng thấy chữ đầu **đúng lúc** E2E. Từ E trở đi có truyền theo luồng, nên thấy chữ đầu lúc có token đầu tiên. Hai dòng F chạy ở tải khác hẳn (12 so với 2 yêu cầu/giây); chúng là một cặp để so với nhau, không so với các dòng trên.

**Chất lượng:**

| Chặng | Có trích dẫn | Hợp lệ | Từ chối đúng (3 câu sạch) | Câu âm khó: cá koi | Từ chối nhầm | Câu trả lời có chữ Hán |
|---|---|---|---|---|---|---|
| A - 0.5B/`transformers` | **5 / 28** | 4 / 28 | 3 / 3 | ✗ mắc bẫy | 0 / 28 | 0 / 32 |
| B - 7B-AWQ/Marlin/vLLM, từng câu | **27 / 28** | 27 / 28 | 3 / 3 | ✓ từ chối | 1 / 28 | **4 / 32** |
| C, D, E, F₀, F | 26 / 28 | 26 / 28 | 3 / 3 | ✓ từ chối | 1 / 28 | 4 / 32 |

Ở câu âm khó cá koi, chặng A **mắc bẫy** - chép lại câu trả lời sai có trong kho - còn các chặng B tới F đều từ chối đúng.

**Bộ nhớ** không có dòng trong bảng, và đó là chủ ý: chặng A đo bằng lượng PyTorch cấp phát (đỉnh **0,99 GiB**), còn các chặng máy chủ đo bằng con số driver báo (khoảng **11 GB**), vì `gpu_memory_utilization=0.85` đặt **ngân sách tối đa 85% VRAM** cho phần thực thi mô hình (model executor): vLLM chạy thử để đo các khoản không phải KV - trọng số, đỉnh activation, đồ thị CUDA và bộ nhớ ngoài Torch - rồi lấy phần ngân sách còn lại làm kích thước bộ nhớ KV, và giữ sẵn chỗ ấy dù có dùng tới hay không. Hai con số đo hai thứ khác nhau - bài 02 đã dặn đừng đặt chúng cạnh nhau.

## 4. Đọc bảng - cả những ô xấu đi

**A → B: phép chuyển nền gộp làm trích dẫn tăng, và ba ô xấu đi.** Câu trả lời có trích dẫn tăng từ **5 lên 27** trên 28. Còn từ chối thì phải đọc theo hai nhóm. Trên **3 câu không có đáp án sạch**, cả A lẫn B đều từ chối đủ 3/3 - **không cải thiện**. Chênh lệch nằm ở **câu âm khó**: cấu hình A mắc bẫy, chép lại câu trả lời sai có trong kho; cấu hình B từ chối đúng. Trước khi tách hai nhóm, bảng ghi "3/4 lên 4/4" - một con số trộn hai loại câu khác hẳn nhau. Tách ra thì kết luận chính xác hơn: B **đứng vững ở một câu có căn cứ gây nhiễu** mà A thì không - trên đúng một ca, nên chỉ là một dấu hiệu. Trích dẫn là thước đo về **làm theo chỉ dẫn**, không phải về câu trả lời có đúng hay không; mục 5 sẽ cho thấy một câu trích dẫn hợp lệ vẫn có thể hỏng. Nhưng cùng lúc, cấu hình B có mỗi câu chậm hơn (E2E p50 **0,73 → 1,42 s**), thông lượng tuần tự thấp hơn (**0,82 → 0,64** req/s), và xuất hiện một câu **từ chối nhầm** - nói "không tìm thấy" dù tài liệu có. Đây là khác biệt giữa hai cấu hình triển khai; phép đo này không cho phép quy riêng bất kỳ ô nào cho số tham số, lượng tử hóa, Marlin hay vLLM.

**B → C: thêm đồng thời mà không đủ sức chứa - hệ sập về độ trễ.** Cho yêu cầu đến 2 cái mỗi giây, E2E p50 nhảy từ **1,42 lên 29,10 giây**, p95 lên **66 giây**. Thông lượng có tăng (0,64 → 1,41) nhưng không theo kịp tốc độ yêu cầu đến, nên hàng đợi phình ra - đúng hiện tượng ở bài 07. Đây là ô xấu đi lớn nhất bảng.

Vì sao lại quá tải ở chỉ 2 yêu cầu mỗi giây, khi bài 07 đo được máy chủ chịu tới khoảng 9? Vì prompt ở đây dài **1.127 token**, còn ở bài 07 chỉ vài chục. Ở chặng B, nạp một prompt mất 0,632 giây, tức khả năng nạp khoảng **1.780 token mỗi giây**. Tải thực tế đến **1,67 yêu cầu mỗi giây** (100 yêu cầu trong 60 giây), cần 1,67 × 1.127 ≈ **1.880 token nạp mỗi giây** - vượt con số ấy. Và bài 01 đã cho thấy pha nạp prompt thiên về **tính toán**, nên gộp lô cứu được pha sinh nhưng không cứu được mấy cho pha nạp. Đây là một **phép tính sức chứa thô**: khả năng 1.780 đo khi chạy từng câu một, còn khi chạy đồng thời thì bộ máy gộp và chia nhỏ các lượt nạp, pha sinh dùng chung GPU, và bộ lập lịch có hiệu ứng riêng. Nó **phù hợp rất khớp** với việc C mất ổn định, nhưng không phải bằng chứng nó là nguyên nhân duy nhất.

**C → D: bộ đệm tiền tố dời hẳn mép sức chứa.** Bật bộ đệm, cùng tải ấy: E2E p50 từ **29,10 xuống 1,39 giây** - giảm 21 lần. System prompt 323 token, tức **28,7%** mỗi prompt, giờ được **tái dùng** mỗi khi các khối KV của nó còn nằm trong bộ đệm - không phải nạp lại mỗi lần. Nhu cầu nạp tụt còn khoảng 1,67 × (1.127 − 323) ≈ **1.340 token mỗi giây** - dưới con số khả năng thô ở trên. Cùng phép tính ấy, nó phù hợp với việc ba mươi phần trăm bớt đi đẩy hệ từ phía mất ổn định sang phía ổn định của cùng một mép.

Bài 09 đo bộ đệm tiền tố làm TTFT nhanh 44 lần - một con số đẹp nhưng nghe như tối ưu phụ. Ở đây nó gắn với một việc lớn hơn: **hệ có đứng vững ở tải này hay không**. Phép tính ở đoạn trên là ước lượng thô, nên phát biểu chắc được là: bật bộ đệm tiền tố là thay đổi **duy nhất** giữa C và D, và nó biến một hệ mất ổn định thành ổn định trên đúng tải này; phép tính sức chứa cho một lời giải thích đúng hướng và đúng cỡ.

Và vẫn có ô chưa đẹp: D có **p95 8,84 giây**, cao gấp sáu lần p50. Hệ đứng vững nhưng đứng gần mép, và những lúc yêu cầu đến dồn là đuôi độ trễ dài ra ngay.

**D → E: truyền theo luồng - chỉ đổi thứ người dùng thấy.** Cùng một lần chạy, người dùng thấy chữ đầu sau **0,06 giây** thay vì **1,39**, và ở p95 là **1,40** thay vì **8,84**. E2E không đổi một chữ số nào, đúng như bài 08: máy chẳng nhanh lên, người dùng bắt đầu nhận giá trị sớm hơn.

**F₀ → F: giới hạn tải - đổi từ chối lấy độ trễ.** Ở 12 yêu cầu mỗi giây, không giới hạn thì mọi yêu cầu được nhận và E2E p95 là **30,3 giây**. Giới hạn 20 yêu cầu đang dở thì E2E p95 còn **5,1 giây**. Con số 20 là một **điểm xuất phát**, lấy từ định luật Little ở bài 07: khoảng 9 yêu cầu mỗi giây nhân với khoảng 2 giây mỗi yêu cầu cho khoảng 18 yêu cầu trong hệ **trung bình** quanh điểm vận hành ấy. Định luật Little không nói giới hạn nào **bảo đảm** được độ trễ - nó chỉ cho số trung bình ở một hệ ổn định. Nên cách làm đúng là chọn một điểm xuất phát như vậy, đo bằng tải mở xem p95 và tỉ lệ từ chối ra sao, rồi chỉnh. Đổi lại, **153 trên 545 yêu cầu (28%) bị từ chối ngay**, và thông lượng giảm 12%.

Đó là một đánh đổi thật chứ không phải cải thiện miễn phí. Nhưng nó là đánh đổi **đúng hướng**: người bị từ chối nhận câu trả lời "đang quá tải, thử lại sau" sau vài mili-giây, thay vì mọi người cùng chờ nửa phút. Và nó bảo vệ máy chủ khỏi loại sập mà bài 11 đã mổ - khi số yêu cầu chạy cùng lúc lên tới 193 thì bộ máy chết.

## 5. Chất lượng: những thứ bảng tốc độ không thấy

**Chạy theo lô đổi câu trả lời, dù giải mã tất định.** So từng câu trả lời giữa các chặng:

| Cặp chặng | Câu trả lời giống hệt từng ký tự |
|---|---|
| B (từng câu) - C (theo lô) | 30 / 32 |
| B - D | 29 / 32 |
| C - D | 31 / 32 |
| D - F₀ | 31 / 32 |

Ba câu khác nhau giữa B và D đều **giống nhau ở đoạn đầu** rồi rẽ sang hướng khác **giữa chừng** - ở ký tự thứ 68, 181 và 278. Bài 09 đã nói trước giới hạn này mà chưa đo: khi nhiều yêu cầu chạy chung một lô, thứ tự cộng các số thực đổi đi, và ở chỗ hai token gần ngang điểm nhau, chênh lệch nhỏ ấy đổi token được chọn - rồi phần sau rẽ theo. Giờ có số: khoảng **một câu trong mười**. Hệ quả thực dụng: **đừng dùng "giải mã tất định" làm lý do để tin hai lần chạy trên máy chủ cho cùng một chữ**, và đừng so chất lượng hai cấu hình bằng cách so từng chữ.

**Cấu hình B-F chuyển sang tiếng Trung giữa câu.** Cấu hình 7B-AWQ/Marlin trên vLLM trả lời có chữ Hán ở **4 trên 32** câu - luôn đúng bốn câu ấy, ở mọi chặng từ B tới F. Cấu hình A dùng 0.5B/`transformers` thì **không câu nào**. Câu về dùng chung khóa-giá trị cho thấy nó xảy ra thế nào:

> *"…Cụ thể, với 14 đầu truy vấn nhưng chỉ 2 bộ khóa-giá* **值，这里应该是**…*"*

Đang viết tiếng Việt thì chữ "trị" bị thay bằng chữ **值** - chính chữ Hán mà "giá trị" mượn từ đó - rồi mô hình đi tiếp bằng tiếng Trung.

Lỗi này ổn định qua mọi chặng B-F, nên các thay đổi phục vụ sau B - đồng thời, bộ đệm tiền tố, truyền theo luồng và giới hạn tải - không tạo ra cũng không chữa được nó. Nhưng vì A → B là thay đổi gộp, dữ liệu chỉ gắn lỗi với **toàn bộ ngăn xếp sinh của cấu hình B**, không đủ quy riêng cho kích thước mô hình, lượng tử hóa, nhân Marlin hay vLLM. Và **các tiêu chí trích dẫn không phát hiện được lỗi ngôn ngữ**: chúng chỉ kiểm trích dẫn, mà những câu ấy vẫn có trích dẫn hợp lệ. Lỗi này phải được theo dõi bằng một phép đo riêng - chính là cột "Câu trả lời có chữ Hán" trong bảng ở mục 3. Đúng bài học của chương Generative AI · Bài 12: **luật chỉ bắt được đúng thứ nó kiểm**. Muốn bắt lỗi này cần một luật riêng - ví dụ đếm ký tự ngoài bảng chữ tiếng Việt - và một hệ thật nên có luật ấy trước khi đưa cho người dùng.

## 6. Còn gì trước khi đưa cho người dùng thật

Bảng ở mục 3 cho thấy hệ chạy được và chạy nhanh. Nó chưa phải một dịch vụ. Những việc còn thiếu, mỗi việc gắn với một bài đã đo:

| Việc | Vì sao | Bài |
|---|---|---|
| Kiểm ngôn ngữ đầu ra | 4/32 câu chuyển sang tiếng Trung mà bộ chấm không thấy | mục 5 |
| Khóa bộ đệm theo quyền người hỏi, nếu có bộ đệm câu trả lời | Dùng chung câu trả lời giữa hai người khác quyền là rò dữ liệu | 09 |
| Ghi đủ trường cho mỗi yêu cầu, **gồm điểm truy hồi** | Không có thì lỗi đổi mô hình embedding không để lại dấu vết | 10, 11 |
| Bộ câu hỏi có nhãn, chạy lại mỗi lần đổi mô hình, prompt hay kho | Bảng chất lượng mục 3 chỉ có giá trị khi đo lại được | 10 |
| Kế hoạch sức chứa bằng **tải mở**, với **đúng độ dài prompt thật** | Chặng C quá tải ở 2 req/s trong khi bài 07 đo được 9 - vì prompt dài gấp hàng chục lần | 07 |
| Giới hạn yêu cầu đang dở và một câu từ chối lịch sự | Chặng F: đổi 28% từ chối lấy p95 giảm sáu lần, và tránh loại sập ở bài 11 | 07, 11 |
| Để dư bộ nhớ ngoài ngân sách của bộ máy | Mức 0,9 đã làm bộ máy chết bốn lần trong chương | 04, 09, 11 |

Không việc nào trong số này làm mô hình thông minh hơn. Tất cả là việc làm cho một mô hình đã đủ thông minh **phục vụ được người thật** mà không trả lời sai ngôn ngữ, rò dữ liệu, sập lúc đông người, hay hỏng mà không ai biết.

> 🔧 **Thử ngay:** trưởng nhóm muốn mở hệ này cho 500 nhân viên, "vì bài 07 đo được máy chủ chịu 9 yêu cầu mỗi giây". Bạn trả lời thế nào?
> Con số 9 đo với prompt vài chục token; prompt của hệ này dài **1.127** token, và chặng C cho thấy nó quá tải ở chưa tới 2 yêu cầu mỗi giây nếu không có bộ đệm tiền tố. Nên trước hết phải đo lại sức chứa bằng **tải mở với đúng phân bố độ dài prompt và câu trả lời của hệ này**. Rồi ước lượng tải thật: 500 người, mỗi người hỏi vài câu một ngày, dồn vào giờ hành chính - đó là một con số yêu cầu mỗi giây ở giờ cao điểm, so được với sức chứa vừa đo. Và dù con số nào ra, vẫn nên bật bộ đệm tiền tố và đặt giới hạn yêu cầu đang dở như chặng D và F - trên tải của bài này, đó là hai thay đổi đi liền với việc hệ đứng vững ở mép.

## Tóm tắt bài học

- Đưa hệ RAG của chương Generative AI qua **sáu chặng**, trên cùng bộ câu hỏi và cùng ba đoạn truy hồi cho mỗi câu. A → B là một **phép chuyển nền gộp**; từ B trở đi, các cặp so sánh giữ chung nền phục vụ và thay yếu tố được nêu ở từng chặng.
- **Chặng 0:** truy hồi mất **4,07 ms**, dưới 1% E2E; article hit@3 **26/28**. Trên 3 câu không có đáp án sạch, điểm truy hồi tách sạch khỏi 28 câu trong kho (0,8251 so với 0,8471) - 3 câu âm không đủ để chọn ngưỡng. Câu cá koi vẫn không có đáp án trong kho, nhưng câu hỏi và một câu trả lời sai **đã rò vào kho**: nó đạt **0,9075**, lọt qua mọi ngưỡng còn giữ được câu trong kho. **Ngưỡng độ giống lọc được câu lạc đề dễ, không giải được bài toán "có trả lời được không".** Nơi đặt việc từ chối là một lựa chọn thiết kế.
- **Chuyển nền A → B** - từ 0.5B/`transformers` sang 7B-AWQ/Marlin/vLLM: có trích dẫn từ **5 lên 27** trên 28; từ chối **không đổi** trên 3 câu sạch (3/3 ở cả hai), nhưng B đứng vững ở câu âm khó cá koi mà A mắc bẫy - một ca, nên chỉ là dấu hiệu. Đổi lại, mỗi câu chậm hơn (0,73 → 1,42 s) và có một câu từ chối nhầm. Đây là khác biệt giữa hai cấu hình, không phải phép đo cô lập tác dụng của kích thước mô hình; các thước đo cũng không chấm câu trả lời đúng hay sai.
- **Thêm đồng thời mà thiếu sức chứa** (B → C): E2E p50 lên **29,1 s**. Phép tính thô - cần khoảng 1.880 token nạp mỗi giây, trong khi từng câu một nạp được khoảng 1.780 - phù hợp với việc hệ mất ổn định.
- **Bộ đệm tiền tố** (C → D) - thay đổi duy nhất giữa hai chặng: system prompt 323 token, 28,7% prompt, được tái dùng khi còn trong bộ đệm; E2E p50 giảm **21 lần** (29,1 → 1,39 s) và hệ từ mất ổn định thành ổn định. Phép tính sức chứa thô cho lời giải thích đúng hướng, không phải bằng chứng nguyên nhân duy nhất.
- **Truyền theo luồng** (D → E): thấy chữ đầu sau **0,06 s** thay vì 1,39 s; E2E không đổi.
- **Giới hạn tải** (F₀ → F): ở 12 req/s, E2E p95 từ **30,3 xuống 5,1 s**, đổi lại **28%** yêu cầu bị từ chối ngay và thông lượng giảm 12%. Con số giới hạn lấy từ định luật Little chỉ là điểm xuất phát; phải đo lại để chỉnh.
- **Chạy theo lô đổi câu trả lời** dù giải mã tất định: B và D giống hệt **29/32**; ba câu rẽ hướng giữa chừng.
- **Cấu hình B-F dùng 7B-AWQ/Marlin trên vLLM chuyển sang tiếng Trung** ở **4/32** câu, ổn định qua mọi chặng; cấu hình A thì không. A → B là thay đổi gộp nên chưa thể quy lỗi cho riêng một thành phần. Bộ chấm trích dẫn không bắt được - luật chỉ bắt đúng thứ nó kiểm.
- Trước khi đưa cho người thật còn thiếu: kiểm ngôn ngữ đầu ra, khóa bộ đệm theo quyền, log đủ trường, bộ câu hỏi đo lại được, kế hoạch sức chứa bằng tải mở với đúng độ dài prompt, giới hạn tải, và bộ nhớ dư.

## Câu hỏi tự kiểm tra

1. Vì sao tầng truy hồi được chạy một lần rồi dùng chung cho mọi chặng, thay vì chạy lại ở mỗi chặng?
2. A → B làm thước đo trích dẫn tăng rõ nhưng có ba ô xấu đi. Nêu cả ba và liệt kê những thành phần đã đổi cùng lúc. Vì sao không được quy các chênh lệch này riêng cho kích thước mô hình? Hai thước đo chất lượng ấy còn bỏ sót điều gì, và cần thêm phép thử nào trước khi nói một cấu hình "tốt hơn"?
3. Chặng C quá tải ở chưa tới 2 yêu cầu mỗi giây, trong khi bài 07 đo được cùng máy chủ chịu khoảng 9. Giải thích bằng số.
4. System prompt chỉ chiếm 28,7% prompt mà bật bộ đệm tiền tố làm E2E p50 giảm 21 lần. Vì sao một thay đổi nhỏ lại gây hiệu ứng lớn như vậy? Liên hệ bài 07.
5. Ở chặng E, vì sao E2E giữ nguyên trong khi thời điểm thấy chữ đầu giảm hơn 20 lần?
6. Con số 20 được chọn làm điểm xuất phát thế nào? Nếu mỗi yêu cầu trung bình mất 4 giây thay vì 2, điểm xuất phát đổi ra sao - và vì sao dù chọn thế nào vẫn phải đo lại bằng tải mở?
7. B và D cho câu trả lời khác nhau ở 3 trên 32 câu dù cùng giải mã tất định. Giải thích cơ chế, và nêu một hệ quả cho việc so sánh chất lượng hai cấu hình.
8. Bộ chấm cho B điểm trích dẫn 27/28 nhưng có 4 câu trả lời chuyển sang tiếng Trung. Thiết kế một luật kiểm bắt được lỗi này, và nêu một trường hợp luật ấy báo nhầm.
9. Bạn phải lập kế hoạch sức chứa cho hệ này phục vụ 2.000 người. Liệt kê các phép đo cần làm, theo thứ tự, và nói mỗi phép đo trả lời câu hỏi gì.
10. Câu cá koi không có đáp án trong kho, nhưng câu hỏi và một câu trả lời sai đã rò vào kho. Vì sao nhãn "không có đáp án" của nó vẫn đúng? Nêu điều nó dạy được ở tầng truy hồi và ở tầng sinh, và đề xuất một bước kiểm phát hiện rò rỉ như vậy **trước khi** chấm.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Generative AI · Bài 12** | Hệ hỏi đáp gốc, và nguyên tắc đo hai tầng riêng |
| **Bài 01 của chương này** | Vì sao pha nạp prompt thiên về tính toán - lý do chặng C quá tải |
| **Bài 07 của chương này** | Tải mở, định luật Little, và mép ổn định |
| **Bài 09 của chương này** | Bộ đệm tiền tố, và giới hạn "chạy theo lô có thể đổi token" |
| **Bài 11 của chương này** | Những trường phải ghi, và lần sập mà chặng F phòng |

**Đi sâu hơn:**

- [Lewis et al. - Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (NeurIPS 2020)](https://arxiv.org/abs/2005.11401) - bài báo gốc của kiến trúc hỏi đáp có truy hồi.
- [He - Defeating Nondeterminism in LLM Inference (Thinking Machines, 2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) - vì sao chạy theo lô làm giải mã tất định cho ra chữ khác nhau, hiện tượng ở mục 5.
- [Beyer et al. - Site Reliability Engineering, chương "Handling Overload"](https://sre.google/sre-book/handling-overload/) - từ chối sớm khi quá tải, nền của chặng F.

**Mã nguồn của bài:** [`code/b12_chung.py`](../../code/ai-systems/b12_chung.py) - tải, prompt, bộ chấm; [`code/b12_chuanbi.py`](../../code/ai-systems/b12_chuanbi.py) - chặng 0, tầng truy hồi; [`code/b12_A.py`](../../code/ai-systems/b12_A.py) - chặng A; [`code/b12_client.py`](../../code/ai-systems/b12_client.py) - client cho các chặng B tới F; [`code/b12_tinh_lai.py`](../../code/ai-systems/b12_tinh_lai.py) - tính lại thước đo từ chối trên 3 câu sạch và báo riêng câu âm khó, từ kết quả đã lưu, không cần chạy lại mô hình.

## Kết thúc chương AI Systems

Chương này bắt đầu bằng một phép trừ và kết thúc bằng một dịch vụ. Mỗi mắt xích ở giữa có một con số đo được:

- **Một lần gọi là hai pha** - nạp prompt 6.602 token/s, sinh token 58 token/s trên cùng một card, vì pha sinh chỉ dùng **0,65%** khả năng tính toán (bài 01).
- **KV cache tính trước được** - công thức cho 28.672 byte mỗi token, đo được đúng 28.672; và logits lớn gấp **10,6 lần** nó (bài 02).
- **Mô hình không vừa** - 15,23 GB trên card 12,49 GB; và lập ngân sách bằng "trọng số cộng KV cache" hụt **59%** (bài 03).
- **Làm cho nó vừa** - AWQ 5,2 GiB; đổi đúng nhân tính toán nhanh thêm **10,4 lần**; biên bộ nhớ hiện ra đúng chỗ phép chia dự đoán (bài 04).
- **Phục vụ nhiều người** - thông lượng tăng 58 lần từ batch 1 tới 64 gần như miễn phí (bài 05); yêu cầu ngắn không còn chờ 27,8 giây vì yêu cầu dài (bài 06).
- **Đo cho đúng** - tải đóng báo p95 3,6 giây, tải mở cho **27** (bài 07); truyền theo luồng không đổi TTFT của máy, chỉ đổi lúc người dùng thấy chữ (bài 08).
- **Quanh mô hình** - bộ đệm tiền tố nhanh 44 lần mà không tái dùng câu trả lời, bộ đệm ngữ nghĩa trả nhầm câu "dưới 500 triệu" với cosine **0,9956** (bài 09); đổi mô hình embedding rơi từ 0,964 xuống **0,036** mà không một dòng lỗi (bài 10).
- **Biết khi nào hỏng** - một lần sập chẩn đoán được từ log của máy chủ và một phép nhân: **193 × 152.064 × 4 byte**, khớp đúng tensor tạm mà stack chỉ vào (bài 11).
- **Ghép lại** - và thấy bộ đệm tiền tố, thứ nghe như tối ưu phụ, là thay đổi duy nhất biến hệ từ mất ổn định thành ổn định trên đúng tải của bài này.

Nếu chỉ mang theo một điều từ chương này, thì đó là: **mọi con số trong một hệ phục vụ đều tính trước được một phần, và phần còn lại phải đo trên đúng tải của mình.** Công thức nói hệ có vừa không, có nhanh không, gãy ở đâu. Phép đo - trên đúng độ dài prompt, đúng cách người dùng đến, đúng phần cứng - nói con số thật.

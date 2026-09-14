---
title: "Hai loại bộ đệm rất khác nhau"
description: "Bộ đệm tiền tố tái dùng phép tính và không đổi câu trả lời, bộ đệm ngữ nghĩa tái dùng câu trả lời cũ và có thể trả sai; đo lợi ích của cái đầu và vì sao cái sau không an toàn chỉ nhờ một ngưỡng cosine."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "two-very-different-kinds-of-cache"
order: 9
pubDate: 2026-09-10
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ phân biệt được hai thứ cùng được gọi là "bộ đệm" trong hệ mô hình ngôn ngữ - một thứ tái dùng **phép tính tương đương** của phần đầu prompt và về nguyên tắc không đổi câu trả lời, một thứ tái dùng **câu trả lời cũ** và có thể trả sai - đo được lợi ích của thứ nhất, và thấy bằng số vì sao thứ hai không thể an toàn chỉ bằng một ngưỡng cosine.

Bài 08 dựng xong máy chủ. Câu hỏi tự nhiên tiếp theo: có cắt bớt được việc của nó không? Nhiều yêu cầu có phần giống nhau - cùng một system prompt dài, cùng một câu hỏi được hỏi lại. Tính lại từ đầu mỗi lần nghe lãng phí.

"Thêm bộ đệm" là câu trả lời quen thuộc. Nhưng trong hệ mô hình ngôn ngữ, cụm từ ấy chỉ **hai thứ khác hẳn nhau**, và nhầm hai thứ là nhầm từ "tối ưu hiệu năng" sang "trả câu trả lời sai".

## 1. Hai thứ cùng tên

| | **Bộ đệm tiền tố** | **Bộ đệm theo ngữ nghĩa** |
|---|---|---|
| Tái dùng cái gì | **Phép tính KV** của một đoạn token đầu giống hệt nhau | **Câu trả lời** của một câu hỏi cũ |
| Điều kiện trúng | Chuỗi token đầu **khớp chính xác** từng token | Câu hỏi mới **gần nghĩa** với câu hỏi cũ, theo cosine |
| Mô hình có chạy không | **Có** - chỉ bỏ qua phần nạp prompt đã tính | **Không** - trả luôn câu trả lời cũ |
| Có thể đổi câu trả lời không | Về nguyên tắc không - phép tính tương đương; phép thử tuần tự ở mục 3 cho 8/8 giống hệt | **Có** - đó chính là việc nó làm |
| Tiết kiệm được gì | Pha nạp prompt, tức TTFT | Toàn bộ lần gọi mô hình |

Bài 02 đã dựng nền cho cột đầu: khóa và giá trị của một token chỉ phụ thuộc vào các token đứng trước nó. Nếu hai yêu cầu có cùng 2.800 token đầu, thì KV của 2.800 token ấy **giống hệt nhau** - tính một lần, dùng cho cả hai là hoàn toàn chính xác.

Cột thứ hai thì không có nền toán học nào như vậy. Nó dựa trên một giả định: câu hỏi gần nghĩa thì có cùng câu trả lời. Phần 4 sẽ đo xem giả định ấy đứng vững tới đâu.

## 2. Bộ đệm tiền tố: đo

Tình huống rất thường gặp: một trợ lý nội bộ có **system prompt dài** - ở đây là một bản quy chế 40 điều, **2.811 token** - và mọi câu hỏi của người dùng đều đi sau đúng system prompt ấy. Mỗi prompt đầy đủ dài **2.843 token**, trong đó chỉ khoảng 32 token cuối là câu hỏi, khác nhau giữa các yêu cầu.

Đo **thời gian nạp prompt** của 24 yêu cầu gửi **tuần tự** (mỗi yêu cầu chỉ sinh 1 token, nên thời gian ấy chính là pha nạp), trên Qwen2.5-7B-Instruct-AWQ, một RTX 3060, tắt rồi bật bộ đệm tiền tố của vLLM:

| | Yêu cầu đầu tiên | 23 yêu cầu sau (trung vị) | Khoảng |
|---|---|---|---|
| **Tắt** bộ đệm tiền tố | 1,535 s | 1,548 s | 1,539 - 1,563 s |
| **Bật** bộ đệm tiền tố | **1,544 s** | **0,0351 s** | 0,033 - 0,055 s |

Khi tắt, **mọi** yêu cầu trả đủ 1,5 giây để nạp lại 2.843 token - kể cả 2.811 token system prompt mà nó vừa nạp đúng y như vậy một giây trước.

Khi bật, yêu cầu đầu tiên vẫn trả đủ - không có gì trong bộ đệm để dùng. Từ yêu cầu thứ hai, phần system prompt được lấy từ bộ đệm, và thời gian nạp tụt còn **0,035 giây**: **nhanh hơn 44 lần**. Bộ máy tự báo: trong **73.965** token phải nạp, **70.432** token trúng bộ đệm - **95,2%**.

Bộ đệm tiền tố chỉ cắt **pha nạp prompt**. Pha sinh token vẫn tốn y như cũ, vì bài 01 đã cho thấy mỗi token sinh ra phải chạy qua toàn bộ mô hình dù ngữ cảnh đến từ đâu. Nên lợi ích lớn nhất khi prompt dài mà câu trả lời ngắn; khi câu trả lời dài hàng trăm token, 1,5 giây tiết kiệm được chỉ là một phần nhỏ của E2E.

> ⚠️ **Lần đo đầu của tôi tự làm hỏng chính nó.** Ở lần chạy đầu, "yêu cầu đầu tiên" khi bật bộ đệm cũng chỉ mất 0,035 giây - vì bước khởi động của script dùng **cùng hàm dựng prompt**, tức cùng system prompt, nên đã nạp sẵn nó vào bộ đệm trước khi đo. Lần đo trong bảng trên khởi động bằng một prompt **không** chứa system prompt. Bài học chung cho mọi phép đo bộ đệm: **phải chứng minh được lần trượt là lần trượt thật**, nếu không bạn đang đo hai lần trúng rồi gọi một trong hai là lần trượt.

## 3. Bộ đệm tiền tố có đổi câu trả lời không?

Phát biểu cho chính xác: bộ đệm tiền tố **không tái dùng câu trả lời**; nó chỉ tái dùng **phép tính KV** của phần tiền tố. Với cùng trạng thái và giải mã tất định, kết quả phải tương đương việc tính lại.

Kiểm: 8 câu hỏi, giải mã tất định (`temperature = 0`), mỗi câu gửi riêng, chạy một lần khi tắt và một lần khi bật bộ đệm.

| | Kết quả |
|---|---|
| Số câu trả lời **giống hệt từng ký tự** giữa tắt và bật | **8 / 8** |

Còn khi **lấy mẫu** (`temperature = 0.8`), gửi cùng một prompt ba lần:

| | Bản khác nhau trong 3 lần |
|---|---|
| Tắt bộ đệm | 2 |
| Bật bộ đệm | 2 |

Hai lần gọi cùng một prompt vẫn cho ra câu khác nhau - **bất kể bộ đệm**. Ngẫu nhiên đến từ bước lấy mẫu, không đến từ bộ đệm. Có một chi tiết đáng để ý thêm: ba câu trả lời ở hai lần chạy **lặp lại y hệt nhau theo đúng thứ tự**, vì cả hai lần chạy khởi tạo bộ máy với cùng một hạt giống ngẫu nhiên. Muốn hai lần chạy cho ra câu khác nhau thì hạt giống phải khác nhau.

Một giới hạn của phép kiểm này: 8 câu, gửi **từng câu một**. Khi nhiều yêu cầu chạy chung một lô, thứ tự cộng các số thực có thể khác đi một chút, và ở những chỗ hai token gần như ngang điểm nhau, chênh lệch nhỏ ấy có thể đổi token được chọn. Bài này không đo trường hợp đó - bài 12 thì có, và chạy theo lô đổi 3 trên 32 câu - nên không được đọc 8/8 thành "không bao giờ khác".

## 4. Bộ đệm theo ngữ nghĩa: đo

Giờ thứ còn lại. Ý tưởng: lưu lại các cặp câu hỏi-câu trả lời đã có. Khi một câu hỏi mới đến, mã hóa nó bằng embedding, tìm câu hỏi cũ gần nhất, và nếu cosine vượt một ngưỡng thì **trả luôn câu trả lời cũ**, không gọi mô hình.

Để đo, tôi dựng 10 câu hỏi gốc - kiểu câu mà một trợ lý doanh nghiệp hay gặp - và với mỗi câu:

- **2 cách nói lại**: cùng một câu hỏi, khác chữ. Bộ đệm **phải** trúng.
- **2 câu gần giống**: gần như cùng chữ, nhưng **đổi một điều kiện then chốt** - người lớn thành trẻ em, trên 500 triệu thành dưới 500 triệu, năm nay thành năm ngoái. Bộ đệm mà trúng là **trả nhầm câu trả lời**.
- Thêm 5 câu lạc đề hẳn.

Mô hình embedding là `multilingual-e5-small`, đúng mô hình ở chương Generative AI · Bài 10.

| Loại câu | Cosine với câu gốc |
|---|---|
| Cách nói lại - phải trúng | **0,9221 - 0,9894** |
| Gần giống, khác điều kiện - trúng là sai | **0,8855 - 0,9956** |
| Lạc đề - cao nhất với bất kỳ câu gốc nào | 0,7511 - 0,8423 |

Hai khoảng đầu **chồng lên nhau gần như hoàn toàn**. Những câu gần giống điểm cao nhất:

| Cosine | Câu gần giống | Câu gốc |
|---|---|---|
| **0,9956** | Chuyển khoản **dưới** 500 triệu có cần xác thực thêm không? | … **trên** 500 triệu … |
| **0,9951** | Hạn chót nộp báo cáo thuế **năm ngoái** là ngày nào? | … **năm nay** … |
| **0,9910** | Văn phòng Hà Nội **đóng** cửa lúc mấy giờ? | … **mở** cửa … |

Còn cách nói lại điểm **thấp nhất** là *"Vé người lớn giá bao nhiêu?"* - **0,9221** so với câu gốc *"Giá vé vào cổng cho người lớn là bao nhiêu?"*.

Tức là: một câu hỏi **ngược nghĩa** - "dưới" thay cho "trên" - được mô hình embedding coi là **gần câu gốc hơn** cả một cách nói lại đúng nghĩa. Và cả 20 câu gần giống đều có câu gốc của chúng là láng giềng gần nhất trong bộ đệm.

Quét ngưỡng:

| Ngưỡng cosine | Cách nói lại trúng đúng | Câu gần giống bị **trả nhầm** | Câu lạc đề bị trúng |
|---|---|---|---|
| 0,80 | 100% | **100%** | 40% |
| 0,85 | 100% | **100%** | 0% |
| 0,90 | 100% | **95%** | 0% |
| 0,92 | 100% | **90%** | 0% |
| 0,94 | 90% | **65%** | 0% |
| 0,96 | 55% | **35%** | 0% |

**Trên bộ thử này, không ngưỡng nào tách sạch hai nhóm.** Ở 0,92, mọi cách nói lại đều trúng - nhưng **9 trên 10** câu đổi điều kiện cũng trúng, tức nhận câu trả lời của một câu hỏi khác. Nâng lên 0,96 thì đã bỏ lỡ gần nửa số cách nói lại mà vẫn trả nhầm **một phần ba**. Ngay cả ở 0,98, ba câu đổi điều kiện đầu bảng trên vẫn vượt ngưỡng.

Chương Generative AI · Bài 10 đã cho thấy ngưỡng cosine không mang từ bộ này sang bộ khác được. Ở đây tình hình tệ hơn: trên **cùng một bộ**, những câu cần bị loại lại **gần hơn** những câu cần được nhận. Không ngưỡng nào tách được hai nhóm khi chúng nằm chồng lên nhau theo chiều ngược.

> ⚠️ **Bộ thử nhỏ, và cố ý khó.** 10 câu gốc, 20 cách nói lại, 20 câu gần giống - do tôi viết, và tôi viết các câu gần giống **để** chúng khó. Bảng này không đo tỉ lệ trả nhầm trên lưu lượng thật của một hệ nào. Nó chứng minh một điều hẹp hơn và chắc hơn: **tồn tại** những câu hỏi khác nghĩa mà cosine chấm cao hơn cả những câu cùng nghĩa, và chúng không hiếm - chỉ cần đổi một chữ "trên" thành "dưới".

## 5. Vì sao embedding không thấy chữ "dưới"

Một sentence embedding nén cả câu thành một vector cố định. Nó giỏi nắm **chủ đề** và **hình dạng** của câu: câu này hỏi về chuyển khoản, về ngưỡng tiền, về xác thực. Đổi "trên" thành "dưới" chỉ đổi **một** token trong khoảng mười lăm; chủ đề, cấu trúc, gần hết từ vựng đều giữ nguyên, nên vector gần như không nhúc nhích.

Nhưng với người hỏi, chữ "dưới" đó là **toàn bộ** câu hỏi. Đây là chỗ lệch căn bản giữa thứ embedding đo - *hai câu nói về cùng một chuyện không* - và thứ bộ đệm câu trả lời cần - *hai câu có cùng một câu trả lời không*. Hai câu hỏi có thể nói về cùng một chuyện mà có câu trả lời ngược nhau.

Chương Generative AI · Bài 09 có một ví dụ cùng họ: mô hình trả con số đúng của **hiện tại** cho câu hỏi về **năm 2019**. Ở đây là bộ đệm trả câu trả lời của **năm nay** cho câu hỏi về **năm ngoái**, với cosine 0,9951. Lỗi mốc thời gian không cần mô hình bịa gì cả; một bộ đệm là đủ.

## 6. Khóa của bộ đệm không chỉ là cosine

Nếu vẫn muốn dùng bộ đệm câu trả lời, câu hỏi thiết kế không phải "ngưỡng bao nhiêu", mà là **"hai yêu cầu phải giống nhau ở những điểm nào thì câu trả lời mới được dùng chung"**. Một câu trả lời được sinh ra dưới rất nhiều điều kiện, và mỗi điều kiện ấy phải nằm trong khóa:

| Thành phần của khóa | Vì sao |
|---|---|
| Phiên bản mô hình | Đổi mô hình thì câu trả lời đổi |
| Phiên bản system prompt | Đổi quy tắc thì câu trả lời đổi |
| Phiên bản kho tài liệu | Tài liệu được cập nhật thì câu trả lời cũ thành sai - bài 10 |
| **Quyền của người hỏi** | Người A được xem tài liệu mà người B không được; dùng chung câu trả lời là **rò dữ liệu** giữa hai người |
| Thời điểm, nếu câu trả lời phụ thuộc thời gian | "Năm nay", "hôm nay", "giá hiện tại" hết hạn |
| Trạng thái công cụ | Câu trả lời dựa trên kết quả tra cứu lúc ấy |

Thành phần thứ tư là thứ nguy hiểm nhất và dễ quên nhất. Một bộ đệm câu trả lời dùng chung cho mọi người, khóa chỉ bằng nội dung câu hỏi, sẽ trả cho người không có quyền đúng câu trả lời mà người có quyền vừa nhận được.

Vậy khi nào bộ đệm câu trả lời chấp nhận được? Khi câu trả lời **đã được người duyệt** và **không phụ thuộc** người hỏi, thời điểm hay tài liệu riêng - một bộ câu hỏi thường gặp chẳng hạn. Và ngay cả khi ấy, bắt đầu bằng **khớp chính xác** sau khi chuẩn hóa văn bản (bỏ khoảng trắng thừa, chữ hoa chữ thường) trước khi nghĩ tới cosine. Khớp chính xác bỏ lỡ nhiều cách nói lại, nhưng nó không bao giờ trả câu trả lời của "dưới 500 triệu" cho câu hỏi "trên 500 triệu".

> 🔧 **Thử ngay:** nhóm của bạn muốn thêm bộ đệm để giảm chi phí gọi mô hình cho một trợ lý chăm sóc khách hàng ngân hàng. Bạn đề xuất gì?
> Tách hai tầng. **Tầng một**, bật bộ đệm tiền tố - gần như luôn nên bật: nó không tái dùng câu trả lời, chỉ tái dùng phép tính, và system prompt dài của trợ lý ngân hàng là đúng trường hợp được lợi nhất, như mục 2 đo được 44 lần ở pha nạp. **Tầng hai**, bộ đệm câu trả lời: chỉ cho những câu hỏi **không phụ thuộc người hỏi** - giờ mở cửa, cách đặt lại mật khẩu - với câu trả lời đã duyệt, khóa gồm phiên bản tài liệu và thời hạn, bắt đầu bằng khớp chính xác. **Không** dùng cho bất cứ gì liên quan tới tài khoản, hạn mức hay số tiền: mục 4 cho thấy "chuyển khoản dưới 500 triệu" và "trên 500 triệu" cách nhau 0,0044 cosine.

## Tóm tắt bài học

- "Bộ đệm" trong hệ mô hình ngôn ngữ là tên chung của **hai thứ khác hẳn nhau**: bộ đệm tiền tố tái dùng **phép tính KV** của phần token đầu khớp chính xác; bộ đệm ngữ nghĩa tái dùng **câu trả lời cũ** cho câu hỏi gần nghĩa.
- **Bộ đệm tiền tố**, đo trên system prompt 2.811 token: nạp prompt từ **1,548 s xuống 0,035 s** - nhanh **44 lần** - cho mọi yêu cầu sau yêu cầu đầu; **95,2%** token trúng bộ đệm. Nó chỉ cắt pha nạp prompt, không cắt pha sinh token.
- Đo bộ đệm phải chứng minh **lần trượt là lần trượt thật**: lần đo đầu của tôi đã vô tình làm nóng bộ đệm ở bước khởi động.
- Bộ đệm tiền tố **không tái dùng câu trả lời**, chỉ tái dùng phép tính tương đương của tiền tố: phép thử tuần tự, giải mã tất định cho **8/8** câu giống hệt khi tắt và bật. Lấy mẫu vẫn cho câu khác nhau giữa các lần gọi - bất kể bộ đệm. Phép kiểm gửi từng câu một; không được đọc thành "không bao giờ khác" khi chạy theo lô.
- **Bộ đệm ngữ nghĩa**: cách nói lại có cosine **0,9221 - 0,9894**, câu đổi điều kiện then chốt có **0,8855 - 0,9956**. Câu ngược nghĩa "dưới 500 triệu" đạt **0,9956** - cao hơn mọi cách nói lại.
- **Trên bộ thử cố ý khó này, không ngưỡng cosine nào tách sạch** cách nói lại khỏi câu đổi điều kiện: ở 0,92 trúng đủ cách nói lại nhưng trả nhầm **90%** câu đổi điều kiện; ở 0,96 bỏ lỡ gần nửa cách nói lại mà vẫn trả nhầm **35%**.
- Lý do: embedding đo *hai câu có nói về cùng một chuyện không*, còn bộ đệm câu trả lời cần *hai câu có cùng câu trả lời không*. Đổi một chữ đổi được câu trả lời mà gần như không đổi vector.
- Khóa của bộ đệm câu trả lời phải gồm phiên bản mô hình, system prompt, kho tài liệu, **quyền của người hỏi**, thời điểm và trạng thái công cụ. Thiếu quyền người hỏi là rò dữ liệu. Chỉ dùng cho câu trả lời đã duyệt, không phụ thuộc người hỏi, và bắt đầu bằng khớp chính xác.

## Câu hỏi tự kiểm tra

1. Vì sao dùng chung KV của một tiền tố giống hệt nhau là **chính xác**, trong khi dùng chung câu trả lời cho hai câu hỏi gần nghĩa thì không? Dùng kiến thức ở bài 02.
2. Bộ đệm tiền tố cắt 1,5 giây ở pha nạp. Với một câu trả lời 500 token ở tốc độ trong bài 08, khoản tiết kiệm ấy chiếm bao nhiêu phần E2E? Khi nào bộ đệm tiền tố đáng giá nhất?
3. Lần đo đầu cho "yêu cầu đầu tiên" chỉ 0,035 giây khi bật bộ đệm. Giải thích vì sao đó là dấu hiệu phép đo sai, và cách chứng minh một lần trượt là trượt thật.
4. Giải mã tất định cho 8/8 câu giống hệt khi bật và tắt bộ đệm. Vì sao bài vẫn không cho phép kết luận "bộ đệm tiền tố không bao giờ đổi câu trả lời"?
5. Vì sao câu "chuyển khoản **dưới** 500 triệu" có cosine với câu gốc cao hơn một cách nói lại đúng nghĩa?
6. Có thể chọn một ngưỡng cosine cho bộ đệm ngữ nghĩa trên bộ thử của bài không? Lập luận bằng bảng quét ngưỡng.
7. Vì sao "quyền của người hỏi" phải nằm trong khóa của bộ đệm câu trả lời? Mô tả một tình huống rò dữ liệu cụ thể nếu thiếu nó.
8. Thiết kế chiến lược bộ đệm cho một trợ lý tra cứu chính sách nhân sự có câu hỏi phụ thuộc cấp bậc nhân viên. Nêu rõ tầng nào dùng loại bộ đệm nào, và khóa gồm những gì.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 02 của chương này** | Vì sao KV của một tiền tố chỉ phụ thuộc các token trước nó |
| **Bài 01 của chương này** | Pha nạp prompt và pha sinh token - bộ đệm tiền tố chỉ chạm vào pha đầu |
| **Chương Generative AI · Bài 10** | Sentence embedding, và vì sao ngưỡng cosine không mang từ bộ này sang bộ khác |
| **Chương Generative AI · Bài 09** | Lỗi mốc thời gian - cùng họ với việc trả câu trả lời "năm nay" cho câu hỏi "năm ngoái" |

**Đi sâu hơn:**

- [vLLM - Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) - cơ chế băm từng khối KV để nhận ra tiền tố trùng.
- [Zheng et al. - SGLang: Efficient Execution of Structured Language Model Programs (NeurIPS 2024)](https://arxiv.org/abs/2312.07104) - RadixAttention, cách tổ chức bộ đệm tiền tố dạng cây cho nhiều yêu cầu dùng chung các đoạn đầu khác nhau.
- [Reimers & Gurevych - Sentence-BERT (EMNLP 2019)](https://arxiv.org/abs/1908.10084) - nền của sentence embedding, và vì sao nó được huấn luyện để đo độ giống về nghĩa chứ không về điều kiện.

**Mã nguồn của bài:** [`code/b09_tiento.py`](../../code/ai-systems/b09_tiento.py) - bộ đệm tiền tố, bật và tắt; [`code/b09_ngunghia.py`](../../code/ai-systems/b09_ngunghia.py) - bộ thử bộ đệm ngữ nghĩa và quét ngưỡng.

> **Bài tiếp theo:** [Kho vector khi vận hành](../vector-stores-in-production-vi/) - bộ đệm ngữ nghĩa vừa cho thấy embedding có thể lừa ta ở mức từng câu. Bài sau nhìn embedding ở mức cả kho: khi nào cần chỉ mục gần đúng, thêm và xóa tài liệu làm gì với chỉ mục, và điều gì xảy ra khi đổi mô hình embedding mà quên một bước.

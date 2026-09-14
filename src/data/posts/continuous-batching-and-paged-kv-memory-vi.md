---
title: "Gộp lô liên tục và KV cache dạng trang"
description: "Tách hai ý tưởng hay bị gọi chung là \"vì sao vLLM nhanh\", thiết kế phép so sánh không quy hết chênh lệch cho một ý tưởng, và vì sao người hỏi câu ngắn phải chờ người hỏi câu dài."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "continuous-batching-and-paged-kv-memory"
order: 6
pubDate: 2026-09-08
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ phân biệt được hai ý tưởng thường bị gọi chung là "vì sao vLLM nhanh" - **gộp lô liên tục** và **KV cache dạng trang** - biết thiết kế một phép so sánh sao cho không quy toàn bộ chênh lệch cho một ý tưởng duy nhất, và thấy bằng số vì sao người hỏi câu ngắn thôi phải chờ người hỏi câu dài.

Bài 05 kết thúc bằng một con số khó chịu: trong lô tĩnh của bài ấy - thứ trả kết quả cả lô một lượt - yêu cầu ngắn chậm đi **7 lần** vì phải chờ yêu cầu dài nhất trong cùng lô, còn chỗ trống của nó bỏ không tới hết lô. Cách chữa nghe hiển nhiên: **ai xong thì cho ra, chỗ trống thì nhét người mới vào ngay**.

Bài này đo cách chữa ấy qua vLLM - và cẩn thận với một cái bẫy: vLLM khác lô tĩnh tự viết ở **rất nhiều** chỗ, không chỉ ở cách lập lịch.

## 1. Hai ý tưởng, giải hai vấn đề khác nhau

**Gộp lô liên tục** (*continuous batching*, còn gọi là lập lịch theo từng bước) giải vấn đề **thời gian**. Thay vì quyết định thành phần lô một lần rồi chạy tới hết, bộ lập lịch quyết định lại **sau mỗi bước sinh token**: yêu cầu nào vừa xong thì rút ra, yêu cầu nào đang chờ thì đưa vào chỗ trống.

```
Gộp lô tĩnh - lô 4 chỗ, 6 yêu cầu dài ngắn khác nhau (mỗi ô là một bước sinh):

chỗ 1  A A A A A A A A          E E E E
chỗ 2  B B . . . . . .          F F F F F F
chỗ 3  C C C . . . . .          (trống)
chỗ 4  D D D D . . . .          (trống)
       └── lô 1: chờ A xong ──┘ └── lô 2 ──┘       "." = ô chạy không

Gộp lô liên tục - cùng 6 yêu cầu:

chỗ 1  A A A A A A A A
chỗ 2  B B E E E E
chỗ 3  C C C F F F F F F
chỗ 4  D D D D
       E vào ngay khi B xong, F vào ngay khi C xong - không ô nào chạy không
```

**KV cache dạng trang** (*paged KV cache*, ý tưởng của PagedAttention) giải vấn đề **bộ nhớ**. Cách ngây thơ là dành trước cho mỗi yêu cầu một vùng KV cache liền mạch, đủ cho độ dài **tối đa** nó có thể đạt tới. Yêu cầu chỉ dùng 300 token mà được dành chỗ cho 2.048 token thì 85% vùng ấy nằm không. Dạng trang chia KV cache thành những **khối nhỏ cỡ cố định**, cấp dần khi yêu cầu thật sự dài ra - y như cách hệ điều hành cấp bộ nhớ theo trang.

Hai ý tưởng bổ trợ nhau - lập lịch liên tục cần một bộ nhớ linh hoạt để nhét người mới vào chỗ trống - nhưng chúng **là hai thứ riêng**, và một hệ có thể có cái này mà không có cái kia.

## 2. Thiết kế phép so sánh cho khỏi tự lừa mình

Cách so sánh dễ nhất là: lấy lô tĩnh tự viết ở bài 05, lấy vLLM, chạy cùng một tải dài ngắn lẫn lộn, rồi quy chênh lệch cho gộp lô liên tục.

Cách ấy sai. vLLM khác lô tĩnh tự viết ở hàng loạt chỗ: nhân tính toán tối ưu hơn, chụp sẵn đồ thị tính toán (*CUDA graph*) cho những kích thước hay gặp, không quay lại Python sau mỗi bước, chia nhỏ pha nạp prompt, cộng thêm cả lập lịch liên tục và bộ nhớ dạng trang. So một lần thì chỉ biết **cả bộ máy** hơn bao nhiêu, không biết phần nào đến từ đâu.

Nên thí nghiệm có **bốn ô**:

| | Tải A: mọi yêu cầu **cùng độ dài** | Tải B: độ dài **lẫn lộn** |
|---|---|---|
| **Lô tĩnh tự viết, 16 chỗ** | ô 1 | ô 2 |
| **vLLM, giới hạn đúng 16 chỗ** | ô 3 | ô 4 |

Cả hai tải có **64 yêu cầu**, prompt 128 token, và **cùng tổng số token cần sinh là 11.776**: tải A là 64 yêu cầu × 184 token, tải B là 16 yêu cầu mỗi loại 32, 64, 128 và 512 token. Mô hình và card như bài 05: Qwen2.5-1.5B-Instruct, bfloat16, RTX 3060.

Logic tách như sau. Ở **tải A**, không có "thằng chậm nhất" - mọi yêu cầu dài bằng nhau, nên lập lịch liên tục không có ô trống nào để lấp. Chênh lệch giữa ô 3 và ô 1 vì thế là khoảng cách giữa hai bộ máy **khi chưa có chỗ trống nào để tái dùng** - gồm nhân tính toán, đồ thị dựng sẵn, và cả những phần của bộ lập lịch vẫn làm việc khi mọi yêu cầu dài bằng nhau, như cách nhận yêu cầu vào hay chia nhỏ pha nạp. Ở **tải B**, độ dài lẫn lộn tạo ra chỗ trống. Chia tỉ lệ ở tải B cho tỉ lệ ở tải A được **phần lợi thêm gắn với việc độ dài trở nên lẫn lộn** - đúng phần mà gộp lô liên tục được thiết kế để giành.

Đó là ước lượng, không phải phép tách sạch: các phần trong một bộ máy không nhân với nhau gọn gàng như thế. Nhưng nó tốt hơn hẳn việc quy tất cả cho một ý tưởng.

## 3. Thông lượng

| | Tải A - cùng độ dài | Tải B - lẫn lộn | Mất bao nhiêu khi lẫn lộn |
|---|---|---|---|
| **Lô tĩnh, 16 chỗ** | 861,2 tok/s | 310,9 tok/s | **−64%** |
| **vLLM, 16 chỗ** | 1.262,5 tok/s | 926,8 tok/s | −27% |
| vLLM, 256 chỗ | 3.707,1 tok/s | 1.731,9 tok/s | −53% |

Lô tĩnh mất **64%** thông lượng khi tải lẫn lộn - khớp với tính toán: mỗi lô 16 yêu cầu chứa đủ bốn loại độ dài nên phải chạy tới 512 bước, trong khi chỉ cần trung bình 184, tức **64,1%** ô chạy không.

Giờ tách:

| | Tỉ lệ vLLM 16 chỗ ÷ lô tĩnh |
|---|---|
| Tải A - khi **chưa có** chỗ trống để tái dùng | 1.262,5 ÷ 861,2 = **1,47 lần** |
| Tải B - **cả bộ máy** | 926,8 ÷ 310,9 = **2,98 lần** |
| Lợi thêm **gắn với** độ dài lẫn lộn | 2,98 ÷ 1,47 ≈ **2,0 lần** |

Đọc cho đúng: ở **cùng 16 chỗ**, bộ máy vLLM đã nhanh hơn lô tự viết khoảng 1,5 lần **khi chưa có chỗ trống nào để tái dùng**. Khi độ dài lẫn lộn, khoảng cách nới thêm khoảng **gấp đôi** - phù hợp với việc gộp lô liên tục tái dùng được chỗ trống, nhưng phép chia này không phải một phép tách nhân quả chính xác. Nếu chỉ chạy ô 2 và ô 4 rồi viết "gộp lô liên tục làm thông lượng tăng gần 3 lần", thì một phần ba câu ấy là của những thứ khác.

Dòng thứ ba cho thấy tác động của việc **tăng số chỗ**: 16 lên 256 chỗ nhân thông lượng tải A lên 2,94 lần - đúng bài học của bài 05, batch lớn hơn thì mỗi lần đọc trọng số phục vụ được nhiều người hơn.

## 4. Độ trễ của từng yêu cầu

Thông lượng là con số của hệ. Người dùng thì cảm nhận độ trễ của **chính họ**. Đo lại tải B, lần này ghi thời điểm từng yêu cầu có token đầu và thời điểm xong; cả 64 yêu cầu đến lúc $t = 0$:

| Độ dài yêu cầu | Lô tĩnh, 16 chỗ | vLLM, 16 chỗ | vLLM, 256 chỗ |
|---|---|---|---|
| | TTFT / **E2E** (trung vị, giây) | TTFT / **E2E** | TTFT / **E2E** |
| 32 token | 16,38 / **27,83** | 1,88 / **2,26** | 0,37 / **0,83** |
| 64 token | 16,38 / **27,83** | 2,10 / **2,91** | 0,37 / **1,32** |
| 128 token | 16,38 / **27,83** | 2,10 / **3,72** | 0,37 / **2,27** |
| 512 token | 16,38 / **27,83** | 2,10 / **8,51** | 0,37 / **7,08** |

Cột lô tĩnh có **cùng một con số cho mọi dòng**. Không phải lỗi chép: lô tĩnh tự viết của bài này trả kết quả khi **cả lô** xong, và mỗi lô đều chứa một yêu cầu 512 token. Người hỏi câu 32 token chờ **27,8 giây** - y hệt người hỏi câu 512 token.

Hai cột vLLM thì khác hẳn: **độ trễ của mỗi người tỉ lệ với độ dài của chính họ**. Yêu cầu 32 token xong sau **2,26 giây** với 16 chỗ, **0,83 giây** với 256 chỗ - nhanh hơn lô tĩnh **12 lần** và **34 lần**.

> ⚠️ **Hai bảng, hai lần chạy - đừng trộn số.** Bảng mục 3 và bảng này đến từ **hai lần chạy riêng**. Giữa hai lần, thông lượng của lô tĩnh trên tải B lệch khoảng **17%** (310,9 so với 257,3 tok/s), còn của vLLM chỉ lệch 3-4%. Lô tĩnh quay lại Python sau mỗi bước sinh, nên nhạy với việc máy chủ dùng chung có một tiến trình khác chiếm gần hết CPU (bài 05 đã gặp đúng hiện tượng này). Vì vậy phép tách ở mục 3 chỉ dùng số của **một** lần chạy, và bảng này chỉ dùng để đọc **hình dạng** của độ trễ theo độ dài.

## 5. Vì sao 256 chỗ lại "lỗ" nhiều hơn 16 chỗ khi tải lẫn lộn

Dòng cuối bảng mục 3 có một chỗ lạ: vLLM với 256 chỗ mất **53%** thông lượng khi tải lẫn lộn, nhiều hơn hẳn mức 27% của 16 chỗ. Nhiều chỗ hơn mà lại tệ hơn?

Nhìn cột 256 chỗ ở mục 4 là thấy. Với 256 chỗ, cả 64 yêu cầu được nhận **ngay từ đầu**. Các yêu cầu ngắn xong nhanh, rồi… không còn ai để nhét vào chỗ trống. Trong 384 bước cuối - **3/4 số bước sinh** của cả tải - chỉ còn 16 yêu cầu 512 token chạy, tức hệ quay về **batch 16**. Cả tải xong sau **7,08 giây**, và 7,08 giây ấy chính là thời gian một yêu cầu 512 token cần để tự đi hết đường của nó.

Đây là giới hạn mà không bộ lập lịch nào vượt được: **một khối yêu cầu hữu hạn, đến cùng lúc, không thể xong sớm hơn yêu cầu dài nhất của nó.** Gộp lô liên tục chỉ lấp được chỗ trống khi **còn người đang chờ** để lấp vào. Với 16 chỗ và 64 yêu cầu thì luôn có người chờ, nên lợi ích hiện ra đầy đủ; với 256 chỗ thì hàng chờ cạn ngay từ đầu.

Người dùng thật không đến thành một khối rồi thôi - họ đến liên tục, như bài 07 sẽ giả lập. Ở tải liên tục gần mức sử dụng cao, yêu cầu mới vẫn tiếp tục tới, nên chỗ vừa trống thường có yêu cầu đang chờ để lấp; còn ở tải thấp thì chỗ trống cứ trống, và GPU rảnh là chuyện bình thường. Nhưng phép đo này nhắc một điều quan trọng về cách **đọc benchmark**: một khối yêu cầu đưa vào cùng lúc đo **thời gian xong cả khối**, và thời gian ấy bị chặn dưới bởi yêu cầu dài nhất, bất kể bộ máy giỏi đến đâu.

## 6. KV cache dạng trang

Bài này **không đo được riêng** tác động của bộ nhớ dạng trang: vLLM không có công tắc để tắt nó, nên không có ô đối chứng. Phần này là **số học** từ con số vLLM tự báo - nguồn thứ ba mà bài 02 đã dặn đọc riêng.

Với Qwen2.5-1.5B và mức dùng bộ nhớ 0,8, vLLM báo có chỗ cho **219.328 token** KV cache, và tự tính "tối đa **107,09** yêu cầu đồng thời nếu mỗi yêu cầu dùng hết 2.048 token".

Con số 107 ấy chính là sức chứa của cách **dành trước theo độ dài tối đa**: nếu mỗi yêu cầu được giữ chỗ cho 2.048 token bất kể dùng bao nhiêu, thì 219.328 ÷ 2.048 ≈ 107 yêu cầu, không hơn. Còn ở tải trong bài, một yêu cầu dài trung bình $128 + 184 = 312$ token. Nếu bộ nhớ chỉ cấp đúng phần đã dùng - theo từng khối nhỏ - thì cùng chỗ ấy chứa được khoảng 219.328 ÷ 312 ≈ **700** yêu cầu, tức gấp khoảng **6,5 lần**.

Đó là một **phép đối chiếu lý thuyết với cách dành trước theo độ dài tối đa**, tính trên một tải cụ thể - không phải lợi ích của bộ nhớ dạng trang nói chung. Một bộ cấp phát không chia trang vẫn có thể cấp động theo độ dài thật, và khi ấy khoảng cách nhỏ hơn nhiều. Cái bộ nhớ dạng trang giải được một cách gọn là **cấp theo khối khi cần mà không đòi một vùng liền mạch**, nên tránh được phân mảnh khi các yêu cầu dài ngắn khác nhau vào ra liên tục. Mỗi chỗ KV cache bị giữ mà không dùng là một người dùng không được phục vụ - nhưng bài này không đo được vLLM thu hồi được bao nhiêu phần trong số ấy.

Công thức ở bài 02 cũng kiểm được con số tự báo: 219.328 token × 28 KiB/token ≈ **5,86 GiB**, khớp với phần còn lại sau khi trừ trọng số 2,89 GiB và các khoản phụ khỏi ngân sách 0,8 × 11,63 GiB.

> 🔧 **Thử ngay:** một bài blog viết "chuyển sang vLLM giúp chúng tôi tăng thông lượng 3 lần nhờ continuous batching". Bạn đọc câu ấy thế nào?
> Hỏi hai câu. **Một:** họ có chạy tải mà mọi yêu cầu cùng độ dài không? Nếu không, thì trong con số 3 lần có cả phần đến từ nhân tính toán, đồ thị tính toán dựng sẵn, và việc không quay lại Python - ở bài này, khi chưa có chỗ trống để tái dùng, khoảng cách ấy đã là 1,5 lần. **Hai:** họ có giữ nguyên số chỗ không? Tăng số chỗ tự nó đã nhân thông lượng lên gần 3 lần trên tải cùng độ dài. Con số của họ có thể đúng; câu quy nó cho **một** ý tưởng thì gần như chắc là sai.

## Tóm tắt bài học

- **Gộp lô liên tục** giải vấn đề thời gian: bộ lập lịch quyết định lại thành phần lô **sau mỗi bước sinh**, rút yêu cầu xong ra, nhét yêu cầu chờ vào. **KV cache dạng trang** giải vấn đề bộ nhớ: cấp KV cache theo khối nhỏ khi cần, thay vì dành trước cho độ dài tối đa. Hai ý tưởng bổ trợ nhau nhưng là hai thứ riêng.
- So vLLM với lô tĩnh tự viết trên một tải duy nhất **không** cô lập được tác dụng của lập lịch, vì vLLM còn khác ở nhân tính toán, đồ thị dựng sẵn, và việc không quay lại Python.
- Thiết kế bốn ô: {lô tĩnh, vLLM cùng 16 chỗ} × {cùng độ dài, lẫn lộn}, cùng 11.776 token cần sinh. Tải cùng độ dài - khi chưa có chỗ trống để tái dùng: vLLM nhanh hơn **1,47 lần**. Tải lẫn lộn: **2,98 lần**. Phần lợi thêm gắn với độ dài lẫn lộn: **khoảng 2 lần** - phù hợp với gộp lô liên tục, không phải phép tách nhân quả chính xác.
- Lô tĩnh mất **64%** thông lượng khi tải lẫn lộn, khớp với 64,1% ô chạy không tính trước; vLLM cùng 16 chỗ chỉ mất 27%.
- Độ trễ từng yêu cầu: ở lô tĩnh của bài - trả kết quả cả lô ở cuối - **mọi** yêu cầu, kể cả yêu cầu 32 token, chờ **27,8 s**. Ở vLLM, độ trễ tỉ lệ với độ dài của chính mỗi yêu cầu: yêu cầu 32 token xong sau **2,26 s** (16 chỗ) và **0,83 s** (256 chỗ).
- Một khối yêu cầu hữu hạn đến cùng lúc **không thể xong sớm hơn yêu cầu dài nhất** - lý do 256 chỗ "lỗ" 53% trên tải lẫn lộn. Gộp lô liên tục chỉ lấp chỗ trống khi còn người chờ.
- Bộ nhớ dạng trang không đo riêng được (không có công tắc tắt). Số học từ sức chứa vLLM tự báo: dành trước theo 2.048 token chỉ chứa **107** yêu cầu; cấp đúng phần dùng ở tải này chứa khoảng **700** - gấp khoảng 6,5 lần, nhưng đó là **đối chiếu với cách dành trước theo độ dài tối đa**, không phải lợi ích của bộ nhớ dạng trang nói chung; một bộ cấp phát động không chia trang cũng thu hẹp được khoảng ấy.
- Hai bảng trong bài là hai lần chạy; lô tĩnh lệch 17% giữa hai lần vì nhạy với CPU bận. Không trộn số giữa hai lần chạy.

## Câu hỏi tự kiểm tra

1. Phân biệt gộp lô liên tục với KV cache dạng trang: mỗi thứ giải vấn đề gì, và một hệ có thể có cái này mà thiếu cái kia không?
2. Vì sao so vLLM với lô tĩnh tự viết trên tải lẫn lộn rồi quy toàn bộ chênh lệch cho gộp lô liên tục là sai?
3. Giải thích logic của thiết kế bốn ô. Tải **cùng độ dài** loại bỏ lợi ích nào của gộp lô liên tục? Vì sao nó vẫn **không** cô lập sạch được toàn bộ phần ngoài lập lịch?
4. Phép chia 2,98 ÷ 1,47 cho một **ước lượng**, không phải một phép tách chính xác. Vì sao?
5. Ở lô tĩnh, yêu cầu 32 token và yêu cầu 512 token có cùng E2E. Giải thích bằng cơ chế trả kết quả của lô tĩnh.
6. vLLM với 256 chỗ mất nhiều thông lượng hơn 16 chỗ khi tải lẫn lộn. Giải thích, và nói điều gì sẽ thay đổi nếu yêu cầu đến liên tục thay vì đến thành một khối.
7. Một mô hình có KV cache chứa được 100.000 token, độ dài tối đa 4.096, độ dài trung bình thực tế 500. Tính sức chứa khi dành trước theo độ dài tối đa và khi cấp đúng phần dùng. Vì sao con số thứ hai là trần chứ không phải con số đạt được?
8. Bạn chạy cùng một benchmark hai lần và thấy mốc tĩnh lệch 17% còn vLLM chỉ lệch 3%. Điều đó nói gì về cách bạn nên trình bày kết quả?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 05 của chương này** | Lô tĩnh, thằng chậm nhất, và vì sao batch lớn làm thông lượng tăng |
| **Bài 02 của chương này** | Công thức KV cache, và vì sao con số engine tự báo phải đọc riêng |

**Đi sâu hơn:**

- [Yu et al. - Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) - bài báo đề xuất lập lịch theo từng bước sinh.
- [Kwon et al. - Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180) - KV cache dạng trang, và phép đo lãng phí của cách dành trước mà mục 6 chỉ tính được bằng số học.
- [Agrawal et al. - Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve (OSDI 2024)](https://arxiv.org/abs/2403.02310) - chia nhỏ pha nạp prompt, một trong những thứ "ngoài lập lịch" mà mục 2 nhắc tới.

**Mã nguồn của bài:** [`code/b06_tinh.py`](../../code/ai-systems/b06_tinh.py) và [`code/b06_vllm.py`](../../code/ai-systems/b06_vllm.py) - bốn ô thông lượng; [`code/b06_tre.py`](../../code/ai-systems/b06_tre.py) - độ trễ từng yêu cầu theo độ dài.

> **Bài tiếp theo:** [Đo tải phục vụ cho đúng](../measuring-serving-load-correctly-vi/) - mọi phép đo tới đây đưa cả khối yêu cầu vào cùng lúc - và mục 5 vừa cho thấy cách ấy có giới hạn riêng của nó. Bài sau cho người dùng giả đến rải rác như người thật, và cho thấy cách giả lập phổ biến nhất lại báo độ trễ tốt hơn thực tế gần tám lần.

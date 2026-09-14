---
title: "Mô hình ngôn ngữ thực chất làm gì"
description: "Mô hình ngôn ngữ được huấn luyện để làm đúng một việc rất nhỏ, vì sao lặp lại việc ấy sinh ra văn bản, và một transformer tự huấn luyện trên chính loạt bài này đi từ ký tự ngẫu nhiên tới câu tiếng Việt."
domain: "Generative AI"
section: "Learn"
language: "vi"
translationKey: "what-a-language-model-really-does"
order: 1
pubDate: 2026-08-31
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết một mô hình ngôn ngữ được huấn luyện để làm đúng **một** việc rất nhỏ, hiểu vì sao lặp lại việc nhỏ ấy lại sinh ra được văn bản, và thấy tận mắt một mô hình đi từ ký tự ngẫu nhiên tới câu tiếng Việt - bằng một transformer chúng ta tự huấn luyện trên chính 51 bài đã viết của loạt bài này.

## 1. Một việc duy nhất

Chương Deep Learning · Bài 10 dừng ngay trước attention, sau khi cho thấy mạng hồi quy không xử lý nổi chuỗi dài. Chương Computer Vision · Bài 10 mở phần attention dành riêng cho ảnh rồi để lại phần ngôn ngữ. Chương này bắt đầu từ chỗ ấy.

Nhưng trước khi đụng tới attention, cần trả lời một câu hỏi cơ bản hơn: **mô hình ngôn ngữ được huấn luyện để làm gì?**

Câu trả lời nhỏ đến mức gây thất vọng: **đoán ký hiệu tiếp theo.**

(Chính xác hơn: đây là họ mô hình **tự hồi quy nhân quả** - loại mà cả chương này bàn tới, và là loại đứng sau mọi trợ lý bạn dùng hàng ngày. Có những họ mô hình ngôn ngữ khác huấn luyện theo cách khác, chẳng hạn đoán những token bị che ở giữa câu; chúng không nằm trong phạm vi chương này.)

Cho một chuỗi ký hiệu đã có, mô hình trả về một **phân bố xác suất** trên toàn bộ từ vựng cho ký hiệu kế tiếp. Chỉ vậy. Không có mô-đun "hiểu", không có kho sự thật, không có bước lập kế hoạch. Toàn bộ những gì trông như hiểu biết đều phải rơi ra từ việc làm tốt phép đoán ấy.

Và sinh văn bản chính là **lặp lại phép đoán đó**: đoán một ký hiệu, nối vào chuỗi, đoán tiếp, nối vào, cứ thế.

```mermaid
flowchart LR
    A["Chuoi da co"] --> B["Mo hinh"]
    B --> C["Phan bo xac suat<br/>tren toan bo tu vung"]
    C --> D["Chon mot ky hieu"]
    D --> E["Noi vao chuoi"]
    E --> A
```

Bài này chứng minh điều đó bằng cách **tự dựng một mô hình như vậy từ số 0** - không dùng mô hình có sẵn của ai - rồi xem nó học được gì.

## 2. Ký tự thay vì từ

Mô hình thật chia văn bản thành **token**, và cách chia ấy đủ phức tạp để chiếm trọn bài 02. Nên bài này dùng cách chia đơn giản nhất có thể: **mỗi ký tự là một ký hiệu.**

Cách này không hiệu quả - mô hình phải học cả chính tả chứ không được cho sẵn từ vựng - nhưng nó cho phép ta bỏ qua tokenizer và nhìn thẳng vào cơ chế.

Dữ liệu là chính **51 bài đã viết** của bốn chương trước: Foundations, Machine Learning, Deep Learning, Computer Vision.

| | |
|---|---|
| Corpus | **886.007 ký tự** |
| Từ vựng | **312 ký tự khác nhau** |
| Chia | 90% train / 10% validation |
| Mô hình | 4 tầng, 6 đầu, 192 chiều, ngữ cảnh 128 ký tự |
| Tham số | **1.924.536** |

Chọn chính bộ sách làm dữ liệu có lý do: bạn biết rõ mô hình đã đọc gì, nên đánh giá được đầu ra thay vì đoán. Con số 312 ký tự cũng đáng chú ý - tiếng Việt có rất nhiều tổ hợp dấu, cộng thêm ký hiệu toán và emoji trong các bài.

## 3. Thước đo: mô hình còn bất ngờ bao nhiêu

Loss của mô hình ngôn ngữ là cross-entropy, thứ chương Machine Learning · Bài 03 đã dựng. Nhưng ở đây có một cách đọc gọn hơn: chia cho $\ln 2$ để ra **bit trên mỗi ký tự**.

$$\text{bit/ký tự} = \frac{\text{cross-entropy}}{\ln 2}$$

Con số này trả lời: *trung bình cần bao nhiêu bit để mã hóa ký tự tiếp theo, nếu dùng chính dự đoán của mô hình làm bảng mã.* Nó có hai mốc rất dễ hiểu:

- **Đoán đều trên 312 ký tự** cần $\log_2 312 = 8{,}29$ bit
- **Biết chắc chắn** cần 0 bit

Nên bit/ký tự đo trực tiếp mức mô hình **bớt bất ngờ** so với đoán mò.

## 4. Bốn nghìn bước, năm lần chụp ảnh

Huấn luyện 4.000 bước, cứ tới vài mốc lại dừng để đo và cho mô hình viết tiếp câu *"Mô hình "*:

| Bước | Loss | bit/ký tự | Thời gian |
|---|---|---|---|
| 0 | 5,8921 | **8,501** | 3 s |
| 100 | 3,9686 | 5,725 | 34 s |
| 500 | 2,6198 | 3,780 | 146 s |
| 1.500 | 1,6644 | 2,401 | 422 s |
| 4.000 | **1,5451** | **2,229** | 1.364 s |

Và đây là thứ mô hình viết ra ở từng mốc.

**Bước 0** - trọng số ngẫu nhiên:

```
Mô hình ₙ(-ệỚöÀ¹┴╱Ự³(l─M╭ỗ-ỉ√ρỖBởểỂ⁻ẺƠö≠ậ'±₇λưq┐jMả&V⁰ỡŨỂ✅a?✗ₜ…ố✓🔧ε📄Ừ◎⁸ỢUy█
```

Chú ý con số **8,501 bit/ký tự** - *cao hơn* mức đoán đều 8,29. Mô hình khởi tạo ngẫu nhiên còn tệ hơn đoán mò, vì nó đang tự tin sai chỗ. Đầu ra là các ký hiệu hiếm bốc đều tay: emoji, ký tự toán, ký tự vẽ khung.

**Bước 100** - đã học được thứ tần suất:

```
Mô hình ch λ  t nỏỳth  | h n‖ Àệ   ₅h n tn Ỉớ 2 jơ2o  g  ti ö] bh x c    t nh noab M u
```

Dấu cách xuất hiện đúng mật độ, chữ cái thường lấn át ký hiệu hiếm. Mô hình chưa biết từ nào, nhưng đã biết **ký tự nào hay gặp**.

**Bước 500** - bắt đầu ra dáng tiếng Việt:

```
Mô hình hì chim vớcầdo m n miết khôn thỉ ca tru chìn trêng cọn đá 1. g chông ta kkhìn
```

Đây là mốc thú vị nhất. "chim", "khôn", "chông ta", "trêng" - chúng **trông như tiếng Việt** dù phần lớn không phải từ thật. Mô hình đã học được quy tắc *tổ hợp* của tiếng Việt: phụ âm đầu nào đi với nguyên âm nào, dấu đặt ở đâu.

**Bước 1.500** - từ thật, ngữ pháp hỏng:

```
Mô hình tho hơn để cách cột điển thể bắt bệnh thợ lệ trường được khi nào, và mà tầng vào
```

Gần như mọi từ đều là từ tiếng Việt có thật. Nhưng ghép lại thì vô nghĩa.

**Bước 4.000** - và đây là chỗ bất ngờ:

```
Mô hình học.

---

## Tóm tắt bài học

- PCA trên dữ liệu mới PCA thì hàm kích hoạt. Con số thêm cứ tham số nhất chương Machine Learning
```

Mô hình đã học **cấu trúc markdown của chính bộ sách**: nó biết sau một đoạn thì xuống dòng, gạch ngang phân cách, rồi tiêu đề `## Tóm tắt bài học`, rồi danh sách gạch đầu dòng. Nó cũng dùng đúng từ chuyên môn - PCA, hàm kích hoạt, tham số, "chương Machine Learning".

Nhưng **nội dung vẫn vô nghĩa**. "PCA trên dữ liệu mới PCA thì hàm kích hoạt" là một câu đúng ngữ pháp, đúng từ vựng, đúng văn phong, và sai hoàn toàn.

Đó là bài học chính của mục này: **hình thức học được trước nội dung.** Một mô hình 1,9 triệu tham số đọc 886 nghìn ký tự đủ để bắt chước *dáng vẻ* của tài liệu kỹ thuật tiếng Việt, nhưng không đủ để nói đúng. Ghi nhớ cảm giác này - bài 09 sẽ gặp lại nó ở mô hình lớn hơn nhiều, dưới cái tên **ảo giác**.

> ⚠️ **Lưu ý nhỏ:** đừng đọc bảng trên thành "cứ train lâu là tốt lên đều". Từ bước 1.500 tới 4.000 - thêm 2.500 bước, gấp ba thời gian - chỉ mua được 0,17 bit/ký tự, trong khi 500 bước đầu mua được 4,7 bit. Đường cong lợi ích giảm rất nhanh, đúng như mọi thứ chương Deep Learning · Bài 06 đã mô tả. Và đây là **một lần chạy minh họa** với một seed, nên các con số cụ thể sẽ khác ở lần chạy khác. Thứ tự các giai đoạn thì khớp với thứ người ta hay mô tả ở mô hình ngôn ngữ mức ký tự - xem bài của Karpathy trong phần Đọc thêm - nhưng một lần chạy không đủ để gọi nó là quy luật.

## 5. Đo chất lượng bằng một câu hỏi trả lời được

"Nhìn thấy hay hơn" không phải thước đo. Có một câu hỏi rẻ mà khách quan: **trong những "từ" mô hình viết ra, bao nhiêu phần là từ có thật trong corpus?**

Cho mô hình ở bước 4.000 viết một đoạn 1.200 ký tự rồi tách từ:

| | |
|---|---|
| Số "từ" sinh ra | 224 |
| Có trong corpus | **221** |
| Tỉ lệ | **98,66%** |

Gần như mọi từ đều là từ thật. Với một mô hình làm việc ở mức **ký tự** - nó phải tự ghép từng chữ cái một, không hề được cho sẵn danh sách từ - thì đó là kết quả đáng kể.

Nhưng thước đo này cũng cho thấy giới hạn của chính nó: 98,66% từ thật mà câu vẫn vô nghĩa. **Chính tả không phải ngữ nghĩa.** Đây là lý do đánh giá mô hình ngôn ngữ khó hơn nhiều so với đánh giá mô hình phân loại của bốn chương trước - ở đó có nhãn đúng để so, ở đây thì không.

## 6. Nhìn thẳng vào phân bố

Đến đây có thể mở nắp ra xem cái mà mục 1 gọi là "phân bố xác suất trên từ vựng" trông ra sao. Cho mô hình ba đoạn dở dang, xem nó đặt cược thế nào cho ký tự kế tiếp:

| Chuỗi đã có | Ký tự kế tiếp, năm ứng viên cao nhất |
|---|---|
| `Mô hình tuyến tí` | **`n` 0,9989** · `c` 0,0007 · `␣` 0,0001 · `ợ` 0,0001 |
| `gradient desc` | **`e` 0,9790** · `o` 0,0077 · `r` 0,0032 · `l` 0,0024 |
| `Học sâu là m` | **`ộ` 0,4663** · `ô` 0,1453 · `ạ` 0,0490 · `a` 0,0456 · `ấ` 0,0358 |

Ba dòng này nói ba chuyện khác nhau, và cả ba đều quan trọng.

**Hai dòng đầu gần như chắc chắn.** Sau "tuyến tí", ứng viên gần như chắc chắn là "tuyến tính"; sau "gradient desc" thì gần như chắc chắn là "descent" - bảng vẫn chừa cho các ký tự khác một phần xác suất rất nhỏ. Mô hình đặt 99,89% và 97,90% vào một ký tự duy nhất. Ở những chỗ như thế **gần như không có gì để chọn**: lấy ký tự cao điểm nhất hay bốc ngẫu nhiên theo phân bố thì hầu hết lượt đều ra cùng một ký tự - với 0,9989 thì trung bình chừng một lượt bốc trong ngót nghìn lượt mới rơi vào ứng viên khác. Khác biệt giữa các cách sinh văn bản nằm ở chỗ khác.

**Dòng thứ ba thì thật sự phân vân.** Sau "Học sâu là m" có thể là "một", "mô hình", "mạnh", "màn"… và mô hình chia xác suất ra: 0,4663 cho `ộ`, 0,1453 cho `ô`, rồi một cái đuôi dài. Đây mới là chỗ cách chọn ký tự quyết định văn bản đi đâu.

Và đó chính là chỗ bài 05 bắt đầu: khi phân bố tù mù như dòng thứ ba, **lấy ký tự có xác suất cao nhất** hay **bốc ngẫu nhiên theo phân bố** cho ra hai loại văn bản rất khác nhau.

> 🔧 **Thử ngay:** mô hình đặt 0,9989 vào ký tự `n` sau "tuyến tí". Con số ấy có nghĩa là mô hình *hiểu* cụm từ "tuyến tính" không?
> Không. Con số ấy là **niềm tin của mô hình**, không phải một tần suất đếm được trong corpus - hai thứ chỉ xấp xỉ trùng nhau ở những chỗ mô hình học tốt. Điều nó phản ánh là: từ 886 nghìn ký tự corpus, mô hình đã rút ra một quy luật thống kê rất mạnh - sau "tuyến tí" thì gần như luôn là `n` - và dồn gần hết khối lượng xác suất vào đúng ký tự đó. Quy luật ấy đủ để đoán đúng - nhưng nó không hàm ý mô hình biết "tuyến tính" nghĩa là gì. Bằng chứng nằm ngay ở mục 4: cùng mô hình ấy viết ra câu "PCA trên dữ liệu mới PCA thì hàm kích hoạt" - đúng chính tả từng chữ, đúng văn phong, vô nghĩa hoàn toàn. Phân biệt *đoán đúng ký tự kế tiếp* với *hiểu* là chuyện sẽ quay lại suốt chương này, và nặng nhất ở bài 09.

## Tóm tắt bài học

- Mô hình ngôn ngữ **tự hồi quy nhân quả** - loại cả chương này bàn tới - được huấn luyện làm đúng **một** việc: trả về **phân bố xác suất** cho ký hiệu kế tiếp. Sinh văn bản là lặp lại việc ấy.
- Bài này dựng một transformer **1.924.536 tham số** từ số 0, huấn luyện trên **886.007 ký tự** của 51 bài đã viết, với từ vựng **312 ký tự**.
- **bit/ký tự** = cross-entropy chia $\ln 2$, đo mức mô hình bớt bất ngờ: đoán đều trên 312 ký tự cần **8,29 bit**, mô hình đạt **2,229**.
- Lúc khởi tạo ngẫu nhiên, mô hình đạt **8,501 bit/ký tự** - *tệ hơn* đoán đều, vì nó tự tin sai chỗ.
- Bốn giai đoạn học rất rõ **trong lần chạy này**: tần suất ký tự (bước 100) → quy tắc tổ hợp tiếng Việt (500) → từ có thật (1.500) → **cấu trúc markdown của chính bộ sách** (4.000).
- **Hình thức học được trước nội dung**: ở bước 4.000 mô hình viết đúng tiêu đề `## Tóm tắt bài học` và dùng đúng từ chuyên môn, nhưng câu vẫn vô nghĩa.
- **98,66%** số "từ" sinh ra là từ có thật trong corpus - nhưng chính tả không phải ngữ nghĩa.
- Phân bố token kế tiếp **sắc hay tù tùy ngữ cảnh**: 0,9989 sau "tuyến tí", nhưng chỉ 0,4663 sau "Học sâu là m". Chỗ tù mù mới là chỗ cách sinh văn bản có ảnh hưởng.
- Lợi ích giảm rất nhanh theo số bước: 500 bước đầu mua được 4,7 bit, 2.500 bước cuối chỉ mua thêm 0,17 bit.

## Câu hỏi tự kiểm tra

1. Mô hình ngôn ngữ được huấn luyện để làm gì? Nêu chính xác đầu vào và đầu ra của một lần gọi mô hình.
2. Vì sao mô hình lúc khởi tạo ngẫu nhiên lại đạt 8,501 bit/ký tự, cao hơn mức đoán đều 8,29?
3. Từ vựng có 312 ký tự. Nếu từ vựng là 1.000 ký tự thì mốc "đoán đều" bằng bao nhiêu bit?
4. Ở bước 500 mô hình viết "chông ta", "trêng" - không phải từ thật nhưng trông rất Việt. Nó đã học được gì ở giai đoạn ấy?
5. Ở bước 4.000 mô hình viết đúng cấu trúc markdown nhưng nội dung vô nghĩa. Giải thích vì sao hình thức lại học được trước.
6. 98,66% từ sinh ra là từ thật. Vì sao con số cao ấy vẫn không nói lên rằng mô hình viết có nghĩa?
7. Phân bố sau "tuyến tí" là 0,9989 còn sau "Học sâu là m" chỉ 0,4663. Điều đó ảnh hưởng thế nào tới việc chọn cách sinh văn bản?
8. Mô hình đặt 0,9989 vào một ký tự. Đưa hai cách giải thích khác nhau cho con số ấy, và nêu bằng chứng nào trong bài phân biệt được chúng.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Deep Learning · Bài 10** | Dữ liệu chuỗi, RNN và chỗ nó hụt hơi - bài này tiếp ngay sau đó |
| **Chương Deep Learning · Bài 06** | Đọc đường cong huấn luyện; lợi ích giảm dần theo số bước |
| **Chương Machine Learning · Bài 03** | Cross-entropy - thước đo được đổi đơn vị thành bit ở mục 3 |
| **Dive into Deep Learning (d2l.ai)** | Chương 9.3 *Language Models* - định nghĩa hình thức và cách đo perplexity |

**Nguồn online bổ sung (miễn phí):**

- [Karpathy - The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - bài kinh điển về mô hình ngôn ngữ mức ký tự, với đúng kiểu tiến trình học ở mục 4.
- [Karpathy - Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) - dựng lại từng dòng một mô hình như trong bài này.
- [Shannon - Prediction and Entropy of Printed English (1951)](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) - nguồn gốc của cách đo bit trên mỗi ký tự, và ước lượng cho tiếng Anh.
- [nanoGPT](https://github.com/karpathy/nanoGPT) - cài đặt gọn của cùng ý tưởng, chạy được trên máy cá nhân.

> **Bài tiếp theo:** [Tokenizer và cái giá của tiếng Việt](../tokenizers-and-the-cost-of-vietnamese-vi/) - bài này cố tình dùng cách chia đơn giản nhất - mỗi ký tự một ký hiệu - để nhìn thẳng vào cơ chế. Mô hình thật thì không làm vậy, và cách chúng chia văn bản có một hệ quả rất cụ thể với người dùng tiếng Việt: cùng một nội dung, tiếng Việt tốn nhiều token hơn tiếng Anh. Bài sau đo xem nhiều hơn bao nhiêu, và vì sao.

---
title: "Regularization cho mạng sâu"
description: "Dropout, weight decay và data augmentation thực sự làm gì, mỗi cái đáng bao nhiêu trên một ca overfitting thật, và vì sao bật cả ba cùng lúc lại làm mô hình tệ đi."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "regularization-for-deep-networks"
order: 7
pubDate: 2026-08-22
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết dropout, weight decay và data augmentation thực sự làm gì, đo được mỗi cái đáng bao nhiêu trên một ca overfitting thật, và hiểu vì sao bật cả ba cùng lúc lại làm mô hình tệ đi.

## 1. Trả một món nợ của Foundations

Foundations · Bài 14 có một cái bảng liệt kê các cách chống overfitting, và nó nói thẳng là chưa dạy: *"Bạn chưa cần thành thạo các kỹ thuật dưới đây ngay - mục tiêu ở đây là biết chúng **tồn tại**"*. Riêng dropout còn ghi *"Chỉ cần nhớ tên ở giai đoạn này"*.

Giai đoạn ấy là bây giờ. Bài 06 vừa chẩn đoán ra một ca overfitting rất rõ, nên ta có sẵn bệnh nhân để thử thuốc:

- **1.000 mẫu** train (thay vì 60.000)
- Mạng **hai tầng ẩn 1.024 nơ-ron** - 2,9 triệu tham số cho 1.000 mẫu
- Adam, 60 epoch

Kết quả cơ sở: validation loss chạm đáy **0,6475** ở epoch 6 rồi leo lên 0,9973; accuracy tốt nhất **0,8093**; train loss cuối 0,0487 - mô hình gần như thuộc lòng.

Mọi con số dưới đây đều đo trên đúng bệnh nhân ấy, mỗi lần chỉ đổi một thứ.

## 2. Dropout: tắt bớt nơ-ron cho khỏi ai làm chủ

**Dropout** làm một việc nghe hơi phá hoại: mỗi lượt huấn luyện, tắt ngẫu nhiên một tỉ lệ $p$ số nơ-ron của tầng - đặt đầu ra của chúng bằng 0.

```
   lượt 1              lượt 2              lượt 3
   ●  ○  ●  ●          ●  ●  ○  ●          ○  ●  ●  ○
   ○  ●  ●  ○          ●  ○  ●  ●          ●  ●  ○  ●
   ● = đang hoạt động   ○ = bị tắt lượt này
```

Vì sao việc này giúp? Mạng không thể trông cậy vào một nơ-ron cụ thể nào, vì lượt sau nó có thể biến mất. Nó buộc phải học những đặc trưng **không quá bám vào nhau**: một nơ-ron không được phép chỉ có nghĩa khi vài nơ-ron bạn cùng tầng có mặt đầy đủ, vì lượt sau chúng có thể không có mặt. Đây là chỗ dropout hay bị nói quá - nó không làm các đặc trưng độc lập với nhau, chỉ phạt kiểu phụ thuộc mong manh vào một tổ hợp cố định. Kiểu như một đội bóng mà ai cũng có thể vắng mặt bất kỳ trận nào thì không thể xây lối chơi quanh đúng một người; các vị trí vẫn phối hợp, chỉ là không ai được là mắt xích duy nhất.

```python
nn.Sequential(nn.Flatten(),
              nn.Linear(784, 1024), nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.5),
              nn.Linear(1024, 10))
```

| Cấu hình | validation loss thấp nhất | accuracy tốt nhất | train loss cuối |
|---|---|---|---|
| Gốc | 0,6475 @ep6 | 0,8093 | 0,0487 |
| Dropout p = 0,2 | 0,6086 @ep12 | 0,8064 | 0,0325 |
| **Dropout p = 0,5** | **0,5943** @ep17 | **0,8110** | 0,0963 |

Ba điều đọc ra. Validation loss tốt lên rõ (0,6475 → 0,5943). Đáy của chữ U **dịch sang phải** - từ epoch 6 sang epoch 17, tức mô hình chịu được train lâu hơn trước khi hỏng. Còn accuracy thì gần như không đổi.

Chuyện accuracy đứng yên trong khi loss cải thiện chính là hiện tượng bài 06 đã gặp: dropout chữa cái **quá tự tin**, không chữa cái xếp hạng.

> ⚠️ **Lưu ý nhỏ:** dropout chỉ bật lúc train. Lúc dự đoán phải dùng đủ nơ-ron - và đó là lý do `model.eval()` ở bài 05 quan trọng đến vậy. PyTorch còn tự lo giúp một chi tiết: nếu lúc train tắt 50% nơ-ron thì tổng đầu vào của tầng sau bị hụt một nửa, nên `nn.Dropout` **nhân bù** các nơ-ron còn sống lên $1/(1-p)$ ngay lúc train. Nhờ vậy hai chế độ cho ra thang giá trị như nhau, và bạn không phải chỉnh gì.

## 3. Weight decay: phạt trọng số lớn

**Weight decay** có cùng mục tiêu với phạt L2 / Ridge của chương Machine Learning · Bài 04 - hạn chế trọng số lớn - nhưng cách nó được áp phụ thuộc vào optimizer. Dạng cơ bản là cộng vào hàm mất mát một khoản phạt theo tổng bình phương trọng số:

$$\text{loss tổng} = \text{cross-entropy} + \lambda \sum_j w_j^2$$

Mô hình bị ép chọn lời giải có trọng số nhỏ, tức là "mềm" hơn, ít phản ứng dữ dội với thay đổi nhỏ ở đầu vào. Trong PyTorch nó là một tham số của optimizer, không phải một tầng:

```python
opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

| Cấu hình | validation loss thấp nhất | accuracy tốt nhất | train loss cuối |
|---|---|---|---|
| Gốc | 0,6475 | 0,8093 | 0,0487 |
| weight decay = 1e-4 | 0,6411 | 0,8065 | 0,0051 |
| weight decay = 1e-2 | **0,5947** | 0,7950 | **0,2415** |

> ⚠️ **Cùng một tên, hai cách áp.** Trong thí nghiệm này, `Adam(..., weight_decay=...)` áp khoản phạt L2 theo kiểu **ghép vào gradient** trước bước cập nhật thích nghi của Adam - đó là mặc định của PyTorch. Vì Adam còn chuẩn hóa bước cập nhật bằng các moment của gradient, hiệu ứng thực tế của cách này **không tương đương với decoupled weight decay** (co trọng số tách rời). **AdamW** tách bước co trọng số ra khỏi gradient và không đưa khoản decay vào các moment - đó là điều Loshchilov & Hutter đề xuất, và ngày nay huấn luyện mạng lớn gần như luôn dùng AdamW. Khi đọc bảng phía trên, nhớ rằng các con số là của `Adam` + phạt L2 ghép, không phải của AdamW.

Mức nhẹ (1e-4) gần như không đổi gì. Mức mạnh (1e-2) cải thiện validation loss đáng kể nhưng **accuracy tụt** xuống 0,7950, và train loss vọt lên 0,2415 - dấu hiệu mô hình đã bị kìm hơi quá tay, bắt đầu ngả sang phía underfit của bài 06.

Đây đúng là sự đánh đổi mà chương Machine Learning · Bài 04 đã mô tả bằng ngôn ngữ bias-variance: phạt làm variance giảm nhưng bias tăng, và chỉ có lãi khi mô hình *có* variance để giảm.

## 4. Data augmentation: chế thêm dữ liệu từ dữ liệu đang có

Ba mục trên đều kìm mô hình. Cách này đi hướng ngược lại - **nới rộng dữ liệu**.

Một chiếc áo lật gương lại vẫn là chiếc áo đó. Dịch nó sang phải 2 pixel cũng vậy. Vậy thì từ 1.000 ảnh, mỗi epoch ta có thể sinh ra 1.000 ảnh hơi khác - mạng không bao giờ nhìn thấy đúng cùng một tấm hai lần.

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(),      # lật gương
    transforms.RandomCrop(28, padding=2),   # đệm rồi cắt ngẫu nhiên → dịch chuyển
    transforms.ToTensor(),
])
```

| Cấu hình | validation loss thấp nhất | accuracy tốt nhất | train loss cuối |
|---|---|---|---|
| Gốc | 0,6475 @ep6 | 0,8093 | 0,0487 |
| Augmentation | 0,6163 **@ep36** | 0,8022 | 0,3684 |

Đáng chú ý nhất là cột cuối cùng: train loss dừng ở 0,3684 thay vì 0,0487. Mô hình **không còn thuộc lòng được nữa**, đơn giản vì tập train đã đổi mặt mỗi epoch. Và đáy chữ U dịch tới tận epoch 36 - trước đó nó ở epoch 6.

> ⚠️ **Lưu ý nhỏ:** phép biến đổi phải **giữ nguyên nhãn**, và điều đó phụ thuộc bài toán. Lật gương một chiếc áo thì vẫn là áo - hợp lệ. Nhưng lật gương chữ số viết tay thì "2" thành một ký tự không tồn tại, và lật chữ "b" thành "d" - sai nhãn hoàn toàn. Trên FashionMNIST lật ngang là an toàn; trên MNIST thì không. Đây là chỗ bạn phải hiểu dữ liệu, không có mặc định đúng cho mọi bài toán.

## 5. Bật cả ba cùng lúc thì sao?

Câu hỏi tự nhiên, và câu trả lời không như mong đợi:

| Cấu hình | validation loss thấp nhất | accuracy tốt nhất | train loss cuối |
|---|---|---|---|
| Gốc | 0,6475 | 0,8093 | 0,0487 |
| Dropout 0,5 | 0,5943 | **0,8110** | 0,0963 |
| **Cả ba cùng lúc** | 0,5958 | **0,7812** | **0,5551** |

Bật cả ba cho accuracy **thấp nhất bảng** - kém cả bản không dùng gì. Nhìn train loss là hiểu: 0,5551, cao gấp mười một lần bản gốc. Mô hình giờ không còn overfit nữa, nhưng nó cũng **không học nổi dữ liệu đã thấy**. Ba liều thuốc chống overfitting cộng lại đã đẩy nó qua hẳn phía bên kia, thành underfit - đúng chứng bệnh D của bài 06.

Đây là bài học quan trọng nhất của cả bài, và nó chống lại một phản xạ rất phổ biến: **regularization không phải thứ càng nhiều càng tốt.** Mỗi công cụ là một lực kéo mô hình về phía đơn giản hơn; cộng nhiều lực lại thì kéo quá đà.

Và có một sự thật cần nói thẳng, nhìn lại toàn bộ bốn bảng: **không cái nào nâng accuracy lên đáng kể.** Tốt nhất là dropout 0,5 với 0,8110 so với 0,8093 của bản gốc - chênh 0,17 điểm, nằm gọn trong nhiễu. Cái thực sự cải thiện đều đặn là **validation loss** (0,6475 → 0,594), tức mô hình bớt tự tin sai, và **đáy chữ U dịch xa hơn**, tức bạn được train lâu hơn trước khi hỏng.

Còn thứ chữa được gốc bệnh thì không có trong bài này: bệnh nhân chỉ có 1.000 mẫu. Foundations · Bài 14 xếp "thêm dữ liệu" ở dòng đầu bảng là có lý do - nó là cách duy nhất tăng thông tin thật, còn ba kỹ thuật ở đây chỉ giúp dùng số thông tin ít ỏi ấy cẩn thận hơn.

> 🔧 **Thử ngay:** train loss 0,55 và validation loss 0,60, gần bằng nhau. Có phải mô hình đã hết overfit và bạn nên mừng không?
> Hết overfit thì đúng, nhưng chưa chắc đáng mừng. Hai đường sát nhau chỉ nói mô hình đối xử với dữ liệu mới và dữ liệu cũ như nhau - nó không nói mô hình *giỏi*. Ở đây train loss 0,55 là cao (bản gốc đạt 0,0487), nên nhiều khả năng bạn đang ở ca underfit. Quy trình của bài 06 vẫn đúng: **nhìn train loss trước**. Khoảng cách hẹp mà train loss cao là tin xấu, không phải tin tốt.

## Tóm tắt bài học

- **Dropout** tắt ngẫu nhiên một tỉ lệ nơ-ron mỗi lượt train, buộc mạng học đặc trưng dư thừa thay vì phụ thuộc vào vài nơ-ron; nó chỉ bật lúc train, nên `model.eval()` là bắt buộc khi đánh giá.
- **Weight decay** cùng mục tiêu hạn chế trọng số lớn với phạt L2 / Ridge của chương Machine Learning · Bài 04, và khai báo trong optimizer chứ không phải một tầng - nhưng cách áp phụ thuộc optimizer: `Adam(weight_decay=...)` ghép khoản phạt vào gradient, còn AdamW tách bước co trọng số ra khỏi các moment.
- **Data augmentation** đi hướng ngược lại: sinh biến thể hợp lệ của dữ liệu để mạng không thuộc lòng được - đo được qua train loss dừng ở 0,3684 thay vì 0,0487.
- Phép biến đổi phải giữ nguyên nhãn, và điều đó phụ thuộc bài toán: lật ngang hợp lệ với quần áo nhưng làm hỏng chữ số viết tay.
- Trên ca overfitting này, cả ba đều cải thiện **validation loss** (0,6475 → 0,594) và đẩy đáy chữ U đi xa (epoch 6 → 17 hoặc 36), nhưng **không cái nào nâng accuracy đáng kể**.
- Bật cả ba cùng lúc cho kết quả *tệ nhất* (accuracy 0,7812, train loss 0,5551) - quá liều thành underfit. Regularization không phải càng nhiều càng tốt.
- Cách chữa gốc cho ca này vẫn là thêm dữ liệu; ba kỹ thuật trên chỉ giúp dùng dữ liệu ít ỏi cẩn thận hơn.

## Câu hỏi tự kiểm tra

1. Dropout tắt ngẫu nhiên nơ-ron lúc train. Vì sao việc phá hoại ấy lại giúp mô hình tổng quát hóa tốt hơn?
2. Vì sao `nn.Dropout` phải nhân bù các nơ-ron còn sống lên $1/(1-p)$? Chuyện gì xảy ra nếu không có bước bù đó?
3. Weight decay trong bài này và Ridge của chương Machine Learning · Bài 04 khác nhau ở điểm nào? Trong PyTorch bạn khai báo nó ở đâu?
4. Bạn định dùng augmentation cho bài phân loại biển báo giao thông. Phép lật gương ngang có an toàn không? Còn xoay 15 độ?
5. Bật cả dropout, weight decay và augmentation cho accuracy thấp hơn bản không dùng gì. Chẩn đoán bằng chỉ số nào, và bạn tháo cái nào ra trước?
6. Ba kỹ thuật đều cải thiện validation loss nhưng không cải thiện accuracy. Nếu đầu ra mô hình được dùng để xếp hạng hồ sơ theo xác suất, bạn có coi chúng là có ích không? Vì sao?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Foundations · Bài 14** | Mục 5 *Hộp đồ nghề chống overfitting* - bảng bốn kỹ thuật mà bài này đo cụ thể từng cái |
| **Chương Machine Learning · Bài 04** | Ridge, Lasso, và đánh đổi bias-variance - nền của khoản phạt mà weight decay ở mục 3 dùng lại |
| **Dive into Deep Learning (d2l.ai)** | Chương 3.7 *Weight Decay*; chương 5.6 *Dropout* - cả cơ chế lẫn phần nhân bù $1/(1-p)$; chương 14.1 *Image Augmentation* |

**Nguồn online bổ sung (miễn phí):**

- [Srivastava và cộng sự - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR 2014)](https://jmlr.org/papers/v15/srivastava14a.html) - bài báo gốc, có phần giải thích trực giác "mạng không được phụ thuộc vào một nơ-ron".
- [`torchvision.transforms` - tài liệu](https://pytorch.org/vision/stable/transforms.html) - danh sách các phép augmentation có sẵn.
- [`torch.optim` - tham số `weight_decay`](https://pytorch.org/docs/stable/optim.html) - công thức cập nhật của `Adam` và của `AdamW` đặt cạnh nhau, thấy rõ khoản decay vào gradient ở bên nào.
- [Loshchilov & Hutter - Decoupled Weight Decay Regularization (ICLR 2019)](https://arxiv.org/abs/1711.05101) - vì sao phạt L2 ghép vào Adam không tương đương decoupled weight decay, và vì sao AdamW tách hai bước ấy ra.

> **Bài tiếp theo:** [Vì sao mạng sâu khó huấn luyện](../why-deep-networks-are-hard-to-train-vi/) - bài này chữa cho mạng học *quá kỹ*. Bài sau xử lý ca ngược lại và nghiêm trọng hơn - mạng 20 tầng không học được gì cả, accuracy đứng nguyên ở mức đoán bừa, và ba cách kéo nó dậy.

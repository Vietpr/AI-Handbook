---
title: "Xếp tầng: mạng tự học lấy đặc trưng"
description: "Một tầng ẩn thực chất tạo ra gì, chữ \"sâu\" mua được bao nhiêu và dừng ở đâu, vì sao hai tầng hẹp thường thắng một tầng rộng cùng số tham số."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "stacking-layers-networks-learn-their-own-features"
order: 3
pubDate: 2026-08-20
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu một tầng ẩn thực chất tạo ra cái gì, thấy bằng số liệu chữ "sâu" mua được bao nhiêu và dừng lại ở đâu, biết vì sao với ngân sách tham số tương đương, hai tầng hẹp thường học tốt hơn một tầng rộng, và đọc được câu chuyện đằng sau việc tầng sau học trên đặc trưng của tầng trước.

## 1. Một tầng ẩn là một bảng đặc trưng do máy tự viết

Bài 02 kết thúc ở một nơ-ron. Đặt 256 nơ-ron cạnh nhau, tất cả cùng nhận một đầu vào, ta được một **tầng (layer)**:

```
              ┌──▶ nơ-ron 1 ──▶ a₁
   784 pixel ─┼──▶ nơ-ron 2 ──▶ a₂     256 con số mới,
              ┼──▶    ⋮                mỗi con số là một
              └──▶ nơ-ron 256 ─▶ a₂₅₆  tổ hợp của cả 784 pixel
```

Nhìn cho kỹ đầu ra bên phải: 256 con số. Đó chính là **một bảng dữ liệu mới**, 256 cột, mô tả cùng một tấm ảnh. Và mỗi cột là một tổ hợp có trọng số của 784 pixel gốc - đúng dạng "feature tự tạo" mà chương Machine Learning · Bài 11 bắt bạn ngồi nghĩ ra bằng tay.

Khác biệt duy nhất, và là khác biệt của cả chương: **không ai viết ra 256 tổ hợp ấy**. Chúng bắt đầu từ những số ngẫu nhiên, rồi gradient descent chỉnh dần chúng sao cho tầng cuối phân loại tốt hơn. Người ta gọi đó là học biểu diễn, và đây là hình dạng cụ thể của nó.

Từ đó có một cách đọc mạng nơ-ron rất tiện:

> Tầng cuối làm đúng việc của hồi quy softmax ở bài 01. Mọi tầng trước nó chỉ có một nhiệm vụ: **biến dữ liệu thô thành thứ dễ phân loại tuyến tính hơn**.

Kiểm chứng được ngay. Lấy mạng đã train, cắt bỏ tầng cuối, dùng 256 con số ở tầng ẩn làm đầu vào cho một hồi quy softmax mới - nó sẽ đạt gần đúng accuracy của cả mạng. Tầng ẩn đã dọn sẵn bàn.

## 2. Sâu thêm được bao nhiêu, và dừng ở đâu

Câu hỏi tự nhiên: nếu một tầng ẩn dọn bàn tốt, thì hai tầng dọn tốt hơn chứ? Đo trên FashionMNIST, mỗi tầng ẩn 256 nơ-ron, 15 epoch, mọi thứ khác giữ nguyên:

| Số tầng ẩn | Tham số | Accuracy validation tốt nhất |
|---|---|---|
| 0 | 7.850 | 0,8357 |
| 1 | 203.530 | 0,8622 |
| 2 | 269.322 | 0,8666 |
| 3 | 335.114 | **0,8674** |
| 4 | 400.906 | 0,8595 |

Đọc bảng này theo ba nhịp.

**Nhịp một: tầng ẩn đầu tiên mua được nhiều nhất.** Từ 0 lên 1 tầng: +2,6 điểm. Đó là bước từ "ranh giới thẳng" sang "ranh giới cong" - thay đổi về chất.

**Nhịp hai: các tầng sau mua được ít dần.** Từ 1 lên 2: +0,4 điểm. Từ 2 lên 3: +0,1 điểm. Vẫn dương, nhưng đã sát mức nhiễu giữa các lần chạy.

**Nhịp ba, và đây là chỗ đáng nhớ: bốn tầng *tệ hơn* ba tầng.** 0,8595 so với 0,8674, dù nhiều tham số nhất bảng. Thêm capacity mà kết quả đi xuống - chuyện này Foundations · Bài 13 đã cảnh báo là có thể xảy ra, nhưng ở đây nguyên nhân không phải overfitting: train loss của mạng 4 tầng cũng không thấp hơn. Nó đơn giản là **khó huấn luyện hơn**. Tín hiệu học phải đi ngược qua nhiều tầng hơn để tới được tầng đầu, và trên đường đi nó yếu dần. Bài 04 đo cụ thể chuyện yếu dần ấy, bài 08 gọi tên và chữa nó.

> ⚠️ **Lưu ý nhỏ:** đừng rút ra "ba tầng là con số tốt nhất". Bảng này đúng cho *mạng kết nối đầy đủ, trên FashionMNIST, với cách khởi tạo và thuật toán tối ưu mặc định*. Các mạng sâu thật ngoài đời - ResNet ở bài 09 chẳng hạn - có hàng chục đến hàng trăm tầng và chạy tốt, nhưng chúng dùng những kỹ thuật mà bài 08 mới giới thiệu. Cái bảng này cho thấy: xếp tầng suông thì đụng trần rất sớm.

## 3. Rộng hay sâu, nếu chỉ được chọn một

Giả sử ngân sách của bạn là khoảng 200-400 nghìn tham số. Chia ngân sách đó thành ít tầng rộng hay nhiều tầng hẹp?

| Kiến trúc | Tham số | Accuracy validation |
|---|---|---|
| 1 tầng ẩn × 512 nơ-ron | 407.050 | 0,8588 |
| **2 tầng ẩn × 256 nơ-ron** | **269.322** | **0,8666** |
| 4 tầng ẩn × 158 nơ-ron | 200.986 | 0,8561 |

Hai tầng thắng, và thắng với **ít hơn 34% tham số** so với một tầng rộng. Nhưng đẩy tiếp lên bốn tầng hẹp thì lại tụt - cùng một hiện tượng của mục 2.

Trực giác đằng sau: mỗi tầng học một tổ hợp *của tầng trước*. Với một tầng, mọi đặc trưng đều phải dựng thẳng từ pixel thô. Với hai tầng, tầng thứ hai được xây trên những mảnh mà tầng đầu đã dựng - như xây nhà từ tường đúc sẵn thay vì từ từng viên gạch. Chiều sâu cho phép **tái sử dụng**: một mảnh mà tầng 1 học được có thể góp mặt trong nhiều đặc trưng khác nhau của tầng 2, nên cùng một ngân sách tham số diễn tả được nhiều thứ hơn.

> 🔧 **Thử ngay:** một mạng có 784 đầu vào, một tầng ẩn 256 nơ-ron và 10 đầu ra. Đếm số tham số.
> Tầng ẩn: mỗi nơ-ron có 784 trọng số cộng 1 hệ số chặn, nhân 256 nơ-ron = $785 \times 256 = 200.960$. Tầng ra: $257 \times 10 = 2.570$. Tổng **203.530** - khớp đúng dòng "1 tầng ẩn" ở bảng mục 2. Để ý phần lớn tham số nằm ở tầng đầu tiên, chỉ vì đầu vào có tới 784 chiều; bài 09 sẽ cho thấy một cách cắt giảm con số này rất mạnh.

## 4. Tầng sau nhìn thấy gì

Có một cách hình dung thứ các tầng học được, và nó đã thành hình ảnh kinh điển của deep learning trên dữ liệu ảnh:

```
   ảnh thô          tầng 1              tầng 2               tầng 3
  (pixel)      cạnh, mảng sáng     góc, đường cong,      bộ phận:
                                    họa tiết lặp         tay áo, gót giày
      │               │                   │                    │
      └───────────────┴───────────────────┴────────────────────┘
                    mỗi tầng dựng trên tầng trước
```

Đây là mô tả định tính, không phải một quy luật đo được ở mọi mạng. Nhưng nó đúng đủ để hữu ích, và nó giải thích vì sao chiều sâu có ý nghĩa: cái tai mèo không phải một tổ hợp tuyến tính của pixel, nhưng nó *là* một tổ hợp của các đường cong, mà đường cong lại là tổ hợp của các cạnh.

Một hệ quả thực tế rất lớn của cách nhìn này: nếu các tầng đầu học những thứ chung chung - cạnh, góc, họa tiết - thì chúng **ít gắn với bài toán cụ thể hơn** các tầng phía sau. Cạnh trong ảnh mèo cũng là cạnh trong ảnh áo sơ mi. Nghĩa là bạn có thể lấy các tầng đầu của một mạng đã học từ hàng triệu ảnh và dùng lại cho bài toán của mình chỉ có vài nghìn ảnh. Đó là toàn bộ ý tưởng của bài 11.

Đây là chuyện mức độ, không phải chuyện có hay không: càng đi sâu, đặc trưng càng dính chặt vào bài toán gốc, và mức dùng lại được giảm dần theo từng tầng. Vì thế trong bài 11, ta giữ nguyên phần đầu của mạng đã học sẵn nhưng thay hẳn phần cuối.

> ⚠️ **Lưu ý nhỏ:** hình ảnh "tầng 1 học cạnh, tầng 2 học góc" được rút ra chủ yếu từ **mạng tích chập** trên ảnh (bài 09), nơi ta trực tiếp nhìn được bộ lọc dưới dạng ảnh. Với mạng kết nối đầy đủ như trong bài này, các đặc trưng học được khó diễn giải hơn nhiều: mỗi nơ-ron tầng 1 nhìn *toàn bộ* 784 pixel cùng lúc, nên trọng số của nó vẽ ra thường trông như nhiễu chứ không thành hình cạnh rõ ràng. Đừng kỳ vọng nhìn vào trọng số của một `nn.Linear` là thấy được cái tai mèo.

## Tóm tắt bài học

- Một tầng ẩn 256 nơ-ron biến bảng 784 cột thành bảng 256 cột mới, mỗi cột là một tổ hợp có trọng số của đầu vào - chính là feature tự tạo của chương Machine Learning · Bài 11, chỉ khác ở chỗ máy tự tìm ra thay vì bạn nghĩ ra.
- Cách đọc tiện: tầng cuối làm việc của hồi quy softmax, mọi tầng trước nó lo biến dữ liệu thô thành thứ dễ phân loại tuyến tính hơn.
- Chiều sâu mua được ít dần: 0 → 1 tầng được +2,6 điểm; 1 → 2 được +0,4; 2 → 3 được +0,1.
- Bốn tầng *tệ hơn* ba tầng (0,8595 so với 0,8674) dù nhiều tham số nhất - không phải overfitting mà là khó huấn luyện, chủ đề của bài 04 và bài 08.
- Cùng ngân sách tham số, hai tầng 256 thắng một tầng 512 với ít hơn 34% tham số, vì tầng sau tái sử dụng được các mảnh tầng trước đã dựng.
- Nếu các tầng đầu học những thứ chung chung, chúng dùng lại được cho bài toán khác - nền tảng của transfer learning ở bài 11.

## Câu hỏi tự kiểm tra

1. Một tầng ẩn tạo ra cái gì, diễn đạt bằng ngôn ngữ "bảng dữ liệu, hàng và cột"? Nó khác feature engineering của chương trước ở đúng điểm nào?
2. Mạng 4 tầng ẩn có nhiều tham số hơn mạng 3 tầng nhưng accuracy thấp hơn. Nêu hai cách giải thích khác nhau, và bạn dùng số liệu nào để phân biệt chúng?
3. Đếm số tham số của mạng 784 → 512 → 10. So với mạng 784 → 256 → 256 → 10, cái nào nhiều tham số hơn?
4. Vì sao với ngân sách tham số tương đương, hai tầng hẹp thường học tốt hơn một tầng rộng trên dữ liệu này? Nêu một lý do khiến lập luận ấy không phải là quy luật luôn đúng.
5. Bạn cắt tầng cuối của một mạng đã train và dùng đầu ra tầng ẩn làm đầu vào cho một hồi quy logistic mới. Bạn kỳ vọng accuracy thế nào so với mạng gốc, và vì sao?
6. Có người bảo "muốn tốt hơn thì cứ thêm tầng". Dựa vào bảng ở mục 2, bạn phản biện thế nào?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 5.1 *Multilayer Perceptrons* - cách xếp tầng và ký hiệu; chương 5.2 *Implementation of MLPs*; chương 5.6 bàn về chiều sâu và độ khó huấn luyện |
| **An Introduction to Statistical Learning (ISLP)** | Chương 10.2 *Multilayer Neural Networks* - mạng nhiều tầng ẩn, và nhận xét về việc tầng sau dựng trên đặc trưng tầng trước |
| **Foundations · Bài 13** | Capacity: vì sao thêm tham số không đảm bảo tốt hơn |

**Nguồn online bổ sung (miễn phí):**

- [TensorFlow Playground](https://playground.tensorflow.org/) - thêm/bớt tầng và nơ-ron rồi xem trực tiếp ranh giới quyết định đổi hình; đây là cách nhanh nhất để cảm nhận mục 2.
- [Zeiler & Fergus - Visualizing and Understanding Convolutional Networks (ECCV 2014)](https://arxiv.org/abs/1311.2901) - nguồn của hình ảnh "tầng 1 học cạnh, tầng sau học bộ phận" ở mục 4, có hình thật cho từng tầng.
- [Distill - Feature Visualization](https://distill.pub/2017/feature-visualization/) - bài tương tác cho thấy từng nơ-ron trong mạng thị giác phản ứng với cái gì.
- [Nielsen - Neural Networks and Deep Learning, chương 5](http://neuralnetworksanddeeplearning.com/chap5.html) - sách miễn phí, chương này bàn đúng chuyện vì sao mạng sâu khó huấn luyện, nối thẳng sang bài 04 và 08.

> **Bài tiếp theo:** [Đồ thị tính toán: một lời gọi backward() lo hàng triệu tham số](../computational-graphs-and-backward-vi/) - bài này nói tầng sau học trên tầng trước, nhưng "học" ở đây nghĩa là ai chỉnh núm nào. Bài sau mở nắp cơ chế đó - và đo luôn cái yếu dần mà mục 2 vừa gặp.

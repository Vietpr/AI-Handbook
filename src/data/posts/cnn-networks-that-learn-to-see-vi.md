---
title: "CNN: mạng học cách nhìn"
description: "Hai ý tưởng làm nên mạng tích chập là nhìn cục bộ và dùng chung trọng số, tích chập và pooling làm gì, và vì sao mạng ít hơn mười lần tham số lại thắng mạng kết nối đầy đủ trên ảnh."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "cnn-networks-that-learn-to-see"
order: 9
pubDate: 2026-08-23
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu hai ý tưởng làm nên mạng tích chập - chỉ nhìn cục bộ và dùng chung trọng số - biết phép tích chập và pooling làm gì, và thấy bằng số vì sao một mạng ít hơn mười lần tham số lại thắng mạng kết nối đầy đủ trên ảnh.

## 1. Tám bài qua, ta vẫn đang vứt bỏ thông tin

Mọi mạng từ bài 01 tới giờ đều bắt đầu bằng đúng một dòng:

```python
nn.Flatten()      # (1, 28, 28) → (784,)
```

Dòng đó duỗi tấm ảnh thành một hàng 784 số. Vị trí tuyệt đối thì không mất: pixel ở hàng 4 cột 12 luôn rơi vào ô thứ 100, và mạng hoàn toàn có thể học rằng ô thứ 100 quan trọng. Cái mất là **quan hệ láng giềng**: với `nn.Linear`, ô thứ 100 và ô thứ 101 - vốn là hai pixel kề nhau - chỉ là hai trong 784 đầu vào ngang hàng, không gần nhau hơn ô thứ 100 và ô thứ 600 chút nào. Mạng có thể học lại quan hệ ấy từ dữ liệu, nhưng phải học từ đầu, trong khi nó vốn đã đúng sẵn với mọi tấm ảnh.

Có một phép thử làm rõ chuyện này. Xáo trộn 784 pixel theo một thứ tự ngẫu nhiên cố định - áp cùng phép xáo cho mọi ảnh - rồi train lại từ đầu. Mạng kết nối đầy đủ đạt kết quả *y hệt như cũ*, vì phép xáo chỉ là đổi tên các đầu vào, mà mọi cách đặt tên đều tương đương với nó. Với nó, ảnh chỉ là một túi số không có hình dạng. Làm cùng phép thử ấy với CNN thì kết quả sập, vì cái CNN dựa vào - pixel gần nhau thì liên quan - đã bị phép xáo phá mất.

Nhưng ảnh thì có hình dạng, và hai sự thật về nó rất đáng khai thác:

- **Pixel gần nhau thì liên quan với nhau.** Một cái cạnh là chuyện xảy ra giữa vài pixel kề nhau, không phải giữa pixel góc trên trái và pixel góc dưới phải.
- **Một mẫu hình có nghĩa ở mọi vị trí.** Cái cạnh dọc ở góc trái vẫn là cái cạnh dọc khi nó xuất hiện ở góc phải. Bộ dò cạnh học được cho vùng này dùng lại được cho vùng khác.

Mạng kết nối đầy đủ không biết cả hai điều đó, nên nó phải học lại chúng từ dữ liệu - tốn tham số và tốn ví dụ. **Mạng tích chập (convolutional neural network - CNN)** thì được xây dựng sẵn với hai sự thật ấy.

## 2. Chỉ nhìn cục bộ, và dùng chung trọng số

Trong tầng `nn.Linear`, mỗi nơ-ron nối tới **cả 784** pixel. Trong tầng tích chập, mỗi nơ-ron chỉ nhìn một ô vuông nhỏ - thường 3×3 - và **cùng một bộ 9 trọng số ấy được trượt qua khắp ảnh**.

```
   bộ lọc 3x3            trượt qua khắp ảnh
   ┌───┬───┬───┐      ┌───────────────────┐
   │ w₁│ w₂│ w₃│      │ ┌───┐             │
   ├───┼───┼───┤      │ │3x3│──────▶      │   cùng 9 trọng số đó
   │ w₄│ w₅│ w₆│  ──▶ │ └───┘             │   dùng ở MỌI vị trí
   ├───┼───┼───┤      │        ┌───┐      │
   │ w₇│ w₈│ w₉│      │        │3x3│───▶  │
   └───┴───┴───┘      └───────────────────┘
    9 trọng số + 1 bias                        kết quả: một "bản đồ đặc trưng"
```

Mỗi lần đặt bộ lọc lên một ô 3×3, ta nhân từng cặp rồi cộng lại - chính là tích vô hướng của Foundations · Bài 07 - và được một con số. Trượt hết ảnh thì được một bảng số mới gọi là **bản đồ đặc trưng (feature map)**: nó cho biết mẫu hình mà bộ lọc này tìm kiếm xuất hiện mạnh ở đâu trong ảnh.

Một tầng thường có nhiều bộ lọc, mỗi cái học một mẫu hình khác nhau - cái này bắt cạnh dọc, cái kia bắt cạnh ngang, cái nọ bắt một mảng sáng.

Hệ quả về số tham số thì gây choáng:

| | Tham số |
|---|---|
| `nn.Linear(784, 784)` | 615.440 |
| `nn.Conv2d(1, 16, 3×3)` - **16 bộ lọc** | **160** |

Và con số 160 ấy **không đổi theo kích thước ảnh**, vì bộ lọc chỉ có 3×3 dù ảnh to đến đâu:

| Kích thước ảnh | `Linear` tới 256 nơ-ron | `Conv2d(1, 16, 3×3)` |
|---|---|---|
| 28 × 28 | 200.960 | 160 |
| 64 × 64 | 1.048.832 | 160 |
| 224 × 224 | **12.845.312** | **160** |

Đây là lý do bài 01 nói "cách chữa cũ hết đường" với ảnh 224×224: chỉ riêng tầng đầu tiên của một mạng kết nối đầy đủ đã ngốn gần 13 triệu tham số, trong khi một tầng tích chập làm việc tương đương với 160.

## 3. Pooling: thu nhỏ lại, giữ lấy cái mạnh nhất

Sau vài tầng tích chập, người ta thường chèn một tầng **pooling** để thu nhỏ bản đồ đặc trưng. Loại quen thuộc nhất là **max pooling 2×2**: chia bản đồ thành các ô 2×2 rồi giữ lại giá trị lớn nhất mỗi ô.

```
   4  1 │ 0  2          max pooling 2x2
   2  3 │ 7  1     ──▶     4   7
   ─────┼─────              9   6
   9  5 │ 3  6
   1  0 │ 2  4
```

Hai tác dụng. Thứ nhất, kích thước giảm bốn lần nên các tầng sau rẻ hơn hẳn. Thứ hai, nó tạo ra một chút **dung sai vị trí**: nếu cái cạnh dịch sang phải một pixel mà vẫn nằm trong cùng ô 2×2 thì đầu ra không đổi. Chỉ "một chút" theo nghĩa đen - dung sai này là cục bộ, cỡ một pixel cho mỗi tầng pooling 2×2, chứ không phải sự chịu đựng dịch chuyển vài pixel như nhiều người tưởng. Mục 5 đo cụ thể chỗ đó.

Ráp lại thành một CNN nhỏ:

```python
cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28×28 → 14×14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14×14 → 7×7
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
```

Để ý `nn.Flatten()` vẫn còn - nhưng giờ nó đứng ở **cuối**, sau khi các tầng tích chập đã rút ảnh thành các đặc trưng có nghĩa. Đó là khác biệt then chốt so với tám bài trước, nơi ta duỗi ảnh ngay từ đầu.

## 4. CNN và MLP: so trên FashionMNIST

Cùng FashionMNIST, cùng 10 epoch, cùng learning rate:

| Mô hình | Tham số | Accuracy | Thời gian train |
|---|---|---|---|
| MLP 1 tầng ẩn 256 | 203.530 | 0,8488 | 55 giây |
| **CNN 2 tầng tích chập** | **20.490** | **0,8781** | 161 giây |

CNN dùng **ít hơn 10 lần tham số** mà accuracy **cao hơn 2,9 điểm**.

Hãy so con số này với bài 01, nơi thêm cả một tầng ẩn 256 nơ-ron - 26 lần tham số - chỉ mua được 2,8 điểm. Ở đây cùng mức lãi ấy đến từ việc *giảm* tham số đi mười lần, chỉ nhờ chọn kiến trúc biết rằng đầu vào là ảnh. Đó chính là điều bài 01 hứa: cú nhảy lớn đến từ kiến trúc hợp dữ liệu, không từ việc thêm tham số.

Đổi lại, CNN train **lâu gấp ba** dù ít tham số hơn. Ít tham số không có nghĩa ít phép tính: mỗi bộ lọc phải trượt qua mọi vị trí trong ảnh, nên số phép nhân rất lớn dù số trọng số rất nhỏ. Đây là lý do deep learning cho ảnh gắn liền với GPU - loại phần cứng làm hàng nghìn phép nhân song song.

## 5. Chuyện "bất biến dịch chuyển" - nói cho đúng

Nhiều tài liệu bảo CNN "bất biến với dịch chuyển". Hãy kiểm chứng: lấy cả hai mô hình đã train, dịch mọi ảnh validation sang ngang **3 pixel**, đo lại.

| Mô hình | Accuracy gốc | Sau khi dịch 3 pixel | Mất |
|---|---|---|---|
| MLP | 0,8550 | 0,3350 | **52,0 điểm** |
| CNN | 0,8810 | 0,4570 | **42,4 điểm** |

CNN chịu đựng tốt hơn - nhưng nó cũng sập. Mất 42 điểm không phải là "bất biến".

Lý do đáng hiểu rõ. Bản thân phép tích chập có tính **tương biến (equivariant)**: dịch ảnh thì bản đồ đặc trưng dịch theo, chứ không phải không đổi. Pooling cho một chút dung sai, nhưng chỉ trong phạm vi ô pooling - 2×2 thì tha thứ được một pixel, không phải ba. Và tầng `nn.Linear` ở cuối lại đọc bản đồ đặc trưng theo đúng vị trí, nên khi mọi thứ dịch đi ba pixel, nó nhận vào một bố cục hoàn toàn khác.

Nói đúng thì phải là: **CNN có thiên hướng phù hợp với ảnh, không phải bất biến với dịch chuyển.** Muốn thực sự chịu được dịch chuyển thì phải huấn luyện cho nó - bằng data augmentation của bài 07, chính là phép `RandomCrop` mà ta đã dùng ở đó.

> ⚠️ **Lưu ý nhỏ:** `padding=1` trong `nn.Conv2d(1, 16, 3, padding=1)` không phải chi tiết trang trí. Bộ lọc 3×3 trượt trên ảnh 28×28 mà không đệm thì chỉ đặt được vào 26×26 vị trí - ảnh teo đi 2 pixel mỗi tầng. Xếp mười tầng là mất 20 pixel, tức ảnh 28×28 biến mất. Đệm một vòng số 0 quanh ảnh giữ nguyên kích thước, nên bạn xếp bao nhiêu tầng cũng được. Quy tắc chung cho bộ lọc $k \times k$ lẻ: `padding = (k-1)/2` thì kích thước không đổi.

> 🔧 **Thử ngay:** vì sao dùng chung trọng số (một bộ lọc trượt khắp ảnh) lại giúp cả về số tham số *lẫn* về khả năng khái quát hóa?
> Về tham số thì rõ: một bộ 9 số thay cho một bộ riêng ở mỗi vị trí. Về khái quát hóa thì tinh tế hơn: nó **nhét sẵn vào mô hình một giả định đúng** - rằng cái cạnh ở góc trái và cái cạnh ở góc phải là cùng một thứ. Mạng kết nối đầy đủ phải học điều đó từ dữ liệu, tức phải nhìn thấy cái cạnh ở đủ mọi vị trí mới học được. CNN được cho không. Foundations · Bài 13 gọi đây là đưa tri thức về bài toán vào cấu trúc mô hình, và nó rẻ hơn nhiều so với để mô hình tự mò ra từ dữ liệu.

## 6. Đây mới là cửa vào

Bài này dừng ở mức trực giác: tích chập, bộ lọc, bản đồ đặc trưng, pooling, và vì sao chúng hợp với ảnh. Còn rất nhiều thứ chưa nói.

**Kiến trúc.** CNN trong bài có 2 tầng tích chập. Các mạng thật có hàng chục: LeNet (1998) cho chữ số viết tay, AlexNet (2012) mở màn làn sóng deep learning ở bài 01, VGG, và **ResNet** với kết nối tắt - thứ mà bài 08 đã hẹn là cách chữa cho mạng thật sự sâu.

**Bài toán khác ngoài phân loại.** Ảnh không chỉ để hỏi "đây là cái gì": còn phát hiện vật thể (cái gì, ở đâu), phân đoạn (từng pixel thuộc về vật nào), OCR, và các mô hình vision transformer.

Tất cả những thứ đó là nội dung của **chương Computer Vision**. Bài này chỉ cần bạn nắm chắc một điều: CNN thắng không phải vì to hơn, mà vì nó *biết trước* đầu vào là ảnh.

## Tóm tắt bài học

- `nn.Flatten()` ở đầu mạng không mất vị trí tuyệt đối nhưng mất quan hệ láng giềng: xáo trộn pixel theo một thứ tự cố định rồi train lại thì mạng kết nối đầy đủ vẫn ra kết quả y hệt, còn CNN thì sập.
- CNN xây sẵn hai sự thật về ảnh: pixel gần nhau thì liên quan (chỉ nhìn cục bộ), và một mẫu hình có nghĩa ở mọi vị trí (dùng chung trọng số).
- Số tham số của một tầng tích chập **không đổi theo kích thước ảnh**: `Conv2d(1, 16, 3×3)` luôn là 160, trong khi `Linear` tới 256 nơ-ron đi từ 200.960 (ảnh 28×28) lên 12.845.312 (ảnh 224×224).
- Pooling thu nhỏ bản đồ đặc trưng và cho một chút dung sai vị trí.
- Trên FashionMNIST, CNN 20.490 tham số đạt 0,8781, thắng MLP 203.530 tham số (0,8488) - ít hơn 10 lần tham số, hơn 2,9 điểm.
- Ít tham số không có nghĩa ít phép tính: CNN train lâu gấp ba lần MLP, vì bộ lọc phải trượt qua mọi vị trí.
- CNN **không** bất biến với dịch chuyển: dịch ảnh 3 pixel làm nó mất 42,4 điểm (MLP mất 52,0). Nó có thiên hướng phù hợp với ảnh, và muốn chịu được dịch chuyển thì phải huấn luyện bằng augmentation.

## Câu hỏi tự kiểm tra

1. Bạn xáo trộn 784 pixel của mọi ảnh theo cùng một thứ tự ngẫu nhiên cố định. Mạng kết nối đầy đủ bị ảnh hưởng thế nào? Còn CNN?
2. Vì sao số tham số của một tầng tích chập không phụ thuộc kích thước ảnh, trong khi tầng `Linear` thì có?
3. CNN ít hơn 10 lần tham số nhưng train lâu gấp ba. Giải thích nghịch lý này.
4. Max pooling 2×2 cho dung sai bao nhiêu pixel? Điều đó giải thích thế nào cho kết quả ở mục 5?
5. Có người nói "CNN bất biến với dịch chuyển nên không cần data augmentation". Dựa vào bảng ở mục 5, bạn phản biện ra sao?
6. Việc dùng chung trọng số đưa vào mô hình một giả định về thế giới. Giả định đó là gì, và bạn hình dung loại dữ liệu nào mà giả định ấy **sai**?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 7.1 *From Fully Connected Layers to Convolutions* - đúng mạch mục 1-2, kể cả lập luận về tính tương biến; 7.2 phép tích chập; 7.5 pooling; 8.1 LeNet |
| **An Introduction to Statistical Learning (ISLP)** | Chương 10.3 *Convolutional Neural Networks* - bộ lọc, bản đồ đặc trưng và pooling giải thích bằng ví dụ số nhỏ |
| **Bài 08 của chương này** | Kết nối tắt và ResNet - lời hẹn được nhắc lại ở mục 6 |

**Nguồn online bổ sung (miễn phí):**

- [`nn.Conv2d` - tài liệu PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - công thức tính kích thước đầu ra theo `padding`, `stride`, `dilation`.
- [CS231n - Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/) - ghi chú bài giảng Stanford, có hình động minh họa bộ lọc trượt.
- [LeCun và cộng sự - Gradient-Based Learning Applied to Document Recognition (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) - bài báo LeNet, CNN thực dụng đầu tiên.
- [He và cộng sự - Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385) - ResNet và kết nối tắt, cách chữa cho mạng sâu mà bài 08 đã hẹn.
- [Distill - Feature Visualization](https://distill.pub/2017/feature-visualization/) - nhìn thấy các bộ lọc của một CNN thật sự học được gì.

> **Bài tiếp theo:** [Dữ liệu chuỗi: RNN và chỗ nó hụt hơi](../sequence-data-rnns-and-where-they-fall-short-vi/) - CNN khai thác cấu trúc *không gian* của ảnh. Bài sau đổi sang loại cấu trúc khác - thứ tự theo thời gian - nơi mỗi đầu vào phụ thuộc vào những gì đã đến trước nó.

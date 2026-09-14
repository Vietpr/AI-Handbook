---
title: "Chỗ mô hình tuyến tính bó tay"
description: "Nơi mô hình tuyến tính ngừng cải thiện nếu thiếu đặc trưng do con người thiết kế, và vì sao cách này không mở rộng tốt sang ảnh hoặc âm thanh."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "where-linear-models-give-up"
order: 1
pubDate: 2026-08-19
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ thấy bằng số liệu chỗ mà mô hình tuyến tính chỉ đi tiếp được khi con người dọn sẵn đặc trưng cho nó, hiểu vì sao việc nghĩ ra feature bằng tay không mở rộng được lên ảnh hay âm thanh, và nắm được câu hỏi mà cả chương này đi trả lời: liệu mô hình có thể tự tìm ra cách biểu diễn dữ liệu thay cho con người.

## 1. Bốn dòng dữ liệu đánh bại hồi quy logistic

Bảng này có bốn dòng, hai cột, nhãn 0/1. Nó nhỏ đến mức nhìn là hiểu:

| $x_1$ | $x_2$ | $y$ |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | 0 |

Quy luật: nhãn là 1 khi *đúng một* trong hai cột bằng 1. Đây là phép **XOR** (hoặc loại trừ) - một quy tắc mà đứa trẻ nào cũng nắm được sau ba giây.

Đưa nó cho hồi quy logistic của chương Machine Learning · Bài 03:

```python
from sklearn.linear_model import LogisticRegression
X = [[0,0],[0,1],[1,0],[1,1]]
y = [0, 1, 1, 0]
print(LogisticRegression().fit(X, y).score(X, y))   # → 0.5
print(LogisticRegression().fit(X, y).predict(X))    # → [0 0 0 0]
```

**Accuracy 0,5.** Không phải trên dữ liệu mới - trên đúng bốn dòng nó vừa học thuộc. Mô hình đầu hàng, phán tất cả là 0.

Lý do nằm ở hình học. Hồi quy logistic vạch một **đường thẳng** rồi tuyên bố: bên này là 1, bên kia là 0. Hãy thử tự kẻ:

```
 x₂
  │
1 ┤  ●(0,1)          ○(1,1)
  │
  │
0 ┤  ○(0,0)          ●(1,0)
  └──┼─────────────────┼──── x₁
     0                 1

  ● = nhãn 1     ○ = nhãn 0
```

Hai điểm nhãn 1 nằm ở hai góc *đối diện* nhau, hai điểm nhãn 0 cũng vậy. Không có đường thẳng nào tách được cặp này khỏi cặp kia - bạn kẻ kiểu gì cũng có một điểm lọt sang nhầm phía. Đây không phải chuyện thuật toán yếu hay dữ liệu ít; nó là giới hạn của *dạng* mô hình.

> 🔧 **Thử ngay:** đổi nhãn dòng cuối từ 0 thành 1 (tức thành phép OR) rồi chạy lại. Accuracy bao nhiêu?
> **1,0.** Với OR, ba điểm nhãn 1 gom về một phía và điểm nhãn 0 nằm lẻ ở góc (0,0) - một đường thẳng tách gọn. Cũng bốn dòng, cũng hai cột, cũng mô hình đó: cái thay đổi duy nhất là *hình dạng* của quy luật.

## 2. Cách chữa cũ: người nghĩ hộ máy một cột mới

Chương Machine Learning · Bài 11 đã đưa lời giải cho tình huống này - **tạo đặc trưng**. Với XOR, thêm một cột là xong: $x_3 = x_1 \times x_2$.

| $x_1$ | $x_2$ | $x_1 x_2$ | $y$ |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 1 | 0 | **1** |
| 1 | 0 | 0 | **1** |
| 1 | 1 | **1** | 0 |

Giờ nhãn 0 ứng với "tổng hai cột đầu bằng 0 **hoặc** cột thứ ba bằng 1" - tách được bằng một mặt phẳng trong không gian ba chiều:

```python
X3 = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]
print(LogisticRegression().fit(X3, y).score(X3, y))          # → 0.75
print(LogisticRegression(C=10).fit(X3, y).score(X3, y))      # → 1.0
```

Chi tiết nhỏ ở dòng đầu đáng dừng lại một nhịp. Cột mới đã làm dữ liệu tách được bằng mặt phẳng, nhưng `LogisticRegression()` mặc định có kèm regularization `C=1.0`, và với đúng bốn dòng dữ liệu thì mức phạt ấy đủ để kéo hệ số về nhỏ, mô hình chưa dựng nổi mặt phẳng cần thiết nên vẫn sai một dòng. Nới phạt bằng `C=10` là đủ 100%. Chuyện này không liên quan gì tới XOR - nó chỉ nhắc lại điều chương Machine Learning · Bài 06 đã nói: mặc định của thư viện là một lựa chọn, không phải chân lý.

Bài toán được giải, nhưng chú ý ai giải nó: **bạn**. Bạn nhìn bảng, nhận ra quy luật là XOR, và nghĩ ra rằng phép nhân hai cột sẽ bắt được nó. Mô hình chỉ khớp hệ số cho cái cột bạn đã dọn sẵn.

Trên dữ liệu thật, chuyện không dễ như vậy. Bộ `make_moons` là hai vòng cung lồng nhau - bạn đã gặp ở chương Machine Learning · Bài 09:

| Mô hình | Accuracy trên test |
|---|---|
| Hồi quy logistic | 0,8767 |
| Logistic + đặc trưng đa thức bậc 2 | 0,8733 |
| Logistic + đặc trưng đa thức bậc 10 | 0,9667 |

Đọc bảng này kỹ một chút. Thêm đa thức bậc 2 - phản xạ đầu tiên của hầu hết mọi người - **không giúp gì**, thậm chí kém đi một chút. Phải lên tới bậc 10 mới khá. Nhưng bậc 10 trên 2 cột sinh ra 66 cột, phần lớn vô dụng, và con số 10 ấy bạn tìm ra bằng cách mò.

Đó mới là dữ liệu hai chiều, vẽ ra nhìn được bằng mắt.

## 3. Chỗ cách chữa cũ hết đường

Một tấm ảnh màu 224×224 pixel là **150.528 con số**. Hãy thử áp đúng quy trình vừa rồi lên nó.

Bước đầu tiên: nhìn dữ liệu để nghĩ ra feature. Nhưng nhìn cái gì? Cột thứ 74.311 là độ sáng kênh xanh lục của một pixel nằm đâu đó trong ảnh. Nó không có ý nghĩa riêng. Không như `thu nhập` hay `số năm đi làm` ở chương trước - những cột mà bạn hiểu ngay và biết cách kết hợp - pixel thứ 74.311 chẳng nói gì cả, và nó cũng không nói gì khác đi khi con mèo trong ảnh dịch sang phải ba pixel.

Cái *có* nghĩa trong một tấm ảnh là những thứ ở tầng cao hơn hẳn: một cạnh, một góc, một mảng lông vằn, một cái tai nhọn. Không cái nào trong số đó là một cột trong bảng. Mỗi cái là một **tổ hợp** của hàng trăm pixel, và tổ hợp ấy còn phải nhận ra được dù vật thể dịch chỗ, xoay nghiêng, hay đổi ánh sáng.

Trong nhiều năm, người ta đã làm đúng việc đó bằng tay. Thị giác máy tính trước 2012 có cả một ngành con chuyên thiết kế **bộ mô tả đặc trưng (feature descriptor)** - SIFT, HOG, SURF - là những công thức do con người nghĩ ra để rút từ ảnh thô ra vài nghìn con số "có nghĩa" hơn, rồi mới đưa vào một bộ phân loại như SVM của chương Machine Learning · Bài 08. Cách này chạy được, và từng là chuẩn mực. Nhưng nó có hai cái giá: mỗi lĩnh vực cần một bộ mô tả riêng do chuyên gia lĩnh vực đó thiết kế, và trần chất lượng bị chặn bởi trí tưởng tượng của người thiết kế.

Câu hỏi tự nhiên nảy ra ở đây, và nó là câu hỏi của cả chương này:

> Nếu con người không biết phải tạo ra feature nào, liệu mô hình có thể **tự tìm ra** cách biểu diễn dữ liệu hay không?

Cách tiếp cận trả lời "có" cho câu hỏi ấy gọi là **học biểu diễn (representation learning)**: thay vì đưa mô hình các đặc trưng đã dọn sẵn, ta đưa nó dữ liệu thô và để chính quá trình huấn luyện tìm ra cách biểu diễn hữu ích.

## 4. Một tầng ẩn đủ để học XOR

Quay lại bốn dòng ở mục 1, nhưng lần này thay vì tự thêm cột $x_1 x_2$, ta đặt giữa đầu vào và đầu ra một **tầng trung gian** - vài nơ-ron, mỗi cái tự học lấy một tổ hợp của $x_1$ và $x_2$:

```
   x₁ ──┐
        ├──▶ [ tầng ẩn: 4 nơ-ron ]──▶ đầu ra ──▶ y
   x₂ ──┘
        ↑                    ↑
   dữ liệu thô        tổ hợp do MÁY tự tìm,
                      không do bạn nghĩ ra
```

```python
import torch
from torch import nn

X = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
y = torch.tensor([0., 1., 1., 0.])
torch.manual_seed(0)

net = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
opt = torch.optim.Adam(net.parameters(), lr=0.1)
lossf = nn.BCEWithLogitsLoss()

for _ in range(2000):
    opt.zero_grad()
    loss = lossf(net(X).squeeze(), y)
    loss.backward()
    opt.step()

print(torch.sigmoid(net(X).squeeze()).detach().numpy().round(4))
```

Kết quả: `[0. 1. 1. 0.]`, loss còn 0,00002. Đúng cả bốn dòng - và lần này **không ai mách cho nó cột $x_1 x_2$**. Bốn nơ-ron ở tầng giữa tự tìm ra các tổ hợp đủ dùng để tách hai lớp.

Đó là toàn bộ ý tưởng của chương này, thu nhỏ vào bốn dòng dữ liệu.

> ⚠️ **Lưu ý nhỏ:** đừng đọc kết quả này thành "cứ thêm tầng ẩn là xong". Chạy lại đúng đoạn code trên với các `manual_seed` khác nhau, chỉ đổi số nơ-ron ở tầng ẩn, và đếm số lần giải đúng XOR trong 20 lần thử:
>
> | Số nơ-ron ẩn | 2 | 3 | 4 | 8 |
> |---|---|---|---|---|
> | Số seed giải đúng | 7/20 | 17/20 | 19/20 | 20/20 |
>
> Hai nơ-ron là mức tối thiểu về lý thuyết, và trong thực tế nó hỏng quá nửa số lần - huấn luyện mắc kẹt ở một lời giải tồi tùy theo chỗ xuất phát ngẫu nhiên. Mạng nơ-ron không có lời giải bằng công thức đóng như hồi quy tuyến tính; nó phải đi xuống núi trong sương mù (Foundations · Bài 11), và điểm xuất phát có ảnh hưởng thật. Bài 08 quay lại chuyện này với tên gọi **khởi tạo (initialization)**.

## 5. Trên dữ liệu thật thì được bao nhiêu?

XOR là đồ chơi. Thử trên **FashionMNIST**: 70.000 ảnh xám 28×28 của quần áo, 10 lớp (áo phông, quần, áo len, váy, áo khoác, dép, sơ mi, giày thể thao, túi, bốt). Nó cùng cỡ với MNIST chữ số viết tay nhưng khó hơn hẳn - và sẽ theo bạn suốt chương này.

Hai mô hình, cùng dữ liệu, cùng 20 epoch, cùng thuật toán tối ưu:

| Mô hình | Tham số | Accuracy validation tốt nhất |
|---|---|---|
| Hồi quy softmax (tuyến tính, không tầng ẩn) | 7.850 | 0,8396 |
| Thêm 1 tầng ẩn 256 nơ-ron | 203.530 | **0,8675** |

Tầng ẩn mang lại **2,8 điểm phần trăm**, đổi lấy số tham số gấp 26 lần.

Hãy thành thật với con số này. Nó có thật và lặp lại được, nhưng nó không phải một cú nhảy vọt - và đó là điều đáng nói ngay từ bài đầu chương. Ném thêm tham số vào một kiến trúc không hợp với dữ liệu thì chỉ được chừng đó. Cái làm nên khác biệt lớn là **kiến trúc phù hợp với dạng dữ liệu**: bài 09 sẽ đưa một mạng biết rằng đầu vào là *ảnh* - rằng các pixel cạnh nhau có liên quan, và một cái cạnh vẫn là cái cạnh dù nó nằm ở góc nào của khung hình.

Ba tính chất khiến hướng đi này đáng theo, dù mức lãi ở bài đầu còn khiêm tốn:

- **Nó tự mở rộng.** Cùng một ý tưởng, thêm tầng, thêm nơ-ron, thêm dữ liệu - không cần nghĩ lại bộ đặc trưng từ đầu.
- **Nó chuyển được giữa các lĩnh vực.** Cùng khung ấy chạy trên ảnh, âm thanh và văn bản, trong khi SIFT chỉ dùng cho ảnh.
- **Cái nó học được dùng lại được.** Đặc trưng một mạng học từ hàng triệu ảnh có thể mang sang một bài toán khác chỉ có vài nghìn ảnh - đó là bài 11.

## Tóm tắt bài học

- Mô hình tuyến tính vạch một ranh giới thẳng, nên có những quy luật đơn giản nó không thể học: XOR bốn dòng làm hồi quy logistic tụt xuống accuracy 0,5 ngay trên dữ liệu train.
- Cách chữa của chương Machine Learning là con người tạo thêm feature. Cách này đúng nhưng phụ thuộc vào việc bạn *đoán trúng* feature nào cần: trên `make_moons`, đa thức bậc 2 không giúp gì (0,8733), phải mò lên bậc 10 mới được 0,9667.
- Với ảnh, âm thanh, văn bản thì cách ấy hết đường: từng pixel không mang ý nghĩa riêng, thứ có nghĩa là các tổ hợp ở tầng cao hơn, và chúng phải nhận ra được dù vật thể dịch chỗ hay đổi sáng.
- Câu hỏi của cả chương: mô hình có thể **tự tìm ra cách biểu diễn** thay vì được cho sẵn không? Hướng trả lời "có" gọi là học biểu diễn.
- Một tầng ẩn giải được XOR mà không ai mách cột $x_1 x_2$ - nhưng với 2 nơ-ron thì 13/20 lần thử thất bại, vì mạng phải dò tìm chứ không có công thức đóng.
- Trên FashionMNIST, thêm một tầng ẩn nâng accuracy từ 0,8396 lên 0,8675 với 26 lần số tham số. Lãi thật nhưng vừa phải - cú nhảy lớn đến từ kiến trúc hợp dữ liệu, không từ việc thêm tham số.

## Câu hỏi tự kiểm tra

1. Vì sao hồi quy logistic không học được XOR, trong khi nó học được OR? Trả lời bằng hình học, không bằng công thức.
2. Bạn thêm cột $x_1 x_2$ vào bảng XOR và mô hình đạt 100%. Ai đã thực sự giải bài toán trong trường hợp này, và điều đó nói gì về khả năng mở rộng của cách làm ấy?
3. Trên `make_moons`, đặc trưng đa thức bậc 2 cho kết quả *kém hơn* hồi quy logistic thuần. Nêu một cách giải thích, và bạn sẽ kiểm chứng nó thế nào?
4. Vì sao "nhìn dữ liệu rồi nghĩ ra feature" làm được với bảng thu nhập nhưng không làm được với ảnh 224×224?
5. Mạng 2 nơ-ron ẩn giải đúng XOR ở 7/20 seed. Điều đó cho biết gì về cách mạng nơ-ron được huấn luyện, khác với hồi quy tuyến tính ở chương trước?
6. Thêm một tầng ẩn 256 nơ-ron vào FashionMNIST chỉ được thêm 2,8 điểm accuracy nhưng tốn gấp 26 lần tham số. Với một dự án thật, bạn cần biết thêm điều gì trước khi kết luận đánh đổi này có đáng hay không?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 5.1 *Multilayer Perceptrons* - mục 5.1.1 giải thích vì sao xếp chồng nhiều tầng tuyến tính vẫn chỉ ra một hàm tuyến tính, và vì sao cần hàm kích hoạt; phần mở đầu chương 1 về representation learning |
| **An Introduction to Statistical Learning (ISLP)** | Chương 10.1 - mạng nơ-ron một tầng ẩn, cách nó dựng đặc trưng dẫn xuất từ đầu vào |
| **Foundations · Bài 11, 13, 14** | Gradient descent và chuyện điểm xuất phát; capacity; overfitting - ba thứ quay lại suốt chương này |

**Nguồn online bổ sung (miễn phí):**

- [FashionMNIST - kho dữ liệu gốc của Zalando Research](https://github.com/zalandoresearch/fashion-mnist) - mô tả bộ dữ liệu và bảng so sánh kết quả của nhiều mô hình, tiện để đối chiếu con số bạn tự chạy.
- [TensorFlow Playground](https://playground.tensorflow.org/) - kéo thả số tầng và số nơ-ron rồi xem ranh giới quyết định đổi hình theo thời gian thực; thử bộ dữ liệu XOR ở góc trái.
- [Krizhevsky, Sutskever, Hinton - ImageNet Classification with Deep Convolutional Neural Networks (NeurIPS 2012)](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) - công trình đánh dấu chỗ đặc trưng học được vượt qua đặc trưng thiết kế bằng tay trên ImageNet.
- [Lowe - Distinctive Image Features from Scale-Invariant Keypoints (IJCV 2004)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) - bài báo SIFT, để thấy một bộ mô tả đặc trưng thiết kế bằng tay trông như thế nào.
- [Bengio, Courville, Vincent - Representation Learning: A Review and New Perspectives (2013)](https://arxiv.org/abs/1206.5538) - bài tổng quan đặt tên cho chính câu hỏi ở mục 3.

> **Bài tiếp theo:** [Một nơ-ron và hàm kích hoạt](../a-neuron-and-its-activation-function-vi/) - bài này gọi "tầng ẩn" mà chưa mở ra xem bên trong có gì. Bài sau tháo một nơ-ron ra - hóa ra bạn đã gặp nó rồi, dưới một cái tên khác - và trả lời vì sao thiếu hàm kích hoạt thì xếp bao nhiêu tầng cũng vô ích.

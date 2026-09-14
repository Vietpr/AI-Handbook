---
title: "Một nơ-ron và hàm kích hoạt"
description: "Bên trong một nơ-ron có gì, vì sao thiếu hàm kích hoạt thì xếp bao nhiêu tầng cũng vô ích, chọn hàm kích hoạt cho tầng ẩn, và phân biệt nó với hàm ở tầng ra."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "a-neuron-and-its-activation-function"
order: 2
pubDate: 2026-08-19
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết bên trong một nơ-ron có gì (và nhận ra mình đã gặp nó ở chương trước dưới tên khác), hiểu bằng cả đại số lẫn số liệu vì sao thiếu hàm kích hoạt thì xếp bao nhiêu tầng cũng vô ích, chọn được hàm kích hoạt cho tầng ẩn, và phân biệt được nó với hàm ở tầng ra - chỗ người mới hay nhầm nhất.

## 1. Tháo một nơ-ron ra xem

Bài 01 gọi tầng giữa là "tầng ẩn" rồi đi tiếp, chưa mở ra xem. Mở luôn. Một **nơ-ron** làm đúng hai việc, theo thứ tự:

```
 x₁ ──w₁──┐
 x₂ ──w₂──┤     z = w₁x₁ + w₂x₂ + ... + b        a = f(z)
  ⋮       ├──▶ ────────────────────────  ──▶  ──────────  ──▶ đầu ra
 xₙ ──wₙ──┘        (1) tổng có trọng số          (2) hàm
           b                                    kích hoạt
```

**Bước 1 - tổng có trọng số.** Nhân mỗi đầu vào với một trọng số, cộng hết lại, cộng thêm một hệ số chặn $b$. Đây chính là tích vô hướng của Foundations · Bài 07, và cũng chính là công thức thợ ống nước `tiền = 50 + 100 × giờ` của chương Machine Learning · Bài 02.

**Bước 2 - hàm kích hoạt.** Đẩy con số $z$ qua một hàm phi tuyến $f$. Mục 3 giải thích vì sao bước này không được phép bỏ.

Vậy thôi. Một nơ-ron không phải mô phỏng tế bào thần kinh; nó là một phép nhân-cộng rồi bẻ cong kết quả.

## 2. Bạn đã gặp nó rồi: một nơ-ron sigmoid là hồi quy logistic

Lấy đúng một nơ-ron, cho hàm kích hoạt là **sigmoid** $\sigma(z) = 1/(1+e^{-z})$, huấn luyện bằng log loss. Viết ra công thức:

$$\hat{p} = \sigma(w_1 x_1 + w_2 x_2 + \dots + w_p x_p + b)$$

Đó là *nguyên văn* hồi quy logistic của chương Machine Learning · Bài 03. Không phải "gần giống" - cùng một mô hình, hai cái tên.

Kiểm chứng bằng số trên bộ `adult` quen thuộc (48.842 hồ sơ, 89 cột sau tiền xử lý):

```python
from sklearn.linear_model import LogisticRegression
import torch
from torch import nn

sk = LogisticRegression(max_iter=3000, C=1.0).fit(Xp, y)      # scikit-learn

neuron = nn.Linear(Xp.shape[1], 1)                            # đúng MỘT nơ-ron
opt = torch.optim.LBFGS(neuron.parameters(), max_iter=500)
lossf = nn.BCEWithLogitsLoss()
lam = 1.0 / (2 * len(y))                                      # khớp với C=1 của sklearn

def closure():
    opt.zero_grad()
    loss = lossf(neuron(Xt).squeeze(), yt) + lam * (neuron.weight ** 2).sum()
    loss.backward()
    return loss

opt.step(closure)
```

| | scikit-learn | Một nơ-ron trong PyTorch |
|---|---|---|
| Accuracy | 0,8522 | 0,8523 |

Tương quan giữa 89 hệ số của hai bên: **0,9803**. Tương quan giữa xác suất chúng dự đoán cho 48.842 hồ sơ: **1,0000**, lệch nhiều nhất 0,088.

Con số 0,852 ấy quen chứ? Nó đúng bằng accuracy mà chương Machine Learning · Bài 03 đã báo cáo cho hồi quy logistic trên cùng bộ dữ liệu. Bạn không học một mô hình mới - bạn vừa gọi tên cũ bằng từ vựng mới.

> ⚠️ **Lưu ý nhỏ:** hai bộ hệ số tương quan 0,98 chứ không phải 1,00, dù xác suất trùng khít. Lý do là `adult` có nhiều cột tương quan với nhau sau one-hot (chương Machine Learning · Bài 02 gọi là đa cộng tuyến): nhiều bộ hệ số khác nhau cho ra cùng một dự đoán, và hai thuật toán tối ưu dừng ở hai bộ khác nhau. Khi so hai mô hình, so *dự đoán* đáng tin hơn so *hệ số*.

Nếu một nơ-ron chỉ là hồi quy logistic, thì cái mới nằm ở chỗ **xếp nhiều nơ-ron cạnh nhau và chồng lên nhau**. Đó là bài 03. Nhưng trước đó phải trả lời một câu quan trọng hơn.

## 3. Bỏ hàm kích hoạt thì xếp bao nhiêu tầng cũng vô ích

Giả sử ta bỏ bước 2, chỉ giữ phép tổng có trọng số. Hai tầng nối tiếp:

$$h = W_1 x + b_1 \qquad\qquad \hat{y} = W_2 h + b_2$$

Thay cái trên vào cái dưới:

$$\hat{y} = W_2 (W_1 x + b_1) + b_2 = \underbrace{(W_2 W_1)}_{\text{gọi là } W'} x + \underbrace{(W_2 b_1 + b_2)}_{\text{gọi là } b'} = W' x + b'$$

Kết quả vẫn là **một phép tuyến tính** - đúng dạng $Wx + b$. Bạn có xếp 2 tầng, 10 tầng hay 100 tầng thì tích của các ma trận vẫn là một ma trận, và mô hình vẫn chỉ vẽ được ranh giới thẳng. Toàn bộ công sức thêm tầng bị nuốt mất trong một phép nhân ma trận.

Đo thử trên FashionMNIST, mạng hai tầng ẩn 256 nơ-ron, 10 epoch:

| Hàm kích hoạt ở tầng ẩn | Accuracy validation |
|---|---|
| **Không có** (chỉ toàn `Linear`) | 0,8283 |
| Sigmoid | 0,7611 |
| Tanh | 0,8452 |
| **ReLU** | **0,8549** |

Dòng đầu là bằng chứng của phép đại số bên trên: bỏ hàm kích hoạt, mạng hai tầng ẩn với 270.000 tham số cho **0,8283** - trong khi hồi quy softmax 7.850 tham số ở bài 01 cho 0,8317 với cùng số epoch. Thêm 260.000 tham số, không thêm được gì, vì về bản chất nó vẫn là đúng một mô hình tuyến tính.

Hàm kích hoạt phi tuyến là thứ duy nhất khiến việc xếp tầng có ý nghĩa.

> 🔧 **Thử ngay:** vì sao hàm kích hoạt phải **phi tuyến**? Nếu ta chọn $f(z) = 3z$ thì sao?
> Thay vào: $h = 3(W_1x + b_1)$, rồi $\hat{y} = W_2 h + b_2 = (3W_2W_1)x + (3W_2b_1 + b_2)$ - lại đúng dạng $W'x + b'$. Một hàm tuyến tính lồng trong hàm tuyến tính vẫn là hàm tuyến tính; hằng số 3 chỉ bị nuốt vào trọng số. Phải là hàm *bẻ cong* - có chỗ gãy, có chỗ bão hòa - thì phép lồng mới sinh ra thứ mới.

## 4. Ba hàm kích hoạt bạn sẽ gặp

```
   Sigmoid σ(z)              Tanh(z)                  ReLU(z)
 1 ┤      ╭────         1 ┤       ╭────         3 ┤        ╱
   │     ╱                │      ╱                 │      ╱
   │    ╱               0 ┼─────┼─────           1 ┤    ╱
   │   ╱                  │    ╱                   │  ╱
 0 ┼──╯                -1 ┤────╯                 0 ┼────────
   └──┼──────              └──┼──────              └──┼──────
      0                       0                       0
 bẹt ở cả hai đầu      bẹt ở hai đầu,          gãy tại 0,
 ra (0, 1)             ra (−1, 1), đối xứng    dương thì đi thẳng
```

**Sigmoid** ép mọi số về khoảng $(0, 1)$. Nó là hàm kích hoạt kinh điển của những năm 1990 và giờ gần như không còn dùng ở tầng ẩn nữa - bảng trên cho thấy vì sao: 0,7611, kém cả mô hình không có hàm kích hoạt. Lý do nằm ở hai đầu đồ thị: khi $z$ lớn hoặc nhỏ, đường cong nằm bẹt, độ dốc gần bằng 0, nên tín hiệu học gần như tắt ngóm khi truyền ngược qua nhiều tầng. Bài 08 đo hiện tượng này và gọi đúng tên của nó.

**Tanh** cùng hình dạng nhưng ra khoảng $(-1, 1)$ và đối xứng qua 0. Khá hơn sigmoid, vẫn bẹt ở hai đầu.

**ReLU** - *rectified linear unit* - đơn giản đến mức đáng ngờ: $\text{ReLU}(z) = \max(0, z)$. Âm thì trả về 0, dương thì giữ nguyên. Nó thắng ở bảng trên, và ba lý do đều thực dụng: đạo hàm ở nhánh dương luôn bằng 1 nên bản thân hàm kích hoạt không làm tín hiệu học co lại khi đi qua nhiều tầng (các ma trận trọng số vẫn có thể làm nó co hoặc phình - chuyện của bài 08); tính nó chỉ là một phép so sánh; và việc nó cho ra đúng số 0 ở nhánh âm khiến mạng chỉ dùng một phần nơ-ron cho mỗi đầu vào.

> ⚠️ **Lưu ý nhỏ:** ReLU có một tật gọi là **ReLU chết (dying ReLU)**. Nếu trọng số bị đẩy đến chỗ mà $z$ luôn âm với mọi dữ liệu, nơ-ron đó luôn ra 0; đạo hàm của ReLU tại đó cũng bằng 0, nên gradient không chảy về trọng số của chính nơ-ron ấy và nó ngừng tự học. Nói "chết hẳn" thì hơi quá: các tầng phía trước vẫn đang được cập nhật, nên phân bố đầu vào của nơ-ron này có thể đổi và đẩy $z$ dương trở lại; weight decay hay quán tính của bộ tối ưu cũng có thể làm điều đó. Chỉ là nó không có cách nào tự cứu mình, nên trong thực tế phần lớn ca chết là chết luôn. Đo trên FashionMNIST sau 5 epoch, đếm số nơ-ron không kích hoạt lần nào trên 1.000 ảnh validation:
>
> | Learning rate | Nơ-ron chết |
> |---|---|
> | 0,1 | 2 / 256 |
> | 0,5 | **31 / 256** |
>
> Learning rate lớn đẩy trọng số đi những bước quá đà: từ 2 nơ-ron chết lên 31, tức 12% số nơ-ron của tầng. Các biến thể như **Leaky ReLU** (nhánh âm cho độ dốc nhỏ thay vì 0) sinh ra để chữa đúng chuyện này.

## 5. Hàm ở tầng ra là chuyện khác

Đây là chỗ người mới hay nhầm, nên tách hẳn ra một mục.

"Hàm kích hoạt" ở bốn mục trên nói về **tầng ẩn** - nơi nhiệm vụ của nó là bẻ cong để việc xếp tầng có nghĩa. Ở **tầng ra** thì nhiệm vụ hoàn toàn khác: đưa con số về đúng dạng mà bài toán cần. Chọn theo bài toán, không theo thói quen:

| Bài toán | Tầng ra | Hàm | Hàm mất mát đi kèm |
|---|---|---|---|
| Phân loại nhị phân | 1 nơ-ron | **sigmoid** | log loss / BCE |
| Phân loại nhiều lớp | 1 nơ-ron mỗi lớp | **softmax** | cross-entropy |
| Hồi quy | 1 nơ-ron | **không có gì** | MSE / MAE |

Ba điều rút ra từ bảng:

- **Sigmoid vẫn sống, nhưng ở tầng ra.** Nó bị loại khỏi tầng ẩn vì bẹt hai đầu, nhưng ở tầng ra thì "ép về khoảng (0, 1)" chính là thứ ta cần - một xác suất.
- **Softmax là bản nhiều lớp của sigmoid**, đúng như chương Machine Learning · Bài 03 đã nói: mỗi lớp một điểm số, lũy thừa lên rồi chia cho tổng để được các số cộng lại bằng 1.
- **Hồi quy thì không đặt gì cả.** Giá nhà có thể là bất kỳ số dương nào; nhét sigmoid vào là tự chặn đầu ra trong khoảng (0, 1).

> ⚠️ **Lưu ý nhỏ:** trong PyTorch, `nn.CrossEntropyLoss` và `nn.BCEWithLogitsLoss` **đã bao gồm softmax/sigmoid bên trong**. Nghĩa là mạng của bạn phải xuất ra con số thô (gọi là **logit**), không được tự áp softmax rồi mới đưa vào hàm mất mát - làm vậy là áp hai lần, và mô hình sẽ học rất tệ mà không báo lỗi gì. Đó là lý do mọi mạng phân loại trong chương này đều kết thúc bằng một `nn.Linear` trần, không có hàm nào phía sau. Việc gộp này không chỉ cho tiện: nó còn giúp phép tính ổn định hơn về mặt số học so với làm hai bước rời.

## Tóm tắt bài học

- Một nơ-ron làm hai việc: tổng có trọng số của đầu vào (tích vô hướng, Foundations · Bài 07), rồi đẩy kết quả qua một hàm kích hoạt phi tuyến.
- Một nơ-ron với sigmoid chính là hồi quy logistic - không phải tương tự, mà là cùng một mô hình: trên `adult`, scikit-learn cho 0,8522 và một nơ-ron PyTorch cho 0,8523, tương quan xác suất 1,0000.
- Bỏ hàm kích hoạt thì hai tầng tuyến tính rút gọn thành một: $W_2(W_1x + b_1) + b_2 = W'x + b'$. Đo được: mạng hai tầng ẩn không hàm kích hoạt cho 0,8283, ngang mô hình tuyến tính 7.850 tham số.
- Ở tầng ẩn, ReLU là lựa chọn khởi đầu quen thuộc (0,8549) vì đạo hàm nhánh dương bằng 1 và tính rất rẻ; sigmoid kém hẳn (0,7611) vì bẹt ở hai đầu.
- ReLU có thể chết khi learning rate quá lớn: 31/256 nơ-ron ngừng kích hoạt ở lr = 0,5, so với 2/256 ở lr = 0,1.
- Hàm ở **tầng ra** chọn theo bài toán chứ không theo thói quen: sigmoid cho nhị phân, softmax cho nhiều lớp, không gì cả cho hồi quy.
- Trong PyTorch, hàm mất mát phân loại đã chứa sẵn softmax/sigmoid - mạng phải xuất logit thô, nếu không là áp hai lần.

## Câu hỏi tự kiểm tra

1. Viết ra hai việc một nơ-ron làm, theo đúng thứ tự. Việc nào bạn đã gặp ở chương Machine Learning, dưới tên gì?
2. Chứng minh bằng đại số rằng hai tầng `Linear` nối tiếp nhau, không có hàm kích hoạt ở giữa, tương đương đúng một tầng `Linear`.
3. Một đồng nghiệp dùng $f(z) = 2z + 1$ làm hàm kích hoạt tầng ẩn và thắc mắc vì sao mạng sâu không khá hơn hồi quy tuyến tính. Giải thích cho họ.
4. Vì sao sigmoid bị loại khỏi tầng ẩn nhưng vẫn được dùng ở tầng ra? Hai vai trò đó khác nhau ở chỗ nào?
5. Mô hình của bạn dự đoán giá nhà và bạn đặt sigmoid ở tầng ra. Chuyện gì sẽ xảy ra với các căn nhà đắt?
6. Sau khi train, bạn thấy 40% nơ-ron ở tầng ẩn không kích hoạt lần nào trên tập validation. Chẩn đoán là gì, và bạn thử điều gì trước tiên?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 5.1 *Multilayer Perceptrons* - mục 5.1.2 *Activation Functions*: đồ thị và đạo hàm của ReLU, sigmoid, tanh; mục 5.1.1 chứng minh chuyện xếp tầng tuyến tính vẫn ra tuyến tính |
| **An Introduction to Statistical Learning (ISLP)** | Chương 10.1 - nơ-ron và hàm kích hoạt trong mạng một tầng ẩn, kèm hình so sánh sigmoid với ReLU |
| **Chương Machine Learning · Bài 03** | Hồi quy logistic, sigmoid, softmax và log loss - mục 2 của bài này là chính bài đó nhìn từ góc khác |

**Nguồn online bổ sung (miễn phí):**

- [`torch.nn` - các hàm kích hoạt trong PyTorch](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) - danh sách đầy đủ kèm công thức và đồ thị.
- [`nn.CrossEntropyLoss` - tài liệu PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) - đọc kỹ dòng nói hàm này đã gộp `LogSoftmax` bên trong, đúng cảnh báo ở mục 5.
- [Glorot, Bordes, Bengio - Deep Sparse Rectifier Neural Networks (AISTATS 2011)](https://proceedings.mlr.press/v15/glorot11a.html) - bài báo đưa ReLU thành lựa chọn mặc định, có phần bàn về tính thưa và về nơ-ron không kích hoạt.
- [Maas, Hannun, Ng - Rectifier Nonlinearities Improve Neural Network Acoustic Models (ICML 2013)](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) - bài giới thiệu Leaky ReLU, cách chữa cho ReLU chết ở mục 4.

> **Bài tiếp theo:** [Xếp tầng: mạng tự học lấy đặc trưng](../stacking-layers-networks-learn-their-own-features-vi/) - giờ đã biết một nơ-ron làm gì và vì sao cần hàm kích hoạt, câu còn lại là xếp chúng thành tầng thì được thêm cái gì - và chữ "sâu" trong deep learning thực ra mua được bao nhiêu.

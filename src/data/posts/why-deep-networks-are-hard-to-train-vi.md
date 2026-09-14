---
title: "Vì sao mạng sâu khó huấn luyện"
description: "Vì sao một mạng 20 tầng có thể không học được gì, truy nguyên nhân về độ lớn tín hiệu qua từng tầng, và hai cách chữa là khởi tạo đúng và batch normalization, kèm số đo."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "why-deep-networks-are-hard-to-train"
order: 8
pubDate: 2026-08-22
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu vì sao một mạng 20 tầng có thể không học được gì cả, truy được nguyên nhân về độ lớn tín hiệu truyền qua từng tầng, và biết hai cách chữa - chọn đúng cách khởi tạo và thêm batch normalization - cùng số đo của từng cách.

## 1. Mạng 20 tầng, accuracy 0,1000

Bài 03 dừng lại ở một điều khó chịu: mạng 4 tầng ẩn tệ hơn mạng 3 tầng. Đẩy tiếp lên 20 tầng thì chuyện không còn là "tệ hơn" nữa:

| Mạng | train loss sau 8 epoch | validation accuracy |
|---|---|---|
| 5 tầng ẩn, ReLU | 0,4198 | 0,8166 |
| **20 tầng ẩn, ReLU** | **2,3028** | **0,1000** |

Accuracy 0,1000 trên 10 lớp là đoán bừa, và 2,3028 chính là $\ln(10)$ - con số bài 06 đã gặp: mô hình chia đều xác suất cho cả 10 lớp và không phân biệt gì cả. Nó có 20 tầng, hơn 300 nghìn tham số, chạy 8 epoch không lỗi, và học được **đúng bằng không**.

Điều đáng nói: đây không phải overfitting (train loss cũng 2,3028), không phải learning rate sai (5 tầng cùng learning rate ấy chạy tốt), không phải thiếu dữ liệu (vẫn 60.000 mẫu). Chỉ riêng việc *sâu hơn* đã đủ giết nó.

## 2. Truy nguyên: tín hiệu co lại qua từng tầng

Bài 04 đã đo dòng gradient chảy ngược. Làm lại trên mạng 20 tầng:

| Cấu hình | Tầng 1 | Tầng 10 | Tầng cuối |
|---|---|---|---|
| ReLU, khởi tạo mặc định | **3,3 · 10⁻⁸** | 1,2 · 10⁻⁶ | 3,9 · 10⁻² |
| Sigmoid, khởi tạo mặc định | **2,3 · 10⁻¹⁷** | 2,4 · 10⁻¹⁰ | 6,1 · 10⁻¹ |

Với sigmoid, gradient tới tầng đầu là $2{,}3 \times 10^{-17}$. Con số ấy vẫn tồn tại và số thực 32 bit vẫn biểu diễn được nó - chỗ hỏng nằm ở bước sau. Bước cập nhật là $\eta g$; với learning rate 0,1 thì nó bằng $2{,}3 \times 10^{-18}$, trong khi một trọng số có độ lớn thông thường, chẳng hạn 0,05, có **độ phân giải cục bộ** của số thực 32 bit vào khoảng $3{,}7 \times 10^{-9}$ - lớn hơn cả tỉ lần. Cộng một lượng nhỏ hơn khoảng cách giữa hai số biểu diễn được liền nhau thì kết quả làm tròn về đúng chỗ cũ: trọng số **không hề đổi**. Nên tầng đầu không phải "học chậm", nó gần như không học. Đây là **vanishing gradient (gradient tiêu biến)**.

Cơ chế là phép nhân dồn của chain rule (bài 04). Gradient tới tầng $k$ là tích của các đạo hàm từ tầng cuối ngược về:

$$\frac{\partial L}{\partial W_k} \sim \frac{\partial L}{\partial \text{ra}} \times \underbrace{f'(z_n) \cdot W_n \times \dots \times f'(z_{k+1}) \cdot W_{k+1}}_{\text{càng sâu, càng nhiều thừa số}}$$

Nhân 20 con số lại với nhau thì kết quả rất nhạy: mỗi số chỉ cần hơi nhỏ hơn 1 là tích lao về 0; hơi lớn hơn 1 là tích nổ tung (**exploding gradient**, thường hiện ra dưới dạng loss thành `nan`).

Nhưng nói "đạo hàm nhỏ hơn 1" mới là nửa câu chuyện, vì công thức trên còn có $W$. Hãy đo phía *xuôi* thay vì phía ngược - đưa một batch qua mạng 20 tầng chưa train, xem độ lệch chuẩn của kích hoạt sau mỗi tầng:

| Khởi tạo | std sau tầng 1 | tầng 10 | tầng 20 |
|---|---|---|---|
| Mặc định của PyTorch | 0,1642 | 0,0272 | 0,0314 |
| Xavier / Glorot | 0,3076 | 0,0390 | 0,0506 |
| **He / Kaiming** | 0,3318 | **0,2877** | **0,2940** |

Đây mới là gốc rễ. Với khởi tạo mặc định, tín hiệu co từ 0,16 xuống 0,027 chỉ sau 10 tầng - mỗi tầng nhân với một hệ số nhỏ hơn 1, và 20 lần như thế thì còn lại rất ít. Tín hiệu xuôi đã tắt thì gradient ngược cũng tắt theo.

**He / Kaiming** giữ được độ lệch chuẩn gần như không đổi qua cả 20 tầng (0,33 → 0,29). Đó chính là thứ nó được thiết kế để làm.

## 3. Cách chữa thứ nhất: chọn đúng khởi tạo

Ý tưởng của cả Xavier lẫn He là chọn phương sai ban đầu của trọng số sao cho tín hiệu **không co cũng không nở** khi đi qua tầng. Khác biệt giữa hai cái: Xavier tính cho các hàm đối xứng quanh 0 như tanh, còn He tính riêng cho ReLU - vì ReLU vứt bỏ nửa âm, nó cần trọng số lớn hơn để bù lại đúng phần đã mất.

```python
for layer in model:
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
```

Mạng 20 tầng ReLU, 8 epoch, thử nhiều learning rate:

| Khởi tạo | lr = 0,1 | lr = 0,03 | lr = 0,01 |
|---|---|---|---|
| Mặc định PyTorch | 0,1000 | - | - |
| N(0; 1) - quá lớn | `nan` | - | - |
| N(0; 0,01) - quá nhỏ | 0,1000 | - | - |
| Xavier / Glorot | 0,5358 | 0,3163 | 0,1000 |
| **He / Kaiming** | `nan` | **0,8419** | 0,8237 |

Từ 0,1000 lên **0,8419** - cùng kiến trúc, cùng dữ liệu, cùng thuật toán tối ưu. Thứ duy nhất thay đổi là các con số ngẫu nhiên lúc khởi tạo.

Hai dòng đáng đọc kỹ. **N(0; 1) cho `nan`** - trọng số quá lớn khiến tín hiệu nở ra qua từng tầng cho tới khi tràn số: đó là exploding gradient. **N(0; 0,01) cho 0,1000** - trọng số quá nhỏ, tín hiệu tắt: vanishing. Hai đầu của cùng một trục, và cả Xavier lẫn He đều là cách tính điểm cân bằng ở giữa.

> ⚠️ **Lưu ý nhỏ:** để ý ô **He với lr = 0,1 cũng cho `nan`**, trong khi lr = 0,03 lại cho kết quả tốt nhất bảng. Đừng đọc thành "He không ổn định". Khởi tạo và learning rate tác động lẫn nhau: He cố ý giữ tín hiệu ở mức lớn hơn qua các tầng, nên gradient tới các tầng đầu cũng lớn hơn, và bước nhảy lr = 0,1 trở thành quá đà. Khi đổi cách khởi tạo, hãy dò lại learning rate - đây là một trong những cặp siêu tham số ràng buộc nhau chặt nhất trong deep learning.

## 4. Cách chữa thứ hai: chuẩn hóa ngay giữa mạng

Khởi tạo chỉ lo được lúc *bắt đầu*. Sau vài trăm bước cập nhật, trọng số đã đổi và thống kê tín hiệu trong mạng lại trôi đi. **Batch normalization** ép lại thống kê ấy về chỗ cũ sau mỗi tầng, trong suốt quá trình train.

Với mỗi lô dữ liệu, tại mỗi tầng, nó lấy đầu ra rồi trừ trung bình và chia độ lệch chuẩn *của chính lô đó* - đúng phép `StandardScaler` của chương Machine Learning · Bài 11, nhưng áp ở giữa mạng thay vì ở đầu vào, và tính lại cho từng lô. Sau đó nhân thêm hai tham số học được ($\gamma$, $\beta$) để mạng vẫn có quyền chọn thang giá trị khác nếu cần.

> ⚠️ **Lưu ý nhỏ:** *vì sao* BatchNorm giúp thì đến giờ vẫn còn tranh luận. Bài báo gốc năm 2015 giải thích bằng khái niệm "internal covariate shift" - phân bố đầu vào của mỗi tầng cứ trôi khi các tầng trước thay đổi. Santurkar và cộng sự (2018) đo lại và thấy lời giải thích ấy không đứng vững: họ cố tình bơm nhiễu để làm phân bố trôi mạnh hơn mà mạng có BatchNorm vẫn train tốt, và cho rằng cái BatchNorm thực sự mang lại là bề mặt loss trơn hơn, nhờ đó chịu được learning rate lớn hơn. Kết quả thực nghiệm ở bảng dưới thì không ai cãi; chỉ có lời giải thích cơ chế là chưa ngã ngũ. Cả hai bài đều có trong phần Đọc thêm.

```python
nn.Sequential(nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), ...)
```

| Số tầng | Không BatchNorm | Có BatchNorm |
|---|---|---|
| 5 tầng | 0,8166 | **0,8790** |
| **20 tầng** | **0,1000** | **0,8300** |

Mạng 20 tầng từ chỗ hoàn toàn không học được nhảy lên 0,8300. Nhìn lại bảng gradient ở mục 2 thì hiểu vì sao:

| Cấu hình | Tầng 1 | Tầng 10 | Tầng cuối |
|---|---|---|---|
| ReLU thường | 3,3 · 10⁻⁸ | 1,2 · 10⁻⁶ | 3,9 · 10⁻² |
| **ReLU + BatchNorm** | **50,0** | **2,95** | **0,64** |

Mọi tầng giờ đều nhận gradient trong khoảng dùng được. Chú ý thứ tự cũng đảo ngược - tầng đầu có gradient *lớn nhất* thay vì nhỏ nhất - nhưng cả ba đều nằm trong vài bậc của nhau, chứ không phải cách nhau một triệu lần như trước.

Và để ý dòng 5 tầng: BatchNorm nâng 0,8166 lên 0,8790 dù mạng ấy vốn đã học được bình thường. Nó không chỉ là phao cứu sinh cho mạng sâu - nó thường giúp cả những mạng đang khỏe, vì cho phép dùng learning rate lớn hơn và làm quá trình huấn luyện bớt nhạy với khởi tạo.

> ⚠️ **Lưu ý nhỏ:** BatchNorm là tầng hành xử khác nhau giữa hai chế độ, đúng như bài 05 cảnh báo. Lúc train nó dùng trung bình và độ lệch chuẩn của **lô hiện tại**, đồng thời cập nhật dần một bộ thống kê tích lũy. Lúc `eval()` nó **thôi không dùng thống kê của lô đang dự đoán nữa** mà dùng bộ tích lũy ấy - không phải vì lúc dự đoán không có lô (dự đoán theo lô 32 hay 128 ảnh là chuyện bình thường), mà để kết quả của một mẫu không phụ thuộc vào những mẫu tình cờ đi cùng lô với nó. Quên `model.eval()` với BatchNorm vì thế còn tai hại hơn quên với dropout: cùng một ảnh sẽ cho kết quả khác nhau tùy nó nằm cùng lô với ai. Và vì lúc train nó dựa vào thống kê của lô, lô quá nhỏ làm thống kê rất nhiễu và BatchNorm kém ổn định hẳn - "quá nhỏ" là bao nhiêu thì tùy bài toán và kiến trúc, chứ không có ngưỡng chung. Khi ấy người ta dùng các biến thể như LayerNorm, vốn cũng là chuẩn hóa được dùng trong Transformer mà chương Generative AI sẽ gặp.

> 🔧 **Thử ngay:** cả khởi tạo He lẫn BatchNorm đều cứu được mạng 20 tầng. Nếu chỉ được chọn một cho một dự án mới, bạn chọn cái nào?
> Trong thực tế người ta dùng **cả hai**, vì chúng lo hai giai đoạn khác nhau: khởi tạo lo lúc xuất phát, BatchNorm lo suốt hành trình khi trọng số đã trôi. Nếu ép chọn một, BatchNorm bền hơn - nó tự chuẩn hóa lại liên tục nên tha thứ cho cả khởi tạo tồi lẫn learning rate hơi lệch, đúng như bảng 5 tầng cho thấy nó giúp cả khi mạng vốn đã ổn. Nhưng nó tốn thêm tính toán và thêm một chỗ dễ sai (`model.eval()`), còn khởi tạo thì miễn phí hoàn toàn.

## 5. Còn một mảnh nữa, để dành cho bài sau

Ba cách chữa ở trên đủ để đưa mạng từ 20 tầng lên chạy được. Nhưng chúng chưa đủ cho những mạng thật sự sâu - loại 50, 100 hay 150 tầng.

Cách chữa cho tầm đó là **kết nối tắt (skip connection)**: cho tín hiệu một đường đi thẳng vượt qua vài tầng, thay vì bắt buộc phải chui qua từng tầng một. Khi ấy gradient có một con đường ngắn để chảy ngược về các tầng đầu mà không bị nhân dồn qua hàng chục thừa số. Đó là ý tưởng của kiến trúc **ResNet**, và bài 09 sẽ gặp nó khi bàn về mạng cho ảnh.

## Tóm tắt bài học

- Mạng 20 tầng ReLU với khởi tạo mặc định cho train loss 2,3028 và accuracy 0,1000 - đúng mức đoán bừa. Sâu hơn, tự nó, đủ để giết một mạng.
- Nguyên nhân là phép nhân dồn của chain rule: gradient tới tầng $k$ là tích của hàng chục thừa số, nên hơi nhỏ hơn 1 thì tiêu biến, hơi lớn hơn 1 thì nổ thành `nan`.
- Gốc rễ đo được ở lượt xuôi: với khởi tạo mặc định, độ lệch chuẩn của kích hoạt co từ 0,16 xuống 0,027 sau 10 tầng; với He/Kaiming nó giữ nguyên 0,33 → 0,29 qua cả 20 tầng.
- Chỉ đổi cách khởi tạo, mạng 20 tầng đi từ accuracy 0,1000 lên **0,8419**. Khởi tạo quá lớn cho `nan`, quá nhỏ cho 0,1000 - Xavier và He là cách tính điểm cân bằng.
- Khởi tạo và learning rate ràng buộc nhau: He cho `nan` ở lr = 0,1 nhưng tốt nhất bảng ở lr = 0,03. Đổi khởi tạo thì phải dò lại learning rate.
- **BatchNorm** chuẩn hóa lại tín hiệu ngay giữa mạng theo từng lô: mạng 20 tầng từ 0,1000 lên 0,8300, và mạng 5 tầng vốn đã ổn cũng lên từ 0,8166 lên 0,8790.
- BatchNorm hành xử khác nhau giữa `train()` và `eval()`, và mất tác dụng khi batch quá nhỏ.

## Câu hỏi tự kiểm tra

1. Mạng của bạn có train loss đứng yên ở 2,30 trên bài toán 10 lớp và nó có 30 tầng. Nêu chẩn đoán và hai thứ bạn thử đầu tiên.
2. Vì sao "đạo hàm nhỏ hơn 1" chỉ là nửa lời giải thích cho vanishing gradient? Nửa còn lại nằm ở đâu, và bảng nào trong bài đo nó?
3. Khởi tạo N(0; 1) cho `nan` còn N(0; 0,01) cho accuracy 0,1000. Hai lỗi này ngược nhau ở chỗ nào?
4. He/Kaiming dùng phương sai lớn hơn Xavier. Vì sao ReLU cần trọng số lớn hơn tanh?
5. Bạn đổi từ khởi tạo mặc định sang He và mô hình lập tức ra `nan`. Đây có phải bằng chứng He tệ hơn không? Bạn làm gì tiếp?
6. Vì sao quên `model.eval()` với BatchNorm còn nguy hiểm hơn với dropout? Mô tả cụ thể chuyện xảy ra với một dự đoán đơn lẻ.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 5.4 *Numerical Stability and Initialization* - vanishing/exploding gradient và khởi tạo Xavier, đúng nội dung mục 2-3; chương 8.5 *Batch Normalization* |
| **Foundations · Bài 09** | Chain rule - phép nhân dồn ở mục 2 là chính nó, lặp qua 20 tầng |
| **Chương Machine Learning · Bài 11** | `StandardScaler` - BatchNorm là phép đó áp ở giữa mạng |

**Nguồn online bổ sung (miễn phí):**

- [Glorot & Bengio - Understanding the difficulty of training deep feedforward neural networks (AISTATS 2010)](https://proceedings.mlr.press/v9/glorot10a.html) - bài báo Xavier, có chính các biểu đồ độ lệch chuẩn theo tầng như bảng ở mục 2.
- [He và cộng sự - Delving Deep into Rectifiers (ICCV 2015)](https://arxiv.org/abs/1502.01852) - bài báo He/Kaiming, giải thích vì sao ReLU cần hệ số khác Xavier.
- [Ioffe & Szegedy - Batch Normalization (ICML 2015)](https://arxiv.org/abs/1502.03167) - bài báo gốc của BatchNorm.
- [Santurkar và cộng sự - How Does Batch Normalization Help Optimization? (NeurIPS 2018)](https://arxiv.org/abs/1805.11604) - bài phản biện cho thấy lời giải thích ban đầu của Ioffe & Szegedy chưa đúng, một ví dụ hay về việc kỹ thuật chạy được trước khi được hiểu đúng.
- [`torch.nn.init` - tài liệu](https://pytorch.org/docs/stable/nn.init.html) - mọi cách khởi tạo có sẵn.

> **Bài tiếp theo:** [CNN: mạng học cách nhìn](../cnn-networks-that-learn-to-see-vi/) - tám bài qua đều duỗi ảnh 28×28 thành vector 784 và ném hết vào `nn.Linear` - tức vứt bỏ thông tin pixel nào nằm cạnh pixel nào. Bài sau dùng một kiến trúc biết rằng đầu vào là ảnh, và mức tăng lần này không còn khiêm tốn nữa.

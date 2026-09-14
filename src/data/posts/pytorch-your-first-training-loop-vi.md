---
title: "PyTorch: vòng lặp huấn luyện đầu tiên"
description: "Viết một chương trình huấn luyện mạng nơ-ron chạy trọn trên dữ liệu thật, năm dòng lệnh làm nên vòng lặp huấn luyện, và hai chế độ học với dự đoán mà người mới hay lẫn."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "pytorch-your-first-training-loop"
order: 5
pubDate: 2026-08-21
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ viết được một chương trình huấn luyện mạng nơ-ron chạy từ đầu đến cuối trên dữ liệu thật, hiểu năm dòng lệnh làm nên vòng lặp huấn luyện, và phân biệt rõ hai chế độ mà người mới hay lẫn - lúc học và lúc dự đoán.

## 1. Bốn mảnh ghép, và chúng đều là bài cũ

Bốn bài vừa rồi đã dựng đủ mọi thứ, chỉ chưa ráp lại. Một chương trình huấn luyện có đúng bốn mảnh:

```mermaid
flowchart LR
    D["Dữ liệu<br/>Dataset + DataLoader"] --> M["Mô hình<br/>nn.Module"]
    M --> L["Hàm mất mát<br/>đo sai bao nhiêu"]
    L --> O["Thuật toán tối ưu<br/>chỉnh núm vặn"]
    O -->|"lặp lại"| D
```

Ba mảnh cuối bạn đã gặp: **mô hình** là các tầng nơ-ron của bài 02-03, **hàm mất mát** là cross-entropy của Foundations · Bài 10, **thuật toán tối ưu** là gradient descent của Foundations · Bài 11. Mảnh còn lại là cách đưa dữ liệu vào theo từng lô.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

train_set = datasets.FashionMNIST("data", train=True,  download=True,
                                  transform=transforms.ToTensor())
val_set   = datasets.FashionMNIST("data", train=False, download=True,
                                  transform=transforms.ToTensor())
train_dl = DataLoader(train_set, batch_size=256, shuffle=True)
val_dl   = DataLoader(val_set,   batch_size=1000)
```

`Dataset` biết cách lấy ra mẫu thứ $i$; `DataLoader` gom chúng thành từng lô và xáo trộn mỗi epoch. Với 60.000 ảnh và batch 256, mỗi epoch có **235 lô**, mỗi lô là một tensor `(256, 1, 28, 28)` - 256 ảnh, 1 kênh màu (ảnh xám), 28×28 pixel.

> ⚠️ **Vì sao 10.000 ảnh kia gọi là validation.** `train=False` trả về tập mà FashionMNIST đặt tên là *test*. Nhưng tên một tệp không quyết định vai trò của nó - **cách bạn dùng nó** mới quyết định. Suốt bài 01 đến bài 11, ta liên tục nhìn 10.000 ảnh này để chẩn đoán, so kiến trúc, chọn learning rate, xem kỹ thuật nào ăn tiền. Đó đúng là vai trò của một tập **validation**, nên loạt bài gọi nó bằng đúng tên ấy.
>
> Hệ quả đi kèm phải nói ngay: một tập đã bị nhìn nhiều lần như vậy thì **không còn dùng làm con số báo cáo cuối cùng được nữa** - nó đã mòn, theo đúng nghĩa Foundations · Bài 04 mô tả. Vì thế bài 12, khi làm project thật, bắt đầu lại từ dữ liệu gốc: nó không đụng tới 10.000 ảnh này mà tự chia ba tập của riêng nó, và giữ một phần làm test cho tới bước đánh giá cuối cùng.

Chữ `shuffle=True` chỉ đặt cho tập train, không đặt cho validation. Lý do là chương Machine Learning · Bài 07 đã nói khi bàn về bootstrap: thứ tự dữ liệu ảnh hưởng tới đường đi của gradient descent, nên xáo trộn mỗi epoch giúp các lô khác nhau giữa các vòng. Tập validation chỉ được đọc để chấm điểm, thứ tự không quan trọng.

```python
model = nn.Sequential(nn.Flatten(),
                      nn.Linear(784, 256), nn.ReLU(),
                      nn.Linear(256, 10))          # 203.530 tham số
lossf = nn.CrossEntropyLoss()
opt   = torch.optim.SGD(model.parameters(), lr=0.1)
```

`nn.Flatten()` duỗi ảnh 28×28 thành vector 784 chiều. Tầng cuối là `nn.Linear` trần, không có softmax - đúng cảnh báo ở bài 02, vì `CrossEntropyLoss` đã chứa softmax bên trong.

## 2. Năm dòng làm nên việc học

Đây là trái tim của mọi chương trình deep learning, và nó ngắn đến mức đáng ngạc nhiên:

```python
for epoch in range(1, 11):
    model.train()
    for xb, yb in train_dl:
        opt.zero_grad()            # 1. xóa gradient của vòng trước
        loss = lossf(model(xb), yb)   # 2. lượt xuôi: dự đoán rồi đo sai
        loss.backward()            # 3. lượt ngược: tính gradient mọi tham số
        opt.step()                 # 4. chỉnh núm vặn theo gradient
```

Bốn dòng, lặp 235 lần mỗi epoch. Đọc lại chúng theo ngôn ngữ các bài trước:

| Dòng | Đang làm gì | Bài đã học |
|---|---|---|
| `opt.zero_grad()` | Xóa `.grad` cũ, vì PyTorch cộng dồn gradient | Bài 04 |
| `model(xb)` | Lượt xuôi, vừa tính vừa dựng đồ thị tính toán | Bài 04 |
| `lossf(..., yb)` | Đo khoảng cách giữa dự đoán và nhãn | Foundations · Bài 10 |
| `loss.backward()` | Đi ngược đồ thị, tính gradient cho cả 203.530 tham số | Bài 04 |
| `opt.step()` | Mỗi tham số bước một bước ngược hướng gradient | Foundations · Bài 11 |

Không có gì mới ở đây. Cái mới duy nhất là **thứ tự** - và thứ tự ấy sai một chỗ là hỏng cả. Quên `zero_grad()` thì gradient các lô cộng dồn lên nhau; đặt `backward()` sau `step()` thì bạn cập nhật bằng gradient của lô trước.

Chạy thật, 10 epoch, mỗi epoch khoảng 3,6 giây trên CPU:

> ⚠️ **Về các con số thời gian trong cả chương.** Mọi mốc giây/phút trong loạt bài này đo trên một máy 8 nhân CPU, 15 GB RAM, **không có GPU**, với PyTorch 2.14 chạy 4 luồng. Máy bạn gần như chắc chắn ra con số khác - nhanh hơn hay chậm hơn vài lần đều bình thường, và có GPU thì chênh lệch còn lớn hơn nhiều. Cái đáng mang theo là **tỉ lệ giữa các dòng trong cùng một bảng** (mô hình này đắt gấp mấy lần mô hình kia), vì tỉ lệ ấy khá ổn định giữa các máy; con số tuyệt đối thì không. Các con số accuracy thì ngược lại - chúng lặp lại được, miễn bạn giữ nguyên seed và phiên bản thư viện.

| Epoch | 1 | 3 | 5 | 7 | 10 |
|---|---|---|---|---|---|
| train loss | 0,8654 | 0,4960 | 0,4416 | 0,4071 | 0,3753 |
| validation accuracy | 0,7877 | 0,8212 | 0,8318 | 0,8222 | **0,8528** |

Train loss giảm đều đặn - mô hình đang học. Validation accuracy thì nhấp nhô: 0,8318 ở epoch 5, tụt xuống 0,8222 ở epoch 7, rồi lên 0,8528 ở epoch 10. Sự nhấp nhô ấy là bình thường và bài 06 dành cả bài để đọc nó.

> 🔧 **Thử ngay:** trong bảng trên, epoch 7 có train loss *thấp hơn* epoch 5 nhưng validation accuracy lại *kém hơn*. Đó có phải dấu hiệu overfitting không?
> Chưa kết luận được. Overfitting là khi train loss tiếp tục giảm còn chất lượng trên dữ liệu chưa thấy **xấu đi có xu hướng** (Foundations · Bài 14). Ở đây epoch 10 lại tốt nhất bảng, nên một lần tụt ở epoch 7 chỉ là dao động - gradient descent với mini-batch vốn nhiễu, và mỗi epoch mô hình dừng ở một chỗ hơi khác trên địa hình loss. Muốn kết luận phải nhìn xu hướng qua nhiều epoch, không nhìn một điểm.

## 3. Hai chế độ: lúc học và lúc dự đoán

Đây là chỗ người mới vấp nhiều nhất, nên tách riêng.

Một mô hình đã train sống trong hai chế độ khác nhau:

```
   HUẤN LUYỆN (training)                 SUY LUẬN (inference)
   dữ liệu → mô hình → loss              dữ liệu → mô hình → dự đoán
                        ↓                              ↑
                    backward                      chỉ có lượt xuôi
                        ↓                      không loss, không gradient
                 cập nhật tham số                 tham số đứng yên
```

Nghe hiển nhiên, nhưng có một chi tiết không hiển nhiên chút nào: **một số tầng hành xử khác nhau ở hai chế độ**. Dropout (bài 07) tắt ngẫu nhiên một phần nơ-ron *khi train* nhưng phải dùng đủ nơ-ron *khi dự đoán*. Batch normalization (bài 08) dùng thống kê của lô hiện tại khi train nhưng dùng thống kê trung bình đã tích lũy khi dự đoán.

PyTorch không tự đoán được bạn đang ở chế độ nào. Bạn phải nói:

```python
model.train()   # bật chế độ huấn luyện
model.eval()    # bật chế độ suy luận
```

Hậu quả của việc quên đo được bằng số. Lấy một mạng có `nn.Dropout(0.5)`, train 3 epoch, rồi đưa **cùng một tấm ảnh** qua nhiều lần:

```
chế độ train():  logit 3 lớp đầu = [-3.030  -3.696  -2.315]
                 logit 3 lớp đầu = [-3.305  -6.657  -3.564]   ← khác nhau!
                 logit 3 lớp đầu = [-2.634  -3.875  -2.169]

chế độ eval():   logit 3 lớp đầu = [-3.283  -3.785  -2.854]
                 logit 3 lớp đầu = [-3.283  -3.785  -2.854]   ← luôn giống
```

Ở chế độ `train()`, cùng một ảnh cho ra kết quả khác nhau mỗi lần, vì dropout bốc ngẫu nhiên nơ-ron nào bị tắt. Một hệ thống dự đoán mà đưa cùng đầu vào lại trả lời khác nhau thì không dùng được.

Đo trên cả tập validation:

| Chế độ khi đánh giá | Accuracy |
|---|---|
| `model.train()` (quên đổi) | 0,7965 |
| `model.eval()` (đúng) | **0,8100** |

Mất 1,35 điểm chỉ vì thiếu một dòng. Và đây là kiểu lỗi tệ nhất: chương trình chạy trơn tru, không báo lỗi, chỉ có kết quả kém hơn thực lực mà bạn không biết vì sao.

> ⚠️ **Lưu ý nhỏ:** `model.eval()` và `torch.no_grad()` là **hai việc khác nhau**, hay bị lẫn. `model.eval()` đổi *hành vi của các tầng* như dropout và batch norm. `torch.no_grad()` bảo PyTorch *đừng dựng đồ thị tính toán*, tiết kiệm bộ nhớ (bài 04). Khi đánh giá thì cần cả hai, và chúng không thay thế cho nhau: bật `no_grad()` mà quên `eval()` thì dropout vẫn hoạt động.

## 4. Hàm đánh giá, viết một lần dùng cả chương

Gộp cả hai thứ trên vào một hàm dùng lại được cho mọi bài sau:

```python
@torch.no_grad()                       # không dựng đồ thị
def evaluate(model, dl):
    model.eval()                       # đổi hành vi dropout / batch norm
    correct = total = 0
    total_loss = 0.0
    for xb, yb in dl:
        out = model(xb)
        total_loss += lossf(out, yb).item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        total += len(yb)
    return total_loss / total, correct / total
```

Ba chi tiết đáng chú ý:

- `out.argmax(1)` lấy chỉ số lớp có điểm cao nhất. Không cần softmax - softmax không đổi thứ tự, nên lớp có logit lớn nhất cũng là lớp có xác suất lớn nhất. Chỉ khi cần *xác suất* để báo cáo hay để chọn ngưỡng (chương Machine Learning · Bài 12) thì mới áp softmax.
- Nhân `len(yb)` khi cộng loss vì lô cuối thường ngắn hơn các lô khác; chia tổng cho tổng số mẫu mới ra trung bình đúng.
- `.item()` lấy con số Python ra khỏi tensor. Nếu cộng dồn tensor mà quên `.item()`, bạn vô tình giữ cả đồ thị tính toán của mọi lô trong bộ nhớ - một cách rất hiệu quả để hết RAM.

## Tóm tắt bài học

- Một chương trình huấn luyện có bốn mảnh: dữ liệu (`Dataset` + `DataLoader`), mô hình (`nn.Module`), hàm mất mát, thuật toán tối ưu. Ba mảnh cuối đã học ở Foundations và các bài trước.
- Vòng lặp huấn luyện là bốn dòng lặp lại: `zero_grad()` → lượt xuôi và tính loss → `backward()` → `step()`. Sai thứ tự hoặc thiếu một dòng là hỏng, thường không kèm thông báo lỗi.
- FashionMNIST với batch 256 cho 235 lô mỗi epoch, mỗi epoch khoảng 3,6 giây trên CPU; sau 10 epoch mạng một tầng ẩn đạt accuracy 0,8528.
- Validation accuracy nhấp nhô giữa các epoch là bình thường với mini-batch; phải nhìn xu hướng nhiều epoch mới kết luận được điều gì.
- Huấn luyện và suy luận là hai chế độ khác nhau, và một số tầng hành xử khác nhau ở hai chế độ. Quên `model.eval()` khiến cùng một ảnh cho kết quả khác nhau mỗi lần, và làm accuracy tụt từ 0,8100 xuống 0,7965.
- `model.eval()` đổi hành vi của tầng; `torch.no_grad()` tiết kiệm bộ nhớ. Cần cả hai khi đánh giá, và không thay thế cho nhau.

## Câu hỏi tự kiểm tra

1. Viết lại bốn dòng của vòng lặp huấn luyện theo trí nhớ, kèm một câu giải thích mỗi dòng.
2. Chuyện gì xảy ra nếu bạn đổi thứ tự thành `loss.backward()` rồi `opt.zero_grad()` rồi `opt.step()`?
3. Vì sao đặt `shuffle=True` cho tập train mà không cho tập validation?
4. Mạng của bạn có dropout. Bạn quên gọi `model.eval()` trước khi đánh giá. Nêu hai triệu chứng bạn sẽ quan sát thấy.
5. `model.eval()` và `torch.no_grad()` khác nhau thế nào? Dùng một cái mà quên cái kia thì mất gì?
6. Trong hàm `evaluate`, vì sao phải nhân `len(yb)` khi cộng dồn loss thay vì lấy trung bình đơn giản của các lô?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 3.4 - vòng lặp huấn luyện viết từ đầu; chương 4.2 *The Image Classification Dataset* - chính bộ FashionMNIST dùng trong bài; chương 5.2 - cài đặt MLP gọn bằng API cấp cao |
| **Foundations · Bài 10, 11** | Hàm mất mát và các thuật toán tối ưu - hai trong bốn mảnh ở mục 1 |
| **Chương Machine Learning · Bài 01** | Quy trình một dự án: nêu bài toán, chia dữ liệu, so với baseline - vẫn đúng nguyên với deep learning |

**Nguồn online bổ sung (miễn phí):**

- [PyTorch - Quickstart](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) - hướng dẫn chính thức chạy đúng bộ FashionMNIST của bài này.
- [`torch.utils.data` - tài liệu](https://pytorch.org/docs/stable/data.html) - `Dataset`, `DataLoader`, `num_workers` và cách viết Dataset riêng.
- [`Module.train` / `Module.eval` - tài liệu](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.eval) - danh sách các tầng thực sự đổi hành vi giữa hai chế độ.
- [PyTorch Recipes - Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) - `state_dict`, và vì sao nạp mô hình xong phải gọi `eval()`.

> **Bài tiếp theo:** [Đọc đường cong huấn luyện](../reading-training-curves-vi/) - bài này train xong và nhìn con số cuối. Bài sau nhìn cả đường đi - sáu lần chạy với sáu chứng bệnh khác nhau, và cách chẩn đoán từng cái chỉ bằng hai đường cong.

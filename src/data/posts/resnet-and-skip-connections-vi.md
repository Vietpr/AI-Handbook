---
title: "ResNet và kết nối tắt"
description: "Cách kết nối tắt giúp huấn luyện mạng sâu, cách phân biệt suy giảm do độ sâu với overfitting, và một thí nghiệm nhỏ tái hiện được gì và bỏ lỡ gì."
domain: "Computer Vision"
section: "Learn"
language: "vi"
translationKey: "resnet-and-skip-connections"
order: 3
pubDate: 2026-08-26
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ phân biệt được **suy giảm do độ sâu** với overfitting bằng đúng một cột số liệu, hiểu kết nối tắt đổi bài toán của mỗi khối thành bài toán gì, và thấy một thí nghiệm tái hiện được nửa hiện tượng còn nửa kia thì không - cùng lý do vì sao.

## 1. Món nợ của chương Deep Learning

Chương Deep Learning · Bài 08 đã cho thấy mạng 20 tầng kết nối đầy đủ với khởi tạo mặc định cho accuracy 0,1000 - đúng mức đoán bừa - và chữa được bằng khởi tạo He hoặc BatchNorm. Bài ấy có nhắc **kết nối tắt** như cách chữa thứ ba, nhưng chưa đo nó lần nào, và cũng chưa đo trên mạng tích chập.

Bài này trả món nợ đó, vì có một câu hỏi chưa được trả lời: **sau khi đã có khởi tạo tốt và BatchNorm, xếp sâu thêm nữa thì sao?**

Bài 02 dừng ở VGG-16 với 13 tầng tích chập. Câu trả lời tự nhiên là cứ xếp tiếp - 20, 34, 50 tầng. Nhưng chuyện xảy ra khi làm vậy đã làm cả ngành bất ngờ.

## 2. Suy giảm không phải overfitting

Thí nghiệm được thiết kế để chỉ đổi **đúng một thứ**. Hai mạng có cùng số khối, cùng số kênh, cùng BatchNorm, cùng SGD với momentum 0,9 và lịch cosine, cùng 6 epoch, cùng dữ liệu Imagenette 32×32. Khác biệt duy nhất: một bên có phép cộng `y = F(x) + x` ở cuối mỗi khối, một bên không.

Ngân sách tham số gần như bằng nhau - kết nối tắt chỉ thêm khoảng 0,1% tham số cho các phép chiếu đổi kích thước:

| Mạng | Tham số | Train loss | Accuracy |
|---|---|---|---|
| Phẳng-20 | 269.722 | 1,4001 | 0,5246 |
| **ResNet-20** | 272.474 | **1,1671** | **0,5717** |
| Phẳng-44 | 658.586 | 2,1247 | **0,2003** |
| **ResNet-44** | 661.338 | **1,7607** | **0,3740** |

Cột đáng nhìn trước tiên **không phải** accuracy mà là **train loss**, và bài 06 của chương Deep Learning đã dạy đúng thứ tự này: luôn đọc train loss trước.

**Mạng phẳng 44 tầng có train loss 2,1247.** Với 10 lớp, đoán đều cho $\ln(10) = 2{,}3026$. Mạng ấy gần như không học nổi **dữ liệu nó đã nhìn thấy**. Đây là chẩn đoán quyết định: nếu là overfitting thì train loss phải *thấp* còn accuracy mới tệ. Ở đây cả hai cùng tệ.

Nên hiện tượng này **không phải overfitting**, mà là **suy giảm do độ sâu (degradation)**: mạng sâu hơn khó *tối ưu* hơn, chứ không phải học tủ nhiều hơn.

**Và điều đó rất khó chấp nhận về mặt lý thuyết.** Mạng 44 tầng chứa mạng 20 tầng như một trường hợp riêng - chỉ cần 24 khối cuối học đúng hàm đồng nhất là hai mạng cho kết quả y hệt. Nghĩa là mạng 44 tầng **về nguyên tắc không thể tệ hơn**. Nhưng gradient descent không tìm ra được lời giải ấy: từ 0,5246 xuống 0,2003, tệ đi 32,4 điểm dù có nhiều hơn 2,4 lần tham số.

**Kết nối tắt giúp ở cả hai độ sâu, và giúp đúng chỗ.** Train loss thấp hơn ở cả 20 tầng (1,1671 so với 1,4001) lẫn 44 tầng (1,7607 so với 2,1247). Ở độ sâu 44, khoảng cách accuracy là **17,4 điểm** cho hai mạng chênh nhau 0,1% tham số.

## 3. Chỗ thí nghiệm này không tái hiện được

Bảng trên có một chi tiết đi ngược bài báo ResNet, và cần nói ra thay vì lờ đi: **ResNet-44 cũng tệ hơn ResNet-20** (0,3740 so với 0,5717).

Bài báo gốc cho thấy ResNet càng sâu càng tốt - 20, 32, 44, 56 tầng cải thiện đều. Thí nghiệm ở đây không ra được như vậy.

Giả thuyết hợp lý nhất là **ngân sách huấn luyện**. Bài báo huấn luyện 64.000 bước trên CIFAR-10; ở đây là **6 epoch** trên máy không có GPU, tức khoảng 450 bước. Mạng 44 tầng có nhiều tham số hơn và cần nhiều bước hơn để hội tụ, nên ở mốc 6 epoch nó vẫn còn đang đi - train loss 1,7607 còn rất cao, đúng dấu hiệu "chưa xong" mà chương Deep Learning · Bài 06 mô tả.

Nhưng đó vẫn là giả thuyết: muốn chứng minh thì phải chạy lại với ngân sách lớn hơn và xem khoảng cách có đảo chiều không - việc mà máy ở đây không kham nổi. Nên kết luận trung thực của mục 2 phải viết chính xác thế này:

- ✅ **Tái hiện được:** mạng phẳng suy giảm mạnh khi xếp sâu, và đó là vấn đề tối ưu chứ không phải overfitting
- ✅ **Tái hiện được:** kết nối tắt cải thiện cả train loss lẫn accuracy ở cùng ngân sách tham số
- ❌ **Không tái hiện được:** ResNet sâu hơn thì tốt hơn - giả thuyết là do ngân sách huấn luyện quá ngắn, nhưng chưa kiểm chứng được

> ⚠️ **Lưu ý nhỏ:** mọi con số trong bài là **một lần chạy minh họa** với một seed. Điều đó đủ cho những khoảng cách lớn - 32,4 điểm giữa Phẳng-20 và Phẳng-44 thì không cách nào là nhiễu - nhưng không đủ để nói chắc về những chênh lệch nhỏ. Muốn chắc thì phải chạy vài seed rồi so trung bình, đúng như chương Deep Learning · Bài 12 đã nói khi bàn về ablation. Và ngay cả khi đó, phép so vẫn chỉ nói về *cặp kiến trúc này, ở ngân sách này, trên bộ dữ liệu này*.

## 4. Kết nối tắt đổi bài toán thành gì

Một khối tích chập thường học một hàm $H(x)$. Khối residual đổi cách đặt bài toán:

$$y = F(x) + x$$

Mạng không còn học thẳng $H(x)$ mà học **phần chênh lệch** $F(x) = H(x) - x$ - cái cần *thêm vào* đầu vào, chứ không phải cái thay thế nó.

```
    x ──────────────────────────┐  đường tắt: đi thẳng, không qua trọng số
    │                           │
    ├─▶ conv 3x3 ─▶ BN ─▶ ReLU  │
    │        │                  │
    │        ▼                  ▼
    └─▶ conv 3x3 ─▶ BN ──────▶ (+) ─▶ ReLU ─▶ y
```

Đổi cách đặt bài toán như vậy mang lại hai thứ.

**Hàm đồng nhất trở thành lựa chọn mặc định.** Muốn khối không làm gì cả thì chỉ cần đẩy trọng số của $F$ về gần 0 - điều mà weight decay vốn đã kéo về. Ở mạng phẳng, muốn một khối không làm gì thì các tầng tích chập phải học *đúng* hàm đồng nhất từ trọng số ngẫu nhiên, việc khó hơn nhiều. Đây là lý do trực tiếp cho hiện tượng ở mục 2: mạng sâu chỉ cần "tắt bớt" vài khối là quay về mạng nông, và kết nối tắt làm việc tắt ấy trở nên dễ.

**Gradient có một đường đi thẳng về đầu mạng.** Đạo hàm của $y = F(x) + x$ theo $x$ có dạng $\partial F/\partial x + 1$. Số hạng $+1$ ấy nghĩa là dù $\partial F/\partial x$ có nhỏ đến đâu, gradient vẫn đi qua được - đúng cơ chế mà chương Deep Learning · Bài 08 mô tả với vanishing gradient, và cũng là họ hàng gần của cổng quên trong LSTM ở Bài 10 chương ấy.

> 🔧 **Thử ngay:** nếu kết nối tắt tốt như vậy, sao không gắn nó vào mọi kiến trúc, kể cả mạng chỉ 5 tầng?
> Người ta có gắn, và chi phí tham số rất nhỏ - 0,1% như bảng mục 2. Và trong thí nghiệm này, khoảng cách nó tạo ra **lớn hơn ở mạng sâu hơn**: 4,7 điểm ở 20 tầng so với 17,4 điểm ở 44 tầng. Hai mức độ sâu thì chưa thành quy luật, nhưng nó khớp với cơ chế. Với mạng 5 tầng thì gradient vốn chẳng phải đi xa, và hàm đồng nhất cũng chẳng cần tới - nên kết nối tắt giải quyết một vấn đề chưa tồn tại. Đây là một ví dụ tốt cho nguyên tắc chung: mỗi kỹ thuật sinh ra để chữa một bệnh cụ thể, và thêm nó vào khi chưa có bệnh thì chỉ tốn chỗ. Chương Deep Learning · Bài 07 đã gặp đúng chuyện này khi bật cả ba kỹ thuật regularization cùng lúc.

## 5. Vì sao ResNet vẫn ở khắp nơi

Kết nối tắt là thứ khiến mạng rất sâu huấn luyện được, và hệ quả của nó lan ra ngoài phạm vi bài này:

- **Bài 04** móc hook vào `layer4` của ResNet-50 - bốn khối `layer1`-`layer4` chính là bốn nhóm khối residual
- **Bài 06** dùng ResNet-18 và ResNet-50 làm xương sống, và đo chất lượng đặc trưng đóng băng của chúng
- **Bài 08 và 09** dùng ResNet-50 làm xương sống cho Faster R-CNN, RetinaNet, FCOS, Mask R-CNN và DeepLabV3 - cả năm
- **Bài 10** cho thấy Vision Transformer cũng dùng kết nối tắt bên trong mỗi khối, dù bỏ hoàn toàn phép tích chập

Một chi tiết nữa của ResNet đáng nhắc, vì bài 02 đã chỉ ra chỗ tốn kém nhất của VGG: ResNet **bỏ hẳn khối kết nối đầy đủ ở cuối**, thay bằng một phép lấy trung bình toàn cục rồi một tầng tuyến tính duy nhất. Nhờ vậy ResNet-50 chỉ có **25.557.032** tham số so với 138.357.544 của VGG-16 - **ít hơn 5,4 lần** dù sâu hơn nhiều. (Con số 23,6 triệu xuất hiện ở bài 06 và 10 là cùng mạng ấy sau khi thay tầng phân loại 1.000 lớp bằng 37 lớp của Oxford Pet.)

## Tóm tắt bài học

- **Suy giảm do độ sâu** không phải overfitting: mạng phẳng 44 tầng có train loss 2,1247, sát mức đoán đều $\ln(10) = 2{,}3026$ - nó không học nổi cả dữ liệu đã thấy.
- Luôn đọc **train loss trước accuracy**; đó là cột phân biệt hai chứng bệnh cần hai cách chữa ngược nhau.
- Mạng phẳng 44 tầng tệ hơn mạng phẳng 20 tầng **32,4 điểm** dù nhiều hơn 2,4 lần tham số, trong khi về nguyên tắc nó chứa mạng 20 tầng như một trường hợp riêng.
- **Kết nối tắt cải thiện cả train loss lẫn accuracy ở cùng ngân sách tham số** (chỉ thêm ~0,1%): hơn 4,7 điểm ở độ sâu 20 và **17,4 điểm** ở độ sâu 44.
- Thí nghiệm này **không** tái hiện được kết quả "ResNet càng sâu càng tốt" của bài báo gốc; giả thuyết là 6 epoch trên CPU quá ít để mạng 44 tầng hội tụ, nhưng bài không kiểm chứng được điều đó.
- $y = F(x) + x$ đổi bài toán của mỗi khối thành học **phần chênh lệch**, khiến hàm đồng nhất thành lựa chọn mặc định và cho gradient một đường đi thẳng nhờ số hạng $+1$ trong đạo hàm.
- Trong thí nghiệm này, khoảng cách do kết nối tắt tạo ra **lớn hơn ở độ sâu 44 so với độ sâu 20** (17,4 điểm so với 4,7). Hai mức độ sâu và một seed thì chưa đủ để nói thành quy luật, nhưng nó khớp với lý do cơ chế ở mục 4.
- ResNet bỏ khối kết nối đầy đủ ở cuối, nên ResNet-50 chỉ có 25.557.032 tham số so với 138.357.544 của VGG-16.

## Câu hỏi tự kiểm tra

1. Train loss của mạng phẳng 44 tầng là 2,1247 trên bài toán 10 lớp. Con số đó nói lên điều gì, và vì sao nó loại trừ chẩn đoán overfitting?
2. Vì sao mạng 44 tầng "về nguyên tắc không thể tệ hơn" mạng 20 tầng? Giải thích bằng hàm đồng nhất.
3. Kết nối tắt chỉ thêm 0,1% tham số. Vậy lợi ích của nó đến từ đâu, nếu không phải từ capacity?
4. Viết đạo hàm của $y = F(x) + x$ theo $x$ và chỉ ra số hạng giúp gradient sống sót. Liên hệ với cổng quên của LSTM.
5. Bài này không tái hiện được "ResNet sâu hơn thì tốt hơn". Nêu nguyên nhân, và thiết kế thí nghiệm kiểm chứng nguyên nhân ấy.
6. Vì sao lợi ích của kết nối tắt tăng theo độ sâu? Dựa vào hai con số 4,7 và 17,4 điểm ở mục 2.
7. ResNet-50 sâu hơn VGG-16 nhiều nhưng ít hơn 5,4 lần tham số. Chỗ chênh lệch ấy nằm ở đâu?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Deep Learning · Bài 08** | Vanishing gradient, khởi tạo, BatchNorm - bài này trả nốt phần kết nối tắt còn thiếu |
| **Chương Deep Learning · Bài 06** | Đọc train loss trước accuracy - chính thứ tự dùng ở mục 2 |
| **Bài 02 của chương này** | VGG-16 và chỗ tốn tham số nhất của nó, để hiểu ResNet cắt gọn ở đâu |
| **Dive into Deep Learning (d2l.ai)** | Chương 8.6 *Residual Networks* - cài đặt khối residual đầy đủ |

**Nguồn online bổ sung (miễn phí):**

- [He và cộng sự - Deep Residual Learning for Image Recognition (CVPR 2016)](https://arxiv.org/abs/1512.03385) - bài báo gốc; hình 1 là chính hiện tượng suy giảm ở mục 2, đo trên CIFAR-10 với ngân sách huấn luyện đầy đủ.
- [He và cộng sự - Identity Mappings in Deep Residual Networks (ECCV 2016)](https://arxiv.org/abs/1603.05027) - bài tiếp theo, phân tích vì sao đường tắt nên đi thẳng không qua phép biến đổi nào.
- [Veit, Wilber, Belongie - Residual Networks Behave Like Ensembles of Relatively Shallow Networks (NeurIPS 2016)](https://arxiv.org/abs/1605.06431) - một cách đọc khác về ResNet, đáng xem sau khi nắm mục 4.
- [Li và cộng sự - Visualizing the Loss Landscape of Neural Nets (NeurIPS 2018)](https://arxiv.org/abs/1712.09913) - hình ảnh trực quan cho thấy kết nối tắt làm bề mặt loss trơn hơn hẳn.
- [ResNet - tài liệu torchvision](https://pytorch.org/vision/stable/models/resnet.html) - các biến thể có sẵn và số công bố của chúng.

> **Bài tiếp theo:** [Mô hình CNN thực sự học gì](../what-cnns-actually-learn-vi/) - ba bài đầu lo chuyện đưa ảnh vào đúng và dựng mạng cho huấn luyện được. Bài sau mở nắp ra xem bên trong: mỗi tầng tính được gì, mô hình dựa vào vùng nào của ảnh để quyết định - và một phép thử cho thấy công cụ giải thích phổ biến nhất đôi khi chỉ sai chỗ.

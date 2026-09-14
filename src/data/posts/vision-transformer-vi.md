---
title: "Vision Transformer"
description: "Cách vision transformer chia ảnh thành các patch, những inductive bias nào của CNN được nới lỏng, và điều gì thay đổi trong một phép so sánh có kiểm soát."
domain: "Computer Vision"
section: "Learn"
language: "vi"
translationKey: "vision-transformer"
order: 10
pubDate: 2026-08-29
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu cách một transformer nhìn ảnh - cắt thành mảnh, nhúng vị trí, rồi để mọi mảnh nói chuyện với nhau - biết ViT bỏ đi thiên hướng nào của mạng tích chập và phải đánh đổi gì, và quan trọng nhất: thấy một phép so sánh trông rất thuyết phục **mất hơn một nửa khoảng cách** chỉ vì đổi một thứ không liên quan gì tới kiến trúc.

## 1. Chín bài dựa trên một giả định

Từ bài 02 tới bài 09, mọi kiến trúc đều là mạng tích chập, và tất cả đều xây trên hai giả định về ảnh mà chương Deep Learning · Bài 09 đã nêu:

- **Cục bộ** - pixel gần nhau thì liên quan, nên bộ lọc chỉ cần nhìn một cửa sổ nhỏ
- **Dùng chung trọng số** - một mẫu hình có nghĩa ở mọi vị trí, nên cùng bộ lọc trượt khắp ảnh

Hai giả định này đúng với ảnh, và chính vì đúng nên chúng là **thiên hướng quy nạp** rất có giá trị: mô hình không phải học chúng từ dữ liệu, chúng được nhét sẵn vào kiến trúc.

Nhưng "nhét sẵn" cũng có nghĩa là "không thể bỏ đi khi không cần". Với một tầng tích chập 3×3, hai pixel ở hai góc đối diện của ảnh chỉ ảnh hưởng tới nhau sau khi tín hiệu đi qua rất nhiều tầng - bài 02 đã tính: VGG-16 cần đủ 13 tầng tích chập và 5 tầng gộp mới đạt trường tiếp nhận 212 pixel.

**Vision Transformer** hỏi ngược lại: nếu để mọi vùng của ảnh nói chuyện trực tiếp với nhau **ngay từ tầng đầu** thì sao?

## 2. Ảnh thành một dãy mảnh

Transformer làm việc với một *dãy*. Nên bước đầu tiên là biến ảnh thành dãy, và cách làm đơn giản đến bất ngờ: **cắt ảnh thành các ô vuông không chồng nhau**.

Với ViT-B/16 trên ảnh 224×224 và mảnh 16×16:

$$\frac{224}{16} \times \frac{224}{16} = 14 \times 14 = \mathbf{196 \text{ mảnh}}$$

Mỗi mảnh 16×16×3 = 768 con số được duỗi thẳng rồi nhân với một ma trận để thành **vector nhúng mảnh** (patch embedding). Kết quả là 196 vector - một dãy, đúng thứ transformer cần.

```
ảnh 224x224            cắt thành lưới 14x14 mảnh        dãy 196 vector
┌─────────────┐        ┌──┬──┬──┬──┬──┐
│             │        ├──┼──┼──┼──┼──┤                 [v1] [v2] ... [v196]
│    con mèo  │  ───▶  ├──┼──┼──┼──┼──┤     ───▶         + nhúng vị trí
│             │        ├──┼──┼──┼──┼──┤                  + một token [CLS]
└─────────────┘        └──┴──┴──┴──┴──┘
```

Hai chi tiết phải thêm vào, và cả hai đều lộ ra chỗ ViT khác hẳn CNN:

**Nhúng vị trí (positional embedding).** Sau khi cắt và duỗi, dãy 196 vector **không còn mang thông tin mảnh nào ở đâu**. Với phép tích chập thì vị trí nằm sẵn trong cấu trúc; với transformer thì không, vì phép chú ý xử lý mọi phần tử của dãy như nhau. Nên người ta **cộng thêm** vào mỗi vector một vector vị trí học được. Đây đúng là chuyện bài 09 của chương Deep Learning đã bàn khi nói `nn.Flatten()` làm mất quan hệ láng giềng - chỉ khác chỗ ViT giải quyết bằng cách học lại nó thay vì nhét sẵn.

**Token phân loại `[CLS]`.** Một vector học được thêm vào đầu dãy, không tương ứng với mảnh nào. Sau khi đi hết các tầng, chính vector này được đưa vào tầng phân loại. Nó đóng vai "chỗ tổng hợp", thay cho phép lấy trung bình toàn cục ở cuối một CNN.

Rồi đến phần chú ý. Chương Deep Learning · Bài 10 đã giới thiệu ý tưởng ở mức khái niệm: **mọi vị trí nối thẳng tới mọi vị trí, và tính được song song**. Ở đây nó nghĩa là mảnh ở góc trên trái và mảnh ở góc dưới phải ảnh hưởng tới nhau **ngay ở tầng thứ nhất** - thứ mà CNN phải xếp hàng chục tầng mới làm được.

Cơ chế đầy đủ của phép chú ý - truy vấn, khóa, giá trị, nhiều đầu, và vì sao nó thay được mạng hồi quy - là việc của **chương Generative AI**, nơi transformer được dựng đầy đủ cho ngôn ngữ. Bài này chỉ cần phần riêng của ảnh: cắt mảnh, nhúng vị trí, và cái giá của việc bỏ thiên hướng quy nạp.

## 3. Bỏ thiên hướng thì mất gì

Đây là chỗ lý thuyết nói khá rõ, và cần nói trước khi xem số.

Cần nói chính xác ViT bỏ cái gì, vì chỗ này rất hay bị kể sai.

**ViT vẫn dùng chung trọng số.** Cùng một ma trận chiếu được áp cho cả 196 mảnh; cùng những tầng chú ý và MLP được áp cho mọi token ở mọi vị trí. Về khoản tiết kiệm tham số nhờ dùng chung, hai họ không khác nhau.

Thứ ViT bỏ là **tính cục bộ được cài cứng** và **tính tương biến với dịch chuyển** của phép tích chập. Một bộ lọc 3×3 *bắt buộc* chỉ nhìn 9 pixel kề nhau, và khi ảnh dịch đi thì bản đồ đặc trưng dịch theo - hai ràng buộc ấy nằm trong chính phép toán. Phép chú ý không có ràng buộc nào như vậy: mọi mảnh đều có thể nhìn mọi mảnh, và "mảnh số 1 nằm cạnh mảnh số 2" là thứ nhúng vị trí phải **học** chứ không được cho sẵn.

Hệ quả lý thuyết: **ViT cần nhiều dữ liệu hơn để học lại những thứ CNN được cho sẵn.** Bài báo gốc đo đúng điều này - huấn luyện trên ImageNet 1,3 triệu ảnh thì ViT thua ResNet, nhưng huấn luyện trên JFT-300M thì nó vượt lên.

Đổi lại, khi có đủ dữ liệu, việc không bị ràng buộc cho phép nó học những quan hệ mà kiến trúc tích chập không biểu diễn gọn được.

Nhưng đó là chuyện **huấn luyện từ đầu**. Câu hỏi thực tế của phần lớn người đọc khác hẳn: *"tôi có một mô hình đã học sẵn và vài trăm ảnh - nên chọn cái nào?"* Đó là câu mục 4 đo.

## 4. Ít dữ liệu: đo chuyển giao, không đo kiến trúc

Thí nghiệm: cả hai mô hình đều **đã học ImageNet**, đều **đóng băng hoàn toàn**, chỉ train tầng phân loại cuối trên **370 ảnh** Oxford Pet (37 lớp, tức 10 ảnh mỗi lớp), chấm trên **500 ảnh validation**. Một lần chạy minh họa, 2 epoch.

Gọi là validation chứ không phải test là có lý do: 500 ảnh ấy được dùng để **so hai mô hình và rút ra kết luận**, tức làm đúng vai trò chọn lựa. Đây là quy ước mà chương Deep Learning · Bài 12 đã dựng, và bài này giữ nguyên.

| Mô hình | Tổng tham số | Tham số được train | Accuracy | Giây/ảnh |
|---|---|---|---|---|
| ResNet-50 (`IMAGENET1K_V2`) | 23.583.845 | 75.813 | 0,5140 | **0,056** |
| **ViT-B/16** | 85.827.109 | **28.453** | **0,8320** | 0,123 |

Chênh **31,8 điểm**. Với 500 ảnh, sai số lấy mẫu của một con số accuracy vào khoảng 2 điểm, nên riêng nhiễu lấy mẫu rất khó giải thích nổi một khoảng cách lớn đến thế. Nhưng mỗi mô hình chỉ train một lần, và hai mô hình chấm trên cùng tập ảnh nên phép so chặt chẽ phải tính theo từng ảnh - muốn định lượng khoảng cách này cho chắc thì vẫn cần nhiều seed.

Và đây đúng là chỗ dễ viết ra một câu sai. Câu sai ấy là: *"ViT hiệu quả dữ liệu hơn CNN"*. Bảng trên **không chứng minh** điều đó, vì hai mô hình khác nhau ở **ba** thứ cùng lúc - kiến trúc, kích thước (85,8 triệu so với 23,6 triệu tham số), và **công thức tiền huấn luyện**.

Thứ ba dễ kiểm nhất, nên kiểm nó. Cùng kiến trúc ResNet-50, cùng dữ liệu, cùng cách huấn luyện đầu phân loại - chỉ đổi **gói trọng số**:

| Bộ trọng số ResNet-50 | Accuracy ImageNet **công bố** | Đặc trưng đóng băng trên Pet |
|---|---|---|
| `IMAGENET1K_V1` (công thức gốc 2015) | 76,13% | **0,6920** |
| `IMAGENET1K_V2` (công thức 2021) | **80,858%** | 0,5140 |

Kết quả này đáng đọc chậm. Gói trọng số **kém hơn 4,7 điểm trên ImageNet** lại cho đặc trưng chuyển giao **tốt hơn 17,8 điểm**.

Nói cho chặt thì "chỉ đổi gói trọng số" gồm hai thứ đi kèm nhau: công thức huấn luyện khác, **và** phép tiền xử lý khác - `weights.transforms()` của V1 thu ảnh về cạnh 256 rồi cắt 224, còn V2 thu về 232 rồi cắt 224. Ở đây ta dùng đúng phép tiền xử lý mà mỗi gói khai báo, như bài 01 đã dặn, nên không tách được hai yếu tố ấy.

Vì vậy phát biểu đúng là phát biểu về **độ ổn định**, không phải về nguyên nhân:

> **17,8 trong 31,8 điểm khoảng cách biến mất chỉ bằng cách đổi gói trọng số của ResNet-50.** Một khoảng cách dịch chuyển được nhiều đến thế bởi một thay đổi không đụng gì tới kiến trúc thì không thể quy cho kiến trúc.

Phần 14,0 điểm còn lại - giữa ViT-B/16 và ResNet-50 V1 - vẫn trộn lẫn kiến trúc với kích thước mô hình (85,8 triệu so với 25,6 triệu tham số ở bản gốc 1.000 lớp), nên cũng chưa nói được gì về transformer so với tích chập.

> ⚠️ **Lưu ý nhỏ:** một giả thuyết hợp lý cho chuyện này: công thức `V2` dùng rất nhiều tăng cường dữ liệu và regularization để đẩy accuracy ImageNet lên, tức tối ưu mạnh cho *đúng nghìn lớp ấy*, và điều đó có thể làm đặc trưng bớt chung chung. Nhưng đó là giả thuyết, không phải thứ bảng trên chứng minh - muốn chứng minh phải tách riêng từng thành phần của công thức huấn luyện. Cũng đừng lật sang thái cực ngược lại: nhìn chung accuracy ImageNet **có** tương quan khá mạnh với chất lượng chuyển giao, và khảo sát của Kornblith cùng cộng sự trong phần Đọc thêm chỉ ra rằng chính regularization mạnh là chỗ tương quan ấy gãy. Bài học thực dụng vì thế là: **bảng xếp hạng ImageNet là một chỉ dấu ban đầu, không phải căn cứ để chốt** - chốt thì phải đo trên validation của chính bài toán bạn làm.

## 5. Chọn cái nào

Gộp cả bài lại thành hướng dẫn dùng được, tách rõ chỗ nào có số liệu và chỗ nào là kiến thức chung:

| Tình huống | Chọn gì | Căn cứ |
|---|---|---|
| Ít dữ liệu, dùng mô hình học sẵn | **Thử cả hai**, chấm bằng validation | Bảng mục 4: khoảng cách lớn nhưng nguyên nhân trộn lẫn |
| Ràng buộc tốc độ chặt | Nghiêng về CNN | ViT-B/16 chậm hơn 2,2 lần và nặng gấp 3,6 lần |
| Huấn luyện từ đầu, dữ liệu vừa phải | Nghiêng về CNN | Thiên hướng quy nạp có sẵn giúp ích khi dữ liệu ít |
| Huấn luyện từ đầu, dữ liệu rất lớn | ViT đáng thử | Kết quả công bố của bài báo gốc |
| Chọn xương sống bất kỳ | **Đừng chốt chỉ bằng accuracy ImageNet** | Bảng V1/V2: xếp hạng ImageNet có thể đi ngược chất lượng chuyển giao |

Dòng cuối là bài học đắt nhất của bài này, và nó nối thẳng về quy ước ba loại con số của bài 01: **80,858% là số công bố, 0,514 là số đo trên máy này, và ở trường hợp này chúng chỉ về hai hướng ngược nhau.** Hai con số cùng nói về một mô hình mà không thay thế được cho nhau.

> 🔧 **Thử ngay:** bạn cần chọn xương sống cho bài toán phân loại 5 loại vải, có 600 ảnh. Đồng nghiệp đề nghị lấy mô hình đứng đầu bảng xếp hạng ImageNet. Bạn phản biện thế nào, và đề xuất quy trình gì?
> Phản biện bằng đúng bảng V1/V2 ở mục 4: mô hình cao điểm hơn trên ImageNet cho đặc trưng chuyển giao kém hơn 17,8 điểm trên một bài toán khác. Thứ hạng ImageNet đo khả năng phân biệt đúng 1.000 lớp ấy, không đo chất lượng đặc trưng dùng lại. Quy trình đề xuất: chọn ba đến bốn ứng viên khác họ nhau (một ResNet, một ViT, một mô hình nhẹ), **đóng băng hết** và chỉ train tầng cuối - mỗi lần chạy chỉ vài phút như mục 4 cho thấy - rồi chấm trên validation của chính bài toán vải. Đó là phép đo rẻ và trực tiếp trả lời câu hỏi đang hỏi. Sau khi chọn được xương sống mới tính tới fine-tune sâu hơn, theo đúng bậc thang của bài 06.

## Tóm tắt bài học

- ViT cắt ảnh thành mảnh không chồng nhau - 224×224 với mảnh 16×16 cho **196 mảnh** - rồi xử lý chúng như một dãy.
- Vì phép chú ý coi mọi phần tử của dãy như nhau, ViT phải **cộng thêm nhúng vị trí học được**; vị trí không nằm sẵn trong kiến trúc như ở tích chập. Token `[CLS]` đóng vai chỗ tổng hợp cuối.
- Mảnh ở hai góc đối diện ảnh hưởng tới nhau **ngay tầng đầu tiên**, thứ mà CNN phải xếp hàng chục tầng mới đạt được.
- ViT **vẫn dùng chung trọng số** (một ma trận chiếu cho mọi mảnh, cùng các tầng cho mọi token). Thứ nó bỏ là **tính cục bộ cài cứng** và **tính tương biến với dịch chuyển** của phép tích chập - lý thuyết nói vì thế nó cần nhiều dữ liệu hơn khi huấn luyện từ đầu.
- Trên 370 ảnh Pet với cả hai đều đóng băng: ViT-B/16 đạt **0,8320** so với **0,5140** của ResNet-50 V2 - chênh 31,8 điểm.
- **Nhưng hơn một nửa khoảng cách ấy dịch chuyển được mà không đụng tới kiến trúc**: cùng ResNet-50, chỉ đổi gói trọng số V2 sang V1 đã lấy lại 17,8 điểm - mà gói trọng số kéo theo cả công thức huấn luyện lẫn phép tiền xử lý khác nhau.
- Gói trọng số **kém hơn 4,7 điểm trên ImageNet** cho đặc trưng chuyển giao **tốt hơn 17,8 điểm**. Giả thuyết hợp lý là regularization mạnh làm đặc trưng bớt chung chung, nhưng bảng này chưa chứng minh được điều đó.
- Hệ quả thực dụng: **đừng chốt xương sống chỉ bằng bảng xếp hạng ImageNet**; coi nó là chỉ dấu ban đầu rồi đo trực tiếp bằng cách đóng băng và train tầng cuối trên chính bài toán của bạn.
- ViT-B/16 nặng gấp 3,6 lần và chậm hơn 2,2 lần ResNet-50 trên CPU.

## Câu hỏi tự kiểm tra

1. Ảnh 384×384 với mảnh 16×16 cho bao nhiêu mảnh? Con số ấy ảnh hưởng thế nào tới chi phí tính toán của phép chú ý?
2. Vì sao ViT cần nhúng vị trí còn CNN thì không? Liên hệ với chuyện `nn.Flatten()` ở chương Deep Learning · Bài 09.
3. Nêu hai thiên hướng quy nạp mà CNN có sẵn và ViT không có. Vì sao "dùng chung trọng số" **không** nằm trong hai thứ đó?
4. Bảng mục 4 cho ViT hơn 31,8 điểm. Liệt kê ba khác biệt giữa hai mô hình khiến ta không kết luận được về kiến trúc.
5. ResNet-50 V2 hơn V1 gần 5 điểm trên ImageNet nhưng kém 17,8 điểm khi làm bộ rút đặc trưng. Nêu hai yếu tố đi kèm gói trọng số khiến ta không quy được nguyên nhân cho riêng công thức huấn luyện.
6. Thiết kế một thí nghiệm tách được ảnh hưởng của *kiến trúc* khỏi ảnh hưởng của *kích thước mô hình* trong so sánh ViT với ResNet. Bạn cần gì mà bài này không có?
7. Vì sao bài này không giải thích cơ chế truy vấn-khóa-giá trị của phép chú ý?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Deep Learning · Bài 10** | Attention ở mức khái niệm - mọi vị trí nối thẳng, tính song song |
| **Chương Deep Learning · Bài 09** | Vì sao `nn.Flatten()` làm mất quan hệ láng giềng, và CNN nhét sẵn giả định gì |
| **Bài 02 và 06 của chương này** | Trường tiếp nhận, và bậc thang chiến lược fine-tune mà mục 5 nối vào |
| **Dive into Deep Learning (d2l.ai)** | Chương 11.8 *Transformers for Vision* - cài đặt ViT đầy đủ từ đầu |

**Nguồn online bổ sung (miễn phí):**

- [Dosovitskiy và cộng sự - An Image is Worth 16x16 Words (ICLR 2021)](https://arxiv.org/abs/2010.11929) - bài báo gốc của ViT; hình 1 là sơ đồ cắt mảnh của mục 2, và mục 4.2 là thí nghiệm cho thấy ViT cần dữ liệu lớn khi train từ đầu.
- [Wightman, Touvron, Jégou - ResNet strikes back (2021)](https://arxiv.org/abs/2110.00476) - công thức đằng sau bộ trọng số `V2` ở mục 4.
- [Kornblith, Shlens, Le - Do Better ImageNet Models Transfer Better? (CVPR 2019)](https://arxiv.org/abs/1805.08974) - khảo sát quan hệ giữa accuracy ImageNet và chất lượng chuyển giao, tức chính hiện tượng ở bảng V1/V2.
- [Raghu và cộng sự - Do Vision Transformers See Like Convolutional Neural Networks? (NeurIPS 2021)](https://arxiv.org/abs/2108.08810) - so sánh biểu diễn bên trong hai họ, đọc nếu muốn đi sâu hơn mục 3.
- [Vision Transformer - tài liệu torchvision](https://pytorch.org/vision/stable/models/vision_transformer.html) - các biến thể có sẵn kèm số công bố.

> **Bài tiếp theo:** [Học biểu diễn: từ nhãn cố định tới không gian vector](../representation-learning-fixed-labels-to-vector-space-vi/) - bài này vừa cho thấy đặc trưng đóng băng của một mô hình quan trọng đến mức nào - và đo nó bằng cách gắn thêm một tầng phân loại. Bài sau bỏ luôn tầng ấy, dùng thẳng vector, và mở ra những việc mà không mô hình phân loại nào làm được: tìm ảnh giống, gom cụm không cần nhãn, và phân loại tập lớp mới mà không huấn luyện gì.

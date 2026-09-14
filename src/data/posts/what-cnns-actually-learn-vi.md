---
title: "Mô hình CNN thực sự học gì"
description: "Đọc bản đồ đặc trưng ở từng độ sâu, tự cài Grad-CAM trong hơn chục dòng, và vì sao một heatmap trông hợp lý không chứng minh mô hình suy luận đúng."
domain: "Computer Vision"
section: "Learn"
language: "vi"
translationKey: "what-cnns-actually-learn"
order: 4
pubDate: 2026-08-26
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ đọc được bản đồ đặc trưng ở từng độ sâu và thấy độ thưa của chúng tăng ra sao, tự cài được Grad-CAM trong hơn chục dòng, và - quan trọng nhất - biết vì sao một tấm heatmap trông hợp lý **không** chứng minh mô hình đang suy luận đúng, bằng một phép thử phản bác chính tấm heatmap ấy.

## 1. Câu hỏi mà accuracy không trả lời

Bài 03 để lại một mô hình chạy tốt. Nhưng "chạy tốt" trên tập kiểm tra và "dựa vào đúng thứ trong ảnh" là hai chuyện khác nhau, và chương Machine Learning · Bài 12 đã gặp đúng vấn đề này ở dạng bảng số: mô hình có thể đúng vì lý do sai.

Với ảnh, chuyện ấy có một cái tên quen thuộc trong nghề: mô hình phân biệt chó husky với sói bằng cách nhìn **tuyết ở nền**, vì trong dữ liệu huấn luyện ảnh sói phần lớn chụp trên tuyết. Accuracy đẹp, mô hình vô dụng.

Bài này mở hai cửa sổ vào bên trong mạng. Cửa thứ nhất - bản đồ đặc trưng - cho biết mạng *tính* ra những gì. Cửa thứ hai - Grad-CAM - cho biết vùng ảnh nào đẩy điểm số của lớp được chọn lên cao. Rồi mục 5 kiểm tra xem cửa thứ hai đáng tin đến đâu.

## 2. Bản đồ đặc trưng đặc dần rồi thưa hẳn

Cho một ảnh đi qua ResNet-50 và ghi lại đầu ra của bốn khối chính:

| Khối | Hình dạng đầu ra | Tỉ lệ giá trị bằng 0 | Số kênh im hoàn toàn |
|---|---|---|---|
| `layer1` | 256 × 56 × 56 | 39,8% | 1 / 256 |
| `layer2` | 512 × 28 × 28 | 53,9% | 0 / 512 |
| `layer3` | 1024 × 14 × 14 | 64,5% | 3 / 1024 |
| `layer4` | **2048 × 7 × 7** | **97,6%** | **1.273 / 2048** |

Hai xu hướng ngược chiều, và chúng là cùng một câu chuyện.

**Không gian co lại, số kênh phình ra.** Từ 56×56 xuống 7×7 - mất 64 lần số vị trí - trong khi số kênh tăng từ 256 lên 2048. Mạng đang đổi *ở đâu* lấy *cái gì*: tầng đầu biết rất rõ vị trí nhưng mỗi kênh mang ít ý nghĩa; tầng cuối gần như không còn phân biệt vị trí nhưng mỗi kênh ứng với một khái niệm rõ hơn nhiều.

**Càng sâu càng thưa.** Ở `layer4`, **97,6%** giá trị bằng 0 và **1.273 trong 2.048 kênh im hoàn toàn** - không kích hoạt ở bất kỳ vị trí nào trên tấm ảnh này.

Cách kể quen thuộc là: 2.048 kênh ấy là 2.048 câu hỏi kiểu "có vây cá không", "có bánh xe không", nên hầu hết câu trả lời phải là *không*. Hình ảnh ấy dễ nhớ, nhưng nói quá. Có những kênh phản ứng chọn lọc với một loại hoa văn nhất định và diễn giải được, nhưng biểu diễn của mạng nhìn chung **phân tán trên nhiều kênh** chứ không phải mỗi kênh một khái niệm. Và bản thân việc nhiều giá trị bằng 0 chỉ nói biểu diễn *thưa trên tấm ảnh này* - nó chưa chứng minh mạng đã "chuyên biệt hóa tốt" hay đang hoạt động đúng.

Cái đọc được chắc chắn từ bảng là xu hướng: càng sâu, tỉ lệ giá trị bằng 0 càng cao. Còn diễn giải xu hướng ấy thành ý nghĩa gì thì cần thêm bằng chứng, và mục 4 sẽ cho thấy loại bằng chứng ấy khó kiếm đến đâu.

> ⚠️ **Lưu ý nhỏ:** con số 1.273 kênh im là **với đúng tấm ảnh này**, không phải kênh chết vĩnh viễn kiểu ReLU chết ở chương Deep Learning · Bài 02. Đưa tấm khác vào thì tập kênh im sẽ khác. Muốn tìm kênh chết thật thì phải đếm trên cả tập dữ liệu chứ không phải một ảnh - đúng cách bài 02 của chương trước đã làm với 1.000 ảnh.

## 3. Grad-CAM trong mười lăm dòng

Ý tưởng gọn: ở `layer4` ta có 2.048 bản đồ 7×7. Bản đồ nào quan trọng với lớp đang xét? Hỏi gradient. Nếu tăng giá trị của kênh $k$ lên một chút mà điểm số của lớp $c$ tăng nhiều, thì kênh $k$ quan trọng.

$$w_k = \frac{1}{7 \times 7}\sum_{i,j} \frac{\partial y_c}{\partial A^k_{ij}}, \qquad L = \text{ReLU}\Big(\sum_k w_k A^k\Big)$$

Trọng số $w_k$ là trung bình gradient trên toàn bản đồ; $L$ là tổng có trọng số của 2.048 bản đồ, rồi ReLU để giữ phần *ủng hộ* lớp ấy và bỏ phần chống lại. Cài đặt bằng hook của PyTorch:

```python
acts, grads = {}, {}
model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o.detach()))
model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))

def gradcam(x):
    model.zero_grad()
    out = model(x)
    c = int(out.argmax(1))
    out[0, c].backward()                       # gradient của điểm lớp c
    A, G = acts["v"][0], grads["v"][0]         # (2048, 7, 7)
    w = G.mean(dim=(1, 2))                     # trọng số từng kênh
    cam = torch.relu((w[:, None, None] * A).sum(0))
    return c, cam / (cam.max() + 1e-8)
```

Không có thư viện nào ở đây cả - chỉ hai hook, một `backward()` và một tổng có trọng số. Kết quả là lưới 7×7 cho ảnh vào 224×224.

> ⚠️ **Lưu ý nhỏ:** dễ nói nhầm rằng mỗi ô của lưới 7×7 "ứng với một vùng 32×32 pixel". Con số 32 là **bước nhảy tích lũy** (output stride) - tâm của hai ô kề nhau cách nhau 32 pixel trên ảnh gốc. Còn **trường tiếp nhận** của mỗi ô, tính theo công thức ở bài 02, lớn hơn thế rất nhiều và các ô **chồng lấn nhau** rất nhiều: mỗi ô ở `layer4` của ResNet-50 nhìn được một vùng rộng hơn cả trăm pixel. Nên đừng đọc lưới 7×7 như một tấm lưới cắt ảnh thành 49 ô rời nhau. Phép thử che ở mục 4 vẫn che theo lưới 32×32 - nhưng đó là chọn *vùng để can thiệp*, không phải khẳng định vùng ấy là trường tiếp nhận.

Chạy trên ba ảnh cá tench, in lưới bằng ký tự (đậm dần từ khoảng trắng tới `@`):

```
ảnh A - đoán "tench" 0,6717      ảnh B - đoán "tench" 0,4509      ảnh C - đoán "sturgeon" 0,2512
    .:.
   .:-.
    .                                                              .+#-
 ==                                                               . =@+
+##=#**                             :=:                          .. ..
*%@=##*                            .@@@:
  .                                 ..
```

Ảnh B gọn và rõ: mô hình dồn chú ý vào một vùng nhỏ ở nửa dưới. Ảnh A tản mát hơn, vắt ngang cả dải dưới. Ảnh C - tấm mô hình đoán **sai**, gọi con tench thành cá tầm - dồn chú ý vào góc trên bên phải.

Đến đây, cách kể thông thường là: "thấy chưa, mô hình nhìn vào con cá". Nhưng đó mới là *nhìn hình rồi kể chuyện*. Mục sau kiểm tra bằng số.

## 4. Kiểm chứng heatmap bằng cách che

Nếu vùng sáng thật sự là thứ quyết định, thì **che nó đi phải làm điểm số tụt mạnh** - mạnh hơn hẳn so với che một vùng ngẫu nhiên. Đó là một phép thử **can thiệp**, và nó rẻ: che 8 trong 49 ô (khoảng 16% diện tích ảnh), đo lại xác suất của đúng lớp mà mô hình vừa chọn.

Ba cách che để so: 8 ô **sáng nhất**, 8 ô **ngẫu nhiên** (trung bình 5 lần bốc), và 8 ô **tối nhất**.

| Ảnh | Dự đoán | Xác suất gốc | Che 8 ô sáng nhất | Che 8 ô ngẫu nhiên | Che 8 ô tối nhất |
|---|---|---|---|---|---|
| A | tench (đúng) | 0,6717 | **0,7654** ↑ | 0,6889 | 0,6936 |
| B | tench (đúng) | 0,4509 | **0,3267** ↓ | 0,4334 | 0,4469 |
| C | sturgeon (sai) | 0,2512 | **0,0103** ↓↓ | 0,2173 | 0,2856 |

**Ảnh C là ca Grad-CAM đúng hoàn toàn.** Che vùng sáng làm xác suất sập từ 0,2512 xuống 0,0103 - giảm hơn hai mươi lần, trong khi che ngẫu nhiên gần như không đổi. Vùng ấy đúng là thứ chống đỡ dự đoán. Đáng chú ý: đây lại là tấm mô hình đoán **sai**. Grad-CAM chỉ ra đúng chỗ mô hình dựa vào, và chỗ ấy dẫn nó tới câu trả lời sai.

**Ảnh B đúng theo hướng, nhưng nhẹ.** Che vùng sáng làm tụt 0,124 điểm, che ngẫu nhiên chỉ tụt 0,018. Chênh lệch rõ ràng, dù không kịch tính.

**Ảnh A phản bác chính tấm heatmap.** Che vùng "quan trọng nhất" làm mô hình **tự tin hơn** - từ 0,6717 lên 0,7654. Vùng Grad-CAM tô đậm nhất, nếu có tác dụng gì, thì là đang *kéo* dự đoán xuống. Không có cách đọc nào biến kết quả này thành "heatmap đã chỉ đúng chỗ".

Một phần ba số ảnh thử cho kết quả ngược. Đó là lý do mục sau tồn tại.

> ⚠️ **Lưu ý nhỏ:** phép thử che cũng không phải trọng tài tuyệt đối. Che bằng số 0 (sau chuẩn hóa là màu trung bình của ImageNet) tạo ra một mảng phẳng không tự nhiên - bản thân nó là một thứ mô hình chưa từng thấy lúc huấn luyện, nên kết quả lẫn cả tác dụng của việc "mất thông tin" lẫn tác dụng của việc "thêm nhiễu lạ". Các phép thử nghiêm túc hơn thay vùng che bằng ảnh làm mờ hoặc bằng nội dung sinh ra cho khớp nền. Ở đây ta chấp nhận cách rẻ, và vì thế đọc kết quả theo *so sánh tương đối* giữa ba cách che chứ không theo giá trị tuyệt đối của mức tụt.

## 5. Heatmap là công cụ chẩn đoán, không phải bằng chứng

Gộp bốn mục lại thành một câu nên nhớ: **Grad-CAM cho biết vùng nào ảnh hưởng tới điểm số, chứ không cho biết mô hình "suy nghĩ" đúng hay sai.** Nó là một lời giải thích *cục bộ* - cho đúng ảnh này, đúng lớp này, đúng mô hình này - và như bảng ở mục 4 cho thấy, đôi khi lời giải thích ấy còn không sống sót qua một phép thử nhân quả đơn giản.

Dùng nó vào việc gì thì hợp lý:

**Truy một ca sai cụ thể.** Ảnh C là ví dụ mẫu: mô hình gọi nhầm cá, và heatmap cộng phép thử che cho biết nó dựa vào vùng nào để nhầm. Đó là đầu mối để đi tiếp.

**Phát hiện mô hình bám vào nền.** Nếu trên nhiều ảnh, vùng sáng đều rơi ra ngoài vật thể - vào tuyết, vào watermark, vào viền ảnh - thì đó là dấu hiệu đáng báo động, và nó có giá trị vì lặp lại trên *nhiều* ảnh chứ không phải một.

Dùng nó vào việc gì thì không nên:

**Đưa một tấm heatmap đẹp làm bằng chứng mô hình đáng tin.** Đây là cách dùng phổ biến nhất trong các bài trình bày, và là cách dùng sai. Một heatmap trùng với trực giác con người có thể chỉ là trùng hợp - ảnh A ở trên trông hợp lý mà không sống nổi phép thử che.

**Kết luận từ một ảnh.** Cùng lý do bài 07 nói "mAP trên một ảnh không dùng làm thước đo được": một quan sát đơn lẻ không phân biệt được quy luật với ngẫu nhiên.

> 🔧 **Thử ngay:** bạn có mô hình phân loại ảnh chụp X-quang, accuracy 0,94. Grad-CAM trên năm ca đều tô đậm đúng vùng phổi. Bạn kết luận được gì, và cần làm thêm gì trước khi dùng nó vào việc thật?
> Kết luận được rất ít. Năm ảnh là quá ít, và như ảnh A cho thấy, vùng sáng có thể không phải vùng quyết định. Việc cần làm: (1) chạy phép thử che trên một tập lớn, so mức tụt khi che vùng sáng với khi che vùng ngẫu nhiên - nếu hai mức xấp xỉ nhau thì heatmap không nói lên gì; (2) kiểm riêng những ca mô hình **sai**, vì chúng cho nhiều thông tin hơn các ca đúng; (3) tìm chỉ dấu tắt trong dữ liệu - chữ, nhãn bệnh viện, ký hiệu máy chụp ở góc ảnh, vốn là ca kinh điển khiến mô hình X-quang đoán đúng vì lý do sai; (4) tách chỉ số theo nhóm bệnh nhân và theo thiết bị chụp, đúng như chương Machine Learning · Bài 12 đã làm.

## Tóm tắt bài học

- Đi sâu vào mạng, bản đồ đặc trưng **co lại về không gian và phình ra về số kênh**: từ 256 × 56 × 56 ở `layer1` tới 2048 × 7 × 7 ở `layer4`.
- Độ thưa tăng theo độ sâu: từ 39,8% giá trị bằng 0 ở `layer1` lên **97,6%** ở `layer4`, với 1.273 trong 2.048 kênh im hoàn toàn trên một tấm ảnh. Nhưng độ thưa tự nó chưa chứng minh điều gì về chất lượng, và **đừng đọc mỗi kênh như một khái niệm** - biểu diễn thường phân tán trên nhiều kênh.
- Lưới 7×7 ở `layer4` có **bước nhảy** 32 pixel, không phải trường tiếp nhận 32 pixel; các ô nhìn những vùng rộng hơn nhiều và chồng lấn nhau.
- **Grad-CAM** cài được trong hơn chục dòng bằng hai hook: lấy hoạt hóa và gradient ở tầng tích chập cuối, lấy trung bình gradient làm trọng số kênh, tổng có trọng số rồi ReLU.
- Phép thử che là cách kiểm chứng heatmap bằng **can thiệp**: che vùng sáng, so mức tụt với che vùng ngẫu nhiên. Nó không phải bằng chứng nhân quả chặt chẽ, vì bản thân vùng bị che tạo ra một đầu vào mô hình chưa từng thấy.
- Trên ba ảnh thử, **hai ảnh xác nhận và một ảnh phản bác** - ảnh A che vùng sáng nhất lại làm xác suất *tăng* từ 0,6717 lên 0,7654.
- Ca thuyết phục nhất (ảnh C: 0,2512 → 0,0103) lại là ca mô hình đoán **sai**: Grad-CAM chỉ đúng chỗ mô hình dựa vào, và chỗ ấy dẫn tới câu trả lời sai.
- **Heatmap hợp lý không chứng minh mô hình suy luận đúng.** Nó là công cụ chẩn đoán cục bộ, hữu ích để truy ca sai và phát hiện mô hình bám vào nền, không phải bằng chứng về độ tin cậy.

## Câu hỏi tự kiểm tra

1. Vì sao độ thưa của bản đồ đặc trưng tăng theo độ sâu? Vì sao đó là dấu hiệu tốt chứ không phải dấu hiệu hỏng?
2. Phân biệt "1.273 kênh im trên tấm ảnh này" với "kênh chết" ở chương Deep Learning · Bài 02. Muốn kiểm chuyện thứ hai thì phải đo thế nào?
3. Phân biệt **bước nhảy tích lũy** với **trường tiếp nhận** của một ô trong lưới 7×7. Vì sao lẫn hai thứ này lại dẫn tới cách đọc heatmap sai?
4. Trong công thức Grad-CAM, vì sao trọng số $w_k$ lấy trung bình gradient trên toàn bản đồ 7×7 thay vì dùng gradient tại từng vị trí?
5. Vì sao có bước ReLU ở cuối? Bỏ nó đi thì bản đồ mang nghĩa gì khác?
6. Ảnh A: che vùng sáng nhất làm xác suất tăng. Nêu hai cách giải thích khác nhau cho hiện tượng này.
7. Thiết kế một phép thử che tốt hơn cách che bằng số 0. Nêu điểm mạnh và điểm yếu của cách bạn đề xuất.
8. Đồng nghiệp đưa một slide có bốn tấm Grad-CAM đẹp và kết luận "mô hình nhìn đúng chỗ". Bạn hỏi lại ba câu gì?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Machine Learning · Bài 12** | Phần "mô hình đang dựa vào cái gì" - cùng câu hỏi, ở dữ liệu bảng |
| **Chương Deep Learning · Bài 02** | ReLU chết, để phân biệt với chuyện kênh im trên một ảnh |
| **Dive into Deep Learning (d2l.ai)** | Chương 7.6 và 8.6 - cấu trúc khối của ResNet, giúp hiểu vì sao `layer4` là chỗ để móc hook |

**Nguồn online bổ sung (miễn phí):**

- [Selvaraju và cộng sự - Grad-CAM (ICCV 2017)](https://arxiv.org/abs/1610.02391) - bài báo gốc; mục 3 là đúng công thức được cài ở mục 3 của bài này.
- [Adebayo và cộng sự - Sanity Checks for Saliency Maps (NeurIPS 2018)](https://arxiv.org/abs/1810.03292) - cho thấy vài phương pháp giải thích tạo ra bản đồ trông hợp lý ngay cả khi trọng số mô hình bị xáo ngẫu nhiên. Đọc bài này trước khi tin bất kỳ heatmap nào.
- [Ribeiro, Singh, Guestrin - "Why Should I Trust You?" (KDD 2016)](https://arxiv.org/abs/1602.04938) - nguồn của ví dụ husky và tuyết, kèm ý tưởng giải thích cục bộ.
- [Zeiler & Fergus - Visualizing and Understanding Convolutional Networks (ECCV 2014)](https://arxiv.org/abs/1311.2901) - mục 4.2 là phép thử che có hệ thống, tiền thân của phép thử ở mục 4.
- [`register_forward_hook` - tài liệu PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) - cơ chế hook dùng để lấy hoạt hóa giữa mạng.

> **Bài tiếp theo:** [Dataset ảnh và augmentation](../image-datasets-and-augmentation-vi/) - bốn bài đầu đều dùng mô hình người khác đã huấn luyện. Bài sau bắt đầu phần tự huấn luyện, và câu hỏi đầu tiên không phải kiến trúc nào mà là **dữ liệu vào ra sao** - chia tập thế nào cho ảnh, phép tăng cường nào hợp bài toán nào, và phép nào âm thầm làm sai nhãn.

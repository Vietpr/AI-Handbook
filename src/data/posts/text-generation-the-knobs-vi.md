---
title: "Sinh văn bản: các núm vặn"
description: "Từ phân bố xác suất tới câu chữ còn một bước quyết định, temperature, top-k và top-p làm gì với phân bố, và chọn cấu hình theo số đo thay vì cảm giác."
domain: "Generative AI"
section: "Learn"
language: "vi"
translationKey: "text-generation-the-knobs"
order: 5
pubDate: 2026-09-02
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết từ phân bố xác suất tới câu chữ còn một bước quyết định nữa, hiểu chính xác temperature, top-k và top-p làm gì với phân bố, và chọn được cấu hình phù hợp với việc mình đang làm - dựa trên số đo chứ không phải cảm giác.

## 1. Mô hình không sinh văn bản

Bài 01 kết bằng một quan sát: mô hình trả về **phân bố xác suất** trên toàn bộ từ vựng cho token kế tiếp. Đó là toàn bộ những gì nó làm.

Từ phân bố ấy tới một token cụ thể là **một bước riêng, nằm ngoài mô hình**. Bước ấy có tên: **chiến lược giải mã** (decoding strategy). Cùng một mô hình, cùng một prompt, đổi chiến lược giải mã là ra văn bản khác hẳn.

Và bài 01 cũng đã cho thấy vì sao bước này quan trọng không đều: phân bố có thể **rất sắc** (0,9989 cho một ký tự) hoặc **rất tù** (0,4663 cho ứng viên đầu). Ở chỗ sắc thì chọn kiểu gì cũng ra một kết quả. Ở chỗ tù thì cách chọn quyết định văn bản đi đâu.

Bài này dùng Qwen2.5-0.5B-Instruct, prompt cố định *"Viết một câu mở đầu cho bài giới thiệu về học máy."*, mỗi cấu hình chạy **ba seed khác nhau**.

## 2. Greedy: luôn lấy cái cao nhất

Cách đơn giản nhất: mỗi bước lấy token có xác suất cao nhất.

```python
out = model.generate(**enc, max_new_tokens=45, do_sample=False)
```

Kết quả với ba seed khác nhau:

```
seed 0: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
seed 1: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
seed 2: "Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó không chỉ là một công cụ…"
```

Ba bản **giống hệt nhau từng ký tự**. Đó là tính chất định nghĩa của greedy: nó **tất định**. Không có phép bốc ngẫu nhiên nào, nên seed chẳng ảnh hưởng gì.

Tính tất định ấy là điểm mạnh thật trong nhiều tình huống: trích xuất dữ liệu, phân loại, dịch thuật, hay bất cứ việc gì bạn cần chạy lại ra đúng kết quả cũ. Bài 08 sẽ dùng greedy cho toàn bộ thí nghiệm prompt chính vì lý do này.

Điểm yếu: nó chọn token tốt nhất **ở từng bước**, không phải chuỗi tốt nhất. Một token có xác suất cao ngay bây giờ có thể dẫn vào ngõ cụt. Và với việc cần đa dạng - viết quảng cáo, sinh ý tưởng, viết nhiều phiên bản - thì một câu trả lời duy nhất là vô dụng.

## 3. Temperature: bóp méo phân bố trước khi bốc

Muốn có đa dạng thì phải **bốc ngẫu nhiên** theo phân bố thay vì lấy cực đại. Và **temperature** là núm chỉnh độ sắc của phân bố trước khi bốc:

$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

- $T < 1$ làm phân bố **sắc hơn** - ứng viên mạnh càng mạnh
- $T = 1$ giữ nguyên phân bố mô hình đưa ra
- $T > 1$ làm phân bố **phẳng hơn** - ứng viên yếu được nâng lên

Con số nói rõ hơn lời. Với đúng một bước sinh, đo xem **năm ứng viên cao nhất chiếm bao nhiêu phần trăm tổng xác suất**:

| Temperature | Xác suất ứng viên đầu | Tổng của 5 ứng viên đầu |
|---|---|---|
| 0,3 | 0,9971 | **0,9994** |
| 0,8 | 0,6311 | 0,7799 |
| 1,5 | 0,1142 | 0,2147 |
| **2,5** | **0,0056** | **0,0147** |

Đọc dòng cuối cho kỹ. Ở $T = 2{,}5$, **năm ứng viên tốt nhất chỉ giữ 1,47% xác suất** - nghĩa là **98,5% còn lại nằm ở cái đuôi dài** gồm hàng trăm nghìn token mà mô hình vốn coi là gần như không thể. Bốc theo phân bố ấy thì gần như chắc chắn rơi vào đuôi.

Và đầu ra xác nhận đúng như vậy:

| $T$ | Mẫu |
|---|---|
| 0,3 | *"Để hiểu được cách học máy, chúng ta cần phải hiểu rằng nó là một lĩnh vực nghiên cứu và ứng dụng tiên tiến…"* |
| 0,8 | *"Trong thế giới hiện đại, học máy đang trở thành một công cụ quan trọng để phát triển các ứng dụng…"* |
| 1,5 | *"Bóng đeo tay trong số tất cả những vật dụng quen thuộc của bạn không có khả năng cung cấp giá trị cao nhất…"* |
| **2,5** | *"Bật ngõ vào tương lai**ment** hiện hành: Mở ra một khía cạnh học trò mà có thể dạy người khác…"* |

Ở 0,3 văn bản đúng chủ đề, gần như lặp lại bản greedy. Ở 0,8 vẫn đúng chủ đề nhưng cách mở bài đã khác. Ở 1,5 mô hình **đi lạc đề hoàn toàn** - "bóng đeo tay" chẳng liên quan gì tới học máy. Ở 2,5 thì ngôn ngữ bắt đầu vỡ: chữ `laiment` là tiếng Việt dính đuôi tiếng Anh, một dấu hiệu rõ của việc bốc trúng token mà mô hình gần như không bao giờ chọn.

## 4. Top-k và top-p: cắt đuôi trước khi bốc

Bảng ở mục 3 chỉ ra vấn đề thật của temperature cao: **cái đuôi**. Temperature nâng đều mọi ứng viên, kể cả những token hoàn toàn không hợp.

Hai kỹ thuật giải quyết bằng cách **loại bỏ đuôi trước khi bốc**:

**Top-k** - chỉ giữ $k$ ứng viên cao nhất, bỏ hết phần còn lại, chuẩn hóa lại rồi bốc. Đơn giản, nhưng $k$ cố định không hợp mọi lúc: ở chỗ phân bố sắc (bài 01: 0,9989 cho một ký tự) thì $k=10$ vẫn cho 9 ứng viên vô lý một cơ hội; ở chỗ thật sự phân vân thì $k=10$ có thể cắt mất ứng viên hợp lý.

**Top-p** (nucleus sampling) - giữ nhóm ứng viên nhỏ nhất có **tổng xác suất đạt $p$**, bỏ phần còn lại. Nhóm ấy **tự co giãn**: phân bố sắc thì nó chỉ gồm một hai token, phân bố tù thì nó mở rộng ra. Đây là cách hợp lý hơn top-k, và thường được bật sẵn - nhưng **giá trị mặc định thì mỗi nơi một khác**, nên đừng giả định.

Đầu ra trên cùng prompt:

| Cấu hình | Mẫu |
|---|---|
| top-k 10 | *"Trong thế giới hiện đại, học máy đang trở thành một công cụ quan trọng để phát triển…"* |
| top-p 0,5 | *"Trong thế giới hiện đại, học máy không chỉ là một công cụ giúp chúng ta xử lý thông tin…"* |
| top-p 0,9 | *"Bật tinh thần học máy - khởi chạy hệ thống tích lũy thông tin, đưa ra khả năng xử lý tương lai"* |

Hai cấu hình đầu cho văn bản đúng chủ đề, đọc được. Top-p 0,9 với temperature 1,0 thì đã bắt đầu kỳ quặc - *"Bật tinh thần học máy"* là một cụm không ai viết. Nó cho thấy top-p không phải phép màu: giữ 90% xác suất ở một phân bố vốn đã tù vẫn để lại rất nhiều ứng viên tệ.

**Bảy trên tám cấu hình có bốc ngẫu nhiên đều cho ba bản khác nhau với ba seed.** Chỉ greedy cho ba bản giống hệt. Đó là đánh đổi trung tâm của bài này: **tất định hay đa dạng - chọn một.**

> ⚠️ **Lưu ý nhỏ:** temperature, top-k và top-p **áp cùng lúc được**, và mặc định của thư viện thường không phải thứ bạn nghĩ. Trong `transformers`, đặt `temperature` mà quên `do_sample=True` thì tham số ấy **bị bỏ qua hoàn toàn** và bạn vẫn đang chạy greedy. Nhiều API cũng đặt sẵn `top_p` khác 1 dù bạn không khai báo. Trước khi kết luận "temperature không có tác dụng", hãy kiểm xem cấu hình bạn tưởng mình đang chạy có đúng là cấu hình đang chạy không - đây là loại lỗi âm thầm cùng họ với chuyện tiền xử lý ảnh ở chương Computer Vision · Bài 01.

## 5. Chọn cấu hình theo việc, không theo thói quen

Và "mặc định" cụ thể ra sao thì đáng kiểm tận nơi. Chính checkpoint Qwen2.5-0.5B-Instruct dùng suốt chương này ship sẵn một `generation_config`:

```
do_sample = True · temperature = 0,7 · top_p = 0,8 · top_k = 20
```

Tức nếu bạn gọi `generate()` mà không khai báo gì, bạn **không** chạy greedy và cũng **không** chạy top-p 0,9 - bạn chạy đúng bộ số trên. Mọi thí nghiệm ở mục 2 và 3 vì thế đều truyền tham số tường minh.

Gộp cả bài thành một bảng **điểm xuất phát** - không phải chuẩn chung, vì con số hợp lý phụ thuộc mô hình, bài toán và cách bạn chấm:

| Việc | Cấu hình | Lý do |
|---|---|---|
| Trích xuất, phân loại, gọi công cụ | **greedy** | Cần chạy lại ra đúng kết quả cũ; đa dạng là có hại |
| Hỏi đáp tài liệu, tóm tắt | greedy hoặc $T$ thấp (0,2-0,4) | Ưu tiên bám sát nguồn hơn là câu chữ đẹp |
| Trợ lý hội thoại | $T \approx 0{,}7$-0,8 với top-p 0,9 | Cân bằng tự nhiên và ổn định |
| Viết sáng tạo, sinh nhiều phương án | $T$ cao hơn, top-p 0,9-0,95 | Đa dạng là mục tiêu |
| Cần nhiều phương án rồi chọn | bốc nhiều bản rồi **lọc bằng tiêu chí riêng** | Đừng trông chờ một lần bốc trúng |

Dòng cuối đáng nói thêm. Khi cần chất lượng cao, cách hiệu quả hơn chỉnh temperature là **sinh nhiều bản rồi chọn** bằng một tiêu chí ngoài mô hình - kiểm tra định dạng, chạy thử code, đối chiếu ràng buộc. Chương Computer Vision · Bài 12 đã dùng đúng lối này: mô hình OCR đọc sai, nhưng ba ràng buộc số học của hóa đơn bắt được lỗi.

> 🔧 **Thử ngay:** hệ thống của bạn trích ngày tháng từ văn bản, chạy ở `temperature=0.7`. Đồng nghiệp báo cùng một tài liệu cho hai kết quả khác nhau ở hai lần chạy. Sửa thế nào, và vì sao đây là lỗi thiết kế chứ không phải lỗi mô hình?
> Đổi sang `do_sample=False`. Trích ngày tháng là việc có **đúng một đáp án**, nên đa dạng hoàn toàn là tác hại - bạn đang trả tiền cho một tính năng làm hỏng việc của mình. Đây là lỗi thiết kế vì mô hình đang làm đúng thứ nó được bảo làm: `temperature=0.7` là lệnh "hãy bốc ngẫu nhiên theo phân bố đã bóp", và nó tuân thủ. Sau khi đổi sang greedy, đầu ra trở nên tất định - và cũng nên thêm một bước kiểm định dạng bằng biểu thức chính quy, vì greedy chỉ đảm bảo *lặp lại được*, không đảm bảo *đúng*.

## Tóm tắt bài học

- Mô hình chỉ trả về **phân bố xác suất**; chọn token nào là **một bước riêng nằm ngoài mô hình**, gọi là chiến lược giải mã.
- **Greedy tất định**: ba seed khác nhau cho ba bản **giống hệt từng ký tự**. Hợp với trích xuất, phân loại, gọi công cụ.
- **Temperature** chia logit trước softmax. Đo trên một bước sinh: năm ứng viên đầu giữ **99,94%** xác suất ở $T=0{,}3$ nhưng chỉ **1,47%** ở $T=2{,}5$ - tức 98,5% dồn vào cái đuôi.
- Đầu ra khớp đúng con số ấy: $T=1{,}5$ lạc đề sang "bóng đeo tay", $T=2{,}5$ vỡ ngôn ngữ thành `laiment`.
- **Top-k** giữ $k$ ứng viên cố định; **top-p** giữ nhóm nhỏ nhất đạt tổng xác suất $p$ nên **tự co giãn theo độ sắc của phân bố** - hợp lý hơn top-k.
- **Mặc định của mỗi checkpoint mỗi khác**: Qwen2.5-0.5B-Instruct ship sẵn `temperature=0,7`, `top_p=0,8`, `top_k=20`, `do_sample=True`. Gọi `generate()` không tham số là chạy đúng bộ ấy, không phải greedy.
- Top-p không phải phép màu: top-p 0,9 ở nhiệt độ 1,0 vẫn cho *"Bật tinh thần học máy"*.
- **Bảy trên tám cấu hình có bốc ngẫu nhiên cho ba bản khác nhau**; chỉ greedy lặp lại được. Tất định hay đa dạng - chọn một.
- Bẫy hay gặp: đặt `temperature` mà quên `do_sample=True` thì tham số **bị bỏ qua** và bạn vẫn đang chạy greedy.
- Khi cần chất lượng cao, **sinh nhiều bản rồi lọc bằng tiêu chí ngoài mô hình** hiệu quả hơn chỉnh temperature.

## Câu hỏi tự kiểm tra

1. Vì sao nói "mô hình không sinh văn bản"? Bước nào mới thực sự tạo ra token?
2. Greedy cho ba bản giống hệt với ba seed. Giải thích, và nêu hai việc mà tính chất ấy là điều bạn muốn.
3. Ở $T = 2{,}5$, năm ứng viên đầu chỉ giữ 1,47% xác suất. Con số ấy dự đoán điều gì về đầu ra, và đầu ra thật có khớp không?
4. So sánh top-k với top-p. Vì sao top-p hợp lý hơn khi độ sắc của phân bố thay đổi theo ngữ cảnh?
5. Bài 01 đo được phân bố 0,9989 ở chỗ này và 0,4663 ở chỗ kia. Điều đó ảnh hưởng thế nào tới tác dụng của các núm vặn?
6. Bạn đặt `temperature=1.2` nhưng đầu ra vẫn lặp lại y hệt mỗi lần chạy. Nêu nguyên nhân khả dĩ nhất.
7. Với hệ hỏi đáp tài liệu, vì sao nên dùng temperature thấp thay vì cao?
8. Vì sao "sinh nhiều bản rồi lọc" thường hiệu quả hơn chỉnh temperature khi cần chất lượng? Cho ví dụ tiêu chí lọc trong một bài toán cụ thể.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 01 của chương này** | Phân bố token kế tiếp, và chuyện nó sắc hay tù tùy ngữ cảnh |
| **Chương Machine Learning · Bài 03** | Softmax và cách nhiệt độ bóp méo nó |
| **Chương Computer Vision · Bài 12** | Dùng ràng buộc ngoài mô hình để bắt lỗi đầu ra |

**Nguồn online bổ sung (miễn phí):**

- [Holtzman và cộng sự - The Curious Case of Neural Text Degeneration (ICLR 2020)](https://arxiv.org/abs/1904.09751) - bài giới thiệu top-p, và giải thích vì sao lấy cực đại lại sinh văn bản lặp.
- [Generation strategies - tài liệu Hugging Face](https://huggingface.co/docs/transformers/generation_strategies) - mọi tham số của `generate`, gồm cái bẫy `do_sample`.
- [Fan, Lewis, Dauphin - Hierarchical Neural Story Generation (ACL 2018)](https://arxiv.org/abs/1805.04833) - nguồn của top-k sampling.
- [How to generate text - blog Hugging Face](https://huggingface.co/blog/how-to-generate) - so sánh greedy, beam search, top-k và top-p bằng ví dụ chạy được.

> **Bài tiếp theo:** [Quy mô mô hình: cái gì ta biết và cái gì không](../model-scale-what-we-know-and-what-we-dont-vi/) - năm bài đầu đều dùng một mô hình 0,5 tỉ tham số và nó làm được kha khá. Bài sau đo xem nó **gãy ở đâu** - rồi bàn thẳng một câu hỏi khó: từ việc quan sát một mô hình nhỏ, ta suy ra được gì và **không** suy ra được gì về mô hình lớn.

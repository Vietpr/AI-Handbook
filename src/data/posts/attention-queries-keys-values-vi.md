---
title: "Attention: truy vấn, khóa, giá trị"
description: "Cơ chế mà hai chương trước cố ý để lại, tính một phép attention bằng tay, và vì sao không được đọc ma trận attention như lời giải thích cho câu trả lời của mô hình."
domain: "Generative AI"
section: "Learn"
language: "vi"
translationKey: "attention-queries-keys-values"
order: 3
pubDate: 2026-09-01
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu cơ chế mà hai chương trước cố ý để lại, tính được một phép attention bằng tay, và - quan trọng không kém - biết vì sao **không được đọc ma trận attention như lời giải thích** cho câu trả lời của mô hình, với bằng chứng đo trên một mô hình thật.

## 1. Món nợ của hai chương

Chương Deep Learning · Bài 10 cho thấy mạng hồi quy phải đi tuần tự qua từng bước, nên tín hiệu từ đầu chuỗi bị teo dần và không tính song song được. Bài ấy giới thiệu attention ở mức khái niệm - *mọi vị trí nối thẳng tới mọi vị trí* - rồi dừng.

Chương Computer Vision · Bài 10 dùng đúng cơ chế ấy cho ảnh, nhưng cũng chỉ mở phần riêng của ảnh: cắt mảnh, nhúng vị trí. Cơ chế bên trong thì để lại.

Bài này trả cả hai món nợ.

## 2. Ba vai trò cho cùng một vector

Ý tưởng gốc rất giống tra cứu trong một cuốn từ điển. Bạn có một **câu hỏi**, từ điển có các **mục từ**, và mỗi mục từ có một **nội dung**. Bạn so câu hỏi với các mục từ để biết nên lấy nội dung nào.

Attention làm đúng thế, nhưng "mềm": thay vì chọn một mục, nó lấy **trung bình có trọng số** của mọi nội dung.

Mỗi vector đầu vào $x_i$ được chiếu thành ba thứ bằng ba ma trận học được:

$$q_i = W_Q x_i \qquad k_i = W_K x_i \qquad v_i = W_V x_i$$

- **Truy vấn (query)** $q_i$ - "tôi đang cần thông tin gì"
- **Khóa (key)** $k_j$ - "tôi có thông tin loại gì"
- **Giá trị (value)** $v_j$ - "nội dung tôi mang"

Rồi ba bước:

$$\text{điểm}_{ij} = \frac{q_i \cdot k_j}{\sqrt{d}} \qquad \alpha_{ij} = \text{softmax}_j(\text{điểm}_{ij}) \qquad y_i = \sum_j \alpha_{ij} v_j$$

Viết gọn lại thành công thức quen thuộc:

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

**Vì sao chia cho $\sqrt{d}$.** Tích vô hướng của hai vector $d$ chiều có độ lớn tăng theo $d$. Không chia thì với $d$ lớn, các điểm số phình to, softmax bão hòa thành gần như one-hot, và gradient teo - đúng bệnh sigmoid của chương Deep Learning · Bài 08. Phép chia giữ điểm số ở thang hợp lý.

**Vì sao cần mặt nạ nhân quả.** Mô hình ngôn ngữ phải đoán token kế tiếp, nên khi xử lý vị trí $i$ nó được nhìn mọi vị trí $\le i$ nhưng **không được nhìn** các vị trí $> i$ - tức chặn phía trước, không chặn phía sau. Nếu không, bài toán thành nhìn trộm đáp án - đúng chuyện lộ đề của Foundations · Bài 14, ở dạng cấu trúc. Cách chặn là đặt điểm số của mọi cặp $(i, j>i)$ thành $-\infty$ trước softmax, để chúng nhận trọng số 0.

```
      Hà   Nội   là   thủ   đô
Hà    ✓    ✗     ✗    ✗     ✗
Nội   ✓    ✓     ✗    ✗     ✗
là    ✓    ✓     ✓    ✗     ✗
thủ   ✓    ✓     ✓    ✓     ✗
đô    ✓    ✓     ✓    ✓     ✓
```

## 3. Nhiều đầu: nhiều phép trộn song song

Một phép attention chỉ cho ra **một** cách trộn thông tin. Nhưng trong một câu có nhiều loại quan hệ cùng lúc - chủ ngữ với động từ, đại từ với thứ nó thay thế, danh từ với tính từ bổ nghĩa.

**Multi-head attention** chạy nhiều phép attention song song, mỗi phép có bộ $W_Q, W_K, W_V$ riêng, rồi ghép kết quả lại và chiếu một lần nữa. Mỗi "đầu" tự do học một kiểu quan hệ khác nhau.

Chi phí không tăng, vì người ta chia chiều: mô hình 896 chiều với 14 đầu thì mỗi đầu làm việc ở 64 chiều.

> ⚠️ **Lưu ý nhỏ:** mô tả trên là **multi-head attention cổ điển**, nơi mỗi đầu có bộ khóa và giá trị riêng. Nhiều mô hình hiện đại dùng biến thể tiết kiệm hơn: Qwen2.5-0.5B có **14 đầu truy vấn nhưng chỉ 2 bộ khóa-giá trị**, tức bảy đầu truy vấn dùng chung một bộ khóa-giá trị. Cách này gọi là **grouped-query attention (GQA)**, và nó sinh ra để giảm bộ nhớ lúc suy luận - chuyện của chương AI Systems. Với bài này thì cơ chế không đổi; chỉ là khi bạn đọc config một mô hình thật, đừng ngạc nhiên nếu `num_key_value_heads` nhỏ hơn `num_attention_heads`.

Và các đầu **thật sự** khác nhau. Lấy Qwen2.5-0.5B - **24 tầng, 14 đầu truy vấn** - chạy trên câu *"Hà Nội là thủ đô của Việt Nam. Thành phố này nằm ở miền Bắc."* rồi xem bốn đầu đầu tiên của tầng cuối, token cuối đang nhìn vào đâu:

| Đầu | Ba token được chú ý nhất |
|---|---|
| 0 | `.` 0,2629 · ` Nội` 0,1958 · `.` 0,1688 |
| **1** | **`.` 0,9919** · ` Bắc` 0,0027 · ` này` 0,0024 |
| 2 | `.` 0,2292 · ` Nội` 0,2062 · `.` 0,1516 |
| 3 | `.` 0,2680 · ` Nội` 0,1582 · `.` 0,1495 |

Đầu số 1 dồn **99,19%** trọng số vào đúng một token, trong khi ba đầu kia trải đều. Cùng một tầng, cùng một đầu vào, hành xử hoàn toàn khác nhau. Đó là lý do người ta dùng nhiều đầu.

## 4. Attention nhìn vào đâu - và vì sao đừng đọc nó như lời giải thích

Đến đây là chỗ dễ sa vào một cái bẫy rất phổ biến. Vì ma trận attention nhìn được, người ta hay trỏ vào nó và nói: *"mô hình chú ý vào từ này, nên nó trả lời như kia."*

Chương Computer Vision · Bài 04 đã gặp đúng cái bẫy ấy với Grad-CAM, và phép thử che cho thấy một phần ba số ảnh **phản bác chính tấm heatmap**. Ở đây ta giữ đúng kỷ luật đó.

Nhìn xem token cuối của câu trên chú ý vào đâu, ở ba tầng khác nhau:

| Tầng | Bốn token được chú ý nhất |
|---|---|
| 0 | `.` 0,2367 · ` Bắc` 0,2199 · `.` 0,0967 · `H` 0,0900 |
| 12 | `H` 0,3084 · `.` 0,1225 · ` Bắc` 0,1035 · `.` 0,0826 |
| 23 | `.` 0,2796 · `.` 0,2754 · ` Nội` 0,0960 · `H` 0,0889 |

Thứ chiếm phần lớn trọng số là **dấu chấm** và **token đầu câu** - không phải những từ mang nghĩa. Nếu đọc bảng này như lời giải thích, ta phải kết luận rằng mô hình trả lời dựa vào dấu câu.

Và hiện tượng ấy còn có hệ thống hơn thế. Đo tỉ lệ attention đổ vào **token đầu tiên** của chuỗi, trung bình trên mọi đầu, ở cả 24 tầng:

```
tầng  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
tỉ lệ .15 .24 .15 .71 .59 .65 .61 .73 .45 .79 .56 .78 .51 .71 .56 .59 .91 .73 .73 .63 .83 .82 .25 .15
                          ▲                                    ▲
                       vọt lên 71%                        đỉnh 91%
```

Ở tầng 16, **91% toàn bộ attention đổ vào token đầu tiên**. Trung bình các tầng giữa cũng vào khoảng 60-80%.

Token đầu tiên ấy không mang thông tin gì đặc biệt - nó chỉ là token mở đầu. Hiện tượng này có tên trong nghề: **attention sink** (hố attention). Một cách giải thích được chấp nhận rộng rãi là softmax buộc tổng trọng số bằng 1, nên khi một đầu **không cần lấy thông tin từ đâu cả**, nó phải đổ trọng số đi đâu đó - và nó chọn một vị trí vô hại.

Ba điều rút ra, và chúng quan trọng hơn bản thân con số:

**Trọng số attention cao không có nghĩa là quan trọng.** Phần lớn trọng số ở đây đổ vào chỗ vô nghĩa nhất có thể.

**Attention chỉ là một mảnh của phép tính.** Đầu ra của mỗi tầng còn đi qua phép chiếu, kết nối tắt, khối FFN, rồi 23 tầng nữa. Nhìn một ma trận attention ở một tầng là nhìn một mắt xích trong một chuỗi rất dài.

**Nó vẫn hữu ích - chỉ là hữu ích cho việc khác.** Ma trận attention cho ta **nhìn thấy trực tiếp một phép trộn thông tin bên trong mô hình**, thứ rất hiếm ở học sâu. Dùng nó để chẩn đoán, để hình thành giả thuyết, để phát hiện hành vi lạ. Đừng dùng nó làm bằng chứng về lý do mô hình trả lời.

> ⚠️ **Lưu ý nhỏ:** mọi con số ở mục 3 và 4 đo trên **đúng một câu**, với **một mô hình**. Chúng minh họa các hiện tượng, không đo tần suất của chúng. Attention sink thì đã được nhiều nghiên cứu ghi nhận trên nhiều mô hình khác nhau - có trong phần Đọc thêm - nhưng con số 91% cụ thể ở tầng 16 là của riêng lần chạy này.

## 5. Vì sao attention thay được RNN

Quay lại vấn đề mà chương Deep Learning · Bài 10 để lại.

**Đường đi ngắn.** Với mạng hồi quy, tín hiệu từ token 1 tới token 500 phải đi qua 499 bước, mỗi bước nhân thêm một ma trận - đúng công thức tiêu biến gradient. Với attention, hai vị trí bất kỳ nối **thẳng** qua một phép tính: khoảng cách luôn là 1.

**Tính song song.** Mạng hồi quy phải tính xong bước $t$ mới tính được bước $t+1$. Attention tính $QK^\top$ cho **toàn bộ** các cặp vị trí trong một phép nhân ma trận. Đây mới là lý do thực sự khiến transformer thắng: không phải nó thông minh hơn, mà nó **huấn luyện được trên phần cứng song song**, nên người ta train được mô hình lớn hơn nhiều trên dữ liệu nhiều hơn nhiều.

**Cái giá.** Ma trận $QK^\top$ có kích thước $n \times n$ với $n$ là độ dài chuỗi, nên chi phí tính toán và bộ nhớ tăng **theo bình phương** độ dài. Gấp đôi độ dài ngữ cảnh là gấp bốn chi phí. Đây là lý do cửa sổ ngữ cảnh từng là con số nhỏ, và là bài toán mà rất nhiều nghiên cứu tìm cách né - nhưng đó là chuyện của chương AI Systems.

> 🔧 **Thử ngay:** một mô hình có ngữ cảnh 4.096 token. Bạn muốn nâng lên 16.384. Chi phí phần attention tăng bao nhiêu lần?
> Gấp **16 lần**, không phải 4. Số cặp vị trí là $n^2$, nên tăng $n$ lên 4 lần thì số cặp tăng $4^2 = 16$ lần, kéo theo cả thời gian lẫn bộ nhớ cho ma trận trọng số. Đây là lý do các mô hình ngữ cảnh dài không đơn giản là "cùng kiến trúc, đặt số lớn hơn" - chúng cần những kỹ thuật riêng để né bức tường bình phương ấy.

## Tóm tắt bài học

- Attention chiếu mỗi vector đầu vào thành ba vai trò: **truy vấn** (cần gì), **khóa** (có gì), **giá trị** (nội dung), rồi lấy trung bình có trọng số của các giá trị.
- Chia cho $\sqrt{d}$ để tích vô hướng không phình theo số chiều, tránh softmax bão hòa và gradient teo.
- **Mặt nạ nhân quả** chặn mỗi vị trí nhìn về **phía trước** - vị trí $i$ chỉ được nhìn các vị trí $\le i$; không có nó thì mô hình nhìn trộm đáp án.
- **Nhiều đầu** cho phép học nhiều kiểu quan hệ song song, và chúng thật sự khác nhau: trong cùng một tầng của Qwen2.5-0.5B, một đầu dồn **99,19%** vào một token còn ba đầu kia trải đều.
- **Đừng đọc ma trận attention như lời giải thích.** Trên câu thử, phần lớn trọng số đổ vào **dấu chấm** và **token đầu câu**, không phải từ mang nghĩa.
- **Attention sink**: tỉ lệ attention đổ vào token đầu tiên đạt **91% ở tầng 16**, và 60-80% ở phần lớn các tầng giữa - một cách để đầu attention "không lấy gì cả" khi softmax bắt buộc tổng bằng 1.
- Attention thắng mạng hồi quy nhờ **đường đi luôn dài 1** và **tính song song được**, chứ không phải nhờ thông minh hơn.
- Cái giá là chi phí **bình phương theo độ dài chuỗi**: gấp đôi ngữ cảnh là gấp bốn chi phí.

## Câu hỏi tự kiểm tra

1. Giải thích vai trò của truy vấn, khóa và giá trị bằng ví dụ tra từ điển. Chỗ nào attention khác phép tra thật?
2. Vì sao phải chia cho $\sqrt{d}$? Chuyện gì xảy ra nếu bỏ?
3. Mặt nạ nhân quả chặn cái gì, và vì sao thiếu nó thì việc huấn luyện trở nên vô nghĩa?
4. Trong cùng một tầng, một đầu dồn 0,9919 vào một token còn các đầu khác trải đều. Điều đó biện minh cho thiết kế nào?
5. Qwen2.5-0.5B có 14 đầu truy vấn nhưng chỉ 2 bộ khóa-giá trị. Kỹ thuật ấy tên gì, và nó tiết kiệm cái gì?
6. Phần lớn attention đổ vào dấu chấm và token đầu câu. Vì sao điều đó bác bỏ cách đọc "mô hình chú ý vào X nên trả lời Y"?
7. Attention sink là gì, và cách giải thích nào khớp với việc softmax buộc tổng trọng số bằng 1?
8. Nêu hai lý do khiến attention thay được mạng hồi quy, và lý do nào quan trọng hơn trong thực tế.
9. Ngữ cảnh tăng từ 2.048 lên 8.192 token. Chi phí phần attention tăng bao nhiêu lần? Giải thích.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Chương Deep Learning · Bài 10** | Vì sao mạng hồi quy hụt hơi, và giới thiệu attention ở mức khái niệm |
| **Chương Computer Vision · Bài 10** | Attention áp cho ảnh: cắt mảnh, nhúng vị trí |
| **Chương Computer Vision · Bài 04** | Grad-CAM và phép thử che - cùng kỷ luật "công cụ nhìn được không phải lời giải thích" |
| **Dive into Deep Learning (d2l.ai)** | Chương 11.1-11.5 *Attention Mechanisms and Transformers* |

**Nguồn online bổ sung (miễn phí):**

- [Vaswani và cộng sự - Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) - bài báo gốc; mục 3.2 là đúng công thức ở mục 2.
- [Alammar - The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - bản giải thích bằng hình được dẫn nhiều nhất.
- [Jain & Wallace - Attention is not Explanation (NAACL 2019)](https://arxiv.org/abs/1902.10186) - nền học thuật cho cảnh báo ở mục 4.
- [Xiao và cộng sự - Efficient Streaming Language Models with Attention Sinks (ICLR 2024)](https://arxiv.org/abs/2309.17453) - đặt tên và phân tích hiện tượng đo được ở mục 4.
- [Bertviz](https://github.com/jessevig/bertviz) - công cụ xem ma trận attention tương tác, tiện để tự thử trên câu của bạn.

> **Bài tiếp theo:** [Transformer từ đầu](../transformer-from-scratch-vi/) - attention là mảnh quan trọng nhất nhưng không phải mảnh duy nhất. Bài sau ráp đủ một Transformer - thông tin vị trí, chuẩn hóa, khối FFN, kết nối tắt - rồi **tháo từng viên gạch ra** để đo xem thiếu cái nào thì hỏng đến mức nào. Kết quả có một chỗ rất bất ngờ.

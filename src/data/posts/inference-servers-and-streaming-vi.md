---
title: "Máy chủ suy luận và truyền theo luồng"
description: "Máy chủ suy luận thêm gì quanh mô hình, vì sao mất cả phút mới sẵn sàng, và số đo cho thấy truyền theo luồng không làm mô hình nhanh hơn, chỉ đổi lúc người dùng bắt đầu nhận kết quả."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "inference-servers-and-streaming"
order: 8
pubDate: 2026-09-09
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết một máy chủ suy luận thêm những gì vào quanh mô hình, vì sao nó mất cả phút mới sẵn sàng, và đo được bằng số rằng **truyền theo luồng không làm mô hình nhanh hơn chút nào** - nó chỉ đổi thời điểm người dùng bắt đầu nhận được giá trị.

Tới bài 06, mọi phép đo đều gọi mô hình từ bên trong một script. Người dùng thật không làm thế. Họ gửi một yêu cầu HTTP, và giữa họ với mô hình là một **máy chủ suy luận**: nhận yêu cầu, xếp hàng, gộp lô, gọi mô hình, trả kết quả.

Bài này dựng một máy chủ như vậy trên đúng bản 7B mà bài 04 đã nhét vừa một card, rồi đo một thứ mà trực giác hay nói sai.

## 1. Từ script sang dịch vụ

Một lệnh là đủ để có máy chủ:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --quantization awq_marlin \
    --max-model-len 4096 --gpu-memory-utilization 0.9 --port 8011
```

Máy chủ này mở ra ba đường mà bài này và các bài sau dùng tới:

| Đường | Dùng để làm gì |
|---|---|
| `POST /v1/chat/completions` | Nhận yêu cầu theo đúng định dạng API phổ biến, có hoặc không truyền theo luồng |
| `GET /health` | Trả lời một câu duy nhất: máy chủ đã sẵn sàng nhận việc chưa |
| `GET /metrics` | Xuất các đại lượng đo từ **bên trong** bộ máy - bài 11 dựng cả bài trên đường này |

**Khởi động không tức thì.** Từ lúc gõ lệnh tới lúc `/health` trả lời OK mất **90 giây**. Trong đó nạp trọng số chỉ tốn **23,4 giây**; phần còn lại là bộ máy đo bộ nhớ, dành vùng KV cache, và chuẩn bị sẵn các nhân tính toán cho những kích thước lô hay gặp. Hệ quả rất thực dụng: **đừng đưa lưu lượng vào một máy chủ vừa khởi động** chỉ vì tiến trình đã chạy. Tiến trình sống không có nghĩa là sẵn sàng. Bộ điều phối - dù là một script đơn giản hay Kubernetes - phải hỏi `/health` và đợi.

Lúc khởi động, máy chủ tự báo nó có chỗ cho **77.744 token** KV cache, tức khoảng **18,98 yêu cầu đồng thời** nếu mỗi yêu cầu dùng hết 4.096 token ngữ cảnh. Đây là nguồn số thứ ba mà bài 02 đã dặn đọc riêng: *sức chứa đã dành trước*, không phải *lượng đang dùng*. Nó cũng lệch vài phần trăm so với con số 76.208 token khi chạy cùng mô hình ở chế độ script ở bài 04, vì hai cách chạy dành bộ nhớ cho những thứ phụ khác nhau. Đừng đọc chênh lệch nhỏ ấy thành một phát hiện.

## 2. Ba đại lượng, và hai chế độ trả lời

Một yêu cầu có thể được trả lời theo hai cách.

**Không truyền theo luồng:** máy chủ sinh xong toàn bộ câu trả lời rồi mới gửi một khối duy nhất.

**Truyền theo luồng** (*streaming*): máy chủ gửi từng mẩu ngay khi sinh ra.

Rất dễ nói "truyền theo luồng làm TTFT giảm". Câu ấy sai, và sai theo cách làm hỏng chẩn đoán. Phải tách ba đại lượng:

```
                         mô hình sinh token 1        mô hình sinh token cuối
                                  │                            │
yêu cầu đến ─────────────────────►●────────────────────────────●
                                  │                            │
               TTFT của engine ───┘                            │
                                                               │
KHÔNG LUỒNG:  người dùng thấy gì ... chưa thấy gì ... chưa ... ● ← thấy tất cả cùng lúc
CÓ LUỒNG:     người dùng thấy     ● token 1 ● 2 ● 3 ● ...      ● ← xong
                                  ▲
                    thời điểm thấy đầu ra đầu tiên
```

| Đại lượng | Ai đo | Nghĩa |
|---|---|---|
| **TTFT của engine** | Máy chủ, qua `/metrics` | Từ lúc yêu cầu **đến máy chủ** tới lúc mô hình **sinh ra** token đầu tiên - gồm cả chờ hàng đợi |
| **Thời điểm thấy đầu ra đầu tiên** | Client | Lúc người dùng **nhận được** mẩu chữ đầu tiên |
| **E2E** | Cả hai | Lúc có đủ câu trả lời |

Ở chế độ không luồng, token đầu tiên vẫn được sinh ra đúng lúc ấy. Nó chỉ nằm chờ trong máy chủ cho tới khi cả câu xong.

## 3. Đo: có và không truyền theo luồng

Hai mươi yêu cầu **tuần tự** - không yêu cầu nào tranh chỗ với yêu cầu nào - mỗi yêu cầu sinh đúng 256 token, chạy một lượt ở chế độ có luồng rồi một lượt không luồng. TTFT của engine đọc từ `/metrics` trước và sau mỗi lượt, lấy hiệu.

| | Thấy đầu ra đầu tiên (client) | E2E (client) | **TTFT của engine** | E2E (engine) | Chờ hàng đợi (engine) |
|---|---|---|---|---|---|
| **Có luồng** | **0,026 s** | 3,852 s | **0,0284 s** | 3,847 s | 0,0 s |
| **Không luồng** | **3,850 s** | 3,850 s | **0,0281 s** | 3,850 s | 0,0 s |

Mô hình: Qwen2.5-7B-Instruct-AWQ, nhân Marlin, một RTX 3060.

## 4. Truyền theo luồng không làm mô hình nhanh hơn

**TTFT của engine không đổi: 0,0284 so với 0,0281 giây.** Truyền theo luồng **không làm mô hình sinh token đầu nhanh hơn**. Nó không chạm vào mô hình.

**Thời điểm thấy đầu ra đầu tiên đổi gần 150 lần: 0,026 so với 3,850 giây.** Ở chế độ không luồng, người dùng nhìn màn hình trống gần bốn giây - không phải vì mô hình chậm, mà vì token đầu tiên đã sinh ra từ giây thứ 0,028 và **bị giữ lại** cho tới khi 255 token còn lại xong.

**E2E không đổi: 3,852 so với 3,850 giây.** Máy chẳng nhanh lên hay chậm đi. Ở phép đo này, chi phí thêm của việc gửi từng mẩu nhỏ không đủ lớn để đo thấy.

Nên phát biểu đúng là: **truyền theo luồng không làm mô hình nhanh hơn; nó đổi thời điểm người dùng bắt đầu nhận được giá trị.** Với một câu trả lời 256 token, đó là khác biệt giữa "trợ lý phản hồi ngay" và "trợ lý treo bốn giây".

> ⚠️ **Hai đồng hồ, đừng so ở mức mili-giây.** Ở dòng có luồng, client thấy chữ đầu lúc 0,026 s, còn engine báo TTFT 0,0284 s - client *sớm hơn* engine 2 ms, điều nghe vô lý. Không có gì vô lý: hai con số đo từ **hai điểm bắt đầu khác nhau**, một bên là trung vị và một bên là trung bình, trên hai đồng hồ khác nhau. Đây đúng là bài học "ba nguồn số" ở bài 02. So hai nguồn thì so ở mức **cỡ độ lớn**; chênh lệch vài mili-giây giữa hai nguồn là nhiễu của định nghĩa, không phải phát hiện.

## 5. Khi nào truyền theo luồng không cứu được

Kết quả trên dễ bị đẩy thành "lúc nào cũng nên bật luồng". Có ba trường hợp nó không giúp gì.

**Khi chính TTFT đã lớn.** Truyền theo luồng chỉ cắt khoảng chờ **sau** token đầu. Nếu token đầu đến muộn - vì prompt dài ở bài 01, vì cả lô phải nạp cùng lúc ở bài 05, hay vì yêu cầu nằm trong hàng đợi - thì người dùng vẫn nhìn màn hình trống đúng ngần ấy. Bảng trên có hàng đợi bằng 0 và prompt ngắn nên TTFT chỉ 0,028 s; bài 07 sẽ cho thấy dưới tải, con số ấy phình ra thế nào.

**Khi người đọc là một chương trình.** Nếu đầu ra được đưa vào một trình phân tích JSON, hay là một lệnh gọi công cụ, thì chương trình phải chờ đủ câu mới làm gì được. Nửa câu JSON không dùng được. Lợi ích của truyền theo luồng là lợi ích về **cảm nhận của con người**; máy thì không cảm nhận.

**Khi hạ tầng ở giữa gom lại.** Một proxy hay bộ cân bằng tải có thể **gom** các mẩu nhỏ lại rồi mới gửi đi, xóa sạch lợi ích ở trên mà máy chủ không hề biết. Cách duy nhất để biết là **đo từ phía client thật**, qua đúng đường mạng mà người dùng đi, chứ không đo ngay trên máy chủ như bài này.

Truyền theo luồng còn có giá về mặt vận hành. Khi máy chủ đã gửi mẩu đầu tiên, nó đã trả mã HTTP thành công - nên nếu lỗi xảy ra giữa chừng, không còn cách báo lỗi bằng mã trạng thái nữa, phải báo trong chính luồng dữ liệu. Và nếu người dùng đóng tab giữa chừng, máy chủ nên dừng sinh cho yêu cầu ấy, nếu không nó tiếp tục đốt GPU cho một câu trả lời không ai đọc. Hai điều này bài không đo, nhưng là hai thứ bạn nên **kiểm trên máy chủ của mình** trước khi triển khai.

> 🔧 **Thử ngay:** người dùng phàn nàn trợ lý "lúc nào cũng khựng khoảng một giây rồi mới chạy chữ", dù đã bật truyền theo luồng. Bạn đo gì?
> Tách đúng ba đại lượng của mục 2. Đọc **TTFT của engine** từ `/metrics`: nếu nó cũng khoảng một giây thì vấn đề nằm ở mô hình hoặc hàng đợi - prompt quá dài, lô quá lớn, hay máy chủ đang quá tải - và truyền theo luồng vốn không chạm được vào đó. Nếu TTFT của engine chỉ vài chục mili-giây mà **client** thấy chữ đầu sau một giây, thì một giây ấy mất ở **giữa đường**: một proxy đang gom dữ liệu, hay mạng chậm. Một con số đo ở hai đầu cho biết phải sửa ở đâu; một con số đo ở một đầu thì chỉ cho biết người dùng đang khó chịu.

## 6. Những gì một máy chủ phải lo ngoài mô hình

Bài này dùng một máy chủ có sẵn, nhưng đáng nêu những thứ nó làm hộ - vì bạn sẽ phải lo nếu tự viết, và phải cấu hình đúng nếu dùng sẵn:

- **Sẵn sàng khác với còn sống.** `/health` phải trả lời "chưa" cho tới khi mô hình nạp xong - 90 giây ở đây.
- **Giới hạn đầu vào.** `--max-model-len 4096` từ chối thẳng những yêu cầu dài hơn, thay vì để chúng làm hết bộ nhớ của mọi người khác.
- **Hàng đợi và gộp lô.** Mọi thứ ở bài 05 và bài 06 xảy ra bên trong, người gọi không thấy.
- **Đo từ bên trong.** `/metrics` xuất sẵn TTFT, thời gian giữa hai token, E2E, thời gian chờ hàng đợi, thời gian nạp prompt và thời gian sinh token, số token vào và ra. Bài 11 dùng đúng danh sách này.

## Tóm tắt bài học

- Một máy chủ suy luận thêm vào quanh mô hình: API HTTP, hàng đợi, gộp lô, kiểm tra sẵn sàng qua `/health`, và đo lường nội bộ qua `/metrics`.
- Máy chủ 7B AWQ ở đây mất **90 giây** mới sẵn sàng, dù nạp trọng số chỉ **23,4 giây**. Tiến trình sống không có nghĩa là sẵn sàng - bộ điều phối phải đợi `/health`.
- Phải tách **ba** đại lượng: TTFT của engine, thời điểm client thấy đầu ra đầu tiên, và E2E.
- Đo trên 20 yêu cầu tuần tự, 256 token mỗi yêu cầu: **TTFT của engine gần như không đổi** (0,0284 s có luồng, 0,0281 s không luồng); **thời điểm thấy chữ đầu đổi gần 150 lần** (0,026 s so với 3,850 s); **E2E không đổi** (3,852 so với 3,850 s).
- Phát biểu đúng: **truyền theo luồng không làm mô hình nhanh hơn; nó đổi thời điểm người dùng bắt đầu nhận được giá trị.**
- Hai nguồn đo (client và engine) chỉ nên so ở mức cỡ độ lớn; chênh 2 ms giữa chúng là nhiễu của định nghĩa.
- Truyền theo luồng **không** cứu được khi TTFT đã lớn, khi người đọc là một chương trình, hay khi hạ tầng ở giữa gom dữ liệu lại. Và nó đổi cách báo lỗi giữa chừng, cùng yêu cầu dừng sinh khi người dùng rời đi.

## Câu hỏi tự kiểm tra

1. Vì sao "tiến trình máy chủ đã chạy" chưa đủ để đưa lưu lượng vào? Nêu con số cụ thể trong bài.
2. Giải thích vì sao TTFT của engine gần như bằng nhau ở hai chế độ, trong khi thời điểm người dùng thấy chữ đầu lại chênh gần 150 lần.
3. Ở chế độ không luồng, token đầu tiên được sinh ra lúc nào, và nó ở đâu cho tới khi người dùng nhận được?
4. Client thấy chữ đầu lúc 0,026 s còn engine báo TTFT 0,0284 s. Vì sao không được kết luận client "nhanh hơn" engine?
5. Nêu ba trường hợp mà bật truyền theo luồng không cải thiện được trải nghiệm, và giải thích từng trường hợp bằng một đại lượng ở mục 2.
6. Một đồng nghiệp đo thấy truyền theo luồng hoạt động tốt khi gọi thẳng máy chủ, nhưng người dùng thật vẫn thấy chữ hiện ra từng cục lớn. Bạn nghi ngờ điều gì, và đo ở đâu để xác nhận?
7. Vì sao việc báo lỗi trở nên khó hơn khi đã bắt đầu truyền theo luồng?
8. Một hệ sinh mã JSON để một chương trình khác đọc. Có nên bật truyền theo luồng không? Lập luận bằng các đại lượng trong bài.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 01 của chương này** | Năm đại lượng, công thức E2E ≈ TTFT + (N−1)·TPOT, và quy ước TTFT tính từ lúc yêu cầu đến |
| **Bài 02 của chương này** | Ba nguồn số, và vì sao số của engine không so thẳng được với số của bạn |
| **Bài 05 của chương này** | Vì sao gộp lô tĩnh làm TTFT tăng tuyến tính theo batch |

**Đi sâu hơn:**

- [vLLM - OpenAI-Compatible Server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) - các tham số dòng lệnh dùng ở mục 1.
- [vLLM - Production Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) - ý nghĩa từng đại lượng trong `/metrics`, nền cho bài 11.
- [MDN - Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) - cơ chế truyền theo luồng mà API dạng này dùng.

**Mã nguồn của bài:** [`code/b08_luong.py`](../../code/ai-systems/b08_luong.py) - đo ba đại lượng ở hai chế độ, đọc TTFT của engine từ `/metrics`.

> **Bài tiếp theo:** [Hai loại bộ đệm rất khác nhau](../two-very-different-kinds-of-cache-vi/) - máy chủ đã chạy và đo được. Bài sau thử cắt bớt công việc của nó bằng bộ đệm - và cho thấy "bộ đệm" trong hệ mô hình ngôn ngữ là tên chung của hai thứ khác hẳn nhau: một thứ không tái dùng câu trả lời mà chỉ tái dùng phép tính, một thứ thì thay hẳn câu trả lời bằng một câu cũ.

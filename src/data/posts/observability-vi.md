---
title: "Quan trắc hệ phục vụ"
description: "Cần ghi gì cho mỗi yêu cầu để chẩn đoán về sau, đọc các đại lượng bộ máy phục vụ tự đo, dấu vết mỗi loại sự cố để lại, và mổ một lần máy chủ sập từ chính log của nó."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "observability"
order: 11
pubDate: 2026-09-11
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ biết cần ghi gì cho mỗi yêu cầu để sau này chẩn đoán được, đọc được những đại lượng mà bộ máy phục vụ tự đo, nhận ra **dấu vết** mà mỗi loại sự cố để lại trên các đại lượng ấy - bằng số đo thật chứ không bằng bảng tra thuộc lòng - và mổ được một lần máy chủ sập từ chính log của nó.

Bài 10 kết thúc bằng một lỗi không để lại dòng log nào: đổi mô hình embedding, chất lượng rơi xuống mức đoán mò, và trong những gì phép thử ấy ghi lại, dấu hiệu trực tiếp là phân bố điểm truy hồi lệch đi. Các bài trước cũng đã gặp đủ loại hỏng khác: hàng đợi phình ở bài 07, bộ máy sập vì hết bộ nhớ ở bài 04 và bài 09.

Bài này gom tất cả lại thành một câu hỏi thực dụng: **khi người dùng phàn nàn, nhìn vào đâu thì biết hỏng ở đâu?**

## 1. Ghi gì cho mỗi yêu cầu

Nguyên tắc: mỗi trường được ghi phải trả lời được một câu hỏi chẩn đoán cụ thể. Trường nào không trả lời câu hỏi nào thì không cần; câu hỏi nào không có trường nào trả lời thì sẽ thành một buổi đoán mò.

| Trường | Trả lời câu hỏi |
|---|---|
| Mã yêu cầu, thời điểm đến | Yêu cầu nào, lúc nào - nối log của các tầng với nhau |
| **Thời gian chờ hàng đợi** | Hệ có đang quá tải không - bài 07 |
| **Thời gian nạp prompt** | Prompt có quá dài không - bài 01 |
| **Thời gian sinh token**, TPOT | Pha sinh có chậm bất thường không - bài 05 |
| TTFT, E2E | Người dùng thấy gì - bài 08 |
| **Số token vào, số token ra** | Chi phí, và yêu cầu này có khác thường không |
| **Lý do dừng** (`stop` hay `length`) | Câu trả lời xong tự nhiên hay bị cắt ngang |
| Các đoạn truy hồi, **điểm của đoạn đầu** | Tầng truy hồi có đưa đúng tài liệu không - bài 10, và chương Generative AI · Bài 12 |
| Trúng bộ đệm hay không, loại nào | Câu trả lời có phải tính mới không - bài 09 |
| Phiên bản mô hình, prompt, kho, cấu hình | Hai yêu cầu có chạy trên cùng một hệ không |
| Lỗi, nếu có | Hỏng ở tầng nào |

Hàng "điểm của đoạn đầu" là hàng hay bị bỏ nhất và đắt nhất khi thiếu: chính nó là dấu hiệu trực tiếp của lỗi đổi mô hình embedding ở bài 10 trong những gì phép thử ấy ghi lại. Một hệ ghi thêm **phiên bản mô hình embedding** và **phiên bản chỉ mục** cho mỗi yêu cầu thì còn thấy được lỗi ấy sớm hơn, ngay từ lúc đổi.

## 2. Bộ máy tự đo những gì

Máy chủ vLLM ở bài 08 xuất sẵn qua đường `/metrics` hai loại đại lượng.

**Loại tích lũy theo yêu cầu** - mỗi yêu cầu xong đóng góp một giá trị; đọc trước và sau một khoảng thời gian rồi lấy hiệu là được trung bình trong khoảng ấy. Danh sách đo được trên máy chủ của chương:

`time_to_first_token_seconds` · `request_queue_time_seconds` · `request_prefill_time_seconds` · `request_decode_time_seconds` · `inter_token_latency_seconds` · `request_time_per_output_token_seconds` · `e2e_request_latency_seconds` · `request_inference_time_seconds` · `request_prompt_tokens` · `request_generation_tokens` · `request_params_max_tokens` · `request_prefill_kv_computed_tokens` · `iteration_tokens_total` · …

**Loại tức thời** - trạng thái của bộ máy **ngay lúc hỏi**: `num_requests_running` (số yêu cầu đang chạy), `num_requests_waiting` (số yêu cầu đang chờ), `kv_cache_usage_perc` (phần trăm KV cache đang dùng). Loại này phải **lấy mẫu liên tục** - hỏi một lần lúc mọi thứ đã yên thì không thấy gì.

Chú ý loại thứ nhất đã **tách sẵn** thời gian của một yêu cầu thành các khúc: chờ, nạp, sinh - đúng công thức giải phẫu ở bài 01, giờ được bộ máy đo hộ cho từng yêu cầu. Và **TTFT không phải một khúc riêng để cộng thêm**: nó tính từ lúc yêu cầu đến, nên đã chứa cả khúc chờ lẫn khúc nạp. Ở tình huống quá tải của mục 3, chờ 0,472 s cộng nạp 0,288 s là 0,760 s - nằm trong TTFT 0,853 s; phần còn lại là xử lý đầu vào và bước sinh token đầu. Cộng TTFT với thời gian chờ là cộng đôi.

## 3. Bốn tình huống, bốn dấu vết

Thay vì đưa ra một bảng tra "triệu chứng nào thì hỏng gì" rồi yêu cầu tin, bài này **tạo ra** bốn tình huống trên cùng máy chủ - Qwen2.5-7B-Instruct-AWQ, một RTX 3060 - và đọc xem mỗi tình huống để lại dấu vết trên đại lượng nào:

- **Bình thường**: yêu cầu đến 2 cái mỗi giây, prompt ngắn, sinh 128 token.
- **Quá tải**: như trên nhưng 14 cái mỗi giây - vượt sức phục vụ đo được ở bài 07.
- **Prompt dài**: 1 cái mỗi giây, prompt **2.905** token, sinh 64 token.
- **Đầu ra dài**: 1 cái mỗi giây, prompt ngắn, sinh 1.024 token.

Số trung bình trên mọi yêu cầu của từng tình huống, đọc từ `/metrics`; ba dòng cuối là giá trị cao nhất khi lấy mẫu mỗi 0,5 giây. Mức dùng bộ nhớ 0,85 - lý do ở mục 5.

| | Bình thường | Quá tải | Prompt dài | Đầu ra dài |
|---|---|---|---|---|
| Token vào / ra (trung bình) | 44 / 128 | 44 / 128 | **2.905** / 64 | 44 / **1.024** |
| Chờ hàng đợi (s) | 0,000 | **0,472** | 0,000 | 0,000 |
| Nạp prompt (s) | 0,036 | 0,288 | 0,082 | 0,062 |
| TTFT (s) | 0,051 | **0,853** | 0,120 | 0,083 |
| Giữa hai token (s) | 0,016 | **0,147** | 0,017 | 0,026 |
| Sinh token (s) | 2,02 | **18,61** | 1,06 | **26,54** |
| E2E (s) | 2,07 | 19,46 | 1,18 | **26,62** |
| Đang chạy - cao nhất | 12 | **256** | 4 | 33 |
| Đang chờ - cao nhất | 0 | **44** | 0 | 0 |
| KV cache - cao nhất | 1,4% | 28,0% | 4,1% | 26,1% |

**Quá tải** là tình huống **duy nhất** có thời gian chờ hàng đợi lớn hơn 0 và có yêu cầu nằm chờ. Số đang chạy dừng đúng ở **256** - trần số chỗ mặc định của bộ máy - trong khi KV cache mới dùng 28%. Tức hệ hết **chỗ**, chưa hết **bộ nhớ**. Và vì 256 yêu cầu chạy chung một lô, thời gian giữa hai token tăng **9,2 lần** - đúng vùng mà bài 05 cho thấy pha sinh ra khỏi chế độ chỉ bị băng thông chi phối. Thời gian nạp prompt cũng tăng 8 lần dù prompt vẫn ngắn, vì các lượt nạp phải chen vào giữa một lô sinh token khổng lồ.

**Đầu ra dài** có E2E **tệ nhất bảng** - 26,6 giây - mà TTFT, hàng đợi đều bình thường. Dấu vết trực tiếp nhất của workload là **1.024 token ra**; các đại lượng hệ thống khác cũng đổi như hệ quả của việc phải sinh lâu hơn: thời gian giữa hai token là 0,026 thay vì 0,016 giây, số đang chạy cao nhất là 33 thay vì 12, và KV cache lên 26,1% thay vì 1,4%. Nếu chỉ nhìn E2E, bạn sẽ tưởng hệ chậm; thật ra hệ đang làm đúng việc được yêu cầu, chỉ là việc dài.

**Prompt dài** có dấu vết rõ nhất ở **token vào: 2.905**. Còn thời gian thì gần như không nhúc nhích - nạp prompt chỉ 0,082 giây - và đó là một **cái bẫy của chính phép đo này**: mọi yêu cầu trong tình huống ấy dùng **cùng một** prompt, nên bộ đệm tiền tố của máy chủ (bài 09) nuốt gần hết phần nạp. Bài 09 đo một prompt cỡ 2.843 token mà không có bộ đệm thì mất **1,55 giây**. Nên với prompt dài **khác nhau** - trường hợp thường gặp hơn - TTFT sẽ cao hơn nhiều so với bảng này. Bài học chung: **thời gian là dấu vết phụ thuộc bối cảnh; số token là dấu vết ổn định.** Ghi cả hai.

> ⚠️ **Lần chạy thứ hai của bảng này đã hỏng ở chính tình huống prompt dài.** Tôi dựng prompt bằng cách lặp một điều khoản 120 lần mà không đếm token; kết quả dài hơn giới hạn 4.096 token của máy chủ, và máy chủ trả mã **400** - đúng chức năng giới hạn đầu vào ở bài 08. Nhưng client của tôi coi lỗi đầu tiên là lỗi chết, nên cả tình huống đầu ra dài phía sau cũng không chạy. Hai sửa: dựng prompt **theo số token đếm bằng tokenizer**, và cho client **đếm lỗi theo từng yêu cầu**. Một công cụ đo tải mà chết vì một yêu cầu hỏng thì không đo được đúng lúc hệ đáng đo nhất.

## 4. Từ triệu chứng tới đại lượng

| Người dùng phàn nàn | Nhìn gì trước | Dấu vết đo được trong chương | Giả thuyết cần kiểm |
|---|---|---|---|
| "Lâu lắm mới ra chữ đầu" | **Chờ hàng đợi**, rồi **token vào** | Quá tải: chờ 0,47 s. Prompt dài: 2.905 token vào | Quá tải → giới hạn tải hoặc thêm máy (bài 07). Prompt dài → rút gọn, bộ đệm tiền tố (bài 09) |
| "Chữ hiện ra chậm" | **Giữa hai token**, **số đang chạy** | Quá tải: gấp 9,2 lần ở 256 đang chạy | Lô quá lớn (bài 05) |
| "Trả lời lâu" nhưng chữ ra ngay | **Token ra**, lý do dừng | Đầu ra dài: 1.024 token, TTFT bình thường | Câu trả lời dài - xem `max_tokens` và prompt |
| "Lúc đông người thì treo" | **Đang chờ**, chờ hàng đợi theo thời gian | Quá tải: đang chờ lên 44 | Vượt sức chứa (bài 07) |
| "Máy chủ lỗi 500 rồi mất" | Log: **đang chạy**, **KV**, **số byte xin cấp** | Mục 5: 193 đang chạy, KV 15,9%, xin 112 MiB | Biên bộ nhớ loại một (bài 04) |
| "Trả lời lạc đề", không có lỗi | **Điểm truy hồi đoạn đầu** | Bài 10: tụt từ 0,878 xuống 0,120 | Retrieval pipeline, chỉ mục, corpus hoặc phân bố câu hỏi thay đổi hay không tương thích; bài 10 minh họa bằng lệch phiên bản encoder |
| "Trả lời lẫn tiếng Trung" | Kiểm ký tự ngoài bảng chữ | Bài 12: 4 trên 32 câu | Lỗi nội dung đầu ra - kiểm mô hình, prompt, giải mã và thêm luật riêng |

Hai dòng cuối đáng để ý nhất: **không đại lượng thời gian nào của bộ máy bắt được chúng**. Bộ máy phục vụ chạy hoàn hảo trong cả hai trường hợp. Chỉ những trường do **ứng dụng** tự ghi - điểm truy hồi, kiểm đầu ra - mới thấy. Đó là lý do bảng ở mục 1 có những dòng mà `/metrics` không bao giờ cung cấp.

## 5. Mổ một lần sập thật

Lần chạy đầu tiên của thí nghiệm mục 3 dùng mức bộ nhớ mặc định 0,9. Tình huống "bình thường" chạy trọn. Sang tình huống "quá tải", sau khoảng 25 giây máy chủ trả lỗi 500 cho mọi yêu cầu, rồi tiến trình biến mất.

Không có gì phải đoán. Log của máy chủ ghi đủ, và ta đọc nó như đọc hồ sơ bệnh án.

**Diễn biến trước khi sập** - dòng thống kê định kỳ của bộ máy, cách nhau 10 giây:

| Thời điểm | Đang chạy | Đang chờ | KV cache đang dùng |
|---|---|---|---|
| 15:53:12 | 3 | 0 | 0,5% |
| 15:53:22 | 46 | 0 | 3,6% |
| 15:53:32 | 144 | 0 | 11,4% |
| 15:53:37 - sập | **193** | 0 | **15,9%** |

**Lời trăng trối:**

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 84.88 MiB is free.
```

Đọc ba manh mối cùng lúc.

**Không phải thiếu KV cache.** Lúc chết, KV cache mới dùng **15,9%**. Chừng ấy đã đủ để loại KV cache khỏi danh sách nghi phạm.

**Nó chết khi xin 112 MiB.** Con số cụ thể ấy là manh mối tốt nhất. Thử một phép nhân: **193** yêu cầu đang chạy × **152.064** token trong từ vựng × **4** byte:

$$193 \times 152.064 \times 4 = 117.393.408 \text{ byte} = 111{,}95 \text{ MiB}$$

Khớp. Và không phải khớp tình cờ: log giữ lại đủ **stack** của lỗi, chỉ đúng chỗ xin cấp:

```
vllm/v1/sample/sampler.py, in apply_penalties
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 112.00 MiB.
```

Bộ máy chết **trong bộ lấy mẫu**, ở bước áp phạt tần suất lên logits. Phép nhân `frequency_penalties.unsqueeze(1) * output_bin_counts` tạo ra một **tensor tạm cùng kích thước với logits** - 193 hàng, mỗi hàng một ô cho mỗi token trong từ vựng, số thực 32-bit - trước khi trừ vào logits. Tensor tạm ấy chính là 112 MiB mà bộ máy không xin được. Nó thuộc đúng khoản "hoạt hóa tạm thời" trong phép giải phẫu ở bài 03, cùng họ với logits mà bài 02 đã đo lớn gấp hơn mười lần KV cache mỗi token: một thứ tỉ lệ với **số yêu cầu đang chạy × kích thước từ vựng**, không tỉ lệ với ngữ cảnh.

**Nên đây là biên loại một của bài 04**: không phải bộ máy hết chỗ để xếp hàng, mà là hoạt hóa của một lô lớn không vừa phần bộ nhớ nằm ngoài ngân sách bộ máy tự dành. Lô càng lớn thì tensor ấy càng lớn - với 193 yêu cầu nó vượt qua mép.

Cách chữa suy ra thẳng từ chẩn đoán: **để dư thêm bộ nhớ ngoài ngân sách** - hạ mức dùng bộ nhớ - hoặc **giới hạn số yêu cầu chạy cùng lúc**, tức giới hạn kích thước những tensor cỡ logits. Hai lần chạy lại dùng cách thứ nhất, mức **0,85**: tình huống quá tải **chạy trọn cả hai lần**. Số đang chạy lên tới **256** - trần số chỗ của bộ máy - mà không sập; phần dôi ra nằm chờ, tối đa **44** yêu cầu; KV cao nhất **28%**. Cùng tải ấy, lần này hệ **xếp hàng** thay vì **chết** - tức biên loại một đã được đẩy ra xa, chỉ còn lại biên loại hai của bài 04.

Và chú ý toàn bộ chẩn đoán đến từ **log của chính máy chủ** - bộ đếm yêu cầu đang chạy, mức dùng KV, con số byte và stack của thông báo lỗi - cộng một phép nhân. Không cần gắn thêm công cụ gì. Điều kiện duy nhất là **log ấy được giữ lại**: một máy chủ chết mà log chết theo thì phải chạy lại sự cố mới biết chuyện gì xảy ra.

## 6. Bảng theo dõi tối giản

Một bảng theo dõi chỉ đáng tồn tại nếu mỗi ô trên đó dẫn tới một hành động. Đây là bộ tối giản, mỗi ô gắn với một hiện tượng chương này đã đo:

| Ô trên bảng | Cảnh báo khi | Vì sao - bài nào |
|---|---|---|
| TTFT, TPOT, E2E - **p50 và p95** | p95 vượt mức chấp nhận | Trung bình che mất người dùng tệ nhất - bài 07 |
| Chờ hàng đợi, số đang chờ | Lớn hơn 0 kéo dài | Dấu vết riêng của quá tải - mục 3 |
| Số đang chạy | Chạm trần số chỗ, hoặc gần mức từng làm máy sập | 256 là trần; 193 từng sập ở mức 0,9 - mục 3, mục 5 |
| KV cache đang dùng | Tiến gần 100% | Vượt biên KV thì thông lượng sập 64% - bài 04 |
| Số yêu cầu đang dở theo thời gian | Tăng dần qua nhiều cửa sổ liên tiếp | Hệ mất ổn định - bài 07 |
| Tỉ lệ lỗi theo mã (4xx, 5xx) | 5xx lớn hơn 0; 4xx tăng bất thường | 400 là giới hạn đầu vào làm việc; 500 là máy chết - bài 08, mục 5 |
| Phân bố điểm truy hồi đoạn đầu | Tụt đột ngột | Tín hiệu sự cố ở retrieval, chỉ mục hoặc corpus có thể không để lại dòng log - bài 10 |
| Tỉ lệ đầu ra thiếu trích dẫn, sai ngôn ngữ | Tăng | Luật chỉ bắt đúng thứ nó kiểm - bài 12 |

Mọi ngưỡng cảnh báo nên đặt **so với một ngày bình thường của chính hệ ấy**, không so với con số trong sách: bảng ở mục 3 cho thấy cùng một máy chủ, dấu vết "bình thường" đã phụ thuộc vào độ dài prompt và độ dài câu trả lời của tải.

> 🔧 **Thử ngay:** từ 9 giờ sáng, E2E p95 tăng từ 3 lên 25 giây. TTFT p95 vẫn 0,1 giây, chờ hàng đợi bằng 0, số đang chạy bình thường. Bạn nghi gì và kiểm gì?
> Hàng đợi và TTFT bình thường làm kiểu quá tải **do hàng đợi chi phối** ít khả năng hơn, nhưng chưa đủ loại mọi nguyên nhân: bộ đệm tiền tố có thể che chi phí của prompt dài, còn tranh chấp ở pha sinh có thể làm E2E và TPOT xấu đi trước khi hàng đợi hiện rõ. Kiểm tiếp ba nhánh:
>
> - **Token ra tăng** → nghi workload đầu ra dài; xem lý do dừng, phiên bản prompt và mô hình.
> - **TPOT tăng mạnh** trong khi token ra không đổi → nghi tranh chấp tài nguyên ở pha sinh; đối chiếu số đang chạy và tải đồng thời.
> - **Token vào tăng** → nghi prompt hoặc input workload dài hơn, kể cả khi prefix cache đang che một phần độ trễ nạp prompt.
>
> Nếu token ra tăng vọt và nhiều câu dừng vì chạm `max_tokens`, khi ấy mới kết luận hệ đang phải sinh dài hơn và xử lý nguyên nhân ở prompt, loại câu hỏi hoặc mô hình trước khi nghĩ tới thêm GPU.

## Tóm tắt bài học

- Mỗi trường ghi cho một yêu cầu phải trả lời một câu hỏi chẩn đoán: chờ hàng đợi, nạp prompt, sinh token, TTFT, E2E, số token vào và ra, lý do dừng, **điểm truy hồi**, trúng bộ đệm, phiên bản, lỗi.
- Bộ máy phục vụ tự đo hai loại: đại lượng **tích lũy theo yêu cầu** (đọc hiệu trước và sau) và đại lượng **tức thời** - số đang chạy, số đang chờ, KV đang dùng - phải lấy mẫu liên tục.
- Bốn tình huống trên cùng máy chủ để lại bốn dấu vết khác nhau. **Quá tải**: duy nhất có hàng đợi (chờ 0,47 s, 44 đang chờ), số đang chạy chạm trần **256** trong khi KV mới 28%, giữa hai token tăng **9,2 lần**. **Đầu ra dài**: E2E tệ nhất (26,6 s) mà TTFT và hàng đợi bình thường; dấu vết trực tiếp nhất của workload là **1.024 token ra**, còn TPOT, số đang chạy và KV tăng theo việc phải sinh lâu hơn. **Prompt dài**: dấu vết trực tiếp là **2.905 token vào**.
- Thời gian là dấu vết phụ thuộc bối cảnh: prompt dài lặp lại bị bộ đệm tiền tố nuốt (0,082 s so với 1,55 s không bộ đệm). Số token là dấu vết ổn định. Ghi cả hai.
- Một lần sập chẩn đoán được từ **log của máy chủ và một phép nhân**: 193 đang chạy, KV mới 15,9%, xin 112 MiB = 193 × 152.064 × 4 byte, và stack chỉ vào bước áp phạt trong bộ lấy mẫu - một **tensor tạm cỡ logits**, tức biên loại một của bài 04. Hạ mức bộ nhớ xuống 0,85 thì cùng tải ấy xếp hàng thay vì chết.
- Hai nhóm hỏng mà **không đại lượng nào của bộ máy bắt được**: retrieval, chỉ mục hoặc corpus không tương thích; và lỗi nội dung đầu ra như sai ngôn ngữ. Chỉ các trường do ứng dụng tự ghi mới thấy.
- Công cụ đo tải phải **đếm lỗi theo từng yêu cầu**; và prompt thử phải dựng **theo số token**, không theo số lần lặp.
- Bảng theo dõi tối giản: mỗi ô dẫn tới một hành động, ngưỡng đặt so với một ngày bình thường của chính hệ.

## Câu hỏi tự kiểm tra

1. Chọn ba trường trong bảng ở mục 1 và nói với mỗi trường, nếu thiếu nó thì loại sự cố nào không chẩn đoán được.
2. Vì sao các đại lượng tức thời như số đang chờ phải lấy mẫu liên tục, còn các đại lượng tích lũy thì chỉ cần đọc hai lần?
3. Trong tình huống quá tải, KV cache mới dùng 28% mà hệ đã có yêu cầu nằm chờ. Giải thích.
4. Tình huống đầu ra dài có E2E tệ nhất bảng. Vì sao không được kết luận "hệ đang chậm"?
5. Tình huống prompt dài chỉ tốn 0,082 s ở pha nạp. Vì sao con số ấy thấp, và trong một hệ thật con số tương ứng có thể là bao nhiêu?
6. Đọc lại mục 5: nêu các manh mối trong log - số đang chạy, mức dùng KV, số byte xin cấp, stack - và phép tính xác nhận thủ phạm. Vì sao chỉ khớp số byte thôi thì chưa đủ để gọi tên thủ phạm?
7. Nêu hai nhóm hỏng mà `/metrics` của bộ máy phục vụ không bao giờ thấy được - một ở phía retrieval, một ở nội dung đầu ra - và trường nào do ứng dụng ghi sẽ bắt chúng.
8. Thiết kế ba cảnh báo cho hệ ở bài 12, mỗi cảnh báo nêu đại lượng, ngưỡng so với cái gì, và hành động khi nó kêu.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 01 của chương này** | Công thức giải phẫu E2E ≈ chờ + nạp + sinh - bộ máy đo hộ từng khúc; và vì sao TTFT đã gồm cả chờ |
| **Bài 04 của chương này** | Hai loại biên bộ nhớ - nền của mục 5 |
| **Bài 07 của chương này** | Quá tải trông thế nào từ bên ngoài |
| **Bài 10 của chương này** | Lỗi không để lại dòng log nào |

**Đi sâu hơn:**

- [vLLM - Production Metrics](https://docs.vllm.ai/en/latest/serving/metrics.html) - ý nghĩa từng đại lượng ở mục 2.
- [Beyer et al. - Site Reliability Engineering, chương "Monitoring Distributed Systems"](https://sre.google/sre-book/monitoring-distributed-systems/) - bốn tín hiệu vàng: độ trễ, lưu lượng, lỗi, độ bão hòa.
- [OpenTelemetry - Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) - cách đặt tên thống nhất cho các trường ở mục 1.

**Mã nguồn của bài:** [`code/b11_dauvet.py`](../../code/ai-systems/b11_dauvet.py) - bốn tình huống, đọc `/metrics` và lấy mẫu các đại lượng tức thời.

> **Bài tiếp theo:** [Project end-to-end: triển khai hệ thống RAG](../project-serving-a-rag-system-vi/) - mười một bài đã dựng đủ mảnh. Bài cuối lấy chính hệ hỏi đáp của chương Generative AI, đưa nó qua sáu chặng - từ một script tuần tự tới một máy chủ có lượng tử hóa, gộp lô, bộ đệm, luồng và giới hạn - đo từng chặng trên cùng một tải, và bảng cuối phải cho thấy cả những ô xấu đi.

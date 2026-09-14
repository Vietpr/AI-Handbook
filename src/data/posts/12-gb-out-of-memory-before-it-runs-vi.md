---
title: "12 GB: chưa chạy đã hết chỗ"
description: "Ước tính một mô hình có vừa GPU hay không trước khi tải xuống, rồi kiểm chứng phép tính bằng một lỗi hết bộ nhớ thực tế."
domain: "AI Systems"
section: "Learn"
language: "vi"
translationKey: "12-gb-out-of-memory-before-it-runs"
order: 3
pubDate: 2026-09-07
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ tính được **trước khi tải về** một mô hình có vừa card của mình không, đọc hiểu từng dòng của một thông báo hết bộ nhớ GPU thật, phân biệt được **ba cách đếm bộ nhớ** cho ba con số khác nhau, và biết vì sao phép cộng "trọng số cộng KV cache" luôn ra thiếu.

Hai bài trước đo trên Qwen2.5-1.5B vì nó chạy được. Nhưng mô hình mà chương này thật sự nhắm tới là bản **7B** - đủ lớn để làm được việc thật, và đủ lớn để không vừa.

Bài này là bài mọi thứ hỏng. Đó là chủ ý: bạn phải thấy nó hỏng trước khi bài 04 chữa.

## 1. Phép tính hai dòng, làm trước khi tải 15 GB

Đừng tra số tham số trên trang giới thiệu rồi nhân nhẩm. Checkpoint tự nó khai báo kích thước, trong tệp `model.safetensors.index.json`:

```
"total_size": 15231233024
```

**15.231.233.024 byte = 15,23 GB.** Đó là toàn bộ trọng số, ở kiểu dữ liệu mà mô hình được công bố - **bfloat16**, tức 2 byte mỗi tham số.

Còn card:

```python
torch.cuda.get_device_properties(0).total_memory   # 12.490.457.088 byte
```

**12,49 GB.**

$$15{,}23 - 12{,}49 = \mathbf{2{,}74 \text{ GB thiếu}}$$

Chưa có KV cache. Chưa có hoạt hóa. Chưa có một người dùng nào. Chỉ riêng trọng số đã thừa ra 2,74 GB.

> ⚠️ **Đơn vị là cái bẫy đầu tiên, và nó làm sai lệch ngay ở bước tính.** Card này bán ra với nhãn "12 GB". `nvidia-smi` báo **12.288 MiB**, tức 12 GiB nhị phân, tức **12,88 GB thập phân**. Còn PyTorch báo **12,49 GB**. Ba con số cho cùng một card. Chênh lệch giữa hai số cuối - khoảng **379 MiB** - là phần driver và firmware giữ lại, CUDA không bao giờ chạm tới. Quy tắc: **luôn quy mọi thứ về byte trước khi cộng trừ**, và lấy con số PyTorch báo làm trần, không lấy con số in trên hộp.

## 2. Nạp thử: lỗi trông ra sao

Phép tính nói nó không vừa. Nhưng một dòng số học không dạy bạn nhận ra lỗi lúc 2 giờ sáng. Nên nạp thật:

```python
m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-7B-Instruct", dtype=torch.bfloat16).to("cuda")
```

Và đây là nguyên văn thứ nhận được:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 130.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 30.88 MiB is free.
Including non-PyTorch memory, this process has 11.54 GiB memory in use.
Of the allocated memory 11.34 GiB is allocated by PyTorch, and 105.47 MiB
is reserved by PyTorch but unallocated. If reserved but unallocated memory
is large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid
fragmentation.
```

Thông báo này đáng đọc từng dòng, vì **nó tự liệt kê đúng bốn khoản bộ nhớ** mà mục 4 sẽ dựng lại.

| Mảnh trong thông báo | Nghĩa |
|---|---|
| `Tried to allocate 130.00 MiB` | Nó chết khi xin thêm **một** tensor 130 MiB. Không phải vì thiếu 2,74 GB cùng lúc - mà vì xin thêm một mẩu nhỏ mà không còn |
| `total capacity of 11.63 GiB` | Trần thật của card, tính theo GiB - chính là 12,49 GB ở mục 1 |
| `11.34 GiB is allocated by PyTorch` | Các tensor đang sống: phần trọng số đã nạp được, khoảng **80%** |
| `105.47 MiB is reserved by PyTorch but unallocated` | Bộ cấp phát đã giữ chỗ nhưng chưa dùng - **khoản 4** |
| `Including non-PyTorch memory, this process has 11.54 GiB` | 11,54 − 11,34 = **0,20 GiB** không thuộc PyTorch: CUDA context và phần driver |
| `30.88 MiB is free` | Còn đúng ngần ấy |

Ba điều đáng rút ra ngay.

**Nó không chết ngay từ đầu.** Nó nạp được 11,34 trong số 14,18 GiB trọng số - khoảng **80%** - rồi mới gãy. Nghĩa là bạn ngồi chờ vài chục giây, thanh tiến trình chạy gần xong, rồi mới đổ. Đây là lý do phép tính hai dòng ở mục 1 đáng giá: nó tốn hai phút và tránh được cả một vòng lặp thử-rồi-sai.

**Con số trong thông báo là số nó vừa xin, không phải số nó thiếu.** "Tried to allocate 130.00 MiB" rất dễ khiến người mới tưởng chỉ thiếu 130 MiB rồi đi giảm batch một chút. Thực tế thiếu 2,74 GB.

**Lời khuyên cuối thông báo chỉ đúng trong một trường hợp.** `expandable_segments:True` chữa **phân mảnh** - tình huống tổng bộ nhớ trống thì đủ nhưng không có khối liền nào đủ lớn. Ở đây phần giữ-chỗ-chưa-dùng chỉ có **105,47 MiB** trong khi ta thiếu 2,74 GB, nên phân mảnh **không** phải nguyên nhân, và bật cờ ấy không cứu được gì. Chính thông báo cũng nói *"if reserved but unallocated memory is large"* - điều kiện ấy không thỏa. Đọc kỹ thông báo trước khi làm theo nó.

## 3. Ba cách đếm, ba con số

Trước khi mổ xẻ, phải thống nhất **đo bằng gì**. Có ba cách đếm bộ nhớ GPU, và chúng **không bao giờ bằng nhau**.

| Cách đếm | Trả lời câu hỏi gì |
|---|---|
| `torch.cuda.memory_allocated()` | Các tensor đang sống chiếm bao nhiêu |
| `torch.cuda.memory_reserved()` | Bộ cấp phát của PyTorch đã xin bao nhiêu từ driver |
| `nvidia-smi` báo cho tiến trình | Driver thấy tiến trình này chiếm bao nhiêu |

Đo cả ba tại bốn thời điểm, trên Qwen2.5-1.5B / RTX 3060:

| Thời điểm | Cấp phát (MiB) | Giữ chỗ (MiB) | Driver báo (MiB) |
|---|---|---|---|
| Sau khi khởi tạo CUDA, **chưa cấp phát gì** | 0,0 | 0,0 | **102** |
| Sau khi nạp trọng số | 2.945,3 | 3.134,0 | 3.236 |
| Sau một lượt chạy xuôi 4.096 token | 4.254,4 | 4.722,0 | **4.856** |
| Sau khi giải phóng và dọn dẹp | 2.954,4 | 3.156,0 | 3.290 |

Dòng đầu là dòng gây bất ngờ nhất: **102 MiB biến mất trước khi bạn cấp phát một byte nào.** Đó là **CUDA context** - driver nạp nhân tính toán, dựng bảng, cấp vùng làm việc. Không công thức bộ nhớ nào của mô hình chứa khoản này, và nó không tỉ lệ với mô hình.

Dòng thứ ba là dòng phải nhớ. Cùng một thời điểm, ba cách đếm cho **4.254 · 4.722 · 4.856**. Chênh lệch giữa cái nhỏ nhất và cái lớn nhất là **602 MiB** - bằng cả một mô hình nhỏ.

Dòng cuối cho thêm một chi tiết khó chịu: sau khi xóa kết quả và gọi dọn dẹp, bộ nhớ **không quay về mức ban đầu**. Cấp phát còn 2.954 thay vì 2.945, driver còn 3.290 thay vì 3.236. Giải phóng trong CUDA không phải phép nghịch đảo sạch sẽ của cấp phát.

## 4. Cái gì chiếm chỗ

Giờ mổ dòng thứ ba của bảng trên - lượt chạy xuôi 4.096 token - bằng các số đã đo:

| Khoản | Đo được (MiB) | Nguồn |
|---|---|---|
| Trọng số | **2.944,4** | Cộng kích thước tham số |
| Hoạt hóa tạm thời - **logits** | **1.187,0** | Kích thước tensor trả về |
| KV cache | **112,0** | Bài 02, công thức đã kiểm |
| Tensor lặt vặt khác | ~11,0 | Phần dư |
| **Tổng các tensor đang sống** | **4.254,4** | `memory_allocated` |
| Bộ cấp phát giữ chỗ mà chưa dùng | **+467,6** | `memory_reserved` trừ đi trên |
| CUDA context và phần ngoài PyTorch | **+134** | `nvidia-smi` trừ đi trên |
| **Driver thấy tiến trình chiếm** | **4.856** | `nvidia-smi` |

Viết lại thành bốn khoản, theo đúng thứ tự cộng dồn:

```
   trọng số                              2.944,4 MiB
 + KV cache                                112,0 MiB
 + hoạt hóa và vùng làm việc tạm thời     1.198,0 MiB   ← logits chiếm gần hết
 + chi phí runtime và bộ cấp phát           601,6 MiB   ← gồm CUDA context,
 ───────────────────────────────────────────────────      phần giữ chỗ chưa dùng,
 = bộ nhớ GPU thực sự tiêu thụ            4.856,0 MiB      và phân mảnh
```

> ⚠️ **Phân mảnh không phải một khoản riêng để cộng thêm.** Rất dễ liệt kê năm ô rồi cộng cả năm - và cộng đôi. Phân mảnh là hiện tượng *bên trong* khoản thứ tư: nó làm phần "giữ chỗ mà chưa dùng" phình ra, chứ không chiếm thêm một vùng nào khác. Thông báo lỗi ở mục 2 đo đúng khoản ấy và gọi tên nó là `reserved but unallocated`.

Và chú ý **hoạt hóa khi suy luận không giống khi huấn luyện**. Huấn luyện phải giữ hoạt hóa của mọi lớp để lan truyền ngược, nên chúng tồn tại suốt cả bước. Suy luận thì không: hoạt hóa của lớp $i$ có thể vứt ngay sau khi lớp $i+1$ dùng xong. Cái còn lại lớn là **logits**, mà bài 02 đã đo - **296,8 KiB cho mỗi vị trí token**, gấp 10,6 lần KV cache.

## 5. Vì sao phép cộng quen thuộc luôn hụt

Cách lập ngân sách phổ biến nhất là cộng hai khoản:

$$\text{trọng số} + \text{KV cache} = 2.944{,}4 + 112{,}0 = 3.056{,}4 \text{ MiB}$$

Còn driver báo **4.856 MiB**. Hụt **1.800 MiB, tức 59%**.

Hệ quả rất cụ thể. Giả sử bạn có card 12,49 GB, mô hình 1.5B, và bạn lập kế hoạch bằng phép cộng hai khoản: "trọng số 2,9 GB, còn 9,5 GB cho KV cache, ở 28 KiB mỗi token là 348.000 token, chia cho ngữ cảnh 4.096 là **85 người cùng lúc**." Con số ấy sai không phải vài phần trăm - nó bỏ sót hẳn khoản logits, thứ tỉ lệ với ngữ cảnh *và* với batch.

Nên nguyên tắc là: **đừng lập ngân sách bộ nhớ bằng công thức rồi triển khai thẳng. Lập bằng công thức để loại sớm những phương án chắc chắn không vừa - như bản 7B ở mục 1 - rồi đo thật để chốt.** Công thức trả lời "có vừa không"; chỉ phép đo trả lời "vừa được bao nhiêu".

> 🔧 **Thử ngay:** hệ của bạn đang chạy ổn với 8 người dùng đồng thời. Bạn nâng ngữ cảnh tối đa từ 4.096 lên 8.192 token, nghĩ rằng chỉ tốn thêm ít KV cache. Nó đổ vì hết bộ nhớ. Vì sao?
> Vì bạn chỉ cộng KV cache. Với bản 1.5B, gấp đôi ngữ cảnh làm KV cache của mỗi người tăng từ 112 lên 224 MiB - thêm 896 MiB cho 8 người, nghe chịu được. Nhưng nếu hệ dựng logits cho toàn bộ vị trí ở pha nạp prompt thì khoản ấy cũng gấp đôi, từ 1.187 lên **2.374 MiB cho một lượt nạp**. Chỗ ấy mới là chỗ đổ. Bài 06 sẽ cho thấy bộ máy phục vụ chuyên dụng chỉ dựng logits cho vị trí cuối cùng, và điều đó đổi hẳn phép tính này.

## 6. Vì sao chưa quét biên bộ nhớ

Một câu hỏi tự nhiên lúc này: vậy quét thử `batch × độ dài` tới lúc gãy, vẽ ra đường biên hết bộ nhớ, có phải gọn hơn không?

Có - nhưng **không làm được ở đây**, và lý do đáng nói ra.

Cả bài này vừa chứng minh bản 7B **không nạp nổi** lên card. Không nạp được thì không có gì để quét: mọi cấu hình batch và độ dài đều gãy ở cùng một chỗ, trước khi kịp chạy. Quét trên bản 1.5B thì được, nhưng nó trả lời câu hỏi về một mô hình khác với mô hình chương này nhắm tới.

Nên thứ tự đúng là: **bài 03 chứng minh nó không vừa · bài 04 làm cho nó vừa · rồi mới quét được biên giới bộ nhớ của chính bản 7B ấy.** Đường biên ấy nằm ở bài 04, sau khi có một bản 7B chạy được thật.

## Tóm tắt bài học

- Kích thước trọng số đọc thẳng từ checkpoint (`model.safetensors.index.json` → `total_size`), không suy từ số tham số: Qwen2.5-7B là **15,23 GB**, card RTX 3060 là **12,49 GB**, **thiếu 2,74 GB** trước khi có bất kỳ người dùng nào.
- Đơn vị là bẫy: cùng một card cho **12.288 MiB** theo `nvidia-smi` và **12,49 GB** theo PyTorch. Quy mọi thứ về byte, và lấy con số PyTorch làm trần.
- Lỗi thật đọc được: nó nạp tới **80%** trọng số rồi mới gãy khi xin thêm **130 MiB**. Con số trong thông báo là số **vừa xin**, không phải số **còn thiếu**.
- Lời khuyên `expandable_segments:True` ở cuối thông báo chỉ chữa **phân mảnh**. Ở đây phần giữ-chỗ-chưa-dùng chỉ **105,47 MiB** còn ta thiếu 2,74 GB, nên nó không cứu được gì - đọc kỹ điều kiện trước khi làm theo.
- **Ba cách đếm cho ba con số:** cùng một thời điểm, `memory_allocated` 4.254 · `memory_reserved` 4.722 · `nvidia-smi` **4.856** MiB. Chênh 602 MiB giữa cái nhỏ nhất và cái lớn nhất.
- **CUDA context tốn 102 MiB trước khi cấp phát một byte nào**, và không tỉ lệ với mô hình.
- Bốn khoản bộ nhớ: **trọng số 2.944,4 + KV cache 112,0 + hoạt hóa tạm thời 1.198,0 + runtime và bộ cấp phát 601,6 = 4.856 MiB**. Phân mảnh **không** là khoản thứ năm - nó nằm trong khoản thứ tư, làm phần giữ-chỗ-chưa-dùng phình ra.
- Lập ngân sách bằng "trọng số + KV cache" cho **3.056 MiB** trong khi thực tế là **4.856 MiB** - **hụt 59%**. Công thức trả lời *"có vừa không"*; chỉ phép đo trả lời *"vừa được bao nhiêu"*.
- Bài này cố tình **không quét** `batch × độ dài`: chưa nạp được mô hình thì không có gì để quét. Đường biên ấy nằm ở bài 04, sau khi mô hình đã vừa.

## Câu hỏi tự kiểm tra

1. Vì sao nên đọc `total_size` trong checkpoint thay vì nhân số tham số với số byte?
2. Một card ghi "24 GB" trên hộp. Nêu ba con số khác nhau bạn có thể gặp cho cùng card ấy, và con số nào nên dùng làm trần khi lập kế hoạch.
3. Thông báo lỗi nói `Tried to allocate 130.00 MiB`. Vì sao giảm batch đi một chút thường không cứu được tình huống ở mục 2?
4. Khi nào thì `expandable_segments:True` thực sự có ích? Dùng chính các con số trong thông báo lỗi để giải thích vì sao nó vô ích ở đây.
5. Phân biệt `memory_allocated`, `memory_reserved` và con số `nvidia-smi` báo. Cái nào bạn phải dùng để biết còn bao nhiêu chỗ cho một tiến trình khác trên cùng card?
6. Vì sao hoạt hóa khi **suy luận** nhẹ hơn nhiều so với khi **huấn luyện**, và vì sao logits lại là ngoại lệ?
7. Giải thích vì sao phân mảnh không được liệt kê thành một khoản riêng khi cộng bộ nhớ.
8. Bạn có card 24 GB, mô hình 13B ở 16-bit, ngữ cảnh 8.192, từ vựng 128.000. Ước lượng bốn khoản bộ nhớ cho **một** yêu cầu, rồi nói rõ con số nào trong ước lượng của bạn là kém chắc chắn nhất và vì sao.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Bài 02 của chương này** | Công thức KV cache, và phép đo cho thấy logits lớn gấp 10,6 lần nó |
| **Bài 01 của chương này** | Hai pha, và vì sao pha nạp prompt mới là chỗ logits phình to |

**Đi sâu hơn:**

- [PyTorch - CUDA semantics: Memory management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management) - vì sao `memory_reserved` lớn hơn `memory_allocated`, và `PYTORCH_ALLOC_CONF` làm gì.
- [Kwon et al. - Efficient Memory Management for LLM Serving with PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180) - đo mức lãng phí do cấp phát KV cache thành khối liền; bài 06 sẽ chạy thật hệ này.
- [Korthikanti et al. - Reducing Activation Recomputation in Large Transformer Models (MLSys 2023)](https://arxiv.org/abs/2205.05198) - giải phẫu hoạt hóa, và vì sao con số khi huấn luyện khác hẳn khi suy luận.

**Mã nguồn của bài:** [`code/b03_bonho.py`](../../code/ai-systems/b03_bonho.py) - ba cách đếm và giải phẫu bốn khoản; [`code/b03_oom.py`](../../code/ai-systems/b03_oom.py) - nạp bản 7B lên card 12 GB để bắt thông báo lỗi thật.

> **Bài tiếp theo:** [Khi mô hình không vừa GPU: ba lựa chọn](../three-ways-out-when-a-model-doesnt-fit-vi/) - thiếu 2,74 GB thì có ba lối thoát - chia mô hình qua nhiều GPU, đẩy bớt sang CPU, hoặc thu nhỏ chính mô hình. Bài sau đo cả ba trên đúng phần cứng này, giải thích vì sao mục tiêu "chạy trên **một** card" dẫn tới lựa chọn nào, và chỉ khi bản 7B đã vừa mới quét biên giới bộ nhớ mà bài này còn nợ.

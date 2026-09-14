---
title: "Dữ liệu chuỗi: RNN và chỗ nó hụt hơi"
description: "Đo hai điểm yếu chính của mạng hồi quy và thấy vì sao chúng dẫn tới attention một cách tự nhiên."
domain: "Deep Learning"
section: "Learn"
language: "vi"
translationKey: "sequence-data-rnns-and-where-they-fall-short"
order: 10
pubDate: 2026-08-23
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu vì sao dữ liệu có thứ tự cần một kiến trúc khác, biết mạng hồi quy xử lý chuỗi bằng cách nào, và thấy bằng số hai chỗ nó hụt hơi - chỗ dẫn thẳng tới attention.

## 1. Một loại cấu trúc khác

Bài 09 khai thác cấu trúc **không gian**: pixel gần nhau thì liên quan. Có một loại cấu trúc khác cũng đáng khai thác không kém - **thứ tự**.

- *"Hôm nay trời không đẹp"* và *"Hôm nay trời đẹp không"* dùng đúng năm từ như nhau, nghĩa hoàn toàn khác.
- Giá cổ phiếu hôm nay phụ thuộc vào chuỗi những ngày trước, không phải một tập hợp giá không thứ tự.
- Một câu nói có độ dài thay đổi: ba từ hay ba mươi từ đều là một mẫu hợp lệ.

Mạng ở chín bài trước xử lý loại dữ liệu này rất vụng, vì hai lý do. Nó cần đầu vào **cố định độ dài** - `nn.Linear(784, 256)` chỉ nhận đúng 784 số, nên chuỗi dài ngắn khác nhau phải được đệm thêm hoặc cắt bớt cho bằng một mốc chọn sẵn, và mốc ấy vừa lãng phí với câu ngắn vừa cắt cụt câu dài. Và nó **không có trí nhớ**: mỗi đầu vào được xử lý độc lập, không mang theo gì từ đầu vào trước, nên muốn nó biết "từ này đứng sau từ kia" thì phải tự dựng sẵn thông tin vị trí vào đầu vào. Làm được, nhưng là làm thay cho mô hình - đúng cái vòng luẩn quẩn mà bài 01 đã chỉ ra.

## 2. RNN: một vòng lặp có trí nhớ

**Mạng hồi quy (recurrent neural network - RNN)** giải cả hai bằng một ý tưởng: xử lý chuỗi **từng bước một**, và mang theo một **trạng thái ẩn (hidden state)** từ bước này sang bước sau.

```
   x₁        x₂        x₃              xₜ
   │         │         │               │
   ▼         ▼         ▼               ▼
  ┌───┐ h₁  ┌───┐ h₂  ┌───┐ h₃      ┌───┐
  │ f │────▶│ f │────▶│ f │──▶ … ──▶│ f │──▶ dự đoán
  └───┘     └───┘     └───┘         └───┘
   cùng MỘT bộ trọng số dùng lại ở mọi bước
```

Ở mỗi bước, mạng nhận đầu vào mới $x_t$ cùng trạng thái cũ $h_{t-1}$, rồi sinh ra trạng thái mới:

$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$

Trạng thái $h_t$ là bản tóm tắt của mọi thứ đã đọc tới bước $t$. Đọc hết chuỗi, ta lấy trạng thái cuối cùng đưa vào một tầng phân loại.

Để ý chi tiết **cùng một bộ trọng số dùng lại ở mọi bước** - đúng ý tưởng chia sẻ trọng số của CNN ở bài 09, chỉ khác là trượt theo *thời gian* thay vì theo *không gian*. Nhờ vậy chuỗi dài bao nhiêu cũng xử lý được mà số tham số không đổi.

Để thử RNN mà không cần đổi bộ dữ liệu, ta đọc ảnh FashionMNIST 28×28 như **một chuỗi 28 hàng**, mỗi hàng là một vector 28 số:

```python
class RowRNN(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.rnn = nn.RNN(28, hidden, batch_first=True)   # mỗi bước đọc 1 hàng
        self.fc  = nn.Linear(hidden, 10)
    def forward(self, x):
        out, _ = self.rnn(x.squeeze(1))    # (batch, 28 bước, hidden)
        return self.fc(out[:, -1])         # lấy trạng thái ở bước CUỐI
```

| Mô hình | Tham số | Accuracy | Thời gian (10 epoch) |
|---|---|---|---|
| **RNN** | 21.514 | 0,8289 | 93 giây |
| **LSTM** | 82.186 | **0,8687** | 172 giây |

RNN đạt 0,8289 với chỉ 21.514 tham số - đọc ảnh theo hàng, không hề biết ảnh là gì. Nhưng chú ý dòng thứ hai đã thắng nó gần 4 điểm, và tên của nó cần một mục riêng.

## 3. Vì sao LSTM tồn tại

Nhìn lại công thức ở mục 2: trạng thái $h_t$ được tính từ $h_{t-1}$, mà $h_{t-1}$ tính từ $h_{t-2}$, cứ thế. Muốn tín hiệu từ bước 1 ảnh hưởng tới dự đoán ở bước 28, gradient phải đi ngược qua **28 phép nhân liên tiếp**.

Đó chính xác là tình huống của bài 08, chỉ khác chỗ độ sâu ở đây là *độ dài chuỗi* chứ không phải số tầng. Cùng một bệnh: nhân dồn nhiều thừa số nhỏ hơn 1 thì gradient tiêu biến, và RNN **quên mất phần đầu chuỗi**.

**LSTM (long short-term memory)** chữa bằng cách thêm một đường riêng cho trí nhớ đi qua, kèm ba cái "cổng" học được để quyết định thông tin nào giữ, nào bỏ, nào cho ra:

```
   trạng thái nhớ ──────────────────────────▶  đường gần như thẳng,
                    ↑ cổng quên  ↑ cổng vào     gradient đi qua ít bị teo
```

Ý tưởng cốt lõi là đổi *kiểu* phép nhân mà trí nhớ phải chịu. Trạng thái nhớ vẫn được nhân - công thức thật là $c_t = f_t \odot c_{t-1} + i_t \odot g_t$, tức nó vẫn nhân với cổng quên $f_t$ ở mỗi bước. Khác biệt nằm ở chỗ $f_t$ là một con số trong khoảng 0 đến 1 do mạng **học lấy** cho từng bước, chứ không phải một ma trận trọng số cố định nhân với đạo hàm của hàm kích hoạt như ở RNN. Khi mạng thấy cần nhớ, nó học đẩy $f_t$ về sát 1, và chuỗi phép nhân gần như thành phép cộng dồn - gradient đi ngược qua đó ít bị teo. Nói cách khác, LSTM không xóa bỏ phép nhân, nó trao cho mạng cái núm vặn để tự quyết định khi nào phép nhân ấy được phép làm mờ trí nhớ. Đây là họ hàng gần với kết nối tắt của ResNet mà bài 08 và 09 đã nhắc. Bảng ở mục 2 cho thấy nó ăn tiền: 0,8687 so với 0,8289, đổi lấy gần bốn lần số tham số. **GRU** là một biến thể gọn hơn với hai cổng thay vì ba.

## 4. Chỗ cả hai cùng hụt hơi

Chuỗi 28 bước là ngắn. Thử kéo dài ra: đọc cùng tấm ảnh ấy nhưng **từng pixel một** - chuỗi 784 bước thay vì 28, cùng lượng thông tin y hệt, chỉ dài gấp 28 lần.

| Độ dài chuỗi | RNN | LSTM |
|---|---|---|
| 28 bước (đọc theo hàng) | 0,8289 | 0,8687 |
| **784 bước (đọc từng pixel)** | **0,1138** | **0,1004** |

Cả hai **sập hoàn toàn** về mức đoán bừa 0,1. LSTM không cứu nổi - thậm chí còn thấp hơn RNN một chút.

Và có một con số thứ hai đáng chú ý không kém: LSTM ở 784 bước mất **3.825 giây cho 3 epoch** - 64 phút. Lý do nằm ở bản chất kiến trúc: bước $t$ cần trạng thái của bước $t-1$, nên **784 bước phải chạy tuần tự, không thể tính song song**. Trong khi CNN ở bài 09 tính mọi vị trí cùng lúc, RNN buộc phải xếp hàng.

Hai vấn đề, và chúng là hai vấn đề khác nhau:

- **Chuỗi dài thì nhớ kém.** Thông tin phải đi qua từng bước một, và đường đi giữa hai vị trí cách nhau 700 bước là 700 phép biến đổi.
- **Chuỗi dài thì train chậm.** Tính tuần tự, không tận dụng được phần cứng song song.

> 🔧 **Thử ngay:** cùng một tấm ảnh 28×28, đọc theo 28 hàng thì RNN đạt 0,8289, đọc theo 784 pixel thì còn 0,1138. Lượng thông tin y hệt nhau - vậy cái gì đã mất?
> Không phải thông tin, mà là **khoảng cách**. Ở cách đọc theo hàng, hai pixel nằm cạnh nhau theo chiều dọc cách nhau đúng 1 bước trong chuỗi. Ở cách đọc từng pixel, chúng cách nhau 28 bước, và hai pixel ở hai góc đối diện cách nhau 783 bước. Gradient phải đi ngược qua từng ấy phép nhân mới nối được chúng lại. Chuỗi càng dài thì cùng một quan hệ trong dữ liệu càng bị kéo giãn ra xa, và đó là thứ RNN không chịu được.

> ⚠️ **Lưu ý nhỏ:** thí nghiệm 784 bước chỉ chạy 3 epoch, đúng vì nó quá chậm. Nên con số 0,10 vừa phản ánh chuyện khó nhớ, vừa phản ánh chuyện huấn luyện chưa đủ - và hai chuyện ấy không tách rời được ở đây. Nhưng chính chỗ không tách rời ấy mới là điều đáng nhớ: với chuỗi dài, "khó học" và "chậm đến mức không kịp học" trói vào nhau thành một vấn đề duy nhất.

## 5. Attention: cho mọi vị trí nói chuyện trực tiếp

Cách chữa cho cả hai vấn đề đến từ một câu hỏi đơn giản: vì sao thông tin phải đi *qua từng bước*? Nếu bước 700 cần biết bước 1 nói gì, sao không cho nó **nhìn thẳng** vào bước 1?

Đó là **attention (cơ chế chú ý)**. Thay vì một trạng thái ẩn truyền tay qua từng bước, mỗi vị trí trong chuỗi được nhìn *mọi* vị trí khác cùng lúc, và học xem nên chú ý vào đâu.

```
  RNN:  x₁ → x₂ → x₃ → … → x₇₈₄     đường từ x₁ tới x₇₈₄ dài 783 bước

  Attention:  x₁ ←──────────────→ x₇₈₄   nối thẳng, đường dài 1 bước
              mọi cặp vị trí đều nối trực tiếp
```

Cách này chữa cả hai bệnh cùng lúc. Đường giữa hai vị trí bất kỳ chỉ còn **một bước**, nên không còn nhân dồn 700 thừa số. Và vì không có phụ thuộc tuần tự, mọi vị trí **tính được song song** - đúng thứ GPU làm tốt nhất.

Kiến trúc dựng hoàn toàn trên ý tưởng ấy tên là **Transformer**, và nó là nền của các mô hình ngôn ngữ mà bạn dùng hằng ngày. Đó là nội dung của **chương Generative AI**, không phải bài này.

Cái bài này cần để lại là câu chuyện *vì sao* attention ra đời: không phải vì ai đó nghĩ ra một ý tưởng hay ho từ trên trời, mà vì hai con số ở mục 4 - 0,1138 và 3.825 giây - mô tả một bức tường mà mạng hồi quy không vượt qua được.

## Tóm tắt bài học

- Dữ liệu có thứ tự cần kiến trúc riêng: mạng kết nối đầy đủ đòi đầu vào cố định độ dài và không mang gì từ mẫu trước sang mẫu sau.
- RNN xử lý chuỗi từng bước, mang theo một trạng thái ẩn tóm tắt mọi thứ đã đọc, và dùng chung một bộ trọng số ở mọi bước - tương tự chia sẻ trọng số của CNN, nhưng theo thời gian.
- Đọc ảnh FashionMNIST như chuỗi 28 hàng: RNN đạt 0,8289 với 21.514 tham số; LSTM đạt 0,8687 với 82.186 tham số.
- LSTM tồn tại vì gradient phải đi ngược qua từng bước - cùng bệnh vanishing gradient của bài 08, với độ dài chuỗi đóng vai độ sâu. Nó cho trí nhớ đi qua một cổng quên học được thay vì một ma trận trọng số cố định, nên mạng tự quyết định khi nào giữ nguyên trí nhớ.
- Kéo chuỗi lên 784 bước thì **cả RNN lẫn LSTM đều sập về mức đoán bừa** (0,1138 và 0,1004), và LSTM mất 64 phút cho 3 epoch vì phải chạy tuần tự.
- Hai vấn đề khác nhau: chuỗi dài thì nhớ kém, và chuỗi dài thì train chậm vì không song song hóa được.
- **Attention** chữa cả hai bằng cách nối thẳng mọi cặp vị trí - đường đi còn một bước, và tính được song song. Kiến trúc dựng trên nó là Transformer, nội dung của chương Generative AI.

## Câu hỏi tự kiểm tra

1. Nêu hai lý do mạng kết nối đầy đủ xử lý rất vụng một câu tiếng Việt có độ dài thay đổi, và cái giá phải trả cho mỗi cách chữa tạm.
2. RNN dùng chung trọng số ở mọi bước. Điều đó tương ứng với tính chất nào của CNN ở bài 09, và cả hai cùng mua được lợi ích gì?
3. Vì sao vấn đề vanishing gradient của bài 08 quay lại với RNN, dù RNN chỉ có một tầng?
4. Ở RNN, trí nhớ bị nhân với một ma trận trọng số cố định ở mỗi bước; ở LSTM nó được nhân với cổng quên $f_t$. Vì sao thay đổi ấy giúp gradient sống sót qua chuỗi dài? Kiến trúc nào ở bài 08-09 dùng mẹo họ hàng?
5. Ở 784 bước, LSTM mất 64 phút cho 3 epoch còn CNN ở bài 09 chỉ mất 161 giây cho 10 epoch. Khác biệt về mặt kiến trúc nào gây ra chênh lệch này?
6. Attention chữa hai vấn đề của RNN. Nêu tên hai vấn đề đó và cách attention xử lý từng cái.

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **Dive into Deep Learning (d2l.ai)** | Chương 9 *Recurrent Neural Networks* - trạng thái ẩn, chia sẻ trọng số theo thời gian, và mục 9.7 về gradient qua thời gian; chương 10.1 *Long Short-Term Memory* - ba cổng của LSTM; chương 11.1 mở đầu về attention |
| **Bài 08 của chương này** | Vanishing gradient - cùng một bệnh, với độ dài chuỗi thay cho số tầng |

**Nguồn online bổ sung (miễn phí):**

- [Christopher Olah - Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) - bài giải thích LSTM được dẫn nhiều nhất, hình vẽ từng cổng rất rõ.
- [Karpathy - The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) - RNN sinh văn bản, kèm ví dụ thật.
- [`nn.RNN` / `nn.LSTM` - tài liệu PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) - chú ý tham số `batch_first` và dạng của đầu ra.
- [Hochreiter & Schmidhuber - Long Short-Term Memory (Neural Computation, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf) - bài báo gốc của LSTM.
- [Vaswani và cộng sự - Attention Is All You Need (NeurIPS 2017)](https://arxiv.org/abs/1706.03762) - bài báo Transformer; tên của nó chính là câu trả lời cho mục 5.

> **Bài tiếp theo:** [Transfer learning: đứng trên vai mô hình đã học](../transfer-learning-standing-on-a-trained-model-vi/) - mọi thí nghiệm trong chương đến giờ đều có 60.000 ảnh train - một điều kiện xa xỉ. Bài sau xử lý tình huống thật hơn nhiều: bạn chỉ có vài nghìn ảnh, và cách mượn lại thứ một mô hình khác đã học từ hàng triệu ảnh.

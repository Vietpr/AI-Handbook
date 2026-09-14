---
title: "SVM: tìm lề phân cách rộng nhất"
description: "Trong vô số đường tách được hai lớp, SVM chọn đường có lề rộng nhất: support vector, núm C cho lề mềm, kernel trick và gamma của kernel RBF, chạy SVC trên breast_cancer và khi nào SVM đáng dùng."
domain: "Machine Learning"
section: "Learn"
language: "vi"
translationKey: "svm-finding-the-widest-boundary"
order: 8
pubDate: 2026-08-16
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu vì sao trong vô số đường tách được hai lớp, SVM chọn đường có lề rộng nhất và chỉ vài điểm "support vector" quyết định nó; biết núm $C$ điều khiển độ khoan dung với điểm phạm lề; nắm trực giác kernel trick - nâng dữ liệu lên chiều cao hơn để tách bằng đường thẳng - và núm `gamma` của kernel RBF; chạy được `SVC` trên `breast_cancer` và thấy vì sao phải scale; biết khi nào SVM đáng dùng, khi nào không, và nó khác hồi quy logistic ở đâu.

## 1. Ba đường kẻ đều đúng - chọn đường nào?

Quay lại bàn duyệt vay với hai feature quen thuộc: số năm đi làm và thu nhập. Lần này dữ liệu "đẹp": nhóm trả đúng hạn (○) và nhóm trễ hạn (●) nằm tách hẳn nhau. Bạn cần kẻ một đường thẳng chia đôi mặt phẳng để phân loại khách mới. Rắc rối là có *vô số* đường đúng 100% trên tập train:

```
 thu nhập
   ↑                a   b     c
 8 │  ○  ○         /   /     /
 7 │ ○   ○  ○     /   /     /
 6 │   ○   ○     /   /     /
 5 │  ○      ○  /   /     /
 4 │           /   /     /    ●
 3 │          /   /     /   ●   ●
 2 │         /   /     /  ●   ●
 1 │        /   /     /     ●
   └────────────────────────────────→ số năm đi làm
```

Đường **a** sát nhóm ○: một khách ○ mới hơi lệch là bị xếp nhầm. Đường **c** sát nhóm ● thì ngược lại. Đường **b** nằm giữa, cách *cả hai* nhóm xa nhất - dữ liệu mới lệch đi một chút, b vẫn còn chỗ xoay xở. Khoảng cách từ đường kẻ tới điểm gần nhất gọi là **lề (margin)**; đường b là **maximal margin hyperplane** - đường có lề rộng nhất. Sách *An Introduction to Statistical Learning* mô tả nó là *trục giữa của dải rộng nhất* có thể chèn giữa hai lớp.

```
              lề          lề
      ○ ○    ┊    b    ┊
     ○  ○  ○*┊    /    ┊
       ○   ○ ┊   /     ┊
             ┊  /      ┊●*
             ┊ /       ┊  ●  ●
             ┊/        ┊●*  ●
```

Ba điểm đánh dấu `*` - nằm đúng trên mép lề - là **support vector (vector hỗ trợ)**: chúng "đỡ" đường kẻ. Điều bất ngờ là *chỉ chúng* quyết định đường b: dịch chuyển bất kỳ điểm nào khác (miễn không lấn vào lề) thì b không nhúc nhích. Trong lab chương 9 của sách, 50 điểm tách được hoàn hảo mà chỉ **3** support vector đủ dựng đường kẻ. Cái tên "máy vector hỗ trợ" (Cortes & Vapnik, *Machine Learning*, 1995) đến từ đó. Sách *Mathematics for Machine Learning* nêu thêm một lý do nên chọn lề rộng: Vapnik và Chervonenkis đã chứng minh lề lớn nghĩa là lớp hàm "ít phức tạp", nên học được - tức tổng quát hóa tốt.

"Hyperplane" (siêu phẳng) chỉ là cách gọi chung: 2 feature là đường thẳng, 3 feature là mặt phẳng, 30 feature (bộ `breast_cancer` ở mục 4) là "mặt phẳng" trong không gian 30 chiều - cùng dạng $b_0 + b_1 x_1 + \dots + b_p x_p = 0$ với điểm số $z$ của hồi quy logistic (Bài 03). Khác biệt nằm ở *tiêu chí chọn* $b$: logistic tối đa hóa likelihood, SVM tối đa hóa lề.

## 2. Lề mềm và núm C: khoan dung bao nhiêu?

Thêm một khách vào bảng: thu nhập cao, đi làm lâu, nhưng trễ hạn - một chấm ● rơi lọt vào giữa đám ○. Giờ không đường thẳng nào tách hoàn hảo nữa. Mà kể cả khi vẫn tách được, sách *An Introduction to Statistical Learning* cho thấy (Hình 9.5) chỉ *một* điểm mới cũng đủ làm đường lề rộng nhất xoay hẳn và lề co lại bé xíu: mô hình đã học tủ theo một quan sát.

Lời giải: **lề mềm (soft margin)** - cho phép một số điểm nằm trong lề, thậm chí sai bên, và tính "tiền phạt" cho mỗi vi phạm. Núm $C$ trong scikit-learn là **giá của mỗi vi phạm**:

- **$C$ lớn** = phạt nặng = ít khoan dung: cố không để điểm nào phạm lề → lề hẹp, ít support vector, bám sát tập train - variance cao (Foundations · Bài 14).
- **$C$ nhỏ** = khoan dung: chấp nhận nhiều điểm phạm lề để lề rộng → nhiều support vector, ranh giới "mượt" hơn nhưng có thể quá cứng - bias cao.

Số liệu thật trên `breast_cancer` (455 mẫu train đã scale, kernel RBF của mục 3):

| `C` | 0,01 | 0,1 | 1 (mặc định) | 10 | 100 |
|---|---|---|---|---|---|
| Số support vector | 341 | 199 | 97 | 84 | 64 |
| Accuracy train | 62,6% | 95,8% | 98,2% | 99,3% | 100% |
| Accuracy test | 63,2% | 94,7% | **98,2%** | 97,4% | 94,7% |

Hai đầu bảng quen thuộc: $C = 0{,}01$ khoan dung tới mức 341/455 điểm thành support vector và mô hình đoán cùn theo lớp đông (63% khối u lành); $C = 100$ thuộc lòng tập train rồi thua trên test. Support vector giờ gồm cả điểm nằm trên lề, trong lề và sai bên - nên $C$ càng nhỏ, danh sách càng dài.

> 🔧 **Thử ngay:** kiểm chứng bảng trên - thay `C` trong `make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, random_state=42))` (chia 80/20, `random_state=42`, `stratify=y`) và in `svm[-1].n_support_`.
> Đáp số: với `C=1`, `n_support_` in `[51 46]` - 51 support vector thuộc lớp ác tính, 46 thuộc lớp lành, tổng 97; với `C=100` là `[30 34]`. Muốn chọn $C$ tử tế thì đưa vào `GridSearchCV` trên pipeline như Bài 05, đừng nhìn test.

<details>
<summary><b>Đào sâu: hinge loss - vì sao điểm xa ranh giới "không có tiếng nói"</b></summary>

Có thể viết SVM lề mềm thành "loss + phạt" như Ridge ở Bài 04: tối thiểu $\sum_i \max\{0,\ 1 - y_i f(x_i)\} + \lambda \sum_j b_j^2$, với $y_i = \pm 1$ và $f(x) = b_0 + b_1 x_1 + \dots$. Số hạng thứ nhất là **hinge loss**: bằng **0** khi điểm nằm đúng bên và ngoài lề ($y_i f(x_i) \geq 1$), tăng tuyến tính khi điểm lấn vào lề hoặc sai bên. Vì loss bằng đúng 0 cho các điểm xa, chúng không ảnh hưởng tới nghiệm - đó là lý do chỉ support vector "có tiếng nói". Log loss của hồi quy logistic không bao giờ bằng 0 hẳn: điểm xa vẫn góp một chút. Sách *An Introduction to Statistical Learning* vẽ hai đường loss cạnh nhau (Hình 9.12) và nhận xét chúng khá giống nhau - nên hai mô hình thường cho kết quả gần nhau. $\lambda$ nhỏ ở đây tương ứng $C$ lớn.

</details>

> ⚠️ **Lưu ý nhỏ:** quy ước về $C$ *ngược nhau* giữa các tài liệu. Sách *An Introduction to Statistical Learning* định nghĩa $C$ là **ngân sách** cho tổng vi phạm: $C$ lớn = khoan dung hơn = lề rộng hơn. scikit-learn và sách *Mathematics for Machine Learning* định nghĩa $C$ là **hệ số phạt**: $C$ lớn = ít khoan dung. Bài này theo quy ước của scikit-learn. Đọc sách nào, kiểm tra công thức của sách đó trước khi xoay núm.

## 3. Kernel trick: không tách được thì nâng lên chiều cao hơn

Một cảm biến ghi độ lệch nhiệt độ phòng so với 24 °C, và người dùng đánh giá "dễ chịu" (1) hay "khó chịu" (0). Dữ liệu chỉ có *một* feature $x$:

```
 x:    -3   -2,5  -2   -1,5  -1   -0,5   0   0,5   1   1,5   2   2,5   3
 nhãn:  0    0    0    0     1    1     1    1    1    0    0    0    0
        ●────●────●────●────○────○────○────○────○────●────●────●────●
                        ↑ lạnh quá            nóng quá ↑
```

Một chiều thì "đường thẳng" chỉ là **một ngưỡng**: bên trái 0, bên phải 1. Không ngưỡng nào tách được cảnh "ở giữa là 1, hai bên là 0" - `LinearSVC` trên $x$ chỉ đạt 61,5% train, chưa hơn đoán cùn. Nhưng hãy thêm một trục: $z = x^2$.

```
 z = x²
 9 │ ●                                   ●
 6 │    ●                             ●
 4 │        ●                     ●
 2 │            ●             ●            ← ngưỡng z ≈ 1,6 tách hoàn hảo
 1 │ - - - - - - - -○- - - - - -○- - - - -
 0 │                    ○   ○   ○
   └───────────────────────────────────────→ x
     -3   -2    -1    0     1     2    3
```

Trong mặt phẳng $(x, z)$, các điểm 1 nằm dưới, các điểm 0 nằm trên: một đường *ngang* tách hoàn hảo - `LinearSVC` trên $(x, x^2)$ đạt 100%, ranh giới học được là $-1{,}43\,z + 2{,}30 = 0$, tức $z \approx 1{,}6$ hay $|x| \approx 1{,}27$. Chiếu ngược về trục $x$, đường ngang thành **hai điểm cắt** $\pm 1{,}27$: ranh giới thẳng ở không gian cao hơn = ranh giới *cong* ở không gian gốc. Sách *An Introduction to Statistical Learning* mở rộng đúng ý này (mục 9.3.1): dùng $2p$ feature $X_1, X_1^2, \dots, X_p, X_p^2$ - nhưng cảnh báo số feature phình rất nhanh nếu thêm bậc cao và tích chéo.

Đây là chỗ SVM có lối đi riêng. Khi giải bài toán lề rộng nhất, hóa ra mô hình *chỉ cần* các **tích vô hướng** giữa từng cặp điểm $\langle x_i, x_j \rangle$ (Foundations · Bài 07) - không cần tọa độ từng điểm. Vậy thay tích vô hướng bằng một hàm $K(x_i, x_j)$ trả về *tích vô hướng trong không gian đã nâng* mà không bao giờ phải dựng không gian đó - **kernel trick**. Kernel đa thức $(1 + \langle x_i, x_j \rangle)^d$ tương ứng với việc thêm mọi bậc và tích chéo tới bậc $d$. Kernel **RBF** (radial basis function, mặc định của `SVC`):

$$K(x_i, x_j) = \exp\!\big(-\gamma\,\|x_i - x_j\|^2\big)$$

là một thước đo *độ giống nhau* giảm dần theo khoảng cách - điểm gần thì $K$ gần 1, điểm xa thì $K$ gần 0 - nên chỉ các điểm train *lân cận* ảnh hưởng tới dự đoán, hơi giống k-NN của Bài 05. Sách nhận xét không gian ẩn của RBF có *vô hạn chiều*; ta không thể tính ở đó, nhưng nhờ kernel không cần phải tính.

Núm `gamma`: tài liệu scikit-learn giải thích "gamma quyết định một điểm train có tầm ảnh hưởng bao xa; gamma càng lớn, các điểm khác phải càng gần mới bị ảnh hưởng". Trên `breast_cancer` (scale, $C = 1$): `gamma=0.001` cho 95,6% test; `0.01` và giá trị mặc định `"scale"` (≈ 0,033 ở đây) cho 98,2%; `0.1` tụt về 95,6%; `gamma=1` thuộc lòng train (100%) với **454/455** điểm thành support vector rồi rơi xuống 63,2% trên test - $k = 1$ của Bài 05 trong lốt khác. Sách gặp cảnh này trên dữ liệu Heart: $\gamma = 0{,}1$ cho đường ROC đẹp nhất trên train và *tệ nhất* trên test.

> 🔧 **Thử ngay:** dựng lại ví dụ nhiệt độ - `x = np.arange(-3, 3.5, 0.5)`, nhãn `(np.abs(x) <= 1.2).astype(int)` - rồi chạy `SVC(kernel="rbf", C=10)` trực tiếp trên `x.reshape(-1, 1)` (một chiều, không tự thêm $x^2$).
> Đáp số: accuracy train **100%** với `n_support_ = [2 2]` - bốn điểm sát hai chỗ cắt là đủ; kernel đã tự "nâng chiều" thay bạn. `LinearSVC(C=10)` trên cùng dữ liệu một chiều: ≈ 0,615.

## 4. Scale trước, SVM sau - chạy thật trên breast_cancer

Bộ `breast_cancer` (Bài 05): 569 khối u, 30 số đo, cột `mean area` chạy tới 2.501 còn `mean smoothness` chưa tới 0,2. Lề, khoảng cách trong RBF và tích vô hướng đều tính bằng đơn vị của từng cột - không scale thì vài cột lớn át hẳn tiếng nói của cột nhỏ, y như k-NN. Tài liệu scikit-learn ghi thẳng: các thuật toán SVM *không bất biến theo thang đo*, nên "rất khuyến nghị scale dữ liệu".

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for kernel in ["linear", "rbf"]:
    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, random_state=42))   # scaler chỉ fit trên train
    svm.fit(X_train, y_train)
    print(kernel, round(svm.score(X_test, y_test), 3), "support vectors:", svm[-1].n_support_.sum())
    unscaled_model = SVC(kernel=kernel, random_state=42).fit(X_train, y_train)           # bỏ scale để so sánh
    print("   không scale:", round(unscaled_model.score(X_test, y_test), 3))
```

| Mô hình (114 mẫu test) | Có scale | Không scale | Support vector (có scale) |
|---|---|---|---|
| `SVC(kernel="linear")` | ≈ 97,4% | ≈ 95,6% | 32 / 455 |
| `SVC(kernel="rbf")` | ≈ 98,2% | ≈ 93,0% | 97 / 455 |
| Hồi quy logistic (Bài 03) | ≈ 98,2% | - | - |

Không scale, RBF mất 5 điểm. Có scale, kernel tuyến tính chỉ cần 32 trong 455 điểm train để dựng ranh giới - 423 điểm còn lại có xóa đi cũng không đổi mô hình. Thứ hạng giữa ba mô hình? Cross-validation 5 phần trên cả 569 mẫu: linear 97,2% ± 0,9, RBF 97,4% ± 1,5, logistic 98,1% ± 0,7 - hòa nhau trong biên nhiễu; với 114 mẫu test, chỉ cần đoán khác một ca là accuracy đã lệch 0,9 điểm phần trăm. Bộ dữ liệu này gần như tách tuyến tính được, nên kernel cong không có đất diễn.

> ⚠️ **Lưu ý nhỏ:** `random_state` của `SVC` chỉ có tác dụng khi `probability=True` (mục 5); thuật toán lõi là bài toán tối ưu lồi, cho cùng nghiệm ở mọi lần chạy - đặt nó để code nhất quán với cả chương. Với kernel RBF hãy dò *cặp* `C` và `gamma` cùng lúc bằng `GridSearchCV`, vì hai núm kéo co nhau: lab chương 9 của sách dò `C` trong {0,1; 1; 10; 100; 1000} và `gamma` trong {0,5; 1; 2; 3; 4}.

## 5. Khi nào dùng SVM, khi nào không - và so với hồi quy logistic

Mang `SVC(kernel="rbf")` sang bộ `adult` của Bài 03 - 39.073 dòng train, 105 cột sau one-hot, cùng pipeline có scale. Hồi quy logistic fit trong 1,8 giây. `SVC` mất **61,7 giây** để fit và **10,7 giây** để dự đoán 9.769 dòng, đổi lấy accuracy 86,0% (so với 85,2%) - và giữ lại 13.195 support vector, một phần ba tập train, trong bộ nhớ. Tài liệu scikit-learn giải thích: lõi của SVM là bài toán quy hoạch toàn phương, thời gian fit tăng ít nhất cỡ bậc hai theo số mẫu, nên "khó mở rộng quá vài chục nghìn mẫu"; thời gian dự đoán tỷ lệ với số support vector. Với dữ liệu lớn và ranh giới tuyến tính, `LinearSVC` (0,4 giây, 85,3% trên `adult`) hoặc `SGDClassifier` là lựa chọn thay thế; với ranh giới cong, quay về Bài 07.

Điểm yếu thứ hai: SVM **không sinh xác suất một cách tự nhiên**. `decision_function` trả về điểm số có dấu - dương/âm cho biết bên nào, độ lớn cho biết *ở xa ranh giới tới mức nào* (muốn ra khoảng cách hình học thật thì phải chia cho $\|w\|$) - không phải xác suất. Tùy chọn `probability=True` ghép thêm một hồi quy logistic lên điểm số ấy (*Platt scaling*), fit bằng cross-validation 5 phần bên trong - tốn thời gian, và tài liệu cảnh báo xác suất thu được có thể *không nhất quán* với `predict`. Cần xác suất để chọn ngưỡng theo chi phí (nấm death cap, Foundations · Bài 15) thì hồi quy logistic thẳng thắn hơn.

Vậy SVM tỏa sáng ở đâu? Ở dữ liệu **nhiều chiều so với số mẫu** - tài liệu scikit-learn ghi SVM "vẫn hiệu quả khi số chiều lớn hơn số mẫu":

- **Phân loại văn bản.** Joachims (ECML 1998) biểu diễn mỗi bài báo Reuters thành vector hàng chục nghìn chiều (mỗi từ một chiều, phần lớn bằng 0) và cho thấy SVM tuyến tính vượt các phương pháp cùng thời (naive Bayes, Rocchio, C4.5, k-NN) mà gần như không cần chỉnh tham số - lý do ông nêu: văn bản có *nhiều* feature đều có ích, và SVM chịu được điều đó.
- **Biểu hiện gene.** Lab chương 9 của sách chạy `SVC(kernel="linear")` trên bộ Khan: 2.308 gene, chỉ 63 mẫu train - 0 lỗi train, 2 lỗi trên 20 mẫu test. Guyon và cộng sự (2002) dùng SVM để *chọn* gene liên quan ung thư (SVM-RFE), kỹ thuật vẫn được dùng trong tin sinh học.
- **Chữ viết tay.** Trên bảng kết quả MNIST của LeCun, SVM kernel Gaussian sai 1,4%; "virtual SVM" kernel đa thức bậc 9 với ảnh xê dịch 2 pixel (DeCoste & Schölkopf, 2002) sai 0,56% - thấp hơn mạng tích chập LeNet-5 trên cùng bảng (0,95%).

So với hồi quy logistic - người anh em gần nhất - sách *An Introduction to Statistical Learning* (mục 9.5) tóm gọn: hai hàm mất mát **gần giống nhau về hình dạng** nên kết quả thường giống nhau - khác nhau ở chỗ hinge loss bằng đúng 0 với các điểm nằm ngoài lề (nên chỉ support vector mới ảnh hưởng tới đường kẻ), còn loss của logistic thì không bao giờ đúng 0, chỉ rất nhỏ; **hai lớp tách rõ** thì SVM có xu hướng tốt hơn, **hai lớp chồng lấn nhiều** thì hồi quy logistic thường được ưu tiên. Kernel trick không độc quyền của SVM - có thể ghép cho logistic - nhưng vì lịch sử, nó gắn với SVM. Với hơn hai lớp, `SVC` huấn luyện một SVM cho *từng cặp* lớp rồi bỏ phiếu (one-versus-one).

| Nên cân nhắc SVM khi | Nên tránh khi |
|---|---|
| Dữ liệu cỡ vừa (vài nghìn tới vài chục nghìn mẫu) | Hàng trăm nghìn mẫu trở lên (thời gian fit tăng ít nhất cỡ $n^2$) |
| Nhiều feature so với số mẫu: văn bản, gene | Cần xác suất được hiệu chuẩn để chọn ngưỡng |
| Ranh giới cong nhưng dữ liệu không đủ cho rừng cây | Cần giải thích hệ số cho từng feature (logistic, Bài 03) |
| Hai lớp tách khá rõ | Chưa scale được dữ liệu, hoặc nhiều cột chữ chưa mã hóa |

> 🔧 **Thử ngay (tính tay):** một mô hình trả về $f(x) = 2{,}5$; $0{,}4$; $-1{,}0$ cho ba khách, cả ba đều thuộc lớp $y = +1$. Hinge loss $\max\{0, 1 - y f(x)\}$ của từng người?
> Đáp số: **0** (ngoài lề, đúng bên - không có tiếng nói); **0,6** (đúng bên nhưng lấn vào lề); **2,0** (sai bên). Chỉ hai khách sau mới là support vector.

## Tóm tắt bài học

- Có vô số đường tách được hai lớp; SVM chọn đường có **lề rộng nhất**. Chỉ các **support vector** - điểm nằm trên hoặc phạm lề - quyết định đường kẻ; trên `breast_cancer`, kernel tuyến tính cần 32/455 điểm.
- **Lề mềm**: cho phép phạm lề, mỗi vi phạm bị phạt $C$ (quy ước scikit-learn). $C$ lớn = ít khoan dung, lề hẹp, dễ overfit (C = 100: 100% train, 94,7% test); $C$ nhỏ = khoan dung, có thể quá cứng (C = 0,01: 63%). Sách *An Introduction to Statistical Learning* dùng $C$ theo nghĩa ngược lại.
- **Kernel trick**: mô hình chỉ cần tích vô hướng giữa các cặp điểm, nên thay bằng kernel là tách được trong không gian cao hơn mà không dựng nó. Ví dụ 1 chiều: thêm $x^2$ biến hai điểm cắt thành một đường ngang.
- **RBF** đo độ giống nhau giảm theo khoảng cách; `gamma` lớn = tầm ảnh hưởng hẹp = dễ thuộc lòng (gamma = 1: 454/455 support vector, 63% test).
- **Scale gần như luôn cần**: RBF không scale mất 5 điểm trên `breast_cancer`. Dò `C` và `gamma` bằng `GridSearchCV` trên pipeline.
- Điểm yếu: thời gian fit tăng ít nhất cỡ $n^2$ theo số mẫu (`adult`: 62 giây so với 1,8 giây của logistic); không có xác suất tự nhiên. Điểm mạnh: nhiều chiều ít mẫu - văn bản (Joachims 1998), gene, chữ viết tay (0,56% lỗi MNIST).
- So với hồi quy logistic: hai loss gần giống nhau về hình dạng (hinge bằng đúng 0 ngoài lề, logistic thì không) nên kết quả thường giống nhau; lớp tách rõ → SVM, lớp chồng lấn hoặc cần xác suất/hệ số → logistic.

## Câu hỏi tự kiểm tra

1. Hai đường thẳng cùng phân loại đúng 100% tập train. Vì sao SVM vẫn thích đường có lề rộng hơn? Liên hệ An & Bình (Foundations · Bài 04).
2. Xóa một điểm train nằm xa ranh giới, đúng bên, ngoài lề. Đường SVM có đổi không? Hồi quy logistic có đổi không? Giải thích bằng hinge loss và log loss.
3. **Tính tay:** với dữ liệu nhiệt độ ở mục 3, thêm feature $z = x^2$ rồi dùng ngưỡng $z \leq 1{,}6$ → nhãn 1. Điểm $x = -1{,}5$ và $x = 1$ được xếp vào lớp nào? Kiểm bằng bảng nhãn.
4. Đồng nghiệp tăng `gamma` từ 0,01 lên 1 vì "accuracy train lên 100%". Bạn dự đoán chuyện gì xảy ra trên test và giải thích bằng số support vector.
5. Bạn có 2 triệu giao dịch thẻ, 40 cột, cần xác suất gian lận để đặt ngưỡng theo chi phí. SVM kernel RBF có phù hợp không? Nêu hai lý do và một mô hình thay thế từ các bài trước.
6. Bộ dữ liệu 80 bệnh nhân với 5.000 chỉ số gene. Vì sao SVM kernel tuyến tính là ứng viên hợp lý, và bạn phải cẩn thận điều gì khi đánh giá nó (Foundations · Bài 14)?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **An Introduction to Statistical Learning (ISLP)** | Chương 9: 9.1 - hyperplane, maximal margin classifier, support vector (Hình 9.2-9.3); 9.2 - support vector classifier, tham số $C$ (Hình 9.5-9.7); 9.3 - kernel đa thức, kernel radial, ví dụ Heart (Hình 9.10-9.11); 9.4 - one-versus-one / one-versus-all; 9.5 - hinge loss và quan hệ với hồi quy logistic (Hình 9.12); 9.6 - lab với `SVC`, `GridSearchCV`, bộ Khan |
| **Mathematics for Machine Learning (MML)** | Chương 12: 12.1-12.2 - hyperplane, khái niệm lề, vì sao đặt lề bằng 1, soft margin SVM nhìn từ hình học và từ loss (hinge loss, 12.2.5) - đọc để hiểu $C$ là hệ số phạt |
| **scikit-learn User Guide** | Mục 1.4 *Support Vector Machines* - ưu/nhược điểm, `SVC`/`LinearSVC`, kernel, độ phức tạp, *Tips on Practical Use* (scale, `C`, `gamma`, `probability`) |

**Nguồn online bổ sung (miễn phí):**

- [1.4. Support Vector Machines - scikit-learn User Guide](https://scikit-learn.org/stable/modules/svm.html) - tài liệu chính chủ; mục *Tips on Practical Use* là checklist trước khi chạy SVM.
- [SVC - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - tham số `C`, `kernel`, `gamma`, `probability`, thuộc tính `n_support_`, `support_vectors_`.
- [RBF SVM parameters - scikit-learn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) - lưới hình minh họa ranh giới đổi thế nào khi xoay `C` và `gamma`.
- [Cortes & Vapnik - Support-vector networks (Machine Learning, 1995)](https://link.springer.com/article/10.1007/BF00994018) - bài báo gốc giới thiệu SVM lề mềm.
- [Joachims - Text Categorization with Support Vector Machines: Learning with Many Relevant Features (ECML 1998)](https://link.springer.com/chapter/10.1007/BFb0026683) - vì sao SVM hợp với văn bản.
- [DeCoste & Schölkopf - Training Invariant Support Vector Machines (Machine Learning, 2002)](https://people.eecs.berkeley.edu/~malik/cs294/decoste-scholkopf.pdf) - virtual SVM đạt 0,56% lỗi trên MNIST; bảng kết quả tổng hợp tại [trang MNIST của LeCun](https://yann.lecun.com/exdb/mnist/).
- [Guyon, Weston, Barnhill, Vapnik - Gene Selection for Cancer Classification using Support Vector Machines (Machine Learning, 2002)](https://link.springer.com/article/10.1023/A:1012487302797) - SVM-RFE trong tin sinh học.

> **Bài tiếp theo:** [Phân cụm với k-means: gom nhóm khi không có nhãn](../k-means-clustering-without-labels-vi/) - tám bài qua đều có nhãn để học; bài sau bỏ nhãn đi - chỉ có các điểm, và câu hỏi là chúng tự gom thành mấy nhóm.

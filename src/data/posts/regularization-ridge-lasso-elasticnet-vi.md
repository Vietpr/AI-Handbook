---
title: "Regularization: Ridge, Lasso và nghệ thuật kìm mô hình"
description: "Kìm mô hình có quá nhiều feature bằng khoản phạt hệ số lớn: Ridge co đều, Lasso cắt hẳn về 0, ElasticNet lai hai kiểu, vì sao phải scale trước khi phạt và cách chọn alpha bằng cross-validation."
domain: "Machine Learning"
section: "Learn"
language: "vi"
translationKey: "regularization-ridge-lasso-elasticnet"
order: 4
pubDate: 2026-08-14
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu vì sao mô hình có nhiều feature so với số dòng dữ liệu rất dễ "học tủ", cách kìm nó bằng khoản phạt hệ số lớn - Ridge (L2) co đều, Lasso (L1) cắt hẳn về 0, ElasticNet lai hai kiểu - vì sao phải scale trước khi phạt, và cách chọn mức phạt `alpha` bằng cross-validation trên dữ liệu thật.

## 1. 1.000 cột, 200 dòng: khi "khớp hoàn hảo" là dấu hiệu xấu

Một nhóm nghiên cứu y sinh có 200 bệnh nhân; với mỗi người, máy giải trình tự trả về hàng nghìn chỉ số gene, và họ muốn dự đoán mức đáp ứng thuốc từ đó. Sách *An Introduction to Statistical Learning* nêu hai ví dụ thực tế của dạng bài toán này: khoảng 200 bệnh nhân với 500.000 biến thể gene (SNP) mỗi người; hay bộ NCI60 bạn sẽ gặp ở Bài 10 - 64 dòng tế bào ung thư, 6.830 gene mỗi dòng. Ký hiệu quen thuộc: $n$ dòng, $p$ cột, và ở đây $p \gg n$.

Đưa thẳng vào hồi quy tuyến tính (Bài 02) thì sao? Nhớ đa thức bậc 9 ở Foundations · Bài 13: 10 hệ số cho 8 điểm dữ liệu thì luồn qua *đúng từng điểm*, sai số train bằng 0, dự đoán uốn éo vô lý. Với 1.000 núm vặn cho 200 dòng, chuyện y hệt: bình phương tối thiểu tìm ra bộ hệ số khớp *hoàn hảo* tập train - sách nhấn mạnh điều này xảy ra **bất kể** feature có liên quan thật đến nhãn hay không. Hình 6.23 của sách với $n = 20$: cứ thêm feature nhiễu vô nghĩa, $R^2$ trên train leo lên 1, MSE train về 0, còn MSE test tăng vọt.

> 🔧 **Thử ngay:** tái hiện bằng dữ liệu giả - 1.000 cột, chỉ 10 cột thật sự liên quan đến $y$, 200 dòng train:
> ```python
> from sklearn.datasets import make_regression
> from sklearn.linear_model import LinearRegression, LassoCV
> from sklearn.metrics import root_mean_squared_error as rmse
> X, y = make_regression(n_samples=300, n_features=1000, n_informative=10, noise=10, random_state=42)
> X_tr, X_te, y_tr, y_te = X[:200], X[200:], y[:200], y[200:]        # 200 dòng train, 100 dòng test
> lin = LinearRegression().fit(X_tr, y_tr)
> print(rmse(y_tr, lin.predict(X_tr)), rmse(y_te, lin.predict(X_te)))  # train ≈ 0,0 | test ≈ 182
> las = LassoCV(cv=5, random_state=42, max_iter=20000).fit(X_tr, y_tr)
> print(rmse(y_te, las.predict(X_te)), (las.coef_ != 0).sum())          # test ≈ 12,5 | còn 120/1.000 hệ số
> ```
> Bình phương tối thiểu: sai số train **0,0** - "hoàn hảo" - và sai số test **≈ 182**, gần bằng độ lệch chuẩn của $y$ (≈ 213), tức chẳng hơn đoán trung bình là bao. Lasso (mục 3) đưa 880 hệ số về đúng 0, sai số test còn **≈ 12,5**, sát mức nhiễu 10 cài sẵn trong dữ liệu.

Ngoài y sinh, tài chính là sân chơi lớn của bài toán "nhiều cột, ít dòng": giới nghiên cứu đã đề xuất hàng trăm "nhân tố" giải thích lợi suất cổ phiếu, còn số tháng dữ liệu thì có hạn. Feng, Giglio và Xiu (*Journal of Finance*, 2020) dùng một phương pháp dựa trên Lasso để "thuần hóa sở thú nhân tố" ấy, và kết luận phần lớn nhân tố mới là thừa so với những nhân tố đã có.

## 2. Phạt hệ số lớn: thêm một khoản vào hàm mất mát

Mô hình học tủ để lại một dấu vết chung: **hệ số cực đoan** - đường cong uốn éo cần những hệ số lớn triệt tiêu lẫn nhau. Trên California Housing thêm feature bậc 2 (mục 5), bình phương tối thiểu có hệ số lớn nhất lên tới 35 về trị tuyệt đối, tổng trị tuyệt đối các hệ số là 358; mô hình bị kìm trên cùng dữ liệu chỉ còn tổng 3,4. Foundations · Bài 14 đã hứa "phạt trọng số lớn" - đây là cách làm cụ thể: **cộng vào hàm mất mát một khoản phạt tăng theo độ lớn của hệ số**.

$$\text{Loss mới} = \text{MSE} + \alpha \times \text{(độ "to" của các hệ số)}$$

Hai cách đo độ "to" thường dùng sinh ra hai mô hình:

| | **Ridge** (phạt L2) | **Lasso** (phạt L1) |
|---|---|---|
| Khoản phạt | $\alpha \sum_j b_j^2$ (tổng bình phương) | $\alpha \sum_j \lvert b_j \rvert$ (tổng trị tuyệt đối) |
| Với $b = (3;\ -0{,}5;\ 2)$ | $9 + 0{,}25 + 4 = 13{,}25$ | $3 + 0{,}5 + 2 = 5{,}5$ |
| Với $b = (1;\ -0{,}5;\ 1)$ | $2{,}25$ | $2{,}5$ |
| Tác dụng lên hệ số | co đều về gần 0, nhưng không có cơ chế ép chúng thành đúng 0 | co và **đưa nhiều hệ số về đúng 0** |
| Ra đời | Hoerl & Kennard, 1970 | Tibshirani, 1996 - *least absolute shrinkage and selection operator* |

$\alpha$ (sách *An Introduction to Statistical Learning* viết $\lambda$; scikit-learn viết `alpha`) là **hyperparameter** - núm bạn chọn, không phải thứ mô hình tự học ra (Foundations · Bài 13). $\alpha = 0$: về lại bình phương tối thiểu; $\alpha \to \infty$: mọi hệ số về 0, chỉ còn hệ số chặn - chính là "đoán trung bình" của `DummyRegressor` ở Bài 01. Hệ số chặn $b_0$ không bị phạt: nó là mức nền của $y$, không phải "độ phức tạp".

Vì sao phạt lại giúp dự đoán tốt hơn? Foundations · Bài 14: chấp nhận bias nhích lên để kéo variance xuống. Sách mô phỏng với $p = 45$, $n = 50$ (hình 6.5): khi $\lambda$ tăng từ 0 đến khoảng 10, variance tụt nhanh trong khi bias gần như chưa nhúc nhích nên MSE test giảm rõ; điểm thấp nhất ở $\lambda \approx 30$. Sách *Dive into Deep Learning* diễn đạt trực giác của phạt L2 (giới deep learning gọi là **weight decay**) gọn hơn: trong tất cả các hàm thì hàm $f = 0$ là đơn giản nhất, nên có thể đo độ phức tạp của mô hình bằng khoảng cách từ tham số tới 0.

> 🔧 **Thử ngay:** hai mô hình cùng MSE = 1,0 trên train. Mô hình A có hệ số $(4;\ 0;\ 0)$, mô hình B có $(2;\ 2;\ 0)$. Với $\alpha = 0{,}5$, Ridge thích mô hình nào? Lasso thích mô hình nào?
> Ridge: A bị phạt $0{,}5 \times 16 = 8$, B bị phạt $0{,}5 \times 8 = 4$ → chọn **B** (trải đều hệ số). Lasso: A $0{,}5 \times 4 = 2$, B $0{,}5 \times 4 = 2$ → **hòa**. L1 không "ghét" một hệ số lớn theo kiểu bình phương, nên sẵn sàng dồn hết vào ít cột và cho phần còn lại về 0.

## 3. Ridge co đều, Lasso cắt hẳn: vì sao khác nhau

Sách *An Introduction to Statistical Learning* có một trường hợp đặc biệt tính tay được ($n = p$, các cột trực giao): gọi $\hat b$ là hệ số bình phương tối thiểu. Ridge cho $\hat b/(1 + \lambda)$ - co theo *tỷ lệ*, hệ số nào cũng còn một phần. Lasso trừ đi một *lượng cố định* $\lambda/2$, và hệ số nào nhỏ hơn $\lambda/2$ thì về **đúng 0** (sách gọi là *soft-thresholding*). Với $\lambda = 1$:

| $\hat b$ (bình phương tối thiểu) | Ridge | Lasso |
|---|---|---|
| 3,0 | 1,5 | 2,5 |
| 0,4 | 0,2 | **0** |
| −2,0 | −1,0 | −1,5 |

Hình học của chuyện này (hình 6.7 của sách): tập hệ số được phép của Lasso là một **hình thoi** có góc nằm đúng trên trục tọa độ; của Ridge là một **hình tròn**. Các ellipse đồng mức của MSE cứ loang dần ra: gặp hình thoi, chúng hay chạm đúng vào *góc* - nơi một hệ số bằng 0; gặp hình tròn, điểm chạm gần như không bao giờ rơi trúng trục:

```
   LASSO: |b1| + |b2| ≤ s                    RIDGE: b1² + b2² ≤ s
             b2                                       b2
              │                                        │
              ◆ ← góc nằm ĐÚNG trên trục:            ╭─┴─╮
             ╱ ╲   ellipse MSE hay chạm vào          ╱  │  ╲ ← không có góc: điểm chạm
   ─────────◆───◆────────── b1               ───────┤   │   ├─────── b1   thường lệch khỏi trục
             ╲ ╱   đây → b1 = 0                      ╲  │  ╱     → b1 và b2 đều khác 0
              ◆                                       ╰─┬─╯
              │                                        │
```

Chạy thật trên 8 cột gốc của California Housing (đã scale, tập train của Bài 01) rồi tăng dần `alpha` - mỗi ô là độ lớn hệ số của một cột:

```
Lasso - |hệ số| theo alpha:

            alpha = 0 (không phạt)   alpha = 0,03        alpha = 0,1        alpha = 0,3
MedInc      █████████ 0,85          ███████▍ 0,74       ███████ 0,71       █████ 0,50
Latitude    █████████ 0,90          █████▌ 0,55         ▏0,01              · 0
Longitude   ████████▋ 0,87          █████ 0,50          · 0                · 0
AveBedrms   ███▍ 0,34               ▏0,02               · 0                · 0
AveRooms    ███ 0,29                · 0                 · 0                · 0
HouseAge    █▏ 0,12                 █▍ 0,13             █ 0,11             · 0
AveOccup    ▍ 0,04                  ▏0,01               · 0                · 0
Population  ▏ 0,00                  · 0                 · 0                · 0
Cột còn lại:      8                       6                  3                  1
```

Cột yếu (Population, AveRooms) rụng trước, cột mạnh (MedInc) trụ đến cùng; đến `alpha = 1` thì rụng hết. Ridge trên cùng dữ liệu với `alpha = 10.000`: hệ số lớn nhất còn 0,49, nhỏ nhất 0,001 - co rất mạnh nhưng **không hệ số nào đúng bằng 0**. Vì thế *Lasso làm luôn việc chọn feature*: mô hình cuối chỉ chứa một phần cột - sách gọi là mô hình **thưa (sparse)**.

**ElasticNet** trộn hai khoản phạt theo tỷ lệ `l1_ratio` (1 = thuần Lasso, 0 = thuần Ridge). Chọn cái nào? Sách tổng kết qua hai mô phỏng: khi chỉ vài cột thật sự quan trọng, Lasso thắng; khi nhiều cột cùng đóng góp những phần nhỏ, Ridge thắng - không biết trước dữ liệu thuộc kiểu nào thì để cross-validation quyết (mục 5).

> ⚠️ **Lưu ý nhỏ:** khi nhiều cột tương quan mạnh (các gene cùng biểu hiện, các chỉ số tài chính cùng tăng giảm), Lasso có xu hướng giữ *một* cột trong nhóm và bỏ phần còn lại - cột nào được giữ có thể đổi khi dữ liệu train đổi chút ít; ElasticNet có xu hướng giữ cả nhóm. Sách *An Introduction to Statistical Learning* cảnh báo riêng cho $p > n$: đa cộng tuyến ở mức cực đoan, nên không thể kết luận "17 gene được chọn là 17 gene quyết định" - chỉ là *một trong nhiều* mô hình dự đoán tốt.

## 4. Scale trước, phạt sau - nếu không, mức phạt phụ thuộc đơn vị đo

Cột thu nhập tính bằng **đồng** cho hệ số 0,000002; đổi sang **triệu đồng**, hệ số thành 2 - cùng một mối quan hệ, chỉ đổi đơn vị. Nhưng khoản phạt L2 của riêng hệ số này nhảy từ $4 \times 10^{-12}$ lên $4$: Ridge sẽ kìm phiên bản "triệu đồng" mạnh hơn hàng nghìn tỷ lần. Sách *An Introduction to Statistical Learning* nói bình phương tối thiểu *bất biến với thang đo*, còn Ridge và Lasso thì không - nên sách khuyên chuẩn hóa mọi cột về độ lệch chuẩn 1 trước khi phạt (công thức 6.6), tức `StandardScaler` của Foundations · Bài 12.

Kiểm chứng trên 8 cột California: Population có độ lệch chuẩn ≈ 1.137, MedInc ≈ 1,9. Cùng `alpha = 0,03`: có scale thì AveRooms và Population về 0; không scale thì bộ cột bị loại khác hẳn và hệ số MedInc còn 0,38 thay vì 0,74 - hai mô hình khác nhau chỉ vì đơn vị đo. Trong `Pipeline`, thứ tự luôn là *tạo feature → scale → mô hình có phạt*, và cả ba bước chỉ fit trên train.

> 🔧 **Thử ngay:** cột "diện tích" đo bằng m² có hệ số 0,05. Đổi sang đơn vị "chục m²" thì hệ số bằng bao nhiêu, và khoản phạt L2 của riêng hệ số đó tăng bao nhiêu lần?
> Hệ số thành **0,5**; phạt từ $0{,}05^2 = 0{,}0025$ lên $0{,}5^2 = 0{,}25$ - **gấp 100 lần**, dù mô hình không hề thay đổi về bản chất.

## 5. Chọn alpha bằng cross-validation - và đọc kết quả thật

`alpha` là núm bạn chọn: thử một dãy giá trị, so sai số trên dữ liệu chưa học, lấy giá trị cho sai số thấp nhất - chính là k-fold cross-validation của Foundations · Bài 04. Sách *An Introduction to Statistical Learning* minh họa (hình 6.13): với $p = 45$, $n = 50$ và chỉ 2 cột thật sự có tín hiệu, Lasso kết hợp 10-fold cross-validation nhặt đúng 2 cột đó, phần còn lại bằng 0. scikit-learn gói sẵn `LassoCV` (mặc định 5-fold, tự sinh 100 giá trị `alpha`) và `RidgeCV` (bạn đưa danh sách `alpha`, nó dùng leave-one-out tính rất nhanh); chọn xong, cả hai fit lại trên toàn bộ tập train.

```mermaid
flowchart LR
    T["Tập train<br/>16.512 dòng"] --> F["PolynomialFeatures(2)<br/>8 → 44 cột"]
    F --> S["StandardScaler"]
    S --> CV["LassoCV: 5-fold trên<br/>100 giá trị alpha"]
    CV --> A["alpha có sai số<br/>cross-validation thấp nhất"]
    A --> R["Fit lại Lasso trên<br/>toàn bộ tập train"]
    R --> E["Đánh giá một lần<br/>trên tập test (4.128 dòng)"]
```

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import root_mean_squared_error

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Thêm feature bậc 2 (bình phương và tích từng cặp): 8 cột → 44 cột
lin = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                    StandardScaler(), LinearRegression()).fit(X_train, y_train)
lasso = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                      StandardScaler(),
                      LassoCV(cv=5, random_state=42, max_iter=20000)).fit(X_train, y_train)

print("RMSE test - Linear+poly2:", root_mean_squared_error(y_test, lin.predict(X_test)))
print("RMSE test - LassoCV+poly2:", root_mean_squared_error(y_test, lasso.predict(X_test)))
coefficients = lasso[-1].coef_
print("alpha chọn:", lasso[-1].alpha_, "| số hệ số = 0:", (coefficients == 0).sum(), "/", len(coefficients))
```

| Mô hình (đơn vị giá: trăm nghìn đô) | Số cột | Hệ số = 0 | RMSE train | RMSE test |
|---|---|---|---|---|
| `LinearRegression`, 8 cột gốc (Bài 01) | 8 | 0 | ≈ 0,720 | ≈ 0,746 |
| `LinearRegression` + bậc 2 | 44 | 0 | ≈ 0,649 | ≈ 0,681 |
| `RidgeCV` + bậc 2 (chọn `alpha` = 1.000) | 44 | 0 | ≈ 0,715 | ≈ 0,748 |
| `LassoCV` + bậc 2 (chọn `alpha` ≈ 0,007) | 44 | **27** | ≈ 0,713 | ≈ 0,747 |

Đọc cho trung thực: Lasso đưa 27/44 hệ số về đúng 0, giữ 17 cột (Latitude, Longitude, tích MedInc×Longitude dẫn đầu), RMSE ngang mô hình 8 cột gốc - nhưng *thua* bình phương tối thiểu trên cùng 44 cột (0,747 so với 0,681). Không mâu thuẫn: với 16.512 dòng cho 44 cột, bình phương tối thiểu không hề học tủ (train 0,649, test 0,681), nên chẳng có variance nào để đổi; `LassoCV` cũng "biết" thế - nó chọn `alpha` sát đáy lưới, tức "phạt càng ít càng tốt". Sách gặp đúng tình huống này với Ridge trên bộ `Credit` (hình 6.12) và bình luận: đáy đường cross-validation không rõ rệt thì có thể cứ dùng bình phương tối thiểu. Bài học kép: regularization là công cụ cho tình huống *ít dòng, nhiều cột* (mục 1), không phải thứ cứ thêm là hay; còn khi cần mô hình gọn để giải thích, Lasso vẫn đáng giá - 17 hệ số thay vì 44 mà sai số ngang mô hình gốc.

<details>
<summary><b>Đào sâu: cắt bớt dữ liệu train xem regularization phát huy khi nào</b></summary>

Vẫn 44 cột bậc 2, chỉ dùng $n$ dòng đầu của tập train, đánh giá trên cùng tập test - RMSE test (chạy thật):

| $n$ dòng train | `LinearRegression` | `LassoCV` | Hệ số Lasso = 0 |
|---|---|---|---|
| 100 | ≈ 103 | ≈ 0,91 | 38/44 |
| 200 | ≈ 19,0 | ≈ 0,95 | 32/44 |
| 500 | ≈ 2,93 | ≈ 2,40 | 28/44 |
| 2.000 | ≈ 4,44 | ≈ 3,37 | 24/44 |
| 16.512 | ≈ 0,68 | ≈ 0,75 | 27/44 |

Với 100-200 dòng, bình phương tối thiểu "nổ" (RMSE 19-103, trong khi đoán trung bình chỉ lệch 1,14) còn Lasso giữ được mức 0,9. Ở 500-2.000 dòng, *cả hai* vẫn tệ hơn đoán trung bình - tập test có 8 căn ngoại lai (AveOccup > 30 hoặc AveRooms > 30), và feature bậc 2 thổi giá trị của chúng lên rất lớn; bỏ 8 dòng đó ra, hai mô hình về lại 0,66-0,70. Regularization giảm thiệt hại nhưng không thay bạn xử lý outlier (Foundations · Bài 12).

</details>

> ⚠️ **Lưu ý nhỏ:** `alpha` của `Ridge` và của `Lasso` trong scikit-learn **không cùng thang**: Lasso chia tổng bình phương sai số cho $2n$ trước khi cộng phạt, Ridge cộng thẳng - nên `alpha = 1.000` của Ridge ở bảng trên không "mạnh gấp 140.000 lần" `alpha ≈ 0,007` của Lasso; so sánh mức phạt phải qua kết quả cross-validation. Lưới của `LassoCV` mặc định chỉ dò từ `alpha_max` xuống `alpha_max/1000`; nếu nó chọn giá trị ở sát đáy, hãy nới lưới hoặc coi đó là tín hiệu "dữ liệu đủ nhiều, phạt không cần thiết".

## 6. Regularization không chỉ có trong hồi quy tuyến tính

Ý "thêm khoản phạt vào loss" đi khắp loạt bài này. `LogisticRegression` ở Bài 03 mặc định phạt L2 với `C` đóng vai $1/\alpha$ - `C` nhỏ là phạt mạnh; SVM (Bài 08) có tham số `C` cùng tinh thần. Cây quyết định và các mô hình gom cây (Bài 06-07) không phạt hệ số mà kìm bằng độ sâu, số lá, tốc độ học. Mạng nơ-ron gọi phạt L2 là weight decay, có thêm dropout và early stopping (Foundations · Bài 14). Tên khác nhau, một ý chung: hy sinh một chút độ khớp trên train để đổi lấy tổng quát hóa.

## Tóm tắt bài học

- Khi số cột $p$ lớn so với số dòng $n$ (và ma trận thiết kế đủ hạng), bình phương tối thiểu có thể khớp hoàn hảo tập train **bất kể** feature có ý nghĩa hay không - sai số train bằng 0 là dấu hiệu xấu (1.000 cột / 200 dòng: test RMSE ≈ 182, so với ≈ 12,5 sau khi kìm).
- **Regularization** = cộng vào hàm mất mát một khoản phạt theo độ lớn hệ số; `alpha` là hyperparameter, `alpha = 0` là bình phương tối thiểu, `alpha` → ∞ là đoán trung bình; hệ số chặn không bị phạt.
- **Ridge (L2)** co đều mọi hệ số về gần 0 nhưng không ép chúng thành đúng 0; **Lasso (L1)** thì có thể đưa nhiều hệ số về **đúng 0** - làm luôn việc chọn feature; **ElasticNet** trộn hai kiểu, hợp khi nhiều cột tương quan.
- Cơ chế bên dưới là đánh đổi bias-variance (Foundations · Bài 14): phạt làm variance giảm nhanh hơn bias tăng - chỉ khi mô hình *có* variance để giảm.
- **Scale trước khi phạt**: khoản phạt phụ thuộc đơn vị đo; gói `PolynomialFeatures → StandardScaler → Lasso` vào `Pipeline`, fit chỉ trên train.
- Chọn `alpha` bằng cross-validation (`LassoCV`, `RidgeCV`). Trên California Housing (16.512 dòng, 44 cột), cross-validation chọn phạt rất nhẹ và bình phương tối thiểu vẫn dự đoán tốt hơn (0,681 so với 0,747) - regularization là công cụ cho lúc thiếu dữ liệu.
- Cùng một ý xuất hiện ở logistic (`C`), SVM (`C`), cây (độ sâu), mạng nơ-ron (weight decay, dropout).

## Câu hỏi tự kiểm tra

1. Một đồng nghiệp khoe mô hình hồi quy tuyến tính đạt $R^2 = 1{,}0$ trên tập train với 300 feature và 150 dòng dữ liệu. Vì sao con số này không nói lên điều gì? Bạn đề nghị kiểm tra gì tiếp?
2. **Tính tay:** hệ số bình phương tối thiểu là $(2{,}0;\ 0{,}3;\ -1{,}0)$. Trong trường hợp đặc biệt của sách với $\lambda = 1$, Ridge và Lasso cho ra bộ hệ số nào? Hệ số nào bị Lasso đưa về 0?
3. Giải thích bằng hình học (hình thoi và hình tròn) vì sao Lasso cho hệ số đúng bằng 0 còn Ridge thì không.
4. Bạn fit Lasso lên dữ liệu chưa scale, trong đó cột "dân số phường" tính bằng người và cột "thu nhập" tính bằng tỷ đồng. Cột nào có nguy cơ bị loại oan, vì sao, và sửa thế nào?
5. Trên bộ dữ liệu 20.000 dòng và 44 cột, `LassoCV` chọn `alpha` ở sát đáy lưới và RMSE test vẫn kém `LinearRegression` một chút. Bạn kết luận gì về vai trò của regularization ở bài toán này, và trong tình huống nào kết luận đó đảo ngược?
6. Một hệ thống gợi ý dùng hồi quy logistic với 5.000 feature one-hot nhưng chỉ 3.000 dòng huấn luyện. Bạn sẽ chỉnh tham số nào của `LogisticRegression`, theo hướng nào, và chọn giá trị bằng cách nào?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **An Introduction to Statistical Learning (ISLP)** | Chương 6, mục 6.2 - *Shrinkage Methods*: 6.2.1 Ridge (hình 6.4-6.5, chuẩn hóa theo công thức 6.6), 6.2.2 Lasso (hình 6.6-6.7, soft-thresholding ở hình 6.10), 6.2.3 chọn $\lambda$ bằng cross-validation (hình 6.12-6.13) |
| **An Introduction to Statistical Learning (ISLP)** | Chương 6, mục 6.4 - *Considerations in High Dimensions*: $p > n$, hình 6.23-6.24, cảnh báo về diễn giải kết quả |
| **Dive into Deep Learning (d2l.ai)** | Chương 3 (Linear Neural Networks for Regression) - mục *Weight Decay*: phạt L2 dưới góc nhìn deep learning, demo 200 feature với 20 mẫu train |

**Nguồn online bổ sung (miễn phí):**

- [Linear Models - scikit-learn User Guide](https://scikit-learn.org/stable/modules/linear_model.html) - hàm mục tiêu chính xác của `Ridge`, `Lasso`, `ElasticNet`; ghi chú `RidgeCV` dùng leave-one-out.
- [Lasso model selection: AIC-BIC / cross-validation - scikit-learn](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html) - ví dụ chính chủ so sánh các cách chọn `alpha`.
- [Least Absolute Shrinkage and Selection Operator (LASSO) - Columbia Mailman School of Public Health](https://www.publichealth.columbia.edu/research/population-health-methods/least-absolute-shrinkage-and-selection-operator-lasso) - tổng quan Lasso, có dẫn Tibshirani (1996).
- [Ridge Regression: Biased Estimation for Nonorthogonal Problems - Hoerl & Kennard, Technometrics 1970](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) - bài báo gốc của Ridge.
- [Taming the Factor Zoo: A Test of New Factors - Feng, Giglio & Xiu (NBER)](https://www.nber.org/papers/w25481) - ứng dụng Lasso trong tài chính để sàng lọc hàng trăm nhân tố.
- [Overfitting - Machine Learning cơ bản](https://machinelearningcoban.com/2017/03/04/overfitting/) - mục 3 "Regularization" bằng tiếng Việt: L2, L1, Elastic Net.

> **Bài tiếp theo:** [k-Nearest Neighbors: học từ hàng xóm](../k-nearest-neighbors-learning-from-neighbors-vi/) - rời các mô hình có hệ số để gặp một mô hình không "học" gì lúc train - chỉ nhớ dữ liệu và hỏi hàng xóm; ở đó thứ cần kìm không phải hệ số mà là số hàng xóm $k$.

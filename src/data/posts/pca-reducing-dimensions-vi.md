---
title: "PCA: giảm chiều và chọn số thành phần cần giữ"
description: "PCA tìm trục mới theo hướng nào và vì sao, đọc biểu đồ tỉ lệ phương sai giải thích để chọn số thành phần, dùng PCA để vẽ dữ liệu nhiều chiều và làm tiền xử lý, cùng ba giới hạn của nó."
domain: "Machine Learning"
section: "Learn"
language: "vi"
translationKey: "pca-reducing-dimensions"
order: 10
pubDate: 2026-08-17
featured: false
draft: false
---
> **Mục tiêu bài học:** Sau bài này, bạn sẽ hiểu PCA tìm trục mới theo hướng nào và vì sao, đọc được biểu đồ tỉ lệ phương sai giải thích để chọn số thành phần, dùng PCA vừa để vẽ dữ liệu nhiều chiều lên mặt phẳng vừa để làm bước tiền xử lý, và biết ba giới hạn khiến PCA không phải lúc nào cũng là ý hay.

## 1. Hai con số vẽ lại bản đồ châu Âu

Năm 2008, nhóm của John Novembre công bố trên tạp chí *Nature* một kết quả gây chú ý. Họ lấy mẫu gene của khoảng 3.000 người châu Âu, mỗi người được đọc hơn nửa triệu vị trí biến thiên trên bộ gene. Mỗi người vì thế là một điểm trong không gian hơn 500.000 chiều - không ai vẽ nổi thứ đó.

Họ nén 500.000 chiều ấy xuống còn **hai** con số cho mỗi người, rồi chấm 3.000 điểm lên một mặt phẳng. Kết quả: đám điểm hiện ra gần như đúng hình bản đồ châu Âu. Bán đảo Iberia một góc, bán đảo Ý một góc, nhóm Anh và nhóm Ireland tách bạch. Chỉ dựa vào gene, nhóm nghiên cứu đoán được nơi sinh của hơn 90% số người với sai số dưới 700 km, và của hơn 50% với sai số dưới 310 km. Không ai đưa vào tọa độ địa lý cả - hai con số đó tự trồi lên từ dữ liệu.

Kỹ thuật họ dùng là **PCA (Principal Component Analysis - phân tích thành phần chính)**. Nó trả lời một câu hỏi: nếu buộc phải mô tả dữ liệu bằng ít con số hơn hẳn, thì mất mát ít nhất là bao nhiêu, và những con số ấy là gì.

Nhu cầu này gặp ở khắp nơi. Bộ `digits` có 64 cột (ảnh số viết tay 8×8 pixel). Bộ `breast_cancer` có 30 cột. Dữ liệu biểu hiện gene có hàng chục nghìn cột trên vài chục bệnh nhân. Cột nhiều thì ba thứ cùng khó: vẽ không nổi, mô hình dễ học vẹt (Foundations · Bài 14), và khoảng cách mất dần ý nghĩa như bài 05 đã nói về lời nguyền số chiều.

## 2. Trục mới đặt theo hướng dữ liệu trải rộng nhất

Hình dung một đám điểm hai chiều nằm chéo, kéo dài theo hướng đông bắc - tây nam:

```
  y
  |            . *
  |         .* * .          Trục PC1 chạy dọc theo chiều dài
  |      .* *.*  .          của đám mây - hướng dữ liệu trải
  |    * *.* .              rộng nhất.
  |  .*.* .                 Trục PC2 vuông góc với PC1, đi theo
  | *. .                    bề ngang còn lại.
  +------------------- x
```

Trục $x$ và trục $y$ ban đầu không có gì đặc biệt - chúng chỉ là hai cột ta tình cờ đo. Nếu được chọn lại hệ trục, ta sẽ đặt trục thứ nhất **dọc theo chiều dài của đám mây**, tức hướng mà các điểm chênh lệch nhau nhiều nhất. Trục đó gọi là **thành phần chính thứ nhất (PC1)**. Trục thứ hai vuông góc với nó, hứng phần biến thiên còn lại, gọi là PC2. Cứ thế cho tới hết số chiều.

"Chênh lệch nhau nhiều nhất" ở đây đo bằng **phương sai** của dữ liệu khi chiếu lên trục đó. Phép chiếu một điểm lên một trục chính là tích vô hướng mà Foundations · Bài 07 đã dựng, còn mỗi thành phần chính là một vector đơn vị trong không gian gốc (Foundations · Bài 06). Nói gọn: PCA đi tìm vector $\phi_1$ sao cho khi chiếu toàn bộ dữ liệu lên nó, phương sai thu được là lớn nhất.

Vì sao lại là phương sai? Vì phương sai theo một hướng chính là lượng thông tin phân biệt các điểm theo hướng ấy. Một cột mà mọi dòng đều có giá trị như nhau (phương sai 0) thì bỏ đi chẳng mất gì - nó không giúp phân biệt dòng nào với dòng nào. Hướng nào phương sai lớn thì giữ lại, hướng nào gần phẳng thì bỏ, và đó là toàn bộ chiến lược giảm chiều của PCA.

Một tính chất đi kèm: giữ $M$ thành phần đầu tiên cho ta xấp xỉ $M$ chiều **tốt nhất có thể** của dữ liệu, theo nghĩa tổng bình phương sai lệch giữa điểm gốc và điểm được tái tạo là nhỏ nhất. Tối đa hóa phương sai giữ lại và tối thiểu hóa sai số tái tạo hóa ra là cùng một bài toán nhìn từ hai phía.

> ⚠️ **Lưu ý nhỏ:** PCA luôn trừ đi giá trị trung bình của từng cột trước khi làm việc (`sklearn.decomposition.PCA` tự làm điều này). Không có bước ấy, "hướng phương sai lớn nhất" sẽ bị kéo về hướng nối gốc tọa độ với tâm đám mây, không còn phản ánh hình dạng dữ liệu nữa.

## 3. Bao nhiêu thành phần là đủ?

Mỗi thành phần giữ được một phần phương sai của toàn bộ dữ liệu, gọi là **tỉ lệ phương sai giải thích (explained variance ratio)**. Trên bộ `wine` đã chuẩn hóa (178 chai, 13 chỉ số hóa học):

```python
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

w  = load_wine()
Xs = StandardScaler().fit_transform(w.data)
p  = PCA().fit(Xs)
print(np.round(p.explained_variance_ratio_ * 100, 1))
print(np.round(np.cumsum(p.explained_variance_ratio_) * 100, 1))
```

| Thành phần | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
|---|---|---|---|---|---|---|---|---|
| Giải thích | 36,2% | 19,2% | 11,1% | 7,1% | 6,6% | 4,9% | 4,2% | 2,7% |
| Cộng dồn | 36,2% | **55,4%** | 66,5% | 73,6% | 80,2% | 85,1% | 89,3% | **92,0%** |

Đọc bảng này theo mục đích:

- **Để vẽ hình:** lấy PC1 và PC2, chấp nhận chỉ giữ 55,4%. Bức tranh có méo, nhưng trong mọi cách chiếu xuống 2 chiều thì đây là cách làm sai số tái tạo bình phương nhỏ nhất.
- **Để làm tiền xử lý:** đặt ngưỡng phương sai muốn giữ. `PCA(n_components=0.90)` giữ 90% - với `wine` là 8 thành phần thay cho 13; muốn 95% thì cần 10.

Cách nhìn quen thuộc là vẽ đường cong cộng dồn và tìm chỗ nó thoải ra - cùng một kiểu "khuỷu tay" như khi chọn $k$ ở bài 09, và cũng cùng một nhược điểm: nhiều tập dữ liệu cho đường cong trơn tru không có khuỷu nào rõ ràng. Khi đó, cách chắc tay hơn là đặt PCA vào `Pipeline` rồi để cross-validation chấm điểm cho từng số thành phần - coi số thành phần như một siêu tham số bình thường (Foundations · Bài 13).

## 4. Không chuẩn hóa thì PCA chỉ nghe theo cột có đơn vị to nhất

Bài 09 vừa gặp chuyện này với k-means. Với PCA thì hậu quả còn rõ hơn nữa, vì PCA đi tìm hướng có **phương sai** lớn nhất, mà phương sai phụ thuộc trực tiếp vào đơn vị đo.

Chạy lại `PCA()` trên `wine` **chưa** chuẩn hóa: PC1 giải thích **99,81%** phương sai. Nghe như một tin tuyệt vời - một chiều thay cho mười ba. Nhưng nhìn vào hệ số của PC1: `proline` được 0,9998, còn cột nặng thứ hai (`magnesium`) chỉ được 0,018. PC1 ở đây gần như *chính là* cột `proline`, không hơn.

Lý do nằm ở đơn vị: `proline` có độ lệch chuẩn ≈ 314, `magnesium` ≈ 14,2, `alcalinity_of_ash` ≈ 3,3. Đổi `proline` từ mg/L sang g/L là PC1 đổi hoàn toàn. Một phép phân tích mà kết quả phụ thuộc vào việc bạn ghi đơn vị bằng gam hay miligam thì không dùng được.

> 🔧 **Thử ngay:** vì sao bài 06 và 07 (cây quyết định, random forest) gần như không quan tâm tới thang đo, còn KNN, k-means, SVM và PCA thì rất nhạy với nó?
> Cây quyết định cắt từng cột một bằng câu hỏi dạng "cột này có lớn hơn ngưỡng $t$ không" - đổi đơn vị thì ngưỡng đổi theo, thứ tự các dòng không đổi, cây không đổi. Bốn phương pháp còn lại đều *gộp đóng góp của nhiều cột* lại, nên đơn vị của chúng phải so sánh được - nhưng gộp theo những cách hơi khác nhau. **KNN và k-means** cộng bình phương chênh lệch để ra khoảng cách, nên cột có biên độ lớn áp đảo trực tiếp. **PCA** cộng phương sai theo từng hướng, nên cột phương sai thô lớn nhất chiếm lấy trục đầu. **SVM** thì tuỳ kernel: bản RBF đo bằng bình phương khoảng cách nên giống KNN; bản tuyến tính không đo khoảng cách giữa các điểm, nhưng vẫn rất cần scale vì lề và khoản phạt $C$ được tính trên độ lớn của trọng số, mà độ lớn ấy phụ thuộc đơn vị của cột.

## 5. PCA làm bước tiền xử lý

Đưa PCA vào `Pipeline` trước mô hình là cách dùng thường gặp. Thử trên `digits`: 1.797 ảnh chữ số viết tay 8×8, tức 64 cột.

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=0.25,
                                      stratify=d.target, random_state=42)
knn  = KNeighborsClassifier(n_neighbors=3).fit(Xtr, ytr)
pipe = make_pipeline(PCA(n_components=30, random_state=42),
                     KNeighborsClassifier(n_neighbors=3)).fit(Xtr, ytr)
print(knn.score(Xte, yte), pipe.score(Xte, yte))
```

| Đầu vào cho KNN | Số chiều | Phương sai giữ lại | Accuracy trên test |
|---|---|---|---|
| 64 pixel gốc | 64 | 100% | 0,9844 |
| PCA 30 thành phần | 30 | 95,9% | **0,9844** |
| PCA 20 thành phần | 20 | 89,5% | 0,9778 |
| PCA 10 thành phần | 10 | 73,8% | 0,9667 |

Bỏ đi hơn nửa số chiều mà accuracy không suy suyển. Điều đó nói rằng 64 pixel chứa rất nhiều thông tin trùng lặp - các pixel cạnh nhau trong một chữ số viết tay gần như luôn sáng tối cùng nhau, nên biết một cái là đoán được cái kia. PCA gộp những nhóm pixel "đi cùng nhau" ấy lại thành một chiều duy nhất.

Ép xuống 10 chiều thì bắt đầu mất thật: accuracy tụt về 0,9667. Còn muốn biết bao nhiêu thành phần là đủ, với `digits` con số cụ thể là: 13 thành phần cho 80% phương sai, 21 cho 90%, 29 cho 95% và 41 cho 99%.

Ngoài chuyện chạy nhanh và nhẹ hơn, PCA còn có tác dụng khử nhiễu: những hướng phương sai bé thường là dao động vụn vặt, bỏ chúng đi đôi khi làm mô hình khái quát hóa tốt hơn. Nhưng "đôi khi" là từ cần nhớ - mục 6 cho một ví dụ ngược lại.

> ⚠️ **Lưu ý nhỏ:** PCA phải nằm *trong* `Pipeline`, không được `fit` trên toàn bộ dữ liệu rồi mới chia train/test. Các trục PC được tính từ dữ liệu; tính chúng trên cả tập test là để mô hình biết trước cấu trúc của phần dữ liệu lẽ ra nó chưa từng thấy - đúng kiểu lộ đề mà Foundations · Bài 14 mô tả. Bài 11 sẽ đo xem lộ đề kiểu này thổi phồng kết quả đến mức nào.

## 6. Đọc thành phần, và ba chỗ PCA hụt hơi

Mỗi thành phần chính là một tổ hợp có trọng số của các cột gốc, và bộ trọng số ấy (`components_` - các **trục chính**; nhiều tài liệu thống kê gọi "loading" cho một đại lượng khác, là trục chính đã nhân với căn của trị riêng) đôi khi đọc được thành nghĩa. Với `wine`, các cột có hệ số dương lớn nhất trong PC1 là `flavanoids` (0,423), `total_phenols` (0,395), `od280/od315` (0,376); các cột âm nhất là `nonflavanoid_phenols` (−0,299) và `malic_acid` (−0,245). PC1 vì thế đọc được là "trục hàm lượng phenol tổng thể". PC2 nặng nhất ở `color_intensity` (0,530), `alcohol` (0,484) và `proline` (0,365) - một trục về độ đậm và nồng độ.

Nhưng đó là trường hợp may. Ba giới hạn thường gặp:

**Thành phần khó diễn giải.** PC1 của `digits` là một tổ hợp có trọng số của cả 64 pixel. Nói với người dùng "hồ sơ này bị từ chối vì PC3 của bạn cao" thì không ai hiểu, và bài 06 đã nhắc rằng ở một số lĩnh vực, giải thích được là yêu cầu bắt buộc chứ không phải điểm cộng.

**PCA chỉ xoay trục, tức chỉ tìm được cấu trúc tuyến tính.** Dữ liệu cuộn theo hình xoắn ốc thì không có phép xoay nào duỗi nó ra được. Cho những trường hợp đó có `KernelPCA`, hoặc các kỹ thuật trực quan hóa như t-SNE và UMAP.

**PCA không biết bạn định dự đoán gì.** Nó tối đa hóa phương sai của $X$, hoàn toàn không nhìn $y$. Hướng phân biệt hai lớp có thể là hướng phương sai nhỏ, và PCA sẽ vứt nó đi ngay:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

b = load_breast_cancer()
full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
two  = make_pipeline(StandardScaler(), PCA(n_components=2, random_state=42),
                     LogisticRegression(max_iter=1000))
print(cross_val_score(full, b.data, b.target, cv=5).mean())   # 0.9807
print(cross_val_score(two,  b.data, b.target, cv=5).mean())   # 0.9508
```

Trên `breast_cancer`, hai thành phần đầu giữ 63,2% phương sai (44,27% + 18,97%) và tách hai nhóm khá rõ: trung bình PC1 của nhóm ác tính là +3,71, của nhóm lành tính là −2,21. Đủ tốt để vẽ một bức tranh thuyết phục. Nhưng dùng đúng hai chiều ấy để dự đoán thì accuracy cross-validation là 0,9508, so với 0,9807 khi giữ cả 30 cột. Chênh lệch 3 điểm phần trăm trên bài toán chẩn đoán ung thư không phải chuyện nhỏ.

> 🔧 **Thử ngay:** với `wine`, thành phần nào giữa PC1 (36,2%) và PC13 (0,8%) đáng ngờ hơn nếu bạn thấy nó tách hai nhóm mẫu rất đẹp?
> Câu hỏi này là một cái bẫy: **không kết luận được chỉ từ con số 0,8%.** Chính mục này vừa nói PCA không nhìn $y$, nên một hướng phương sai bé hoàn toàn có thể chứa tín hiệu phân loại thật - đó đúng là kiểu tín hiệu mà PCA sẵn sàng vứt đi, như bộ `breast_cancer` ở dưới cho thấy. Nhưng nó cũng có thể là một trùng hợp trên đúng tập dữ liệu này, hoặc một dấu vết kỹ thuật (mẫu của hai nhóm được đo bằng hai máy khác nhau chẳng hạn). Việc phải làm không phải là đoán, mà là kiểm chứng trên dữ liệu giữ riêng: nếu PC13 vẫn tách được nhóm trên dữ liệu mới thì đó là tín hiệu thật.

## Tóm tắt bài học

- PCA tìm hệ trục mới: trục đầu đặt theo hướng dữ liệu trải rộng nhất, các trục sau vuông góc với những trục trước và hứng phần biến thiên còn lại.
- Giữ $M$ thành phần đầu cho xấp xỉ $M$ chiều tốt nhất của dữ liệu - tối đa hóa phương sai giữ lại và tối thiểu hóa sai số tái tạo là cùng một bài toán.
- Tỉ lệ phương sai giải thích cộng dồn là căn cứ chọn số thành phần: để vẽ hình thì lấy 2, để tiền xử lý thì đặt ngưỡng (`n_components=0.95`) hoặc để cross-validation quyết định.
- Khi các cột khác đơn vị hoặc khác thang đo thì phải chuẩn hóa trước, nếu không cột có phương sai thô lớn nhất sẽ quyết định thay bạn: chưa chuẩn hóa, PC1 của `wine` chiếm 99,81% phương sai và thực chất chỉ là cột `proline`. (`PCA` của scikit-learn tự trừ trung bình nhưng **không** tự chia độ lệch chuẩn.) Nếu mọi cột cùng đơn vị và phương sai thô vốn đã có ý nghĩa - ví dụ pixel ảnh - thì không nhất thiết phải scale.
- PCA giảm chiều mà giữ được kết quả: `digits` từ 64 xuống 30 chiều, accuracy KNN không đổi (0,9844).
- PCA không nhìn $y$ nên có thể vứt đi đúng hướng cần cho việc dự đoán - trên `breast_cancer`, ép xuống 2 chiều làm accuracy tụt từ 0,9807 xuống 0,9508.
- PCA chỉ xoay trục nên chỉ bắt được cấu trúc tuyến tính; các thành phần thường khó diễn giải thành ngôn ngữ nghiệp vụ.

## Câu hỏi tự kiểm tra

1. Vì sao PCA đi tìm hướng có phương sai lớn nhất chứ không phải hướng nào khác? Một hướng có phương sai bằng 0 nói lên điều gì?
2. Bạn chạy PCA trên dữ liệu gồm chiều cao (cm), cân nặng (kg) và thu nhập (VNĐ) mà quên chuẩn hóa. PC1 sẽ gần như trùng với cột nào, và vì sao?
3. `explained_variance_ratio_` của bạn là `[0.62, 0.21, 0.09, 0.05, 0.03]`. Giữ 2 thành phần thì mất bao nhiêu phần trăm phương sai? Cần mấy thành phần để đạt 95%?
4. Đồng nghiệp chạy `PCA()` trên toàn bộ dữ liệu, rồi mới `train_test_split` và huấn luyện. Sai ở đâu, và bạn sửa thế nào?
5. Một mô hình chạy trên 5 thành phần chính cho accuracy tương đương mô hình chạy trên cả 200 cột gốc. Bạn rút ra được điều gì về 200 cột ấy?
6. Nhóm pháp chế yêu cầu giải thích cho khách hàng vì sao hồ sơ bị từ chối. Mô hình của bạn chạy trên 10 thành phần chính. Vấn đề nằm ở đâu, và bạn có những lựa chọn nào?

## Đọc thêm

**Nguồn nền tảng của bài:**

| Nguồn | Đọc phần nào |
|---|---|
| **An Introduction to Statistical Learning (ISLP)** | Chương 12: 12.2.1 - định nghĩa thành phần chính, ví dụ `USArrests` với bảng loading (Bảng 12.1, Hình 12.1); 12.2.2 - cách nhìn khác: thành phần chính là xấp xỉ $M$ chiều tốt nhất theo sai số bình phương (Hình 12.2); 12.2.3 - vì sao phải scale, và tỉ lệ phương sai giải thích (PVE, công thức 12.10); 12.2.4 - dùng bao nhiêu thành phần; 12.5.1 - lab với `PCA` |
| **Mathematics for Machine Learning (MML)** | Chương 10: 10.1 - đặt vấn đề nén dữ liệu; 10.2 - góc nhìn "tối đa hóa phương sai"; 10.3 - góc nhìn "tối thiểu hóa sai số chiếu"; 10.4 - quan hệ với phân rã trị riêng; 10.6 - PCA khi số chiều lớn hơn số mẫu; 10.8 - Kernel PCA |
| **scikit-learn User Guide** | Mục 2.5.1 *Principal component analysis* - `PCA`, `n_components` nhận số nguyên hoặc tỉ lệ phương sai, `whiten`, `IncrementalPCA` cho dữ liệu không nạp hết vào bộ nhớ |

**Nguồn online bổ sung (miễn phí):**

- [2.5. Decomposing signals in components - scikit-learn User Guide](https://scikit-learn.org/stable/modules/decomposition.html) - PCA và các họ hàng (Kernel PCA, NMF, ICA).
- [PCA - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - tham số `n_components`, thuộc tính `explained_variance_ratio_`, `components_`.
- [Faces dataset decompositions](https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html) - "eigenfaces": các thành phần chính của một tập ảnh khuôn mặt, hiển thị lại dưới dạng ảnh.
- [Principal Component Regression vs Partial Least Squares Regression](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html) - minh họa bằng số cho chuyện PCA không nhìn $y$.
- [Novembre và cộng sự - Genes mirror geography within Europe (Nature, 2008)](https://www.nature.com/articles/nature07331) - nghiên cứu ở mục 1; [kho tài nguyên đi kèm của nhóm tác giả](https://github.com/NovembreLab/Novembre_etal_2008_misc) có hình gốc.
- [Pearson - On Lines and Planes of Closest Fit to Systems of Points in Space (1901)](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) - bài báo đặt nền cho PCA, tiếp cận từ phía "sai số chiếu nhỏ nhất".

> **Bài tiếp theo:** [Feature engineering & Pipeline: làm dữ liệu biết nói](../feature-engineering-and-pipelines-vi/) - hai bài vừa rồi biến đổi dữ liệu bằng thuật toán; bài sau bàn phần việc con người làm - tạo ra những cột mới mà không thuật toán nào tự nghĩ ra được, và đóng gói toàn bộ để không rò rỉ đáp án.

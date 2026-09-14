---
title: "SVM: Finding the Widest Margin"
description: "Among the countless lines that separate two classes, SVM picks the one with the widest margin: support vectors, the C knob for the soft margin, the kernel trick and the gamma of the RBF kernel, running SVC on breast_cancer, and when SVM is worth using."
domain: "Machine Learning"
section: "Learn"
language: "en"
translationKey: "svm-finding-the-widest-boundary"
order: 8
pubDate: 2026-08-16
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand why, among the countless lines that separate two classes, SVM chooses the one with the widest margin and why only a few "support vector" points decide it; know how the $C$ knob controls tolerance towards points that violate the margin; grasp the intuition of the kernel trick - lifting the data into a higher dimension so it can be separated by a straight line - and the `gamma` knob of the RBF kernel; be able to run `SVC` on `breast_cancer` and see why scaling is required; and know when SVM is worth using, when it is not, and how it differs from logistic regression.

## 1. Three lines, all correct - which one to pick?

Back to the loan-review desk with the two familiar features: years employed and income. This time the data is "clean": the on-time group (○) and the late group (●) sit completely apart. You need to draw a straight line splitting the plane in two to classify new customers. The trouble is that there are *countless* lines that are 100% correct on the training set:

```
 income
   ↑                a   b     c
 8 │  ○  ○         /   /     /
 7 │ ○   ○  ○     /   /     /
 6 │   ○   ○     /   /     /
 5 │  ○      ○  /   /     /
 4 │           /   /     /    ●
 3 │          /   /     /   ●   ●
 2 │         /   /     /  ●   ●
 1 │        /   /     /     ●
   └────────────────────────────────→ years employed
```

Line **a** hugs the ○ group: a new ○ customer that is slightly off gets misclassified. Line **c** hugs the ● group, with the opposite problem. Line **b** sits in the middle, farthest from *both* groups - if new data drifts a little, b still has room to spare. The distance from the line to the nearest point is called the **margin**; line b is the **maximal margin hyperplane** - the line with the widest margin. *An Introduction to Statistical Learning* describes it as the *midline of the widest slab* that can be inserted between the two classes.

```
           margin      margin
      ○ ○    ┊    b    ┊
     ○  ○  ○*┊    /    ┊
       ○   ○ ┊   /     ┊
             ┊  /      ┊●*
             ┊ /       ┊  ●  ●
             ┊/        ┊●*  ●
```

The three points marked `*` - lying exactly on the edge of the margin - are the **support vectors**: they "support" the line. The surprising part is that *only they* decide line b: move any other point (as long as it does not intrude into the margin) and b does not budge. In the chapter 9 lab of the book, 50 perfectly separable points need only **3** support vectors to build the line. The name "support vector machine" (Cortes & Vapnik, *Machine Learning*, 1995) comes from there. *Mathematics for Machine Learning* adds one more reason to prefer a wide margin: Vapnik and Chervonenkis proved that a large margin means the function class is "less complex", hence learnable - that is, it generalizes well.

"Hyperplane" is just the general name: with 2 features it is a line, with 3 features a plane, with 30 features (the `breast_cancer` set in section 4) a "plane" in 30-dimensional space - the same form $b_0 + b_1 x_1 + \dots + b_p x_p = 0$ as the score $z$ of logistic regression (Lesson 03). The difference lies in the *criterion for choosing* $b$: logistic maximizes the likelihood, SVM maximizes the margin.

## 2. Soft margin and the C knob: how much tolerance?

Add one customer to the table: high income, long employment, but late - a ● dot lands right in the middle of the ○ crowd. Now no straight line separates perfectly. And even when separation is still possible, *An Introduction to Statistical Learning* shows (Figure 9.5) that just *one* new point is enough to swing the widest-margin line around and shrink the margin to almost nothing: the model has overfit to a single observation.

The fix: the **soft margin** - allow some points to lie inside the margin, even on the wrong side, and charge a "fine" for each violation. The $C$ knob in scikit-learn is the **price of each violation**:

- **Large $C$** = heavy fine = little tolerance: tries not to let any point violate the margin → narrow margin, few support vectors, hugs the training set - high variance (Foundations · Lesson 14).
- **Small $C$** = tolerant: accepts many margin violations in exchange for a wide margin → many support vectors, a "smoother" boundary that may be too rigid - high bias.

Real numbers on `breast_cancer` (455 scaled training samples, the RBF kernel of section 3):

| `C` | 0.01 | 0.1 | 1 (default) | 10 | 100 |
|---|---|---|---|---|---|
| Number of support vectors | 341 | 199 | 97 | 84 | 64 |
| Train accuracy | 62.6% | 95.8% | 98.2% | 99.3% | 100% |
| Test accuracy | 63.2% | 94.7% | **98.2%** | 97.4% | 94.7% |

Both ends of the table are familiar: $C = 0.01$ is so tolerant that 341/455 points become support vectors and the model falls back to blindly guessing the majority class (63% benign tumours); $C = 100$ memorizes the training set and then loses on test. Support vectors now include points on the margin, inside the margin and on the wrong side - so the smaller $C$, the longer the list.

> 🔧 **Try it now:** verify the table above - change `C` in `make_pipeline(StandardScaler(), SVC(kernel="rbf", C=C, random_state=42))` (80/20 split, `random_state=42`, `stratify=y`) and print `svm[-1].n_support_`.
> Answer: with `C=1`, `n_support_` prints `[51 46]` - 51 support vectors belong to the malignant class, 46 to the benign class, 97 in total; with `C=100` it is `[30 34]`. To choose $C$ properly, put it into `GridSearchCV` on the pipeline as in Lesson 05 - don't look at the test set.

<details>
<summary><b>Going deeper: hinge loss - why points far from the boundary "have no say"</b></summary>

The soft-margin SVM can be written as "loss + penalty", like Ridge in Lesson 04: minimize $\sum_i \max\{0,\ 1 - y_i f(x_i)\} + \lambda \sum_j b_j^2$, with $y_i = \pm 1$ and $f(x) = b_0 + b_1 x_1 + \dots$. The first term is the **hinge loss**: it equals **0** when a point is on the correct side and outside the margin ($y_i f(x_i) \geq 1$), and grows linearly as the point intrudes into the margin or lands on the wrong side. Because the loss is exactly 0 for far-away points, they have no effect on the solution - that is why only the support vectors "have a say". The log loss of logistic regression never quite reaches 0: far-away points still contribute a little. *An Introduction to Statistical Learning* plots the two loss curves side by side (Figure 9.12) and remarks that they are quite similar - so the two models often give results close to each other. A small $\lambda$ here corresponds to a large $C$.

</details>

> ⚠️ **A small note:** the convention for $C$ is *reversed* between references. *An Introduction to Statistical Learning* defines $C$ as a **budget** for the total violation: large $C$ = more tolerant = wider margin. scikit-learn and *Mathematics for Machine Learning* define $C$ as a **penalty coefficient**: large $C$ = less tolerant. This lesson follows the scikit-learn convention. Whichever book you read, check that book's formula before turning the knob.

## 3. The kernel trick: if it can't be separated, lift it to a higher dimension

A sensor records the deviation of the room temperature from 24 °C, and users rate it "comfortable" (1) or "uncomfortable" (0). The data has only *one* feature $x$:

```
 x:    -3   -2.5  -2   -1.5  -1   -0.5   0   0.5   1   1.5   2   2.5   3
 label: 0    0    0    0     1    1     1    1    1    0    0    0    0
        ●────●────●────●────○────○────○────○────○────●────●────●────●
                        ↑ too cold             too hot ↑
```

In one dimension a "straight line" is just **a threshold**: 0 to the left, 1 to the right. No threshold can separate the pattern "1 in the middle, 0 on both sides" - `LinearSVC` on $x$ reaches only 61.5% on train, no better than blind guessing. But add one axis: $z = x^2$.

```
 z = x²
 9 │ ●                                   ●
 6 │    ●                             ●
 4 │        ●                     ●
 2 │            ●             ●            ← threshold z ≈ 1.6 separates perfectly
 1 │ - - - - - - - -○- - - - - -○- - - - -
 0 │                    ○   ○   ○
   └───────────────────────────────────────→ x
     -3   -2    -1    0     1     2    3
```

In the $(x, z)$ plane, the 1-points sit below and the 0-points sit above: a *horizontal* line separates perfectly - `LinearSVC` on $(x, x^2)$ reaches 100%, and the learned boundary is $-1.43\,z + 2.30 = 0$, i.e. $z \approx 1.6$ or $|x| \approx 1.27$. Projected back onto the $x$ axis, the horizontal line becomes **two cut points** $\pm 1.27$: a straight boundary in the higher space = a *curved* boundary in the original space. *An Introduction to Statistical Learning* extends exactly this idea (section 9.3.1): use $2p$ features $X_1, X_1^2, \dots, X_p, X_p^2$ - but warns that the number of features balloons very quickly once you add higher powers and cross terms.

This is where SVM has a way of its own. When solving the widest-margin problem, it turns out the model *only needs* the **dot products** between each pair of points $\langle x_i, x_j \rangle$ (Foundations · Lesson 07) - not the coordinates of each point. So replace the dot product with a function $K(x_i, x_j)$ that returns *the dot product in the lifted space* without ever having to build that space - the **kernel trick**. The polynomial kernel $(1 + \langle x_i, x_j \rangle)^d$ corresponds to adding every power and cross term up to degree $d$. The **RBF** kernel (radial basis function, the default of `SVC`):

$$K(x_i, x_j) = \exp\!\big(-\gamma\,\|x_i - x_j\|^2\big)$$

is a measure of *similarity* that decays with distance - nearby points have $K$ close to 1, distant points have $K$ close to 0 - so only the *neighbouring* training points influence the prediction, a bit like k-NN in Lesson 05. The book notes that the implicit space of RBF has *infinitely many dimensions*; we cannot compute there, but thanks to the kernel we never have to.

The `gamma` knob: the scikit-learn documentation explains that "gamma defines how far the influence of a single training example reaches; the larger gamma, the closer other examples must be to be affected". On `breast_cancer` (scaled, $C = 1$): `gamma=0.001` gives 95.6% on test; `0.01` and the default `"scale"` (≈ 0.033 here) give 98.2%; `0.1` drops to 95.6%; `gamma=1` memorizes train (100%) with **454/455** points becoming support vectors and then falls to 63.2% on test - Lesson 05's $k = 1$ in another guise. The book runs into the same scene on the Heart data: $\gamma = 0.1$ gives the best-looking ROC curve on train and the *worst* on test.

> 🔧 **Try it now:** rebuild the temperature example - `x = np.arange(-3, 3.5, 0.5)`, labels `(np.abs(x) <= 1.2).astype(int)` - then run `SVC(kernel="rbf", C=10)` directly on `x.reshape(-1, 1)` (one dimension, without adding $x^2$ yourself).
> Answer: train accuracy **100%** with `n_support_ = [2 2]` - the four points next to the two cut points are enough; the kernel has "lifted the dimension" for you. `LinearSVC(C=10)` on the same one-dimensional data: ≈ 0.615.

## 4. Scale first, SVM second - a real run on breast_cancer

The `breast_cancer` set (Lesson 05): 569 tumours, 30 measurements; the `mean area` column runs up to 2,501 while `mean smoothness` stays under 0.2. The margin, the distances in RBF and the dot products are all computed in each column's own units - without scaling, a few large columns drown out the voice of the small ones, exactly as with k-NN. The scikit-learn documentation says it outright: SVM algorithms are *not scale invariant*, so it is "highly recommended to scale your data".

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for kernel in ["linear", "rbf"]:
    svm = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, random_state=42))   # scaler is fit on train only
    svm.fit(X_train, y_train)
    print(kernel, round(svm.score(X_test, y_test), 3), "support vectors:", svm[-1].n_support_.sum())
    unscaled_model = SVC(kernel=kernel, random_state=42).fit(X_train, y_train)           # drop scaling for comparison
    print("   no scaling:", round(unscaled_model.score(X_test, y_test), 3))
```

| Model (114 test samples) | Scaled | Unscaled | Support vectors (scaled) |
|---|---|---|---|
| `SVC(kernel="linear")` | ≈ 97.4% | ≈ 95.6% | 32 / 455 |
| `SVC(kernel="rbf")` | ≈ 98.2% | ≈ 93.0% | 97 / 455 |
| Logistic regression (Lesson 03) | ≈ 98.2% | - | - |

Unscaled, RBF loses 5 points. Scaled, the linear kernel needs only 32 of the 455 training points to build the boundary - the remaining 423 could be deleted without changing the model. The ranking among the three models? 5-fold cross-validation on all 569 samples: linear 97.2% ± 0.9, RBF 97.4% ± 1.5, logistic 98.1% ± 0.7 - a tie within the noise; with 114 test samples, guessing a single case differently already shifts accuracy by 0.9 percentage points. This dataset is nearly linearly separable, so a curved kernel has no room to shine.

> ⚠️ **A small note:** `random_state` in `SVC` only has an effect when `probability=True` (section 5); the core algorithm is a convex optimization problem and gives the same solution on every run - we set it so the code stays consistent across the chapter. With the RBF kernel, search the *pair* `C` and `gamma` together with `GridSearchCV`, because the two knobs pull against each other: the chapter 9 lab of the book searches `C` in {0.1; 1; 10; 100; 1000} and `gamma` in {0.5; 1; 2; 3; 4}.

## 5. When to use SVM, when not to - and compared with logistic regression

Take `SVC(kernel="rbf")` over to the `adult` set of Lesson 03 - 39,073 training rows, 105 columns after one-hot encoding, the same pipeline with scaling. Logistic regression fits in 1.8 seconds. `SVC` takes **61.7 seconds** to fit and **10.7 seconds** to predict 9,769 rows, in exchange for 86.0% accuracy (versus 85.2%) - and keeps 13,195 support vectors, a third of the training set, in memory. The scikit-learn documentation explains: the core of SVM is a quadratic programming problem, fit time grows at least roughly quadratically in the number of samples, so it is "hard to scale beyond tens of thousands of samples"; prediction time is proportional to the number of support vectors. For large data with a linear boundary, `LinearSVC` (0.4 seconds, 85.3% on `adult`) or `SGDClassifier` are the alternatives; for a curved boundary, go back to Lesson 07.

The second weakness: SVM **does not produce probabilities naturally**. `decision_function` returns a signed score - positive/negative tells you which side, the magnitude tells you *how far from the boundary* the point is (to get the true geometric distance you must divide by $\|w\|$) - not a probability. The `probability=True` option bolts a logistic regression onto that score (*Platt scaling*), fit with an internal 5-fold cross-validation - it costs time, and the documentation warns that the resulting probabilities may be *inconsistent* with `predict`. If you need probabilities to pick a cost-based threshold (the death cap mushroom, Foundations · Lesson 15), logistic regression is more straightforward.

So where does SVM shine? On data with **many dimensions relative to the number of samples** - the scikit-learn documentation notes that SVM is "still effective in cases where the number of dimensions is greater than the number of samples":

- **Text classification.** Joachims (ECML 1998) represented each Reuters article as a vector with tens of thousands of dimensions (one per word, mostly zeros) and showed that a linear SVM beat the methods of its day (naive Bayes, Rocchio, C4.5, k-NN) with almost no parameter tuning - the reason he gave: text has *many* features that are all useful, and SVM can handle that.
- **Gene expression.** The chapter 9 lab of the book runs `SVC(kernel="linear")` on the Khan set: 2,308 genes, only 63 training samples - 0 training errors, 2 errors on 20 test samples. Guyon et al. (2002) used SVM to *select* cancer-related genes (SVM-RFE), a technique still used in bioinformatics.
- **Handwriting.** On LeCun's MNIST results table, a Gaussian-kernel SVM errs 1.4%; a "virtual SVM" with a degree-9 polynomial kernel and images shifted by 2 pixels (DeCoste & Schölkopf, 2002) errs 0.56% - lower than the LeNet-5 convolutional network on the same table (0.95%).

Compared with logistic regression - its closest sibling - *An Introduction to Statistical Learning* (section 9.5) sums it up: the two loss functions are **nearly identical in shape**, so the results are usually similar - the difference is that the hinge loss is exactly 0 for points outside the margin (so only the support vectors affect the line), while the logistic loss is never exactly 0, only very small; when **the two classes are well separated**, SVM tends to do better; when **the two classes overlap heavily**, logistic regression is usually preferred. The kernel trick is not exclusive to SVM - it can be attached to logistic regression too - but for historical reasons it is tied to SVM. With more than two classes, `SVC` trains one SVM for *each pair* of classes and then votes (one-versus-one).

| Consider SVM when | Avoid it when |
|---|---|
| Medium-sized data (a few thousand to a few tens of thousands of samples) | Hundreds of thousands of samples or more (fit time grows at least roughly as $n^2$) |
| Many features relative to the number of samples: text, genes | You need calibrated probabilities to pick a threshold |
| A curved boundary but not enough data for a forest of trees | You need to explain a coefficient per feature (logistic, Lesson 03) |
| Two classes fairly well separated | The data cannot be scaled yet, or many text columns are still unencoded |

> 🔧 **Try it now (by hand):** a model returns $f(x) = 2.5$; $0.4$; $-1.0$ for three customers, all three belonging to class $y = +1$. What is each one's hinge loss $\max\{0, 1 - y f(x)\}$?
> Answer: **0** (outside the margin, correct side - no say); **0.6** (correct side but intruding into the margin); **2.0** (wrong side). Only the latter two customers are support vectors.

## Lesson summary

- There are countless lines that separate two classes; SVM picks the one with the **widest margin**. Only the **support vectors** - points lying on or violating the margin - decide the line; on `breast_cancer`, the linear kernel needs 32/455 points.
- **Soft margin**: margin violations are allowed, each one penalized by $C$ (scikit-learn convention). Large $C$ = little tolerance, narrow margin, prone to overfitting (C = 100: 100% train, 94.7% test); small $C$ = tolerant, may be too rigid (C = 0.01: 63%). *An Introduction to Statistical Learning* uses $C$ in the opposite sense.
- **Kernel trick**: the model only needs dot products between pairs of points, so swapping in a kernel separates in a higher space without building it. One-dimensional example: adding $x^2$ turns two cut points into a horizontal line.
- **RBF** measures similarity that decays with distance; large `gamma` = narrow reach = prone to memorizing (gamma = 1: 454/455 support vectors, 63% test).
- **Scaling is almost always needed**: unscaled RBF loses 5 points on `breast_cancer`. Search `C` and `gamma` with `GridSearchCV` on the pipeline.
- Weaknesses: fit time grows at least roughly as $n^2$ in the number of samples (`adult`: 62 seconds versus 1.8 seconds for logistic); no natural probabilities. Strengths: many dimensions, few samples - text (Joachims 1998), genes, handwriting (0.56% error on MNIST).
- Versus logistic regression: the two losses are nearly identical in shape (hinge is exactly 0 outside the margin, logistic is not), so the results are usually similar; well-separated classes → SVM, overlapping classes or a need for probabilities/coefficients → logistic.

## Self-check questions

1. Two straight lines both classify 100% of the training set correctly. Why does SVM still prefer the one with the wider margin? Relate it to An & Binh (Foundations · Lesson 04).
2. Delete one training point that lies far from the boundary, on the correct side, outside the margin. Does the SVM line change? Does logistic regression change? Explain using hinge loss and log loss.
3. **By hand:** with the temperature data of section 3, add the feature $z = x^2$ and use the threshold $z \leq 1.6$ → label 1. Which class do the points $x = -1.5$ and $x = 1$ get? Check against the label table.
4. A colleague raises `gamma` from 0.01 to 1 because "train accuracy went up to 100%". Predict what happens on test and explain using the number of support vectors.
5. You have 2 million card transactions, 40 columns, and need fraud probabilities to set a cost-based threshold. Is an RBF-kernel SVM a good fit? Give two reasons and one alternative model from earlier lessons.
6. A dataset of 80 patients with 5,000 gene measurements. Why is a linear-kernel SVM a reasonable candidate, and what must you be careful about when evaluating it (Foundations · Lesson 14)?

## Further reading

**Foundational sources for this lesson:**

| Source | Which parts to read |
|---|---|
| **An Introduction to Statistical Learning (ISLP)** | Chapter 9: 9.1 - hyperplane, maximal margin classifier, support vectors (Figures 9.2-9.3); 9.2 - support vector classifier, the parameter $C$ (Figures 9.5-9.7); 9.3 - polynomial kernel, radial kernel, the Heart example (Figures 9.10-9.11); 9.4 - one-versus-one / one-versus-all; 9.5 - hinge loss and the relation to logistic regression (Figure 9.12); 9.6 - lab with `SVC`, `GridSearchCV`, the Khan set |
| **Mathematics for Machine Learning (MML)** | Chapter 12: 12.1-12.2 - hyperplane, the notion of margin, why the margin is set to 1, the soft margin SVM from the geometric view and from the loss view (hinge loss, 12.2.5) - read it to understand $C$ as a penalty coefficient |
| **scikit-learn User Guide** | Section 1.4 *Support Vector Machines* - pros/cons, `SVC`/`LinearSVC`, kernels, complexity, *Tips on Practical Use* (scaling, `C`, `gamma`, `probability`) |

**Additional online sources (free):**

- [1.4. Support Vector Machines - scikit-learn User Guide](https://scikit-learn.org/stable/modules/svm.html) - the official documentation; the *Tips on Practical Use* section is a checklist before running SVM.
- [SVC - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) - the parameters `C`, `kernel`, `gamma`, `probability`, the attributes `n_support_`, `support_vectors_`.
- [RBF SVM parameters - scikit-learn](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) - a grid of figures showing how the boundary changes as `C` and `gamma` are turned.
- [Cortes & Vapnik - Support-vector networks (Machine Learning, 1995)](https://link.springer.com/article/10.1007/BF00994018) - the original paper introducing the soft-margin SVM.
- [Joachims - Text Categorization with Support Vector Machines: Learning with Many Relevant Features (ECML 1998)](https://link.springer.com/chapter/10.1007/BFb0026683) - why SVM suits text.
- [DeCoste & Schölkopf - Training Invariant Support Vector Machines (Machine Learning, 2002)](https://people.eecs.berkeley.edu/~malik/cs294/decoste-scholkopf.pdf) - the virtual SVM reaching 0.56% error on MNIST; the aggregated results table is on [LeCun's MNIST page](https://yann.lecun.com/exdb/mnist/).
- [Guyon, Weston, Barnhill, Vapnik - Gene Selection for Cancer Classification using Support Vector Machines (Machine Learning, 2002)](https://link.springer.com/article/10.1023/A:1012487302797) - SVM-RFE in bioinformatics.

> **Next lesson:** [Clustering with k-means: Grouping Without Labels](../k-means-clustering-without-labels/) - all eight lessons so far had labels to learn from; the next one drops the labels - there are only points, and the question is how many groups they form on their own.

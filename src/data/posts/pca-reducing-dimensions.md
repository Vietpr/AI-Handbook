---
title: "PCA: Reducing Dimensions and Choosing How Many Components to Keep"
description: "Which direction PCA picks for its new axes and why, how to read the explained variance ratio chart to choose the number of components, how to use PCA both to plot high-dimensional data and as a preprocessing step, and its three limitations."
domain: "Machine Learning"
section: "Learn"
language: "en"
translationKey: "pca-reducing-dimensions"
order: 10
pubDate: 2026-08-17
featured: false
draft: false
---
> **Learning objectives:** After this lesson, you will understand which direction PCA picks for its new axes and why, be able to read the explained variance ratio chart to choose the number of components, use PCA both to draw high-dimensional data on a plane and as a preprocessing step, and know the three limitations that make PCA not always a good idea.

## 1. Two numbers redraw the map of Europe

In 2008, John Novembre's group published a striking result in the journal *Nature*. They took gene samples from about 3,000 Europeans, each read at more than half a million variable positions across the genome. Each person was therefore a point in a space of more than 500,000 dimensions - nobody can draw such a thing.

They compressed those 500,000 dimensions down to **two** numbers per person, then plotted the 3,000 points on a plane. The result: the cloud of points came out almost exactly in the shape of the map of Europe. The Iberian peninsula in one corner, the Italian peninsula in another, the British and Irish groups clearly separated. From genes alone, the researchers could guess the birthplace of more than 90% of the people to within 700 km, and of more than 50% to within 310 km. Nobody fed in any geographic coordinates - those two numbers rose out of the data on their own.

The technique they used is **PCA (Principal Component Analysis)**. It answers one question: if you are forced to describe the data with far fewer numbers, what is the smallest possible loss, and what are those numbers?

This need shows up everywhere. The `digits` dataset has 64 columns (8×8-pixel images of handwritten digits). The `breast_cancer` dataset has 30 columns. Gene expression data has tens of thousands of columns over a few dozen patients. With many columns, three things get hard at once: you cannot plot them, the model memorizes more easily (Foundations · Lesson 14), and distances gradually lose their meaning, as lesson 05 said about the curse of dimensionality.

## 2. The new axis is placed along the direction where the data spreads the most

Picture a two-dimensional cloud of points lying diagonally, stretched along the northeast-southwest direction:

```
  y
  |            . *
  |         .* * .          The PC1 axis runs along the length
  |      .* *.*  .          of the cloud - the direction in which
  |    * *.* .              the data spreads the most.
  |  .*.* .                 The PC2 axis is perpendicular to PC1 and
  | *. .                    follows the remaining width.
  +------------------- x
```

The original $x$ and $y$ axes are nothing special - they are just the two columns we happened to measure. If we could choose the axes again, we would place the first axis **along the length of the cloud**, that is, the direction in which the points differ from one another the most. That axis is called the **first principal component (PC1)**. The second axis is perpendicular to it and catches the remaining variation; it is called PC2. And so on until the number of dimensions runs out.

"Differ the most" here is measured by the **variance** of the data when projected onto that axis. Projecting a point onto an axis is exactly the dot product built in Foundations · Lesson 07, and each principal component is a unit vector in the original space (Foundations · Lesson 06). In short: PCA looks for a vector $\phi_1$ such that when the whole dataset is projected onto it, the resulting variance is as large as possible.

Why variance? Because the variance along a direction is precisely the amount of information that distinguishes the points along that direction. A column in which every row has the same value (variance 0) can be dropped at no cost - it does not help tell any row apart from any other. Directions with large variance are kept, directions that are nearly flat are dropped, and that is the whole dimensionality-reduction strategy of PCA.

An accompanying property: keeping the first $M$ components gives us the **best possible** $M$-dimensional approximation of the data, in the sense that the sum of squared deviations between the original points and the reconstructed points is smallest. Maximizing the retained variance and minimizing the reconstruction error turn out to be the same problem seen from two sides.

> ⚠️ **A small note:** PCA always subtracts the mean of each column before doing its work (`sklearn.decomposition.PCA` does this automatically). Without that step, the "direction of largest variance" would be pulled toward the direction joining the origin to the center of the cloud, and would no longer reflect the shape of the data.

## 3. How many components are enough?

Each component retains a share of the variance of the whole dataset, called the **explained variance ratio**. On the standardized `wine` dataset (178 bottles, 13 chemical measurements):

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

| Component | PC1 | PC2 | PC3 | PC4 | PC5 | PC6 | PC7 | PC8 |
|---|---|---|---|---|---|---|---|---|
| Explained | 36.2% | 19.2% | 11.1% | 7.1% | 6.6% | 4.9% | 4.2% | 2.7% |
| Cumulative | 36.2% | **55.4%** | 66.5% | 73.6% | 80.2% | 85.1% | 89.3% | **92.0%** |

Read this table according to your purpose:

- **For plotting:** take PC1 and PC2 and accept keeping only 55.4%. The picture is distorted, but of all projections down to 2 dimensions, this is the one with the smallest squared reconstruction error.
- **For preprocessing:** set a threshold for the variance you want to keep. `PCA(n_components=0.90)` keeps 90% - for `wine` that is 8 components instead of 13; for 95% you need 10.

The familiar approach is to plot the cumulative curve and look for where it flattens out - the same kind of "elbow" as when choosing $k$ in lesson 09, and with the same drawback: many datasets give a smooth curve with no clear elbow at all. In that case, the more reliable approach is to put PCA into a `Pipeline` and let cross-validation score each number of components - treating the number of components as an ordinary hyperparameter (Foundations · Lesson 13).

## 4. Without standardization, PCA only listens to the column with the largest units

Lesson 09 just ran into this with k-means. With PCA the consequences are even clearer, because PCA looks for the direction of largest **variance**, and variance depends directly on the unit of measurement.

Run `PCA()` again on `wine` **without** standardizing: PC1 explains **99.81%** of the variance. That sounds like wonderful news - one dimension instead of thirteen. But look at the coefficients of PC1: `proline` gets 0.9998, while the second-heaviest column (`magnesium`) gets only 0.018. PC1 here is almost *exactly* the `proline` column, nothing more.

The reason lies in the units: `proline` has a standard deviation ≈ 314, `magnesium` ≈ 14.2, `alcalinity_of_ash` ≈ 3.3. Change `proline` from mg/L to g/L and PC1 changes completely. An analysis whose result depends on whether you recorded the unit in grams or milligrams is unusable.

> 🔧 **Try it now:** why do lessons 06 and 07 (decision trees, random forests) barely care about the scale of the columns, while KNN, k-means, SVM and PCA are very sensitive to it?
> A decision tree cuts one column at a time with a question of the form "is this column greater than threshold $t$" - change the unit and the threshold changes with it, the ordering of the rows does not change, and the tree does not change. The other four methods all *combine contributions from many columns*, so their units have to be comparable - but they combine them in slightly different ways. **KNN and k-means** add squared differences to get a distance, so a column with a large range dominates directly. **PCA** adds variances along each direction, so the column with the largest raw variance takes over the first axis. **SVM** depends on the kernel: the RBF version measures squared distances, so it behaves like KNN; the linear version does not measure distances between points, but still needs scaling badly because the margin and the penalty $C$ are computed on the size of the weights, and that size depends on the units of the columns.

## 5. PCA as a preprocessing step

Putting PCA into a `Pipeline` before the model is a common use. Try it on `digits`: 1,797 images of handwritten digits at 8×8, that is, 64 columns.

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

| Input to KNN | Dimensions | Variance retained | Accuracy on test |
|---|---|---|---|
| 64 original pixels | 64 | 100% | 0.9844 |
| PCA, 30 components | 30 | 95.9% | **0.9844** |
| PCA, 20 components | 20 | 89.5% | 0.9778 |
| PCA, 10 components | 10 | 73.8% | 0.9667 |

More than half the dimensions dropped and accuracy does not budge. That tells us the 64 pixels contain a great deal of redundant information - neighbouring pixels in a handwritten digit are almost always light or dark together, so knowing one lets you guess the other. PCA merges those groups of pixels that "move together" into a single dimension.

Squeezing down to 10 dimensions is where real loss begins: accuracy drops to 0.9667. And if you want to know how many components are enough, for `digits` the specific numbers are: 13 components for 80% of the variance, 21 for 90%, 29 for 95% and 41 for 99%.

Beyond running faster and lighter, PCA also has a denoising effect: the directions with small variance are often petty fluctuations, and dropping them sometimes makes the model generalize better. But "sometimes" is the word to remember - section 6 gives a counter-example.

> ⚠️ **A small note:** PCA must sit *inside* the `Pipeline`; it must not be `fit` on the whole dataset before splitting train/test. The PC axes are computed from the data; computing them on the test set as well lets the model know in advance the structure of data it should never have seen - exactly the kind of leakage that Foundations · Lesson 14 describes. Lesson 11 will measure how much this kind of leakage inflates results.

## 6. Reading the components, and three places where PCA runs out of breath

Each principal component is a weighted combination of the original columns, and that set of weights (`components_` - the **principal axes**; many statistics texts use "loadings" for a different quantity, the principal axes multiplied by the square root of the eigenvalues) can sometimes be read as meaning. For `wine`, the columns with the largest positive coefficients in PC1 are `flavanoids` (0.423), `total_phenols` (0.395), `od280/od315` (0.376); the most negative are `nonflavanoid_phenols` (−0.299) and `malic_acid` (−0.245). PC1 can therefore be read as the "overall phenol content axis". PC2 is heaviest on `color_intensity` (0.530), `alcohol` (0.484) and `proline` (0.365) - an axis of intensity and strength.

But that is the lucky case. Three common limitations:

**Components are hard to interpret.** PC1 of `digits` is a weighted combination of all 64 pixels. Telling a user "this application was rejected because your PC3 is high" means nothing to anyone, and lesson 06 already reminded us that in some fields, explainability is a mandatory requirement, not a bonus.

**PCA only rotates the axes, so it can only find linear structure.** Data curled into a spiral cannot be straightened out by any rotation. For those cases there is `KernelPCA`, or visualization techniques such as t-SNE and UMAP.

**PCA does not know what you intend to predict.** It maximizes the variance of $X$ and never looks at $y$. The direction that separates two classes may be a direction of small variance, and PCA will throw it away immediately:

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

On `breast_cancer`, the first two components retain 63.2% of the variance (44.27% + 18.97%) and separate the two groups fairly clearly: the mean PC1 of the malignant group is +3.71, of the benign group −2.21. Good enough to draw a convincing picture. But using exactly those two dimensions for prediction gives a cross-validation accuracy of 0.9508, versus 0.9807 when all 30 columns are kept. A gap of 3 percentage points on a cancer diagnosis problem is no small matter.

> 🔧 **Try it now:** with `wine`, which component, PC1 (36.2%) or PC13 (0.8%), is more suspicious if you see it separating two groups of samples beautifully?
> This question is a trap: **you cannot conclude anything from the 0.8% figure alone.** This very section just said that PCA does not look at $y$, so a low-variance direction can perfectly well contain a real classification signal - that is exactly the kind of signal PCA is willing to throw away, as the `breast_cancer` dataset below shows. But it could also be a coincidence in this particular dataset, or a technical artefact (the samples of the two groups were measured on two different machines, for instance). What you need to do is not guess but verify on held-out data: if PC13 still separates the groups on new data, the signal is real.

## Lesson summary

- PCA finds a new set of axes: the first is placed along the direction where the data spreads the most, and each subsequent axis is perpendicular to the previous ones and catches the remaining variation.
- Keeping the first $M$ components gives the best $M$-dimensional approximation of the data - maximizing retained variance and minimizing reconstruction error are the same problem.
- The cumulative explained variance ratio is the basis for choosing the number of components: take 2 for plotting, and for preprocessing set a threshold (`n_components=0.95`) or let cross-validation decide.
- When the columns have different units or scales, standardize first, otherwise the column with the largest raw variance decides for you: unstandardized, PC1 of `wine` takes 99.81% of the variance and is in reality just the `proline` column. (scikit-learn's `PCA` subtracts the mean automatically but does **not** divide by the standard deviation.) If all columns share the same unit and the raw variance is already meaningful - image pixels, for example - scaling is not strictly necessary.
- PCA reduces dimensions while keeping results: `digits` from 64 down to 30 dimensions, KNN accuracy unchanged (0.9844).
- PCA does not look at $y$, so it can throw away exactly the direction needed for prediction - on `breast_cancer`, squeezing to 2 dimensions drops accuracy from 0.9807 to 0.9508.
- PCA only rotates the axes, so it only captures linear structure; the components are usually hard to interpret in business language.

## Self-check questions

1. Why does PCA look for the direction of largest variance rather than any other direction? What does a direction with zero variance tell you?
2. You run PCA on data consisting of height (cm), weight (kg) and income (VND) and forget to standardize. Which column will PC1 almost coincide with, and why?
3. Your `explained_variance_ratio_` is `[0.62, 0.21, 0.09, 0.05, 0.03]`. If you keep 2 components, what percentage of the variance is lost? How many components are needed to reach 95%?
4. A colleague runs `PCA()` on the whole dataset, and only then does `train_test_split` and trains. What is wrong, and how would you fix it?
5. A model running on 5 principal components gives accuracy equivalent to a model running on all 200 original columns. What can you conclude about those 200 columns?
6. The legal team requires an explanation to the customer of why their application was rejected. Your model runs on 10 principal components. Where is the problem, and what options do you have?

## Further reading

**Foundational sources for this lesson:**

| Source | Which part to read |
|---|---|
| **An Introduction to Statistical Learning (ISLP)** | Chapter 12: 12.2.1 - definition of principal components, the `USArrests` example with the loading table (Table 12.1, Figure 12.1); 12.2.2 - another view: principal components as the best $M$-dimensional approximation under squared error (Figure 12.2); 12.2.3 - why scaling is needed, and the proportion of variance explained (PVE, formula 12.10); 12.2.4 - how many components to use; 12.5.1 - lab with `PCA` |
| **Mathematics for Machine Learning (MML)** | Chapter 10: 10.1 - framing the data compression problem; 10.2 - the "maximize variance" view; 10.3 - the "minimize projection error" view; 10.4 - relation to eigendecomposition; 10.6 - PCA when the number of dimensions exceeds the number of samples; 10.8 - Kernel PCA |
| **scikit-learn User Guide** | Section 2.5.1 *Principal component analysis* - `PCA`, `n_components` accepts an integer or a variance ratio, `whiten`, `IncrementalPCA` for data that does not fit in memory |

**Additional online sources (free):**

- [2.5. Decomposing signals in components - scikit-learn User Guide](https://scikit-learn.org/stable/modules/decomposition.html) - PCA and its relatives (Kernel PCA, NMF, ICA).
- [PCA - scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) - the `n_components` parameter, the `explained_variance_ratio_` and `components_` attributes.
- [Faces dataset decompositions](https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html) - "eigenfaces": the principal components of a set of face images, displayed again as images.
- [Principal Component Regression vs Partial Least Squares Regression](https://scikit-learn.org/stable/auto_examples/cross_decomposition/plot_pcr_vs_pls.html) - a numerical illustration of PCA not looking at $y$.
- [Novembre et al. - Genes mirror geography within Europe (Nature, 2008)](https://www.nature.com/articles/nature07331) - the study in section 1; [the authors' accompanying resource repository](https://github.com/NovembreLab/Novembre_etal_2008_misc) has the original figure.
- [Pearson - On Lines and Planes of Closest Fit to Systems of Points in Space (1901)](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720) - the paper that laid the foundation for PCA, approached from the "smallest projection error" side.

> **Next lesson:** [Feature Engineering & Pipelines: Making Data Speak](../feature-engineering-and-pipelines/) - the last two lessons transformed data with algorithms; the next one covers the part humans do - creating new columns that no algorithm could think up on its own, and packaging the whole thing so that the answer does not leak.

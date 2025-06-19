# K-Mean Clustering - Customer Segmentation

## 📌 Table of Contents

- [Dataset Description](#dataset-description)
- [Build a K-means Model](#build-a-k-means-model)
- [inertia: Evaluating Clustering Quality](#inertia-evaluating-clustering-quality)
- [Visualizing the Clustering Results](#visualizing-the-clustering-results)
- [How to Choose the Best K](#how-to-choose-the-best-k)
- [Application of Dimensionality-Reduction Techniques in Machine Learning](#application-of-dimensionality-reduction-techniques-in-machine-learning)
- [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
- [t-Distributed Stochastic Neighbor Embedding (t-SNE)](#t-distributed-stochastic-neighbor-embedding-t-sne)

-----------------------------------

In this case study, we’ll load and analyze a publicly available customer purchase dataset from Kaggle, then apply the K-means algorithm to segment customers. The goal is to better understand groups’ spending preferences and characteristics. 

Download dataset:
[Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data)

### Dataset Description

This dataset includes the following columns:

-   **Customer ID**

-   **Gender**

-   **Age**

-   **Annual Income** (in thousands of dollars)

-   **Spending Score** (1 - 100)

The feature “Annual Income (k$)” represents each customer’s yearly income in thousands of dollars, and “Spending Score (1–100)” indicates a score assigned based on the customer’s spending behavior and habits. By analyzing these two attributes, we can understand differences in customers’ spending patterns and income levels, perform clustering analysis, and then design targeted marketing strategies and services. 

### Build a K-means Model

The K-means implementation in scikit-learn offers a convenient interface, allowing users to perform clustering via a simple API. You can adjust parameters (like the number of clusters) to suit your needs and then run K-means to segment the data.

**Parameters**

Below are the key parameters you can tune when using the K-means algorithm:
-   **`n_clusters`**: The number of clusters **K** to form.

-   **`init`**: Strategy for initializing the cluster centers.
    -   `"k-means++"`: Selects initial centers in a way that speeds up convergence and reduces the chance of empty clusters.
    -   `"random"`: Chooses **K** observations at random from the data as initial centroids.

-   **`n_init`**: Number of times the algorithm will be run with different centroid seeds. The best output (lowest inertia) is kept.

-   **`max_iter`**: Maximum number of iterations for a single run (default: 300).

-   **`random_state`**: Seed for the random number generator---set this for reproducible results.


**Attributes**

-   **`inertia_`**: *float*---sum of squared distances of samples to their closest cluster center (a measure of cluster compactness).

-   **`cluster_centers_`**:
    Array of shape `(K, n_features)` giving the coordinates of the cluster centers after fitting.

**Methods**

-   **`fit(X)`**\
    Compute K-means clustering on data **X**.

-   **`predict(X)`**\
    Given fitted centers, assign each sample in **X** to the nearest cluster and return its label.

-   **`fit_predict(X)`**\
    Convenience method: run `fit(X)` followed by `predict(X)` in one call, returning cluster labels directly.

-   **`transform(X)`**\
    Compute the distance from each sample in **X** to each cluster center (L₂ distance), returning an array of shape `(n_samples, K)`.
    

### inertia: Evaluating Clustering Quality

After you choose *K* and fit the model, the algorithm will quickly locate the *K* centroids and assign each sample to its nearest cluster. Once fitting is complete, you can compute the sum of squared distances from each sample to its assigned cluster center---this quantity is called **inertia**. A larger inertia value indicates poorer clustering (i.e. the clusters are less compact). By inspecting inertia, we can judge how well K-means has performed.

```markdown
kmeansModel.inertia_ (sum of squared distance within cluster)
44448.45544793369

kmeansModel.cluster_centers_ (coordinates of each cluster’s centroid)
[[55.2962963  49.51851852]
 [86.53846154 82.12820513]
 [25.72727273 79.36363636]
 [88.2        17.11428571]
 [26.30434783 20.91304348]]
```

### Visualizing the Clustering Results

Each cluster is drawn in a different color and marker shape, and the centroids are highlighted with black stars. By plotting "Annual Income -- Spending Score," we can see how each group is distributed and clearly compare the clusters.

![Scatter Plot](/assets/scatterplot.png)

From the resulting, we can describe each cluster as follows:

-   **Cluster 1:** Income and spending both around the average.
-   **Cluster 2:** High income and high spending.
-   **Cluster 3:** Low income but relatively high spending.
-   **Cluster 4:** High income but relatively low spending.
-   **Cluster 5:** Low income and low spending.

### How to Choose the Best K

To select the optimal number of clusters, we use the sum of squared distances between samples and their nearest centroid---**inertia**---as our evaluation metric. We run K-means for different values of *K* (e.g. 1 through 10), record the inertia each time, and then plot inertia versus *K*. The point where the inertia curve begins to flatten (the "elbow") indicates the most appropriate *K*. By choosing the *K* with the smallest inertia before diminishing returns, we ensure a well-balanced clustering solution.

![Elbow Method](/assets/elbow_methods.png)


### Application of Dimensionality-Reduction Techniques in Machine Learning

Dimensionality reduction refers to a set of methods that transform high-dimensional data into a lower-dimensional form while preserving as much of the original information (variance) as possible. In machine learning, dimensionality reduction is widely used to:

-   **Mitigate the curse of dimensionality**, improving model performance and generalization.

-   **Speed up computation** by reducing feature count, which lowers training and inference time.

-   **Enhance interpretability** and reduce noise by discarding redundant or irrelevant features.

-   **Facilitate visualization** of complex data in 2D or 3D plots.

Common techniques include Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), and t-Distributed Stochastic Neighbor Embedding (t-SNE). In the following sections, following section will introduce these methods in detail and illustrate how to apply them to real-world datasets.

**Concept of Dimensionality Reduction**

The core idea of dimensionality reduction can be compared to file compression. Imagine you have a very large file---perhaps containing text, images, or other media---that is cumbersome to store and transmit. You use a compression algorithm to pack it into a much smaller archive. Although the compressed file is greatly reduced in size, it still retains the essential information of the original. When needed, you can decompress it and restore the original file contents.

This is precisely the spirit of dimensionality reduction: **compressing and simplifying data without losing its overall structure or key information**, so that it can be processed more efficiently.


### Principal Component Analysis (PCA)

**Principal Component Analysis** (PCA) is one of the most common **linear** dimensionality-reduction methods. It finds new orthogonal axes (principal components) that successively maximize the variance of the data, and then projects the data onto the subspace spanned by the top components. PCA is especially effective when the number of features is large relative to the number of samples, helping to reveal underlying structure while discarding redundancy.

The standard PCA workflow comprises five steps:

1.  **Standardize the data**\
    Scale each feature so they have the same units (e.g., zero mean and unit variance).

2.  **Compute the covariance matrix**\
    Measure pairwise covariances between all features.

3.  **Eigen decomposition**\
    Solve for the eigenvalues and eigenvectors of the covariance matrix.

4.  **Select principal components**\
    Rank the eigenvectors by their eigenvalues (from largest to smallest) and choose the top *k* that capture the most variance.

5.  **Transform the data**\
    Project the original data onto the subspace defined by the selected eigenvectors, yielding a *k*-dimensional representation.

**Parameters**

- **`n_components`** int\
  Number of principal components to keep (i.e. target dimension).

- **`whiten`** bool (default=False)\
  Whether to whiten (scale) components to unit variance (zero mean, variance 1).

- **`random_state`** int or None\
  Random‐state seed for reproducible results.

**Attributes**

- **`explained_variance_`** array, shape=(n_components,)\
  Variance of each principal component (the larger, the more information retained).

- **`explained_variance_ratio_`** array, shape=(n_components,)\ 
  Proportion of the dataset's total variance explained by each component.

- **`n_components_`** int\ 
  Number of components kept.

**Methods**

- **`fit(X[, y])`**: Fit the PCA model with the data X.

- **`fit_transform(X[, y])`**: Fit the model and return the data transformed into the principal‐component space.

- **`transform(X)`**: Apply the existing PCA transformation to new data X.


- **Variance ratio** measures each component's share of the total variance.

- **Variance** is the absolute variance along each principal axis.

![PCA Train](/assets/pca_train.png)

![PCA Test](/assets/pca_test.png)

PCA is straightforward and often very effective at compressing data and removing noise. However, because it relies on global variance (the covariance matrix), it can be sensitive to outliers and **cannot capture nonlinear structure**. In practice, applying PCA to data with strong nonlinear relationships may lead to cluster overlap or loss of important structure. Therefore, PCA is best suited for datasets where the dominant variations are roughly linear; for more complex, nonlinear manifolds, other methods such as t-SNE or kernel PCA may be more appropriate.


### t-Distributed Stochastic Neighbor Embedding (t-SNE)

We can use the t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation in sklearn.manifold to project our high-dimensional digit data into two dimensions. 

**Parameters**

-   **`n_components`**: The number of dimensions of the embedded space after t-SNE.

-   **`perplexity`**: The number of nearest neighbors considered during optimization. Default is 30; recommended range is 5--50.

-   **`learning_rate`**: The learning rate---usually set between 10 and 1000. Default is 200.

-   **`n_iter`**: The maximum number of iterations. Default is 1000.

-   **`random_state`**: The random seed, to ensure that each run of t-SNE produces the same result.

**Attributes**

-   **`embedding_`**: The array of shape (n_samples, n_components) with the embedded coordinates.

-   **`kl_divergence_`**: The final Kullback--Leibler divergence after optimization.

-   **`n_iter_`**: The actual number of iterations run.

**Methods**

-   **`fit_transform(X, y)`**: Fit the model to X (and optional labels y) and return the transformed (low-dimensional) data.

![t-SNE](/assets/t-SNE.png)

After applying t-SNE to reduce the dimensionality of the training set, the resulting two-dimensional plot shows clear clusters corresponding to each handwritten digit. Compared with PCA, t-SNE more effectively captures the data’s nonlinear structure, so points from different digit classes are better separated. This crisp clustering makes t-SNE especially powerful for visualization and unsupervised grouping. In practice, you can even feed the 2D t-SNE embedding directly into a K-Means algorithm to discover ten cluster centers, or use the embedding as input to a supervised classifier for training and prediction.

One important caveat is that **t-SNE cannot be applied to new (unseen) data**. PCA, by contrast, simply exposes a transform() method so you can project fresh samples into the existing PCA space. Because scikit-learn’s t-SNE implementation is based on pairwise neighbor relationships, it does not provide a transform() routine. As a result, t-SNE is really only suitable for one-off dimensionality reduction and visualization—you cannot directly embed new points into the learned t-SNE map. Each run of t-SNE must recompute all similarities and the low-dimensional layout from scratch, so there’s no guarantee that new data will be embedded consistently with the original training set.

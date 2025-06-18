# K-Mean Clustering - Customer Segmentation

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
-   **n_clusters**: The number of clusters **K** to form.
-   **init**: Strategy for initializing the cluster centers.
    -   `"k-means++"`: Selects initial centers in a way that speeds up convergence and reduces the chance of empty clusters.
    -   `"random"`: Chooses **K** observations at random from the data as initial centroids.
-   **n_init**: Number of times the algorithm will be run with different centroid seeds. The best output (lowest inertia) is kept.
-   **max_iter**: Maximum number of iterations for a single run (default: 300).
-   **random_state**: Seed for the random number generator---set this for reproducible results.


**Attributes**

-   **`inertia_`**: *float*---sum of squared distances of samples to their closest cluster center (a measure of cluster compactness).

-   **`cluster_centers_`**
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

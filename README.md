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

-   **Spending Score** (1 - 100)

The feature “Annual Income (k$)” represents each customer’s yearly income in thousands of dollars, and “Spending Score (1–100)” indicates a score assigned based on the customer’s spending behavior and habits. By analyzing these two attributes, we can understand differences in customers’ spending patterns and income levels, perform clustering analysis, and then design targeted marketing strategies and services. 

### Build a K-means Model

The K-means implementation in scikit-learn offers a convenient interface, allowing users to perform clustering via a simple API. You can adjust parameters (like the number of clusters) to suit your needs and then run K-means to segment the data.

#### Parameters

```python
kmeansModel = KMeans(
    n_clusters=5,       # form 5 clusters
    init='k-means++',   # smart centroid seeding
    n_init=10,          # run 10 times, keep the best
    max_iter=300,       # up to 300 iterations per run
    tol=1e-4,           # convergence threshold
    random_state=42,    # fixed seed for reproducibility
    algorithm='auto'    # let sklearn pick the best method
)
```

#### Attributes

**inertia_**

-   Interpretation
    -   It's the total within-cluster sum of squared distances.
    
    -   Lower inertia means tighter (more compact) clusters.

-   Usage
    -   Used as the optimization objective that K-means minimizes.
    
    -   Helpful in the "elbow method" to choose a good number of clusters: plot inertia vs. kkk and look for the point where the decrease levels off.

**cluster_centers_**

The cluster_centers_ attribute of a fitted KMeans object holds the coordinates of each cluster’s centroid

-   Interpretation: Gives you the prototypical point for each cluster.

-   Prediction: You can compute distances from new samples to these centers to assign clusters.

-   Visualization: Plot them (e.g. on a scatter of your data) to see where centroids lie.

#### Methods

-   `fit(X)`\
    Learns cluster centroids from data `X` (shape `(n_samples, n_features)`) by running k-means (init → iterate → converge).\

-   `predict(X)`\
    Assigns each sample in `X` to the nearest learned centroid.\

-   `fit_predict(X)`\
    Convenience: runs `fit(X)` then `predict(X)` on the same data.\

-   `transform(X)`\
    Computes the distance of each sample in `X` to each cluster center.\

-   `fit_transform(X)`\
    Convenience: runs `fit(X)` then `transform(X)`.\

### inertia: Evaluating Clustering Quality

After you choose *K* and fit the model, the algorithm will quickly locate the *K* centroids and assign each sample to its nearest cluster. Once fitting is complete, you can compute the sum of squared distances from each sample to its assigned cluster center---this quantity is called **inertia**. A larger inertia value indicates poorer clustering (i.e. the clusters are less compact). By inspecting inertia, we can judge how well K-means has performed.


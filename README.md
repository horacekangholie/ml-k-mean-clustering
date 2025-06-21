# 📘 K-Means Clustering

## 📌 Introduction

**K-Means Clustering** is an **unsupervised machine learning algorithm** used to partition a dataset into **K distinct, non-overlapping clusters**. It aims to group data points so that points within the same cluster are more similar to each other than to those in other clusters, based on distance metrics (typically Euclidean distance).

## ⚙️ What the Model Does

K-Means works by:

1. Choosing K initial cluster centroids (randomly or using a method like *k-means++*).
2. Assigning each data point to the nearest centroid (forming K clusters).
3. Updating each centroid to be the **mean** of the points in its cluster.
4. Repeating steps 2 and 3 until convergence (e.g., centroids don't move or maximum iterations reached).

## 🚧 Limitations

| Limitation                             | Description |
|----------------------------------------|-------------|
| Requires Predefined K                  | You must specify the number of clusters beforehand. |
| Sensitive to Initialization            | Bad initialization can lead to poor convergence. |
| Not Suitable for Non-Spherical Clusters| Struggles with complex cluster shapes and different densities. |
| Sensitive to Outliers                  | Outliers can skew centroids significantly. |

---

## 🔧 Common Parameters, Attributes, and Methods

### Parameters (in `sklearn.cluster.KMeans`)

| Parameter         | Description |
|------------------|-------------|
| `n_clusters`      | Number of clusters to form (K). |
| `init`            | Initialization method (`'k-means++'`, `'random'`). |
| `n_init`          | Number of times the algorithm will be run with different centroid seeds. |
| `max_iter`        | Maximum number of iterations per run. |
| `random_state`    | Random seed for reproducibility. |
| `tol`             | Tolerance to declare convergence. |

### Attributes (after fitting)

| Attribute          | Description |
|-------------------|-------------|
| `cluster_centers_` | Coordinates of the cluster centers. |
| `labels_`          | Cluster labels for each data point. |
| `inertia_`         | Sum of squared distances of samples to their closest cluster center. |

### Methods

| Method             | Description |
|-------------------|-------------|
| `fit(X)`           | Computes K-means clustering on data `X`. |
| `predict(X)`       | Predicts closest cluster each sample in `X` belongs to. |
| `fit_predict(X)`   | Computes cluster centers and predicts cluster index for each sample. |
| `transform(X)`     | Transforms `X` to a cluster-distance space. |
| `fit_transform(X)` | Fits to data, then transforms it. |

---

## 📏 Evaluation Metrics

### 1. **Inertia**
- **Definition**: Total within-cluster sum of squares (WSS).
- **Interpretation**: Lower values indicate tighter clusters. Used for **Elbow Method** to find optimal K.

### 2. **Silhouette Score**
- **Definition**: Measure of how similar an object is to its own cluster compared to others.
- **Range**: `-1` (poor clustering) to `1` (well-clustered).
- **Interpretation**: Higher is better. Scores near 0 indicate overlapping clusters.

### 3. **Davies-Bouldin Index**
- **Definition**: Ratio of intra-cluster distance to inter-cluster separation.
- **Interpretation**: Lower values suggest better clustering.

---

## 💡 Example Use Case

**Customer Segmentation**

A retail company wants to group customers based on purchasing behavior. By applying K-Means on features like total spending, frequency of purchase, and product categories, they can identify distinct customer groups (e.g., budget shoppers, loyal spenders), and tailor marketing strategies accordingly.

## 📈 Sample Dataset

Description

The Mall Customer Segmentation dataset from Kaggle is a small, fictional dataset (200 rows) designed to teach unsupervised learning techniques in Python—specifically customer segmentation using clustering.\
This dataset helps demonstrate how clustering can reveal natural groupings within customer data. It allows you to profile distinct segments (by age, income, spending) and derive strategic insights—like which segments to attract, retain, or upsell—making it a classic tutorial example for customer segmentation in retail and marketing analytics.

Dataset

[Mall Customer Segmentation Data](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python/data)

This dataset includes the following columns:

-   Customer ID: unique identifier.

-   Gender: Male or Female.

-   Age: customer age in years.

-   Annual Income: yearly income in thousands of dollars.

-   Spending Score (1 - 100): a synthetic metric representing how much and how “good” a customer is, based on their shopping behavior.

The feature “Annual Income (k$)” represents each customer’s yearly income in thousands of dollars, and “Spending Score (1–100)” indicates a score assigned based on the customer’s spending behavior and habits. By analyzing these two attributes, we can understand differences in customers’ spending patterns and income levels, perform clustering analysis, and then design targeted marketing strategies and services. 

Visualizing the Clustering Results

Each cluster is drawn in a different color and marker shape, and the centroids are highlighted with black stars. By plotting "Annual Income -- Spending Score," we can see how each group is distributed and clearly compare the clusters.

![Scatter Plot](/assets/scatterplot.png)

From the resulting, we can describe each cluster as follows:

-   **Cluster 1:** Income and spending both around the average.
-   **Cluster 2:** High income and high spending.
-   **Cluster 3:** Low income but relatively high spending.
-   **Cluster 4:** High income but relatively low spending.
-   **Cluster 5:** Low income and low spending.

# K-Mean Clustering - Customer Segmentation

- [Introduction to K-means](#introduction-to-k-means)

-----------------------

## What Is Unsupervised Learning?

In machine learning, **unsupervised learning** is a method that does not require "correct answers" (ground-truth labels). During training, the model automatically learns the relationships among features in the data without any pre-assigned target values. One of the most common unsupervised learning tasks is **clustering**, in which the machine attempts to discover the inherent structure and groupings within the data on its own.

The primary advantage of unsupervised learning is that it does not rely on large volumes of pre-labeled training data; instead, it uncovers valuable information by autonomously learning the features and structures present in the dataset. This makes unsupervised learning highly adaptable and broadly applicable when working with datasets whose underlying structure is unknown.

### Characteristics of Unsupervised Learning

-   **No "Right Answer"**
    Unlike supervised learning, unsupervised learning does not require pre-labeled target values during training.

-   **Self-Directed Discovery**\
    The model learns features and patterns in the data on its own and uncovers structure without guidance from explicit labels.

### Common Unsupervised Clustering Algorithms

-   **K-means Clustering**\
    Partitions the data into *K* clusters by assigning each point to the nearest cluster centroid.

-   **Hierarchical Clustering**\
    Builds a tree-like (dendrogram) hierarchy of clusters, grouping points step by step.

-   **DBSCAN** (Density-Based Spatial Clustering of Applications with Noise)\
    Forms clusters based on the density of points in their neighborhood, allowing detection of clusters of varying shape and size as well as noise.

## Introduction to K-means

### 1. How Does K-means Cluster?

By applying a clustering algorithm, we can effectively segment multi-dimensional data into groups. K-means is one of the simplest and most widely used methods: its objective is to partition data in an *n*-dimensional space into a user-specified number *K* of clusters so that:

-   Points within each cluster are as similar as possible to one another, and

-   Points in different clusters are as dissimilar as possible.

**Optimization Objective of K-means**

The primary goal of the K-means algorithm is to partition all data points into *K* clusters---each represented by a centroid---so that the average squared distance from each point to its assigned centroid is minimized. Formally, if we denote the set of points in cluster *Cᵢ* by Cᵢ and its centroid by μᵢ, the K-means objective function is:

$$
min⁡C,μJ=∑i=1K∑x∈Ci∥x-μi∥2.\min_{C,\;\mu}\;J \;=\;\sum_{i=1}^{K}\;\sum_{x\in C_i}\|\,x - \mu_i\|^2.C,μmin​J=i=1∑K​x∈Ci​∑​∥x-μi​∥2.
$$

When running K-means, a few practical considerations can greatly affect the stability and quality of your clusters:

🔺 **Initialization -- Random Initialization**\
K-means is sensitive to how the centroids are initialized. Different random starts can lead to very different final clusters. Since K-means is unsupervised, you don't know the "best" centroids beforehand---initial centers are typically chosen at random or by a simple heuristic.

🔺 **Choosing the Number of Clusters**\
Because K-means does not require labels, you must pick *K* yourself based on the data. A common strategy is to compute the **silhouette coefficient**, which balances within-cluster cohesion against between-cluster separation, and choose the *K* that maximizes this score.
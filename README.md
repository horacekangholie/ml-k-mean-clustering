- [Introduction to K-means](#introduction-to-k-means)

-----------------------

### What Is Unsupervised Learning?

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


-----------------------
## Introduction to K-means

### 1. How Does K-means Cluster?

By applying a clustering algorithm, we can effectively segment multi-dimensional data into groups. K-means is one of the simplest and most widely used methods: its objective is to partition data in an *n*-dimensional space into a user-specified number *K* of clusters so that:

-   Points within each cluster are as similar as possible to one another, and

-   Points in different clusters are as dissimilar as possible.
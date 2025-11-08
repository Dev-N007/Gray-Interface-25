# PCA + K-Means Clustering on the Digits Dataset

This project applies Principal Component Analysis (PCA) and K-Means clustering to the handwritten digits dataset available in scikit-learn. It aims to reduce the dimensionality of the image data while preserving most of the variance, and then group the images into clusters that represent different digits.

## Overview

The digits dataset consists of 1,797 grayscale handwritten digit images (0–9), each represented as an 8×8 image. Each image is flattened into a 64-feature vector of pixel intensities. Working with all 64 features can be inefficient and noisy, so PCA is used to reduce dimensionality.

After PCA reduces the dataset to a lower-dimensional subspace that keeps at least 95% of the variance, K-Means is applied to cluster the digits into 10 groups. The results are then visualized using a confusion matrix and sample images from each cluster.

## Colab Link
https://colab.research.google.com/drive/1rBpyd32QFudwq4chuSR0n1rdtUe1bamu?usp=sharing

## Steps Performed

1. The digits dataset was loaded from scikit-learn.
- X contains the flattened pixel values
- y contains the true digit labels
- images contains the original 8×8 images used for visualization

2. A 3×3 grid of sample digit images was plotted to understand what the dataset looks like before processing.

3. Standardized pixel values to ensure each feature has zero mean and unit variance because PCA and K-Means are scale-sensitive.

4. Finding optimal no of components for PCA
- PCA was fitted on the standardized data without specifying the number of components.
- The cumulative explained variance was plotted to determine how many components were required to preserve most of the information.
- The model found that 40 principal components are enough to retain 95% or more of the variance.

5. Created a scikit-learn pipeline with:
- StandardScaler
- PCA with 40 components(as determined above)
- K-Means with 10 clusters(as there are 10 digits)

6. Fitted the pipeline to the original dataset, assigning each sample a cluster label.

7. Used a confusion matrix to compare actual digits with cluster assignments.

8. Visualized Cluster Results by displaying sample images from each of the 10 clusterss.

## Requirements

- Python 3.x
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn

## Conclusion

This project demonstrates how dimensionality reduction and clustering can be applied to image data.
By combining PCA and K-Means into an automated pipeline, handwritten digits can be grouped into visually meaningful clusters while keeping most of the dataset’s information intact.
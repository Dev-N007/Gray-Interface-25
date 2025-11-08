# K-Means Clustering of Credit Card Customer Data

This project performs clustering on a credit card customer dataset. The dataset contains information about customer transactions and spending behavior, making it suitable for unsupervised learning.

## Overview

Customer clustering can help identify spending patterns, customer types, and behavioral groups without using labeled data. This can be useful for targeted marketing, fraud detection, risk analysis, and personalized financial services. The technique used here is K-Means, along with exploratory data analysis and preprocessing.

## Dataset link:
https://www.kaggle.com/datasets/arjunbhasin2013/ccdata

## Colab link:
https://colab.research.google.com/drive/1LRn00h-GlvxAlJWQkRypHQHHr66y4E7_?usp=sharing

## Steps Performed

1. Importing Libraries (NumPy, Pandas, Matplotlib, Seaborn, and Scikit-Learn were used).
2. Loading the dataset from the CSV file and visualized using head, shape, info, and describe.
3. The column CUST_ID was removed as it does not contribute to clustering.
4. Missing values were replaced with each column’s mean.
5. Exploratory Data Analysis using a histogram of all features to observe data distribution and a heatmap showing correlations between variables.
6. Feature Scaling using StandardScaler as K-Means clustering is distance-based.
7. To determine the optimal number of clusters, the Elbow Method was implemented in which the K-Means model was trained for values of k from 1 to 10, and inertia scores were plotted and the “elbow point” signifies the ideal number of clusters where adding more clusters stops significantly reducing inertia (4 in this case).
8. Applying K-Means Clustering, each customer was assigned a cluster label(0,1,2,3).
7. Visualization of Clusters through pairplots


## Requirements

- Python 3.x
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

## Conclusion
This project demonstrates how unsupervised learning techniques like K-Means can segment credit card customers based on spending habits. After preprocessing, visualization, and the use of the Elbow Method, customers are grouped into meaningful clusters, providing valuable insights for credit card services and data analysis.
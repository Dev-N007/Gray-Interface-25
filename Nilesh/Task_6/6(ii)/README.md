# PCA on the Iris Dataset

This project demonstrates how to perform Principal Component Analysis (PCA) on the Iris dataset using Python, Scikit-Learn, and matplotlib. 

## Overview

The Iris dataset has four numerical features: sepal length, sepal width, petal length, and petal width. In this project, the dataset is reduced from four dimensions to two using PCA. The two resulting components preserve most of the original variance, and the reduced data is plotted to show the separation between different Iris flower species.

## Link
https://colab.research.google.com/drive/1eciLwHYCS1u2g21pbPyqSfIfG4bsXKgX?usp=sharing

## Steps Performed

1. Imported required libraries (NumPy, Pandas, Matplotlib, and modules from Scikit-Learn).
2. Loaded the Iris dataset.
3. Extracted the input features and target class labels.
4. Standardized the feature values to zero mean and unit variance (because PCA is sensitive to the relative        magnitude of each feature)
5. Applied PCA to reduce the dataset from 4 components to 2 principal components.
6. Displayed the explained variance ratio (72.96% for one principal component and 22.85% for the other so overall >95%).
7. Visualized the transformed data using a 2D scatter plot (the 3 different kinds of irises can be clearly seen segregated into 3 distinct groups).

## Output and Interpretation

- The output includes the variance explained by each of the principal components (72.96% and 22.85%). 
- The sum of these values indicates how much of the original dataset’s information is preserved (95.81%). 
- The 2D scatter plot shows clear clustering that corresponds to the three Iris species, demonstrating that even after reducing the dimensionality, the structure of the data remains well separated.

## Requirements

- Python 3.x
- NumPy
- matplotlib
- scikit-learn

## Conclusion

This project shows how PCA can significantly reduce data dimensionality while preserving important information. 
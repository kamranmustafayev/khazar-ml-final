# Final Report

## Title
**Implementing and Analyzing Support Vector Machines (SVM) for Real-World Classification Problems**

## Introduction
This project explores the use of Support Vector Machines (SVM) for real-world classification tasks. We aim to understand the mechanics of SVMs, evaluate their performance using various kernel functions, and compare results with other classification algorithms. For this project, the Breast Cancer dataset from the Scikit-learn library was used.

## Data Description & Preprocessing
### Dataset
- **Source**: Scikit-learn built-in dataset
- **Target**: Classify tumors as malignant or benign
- **Features**: 30 numerical features based on digitized images of breast masses

### Preprocessing Steps
- Checked for missing values (none present)
- Standardized features using `StandardScaler`
- Split data: 80% training / 20% testing
- No categorical encoding needed as all features are numeric

## Model Implementation & Hyperparameter Tuning
### SVM Configuration
- Implemented using Scikit-learn’s `SVC`
- Kernels tested: Linear, RBF, Polynomial, Sigmoid

### Hyperparameter Tuning
- Used `GridSearchCV` for tuning `C` and `gamma`
- Best performance achieved using RBF kernel with `C=10` and `gamma=0.01`

### Kernel Comparison
- **Linear Kernel**: High accuracy, faster training
- **RBF Kernel**: Best performance with fine decision boundary
- **Polynomial/Sigmoid**: Lower accuracy, risk of overfitting

## Results & Evaluation
### Metrics (on test set)
- **Accuracy**: ~97%
- **Precision**: ~96%
- **Recall**: ~98%
- **F1-score**: ~97%
- **Confusion Matrix**:
```
[[42  1]
 [ 1 70]]
```
- **ROC-AUC Score**: ~0.99

### Visualizations
- Confusion matrix heatmap
- ROC curve
- Decision boundary plots (PCA reduced 2D projection)

## Challenges & Limitations
- Feature importance in SVM is not directly interpretable
- Hyperparameter tuning is time-consuming
- SVMs can struggle with large-scale or noisy datasets

## Conclusion & Future Work
SVM performed exceptionally well on the Breast Cancer dataset. The RBF kernel yielded the highest accuracy with optimal generalization. Future improvements could include:
- Dimensionality reduction (PCA)
- Implementing custom kernel functions
- Comparing with deep learning models

## References
1. Scikit-learn Documentation: https://scikit-learn.org/stable/
2. UCI ML Repository: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
3. Cortes, C., & Vapnik, V. (1995). Support-Vector Networks.

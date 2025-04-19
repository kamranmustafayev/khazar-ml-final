# Support Vector Machine (SVM) Classification Project

## Project Title
**Implementing and Analyzing Support Vector Machines (SVM) for Real-World Classification Problems**

## Overview
This project applies Support Vector Machines (SVM) using Scikit-learn to solve a real-world classification task. The Breast Cancer dataset is used to classify tumors as benign or malignant. The project includes data preprocessing, hyperparameter tuning with GridSearchCV, evaluation metrics, and visualization.

## Dataset
- **Source**: Scikit-learn built-in Breast Cancer Dataset
- **Features**: 30 numerical features describing cell nuclei characteristics
- **Target**: Binary classification – Benign (0) or Malignant (1)

## Features
- Data normalization using StandardScaler
- SVM model with RBF kernel
- Grid search for tuning hyperparameters `C` and `gamma`
- Evaluation: classification report, confusion matrix, ROC curve
- PCA-based 2D decision boundary visualization

## Technologies Used
- Python 3.x
- Scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## Folder Structure
```
ProjectNumber_SVM_YourName/
├── code/
│   └── svm_classification.py        # Main script
├── results/
├── report/
│   └── final_report.md
├── README.md
└── .gitignore
```

## Setup Instructions
1. **Clone the Repository**
```bash
git clone https://github.com/kamranmustafayev/khazar-ml-final.git
cd ProjectNumber_SVM_YourName
```

2. **Install Dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

3. **Run the Script**
```bash
cd code
python svm_classification.py
```

## Output
- Best hyperparameters printed in terminal
- Confusion matrix and ROC curve saved in `/results/`
- Decision boundary plot (PCA 2D) saved in `/results/`

## Notes
- All results are based on a random 80/20 train/test split.
- You can modify kernel types and parameters in the script.



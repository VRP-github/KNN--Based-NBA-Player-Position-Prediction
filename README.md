# NBA Player Position Prediction using K-Nearest Neighbors (KNN)

## Project Overview
This project implements a machine learning pipeline to classify NBA players' positions based on their performance statistics using the K-Nearest Neighbors (KNN) algorithm. It involves data preprocessing, model training, validation, and evaluation, including detailed metrics and insights.

---

## Features
- **Exploratory Data Analysis (EDA):** Ensures no null or zero values in the dataset.
- **Label Encoding:** Encodes categorical data (positions) into numerical labels for modeling.
- **Model Training and Validation:** Splits the dataset into training and validation subsets.
- **Performance Evaluation:**
  - Calculates training, validation, and testing accuracy.
  - Produces confusion matrices for in-depth error analysis.
  - Computes position-wise classification accuracy.
- **Cross-Validation:** Employs 10-fold stratified cross-validation for robust evaluation.

---

## Key Results
### Accuracy
- **Training Accuracy:** 63.60%
- **Validation Accuracy:** 61.40%
- **Testing Accuracy on Dummy Dataset:** 62.14%

### Confusion Matrices
#### Training Confusion Matrix:
```
[[106  18   0   5   0]
 [ 27  79   2  19  11]
 [  4   8 100   2  20]
 [  9  29   6  64  23]
 [  3  15  23  25  86]]
```
#### Validation Confusion Matrix:
```
[[23  4  0  2  0]
 [11 16  1  7  6]
 [ 0  0 28  0  4]
 [ 0  6  2 13  8]
 [ 0  4  8  3 25]]
```
#### Testing Confusion Matrix:
```
[[15  1  0  0  0]
 [ 5 10  0  1  4]
 [ 0  0 17  0  1]
 [ 1  5  2  9 10]
 [ 1  3  1  4 13]]
```

### Cross-Validation
- **Scores per Fold:** [0.4419, 0.5116, 0.5698, 0.5000, 0.5000, 0.5059, 0.5529, 0.5294, 0.6118, 0.4824]
- **Mean Cross-Validation Accuracy:** 52.06%

### Position-Wise Classification Accuracy on Test Set:
- **C:** 93.75%
- **PF:** 50.00%
- **PG:** 94.44%
- **SF:** 33.33%
- **SG:** 59.09%

---

## Dataset
The project uses the following datasets:
1. **Training Dataset:** `nba_stats.csv`
2. **Testing Dataset:** `dummy_test.csv`

### Features Used:
- AST, STL, BLK, TRB, FGA, FG%, 3P, 3PA, PF, eFG%, FT%, DRB, 3P%, 2P, 2P%

---

## Prerequisites
- Python (3.x)
- Required Libraries:
  - pandas
  - scikit-learn

### Installation
Install the dependencies using pip:
```bash
pip install pandas scikit-learn
```

---

## Usage
1. **Dataset Preparation:** Place `nba_stats.csv` and `dummy_test.csv` in the working directory.
2. **Run the Script:** Execute the `NBA_KNN.py` file:
   ```bash
   python NBA_KNN.py
   ```
3. **Outputs:**
   - Accuracy scores
   - Confusion matrices
   - Cross-validation results
   - Position-wise classification accuracies

---

## References
- [K-Nearest Neighbors Algorithm in Python](https://www.geeksforgeeks.org/k-nearest-neighbor-algorithm-in-python/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---



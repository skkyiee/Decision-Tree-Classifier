# Bank Marketing Campaign Prediction

This project applies Decision Tree Classification to predict the outcome of a bank's marketing campaign. Using the bank's customer data, the model predicts whether a customer will subscribe to a term deposit (`yes` or `no`).

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Introduction

In this project, we utilize the UCI Bank Marketing Dataset to predict the success of a direct marketing campaign run by a Portuguese banking institution. Specifically, the goal is to determine whether a client will subscribe to a term deposit (`yes`) based on various features, such as their demographic and economic factors.

## Dataset

The dataset used is the `bank-additional-full.csv` from the UCI Machine Learning Repository. It includes features such as:
- Client attributes (age, job, marital status, etc.)
- Previous campaign contact outcomes
- Social and economic context attributes

Target variable: `y` (whether the client subscribed to a term deposit: `yes` or `no`).

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/bank-marketing-prediction.git
    cd bank-marketing-prediction
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have the dataset (`bank-additional-full.csv`) placed in the root directory.

### Dependencies
- pandas
- scikit-learn
- matplotlib

You can install these using the `requirements.txt` or individually as shown below:
```bash
pip install pandas scikit-learn matplotlib
```

## Modeling Process

1. **Data Preprocessing**:
   - Encoded categorical variables using `LabelEncoder`.
   - Split the data into features (`X`) and target (`y`).
   - Created training and testing datasets with a 70-30 split.

2. **Model**:
   - Trained a Decision Tree Classifier on the training data.
   - Evaluated the model's performance using accuracy, confusion matrix, and classification report.

3. **Evaluation**:
   - Measured model performance through accuracy, precision, recall, F1-score, confusion matrix, and visualization of the decision tree.

## Results

- **Accuracy**: The model achieved an accuracy of **X.XX%** on the test dataset.
- **Confusion Matrix**:
    |        | Predicted No | Predicted Yes |
    |--------|--------------|---------------|
    | Actual No |  XXXX         | XXX          |
    | Actual Yes |  XXX         | XXXX         |

- **Classification Report**:
  - **Class 0 (No)**:
    - Precision: X.XX
    - Recall: X.XX
    - F1-Score: X.XX
  - **Class 1 (Yes)**:
    - Precision: X.XX
    - Recall: X.XX
    - F1-Score: X.XX

- **Overall Accuracy**: X.XX

## Visualization

The Decision Tree model is visualized below:

![Decision Tree](decision_tree.png)



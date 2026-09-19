# 🏦 Loan Default Risk Prediction

> **Predicting Loan Defaults Using Machine Learning**  
> An end-to-end Machine Learning pipeline designed to evaluate credit risk and predict loan default probabilities based on applicant demographics, financial profiles, and loan characteristics.

---

## 📌 Table of Contents
- [📂 Project Overview](#-project-overview)
- [📊 Dataset Information](#-dataset-information)
- [📁 Project Structure](#-project-structure)
- [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [⚙️ Data Preprocessing & Feature Engineering](#️-data-preprocessing--feature-engineering)
- [🤖 Model Development](#-model-development)
- [📊 Sample Inference Code](#-sample-inference-code)
- [🛠️ Tech Stack](#️-tech-stack)


---

## 📂 Project Overview

Understanding loan default risk is crucial for financial institutions to manage credit risk, minimize financial losses, and optimize lending decisions.

- **Goal:** Predict whether a loan applicant will default on their loan (`DEFAULT / HIGH RISK` vs. `APPROVED / LOW RISK`) and calculate their default probability.
- **Algorithms & Methods:** Machine Learning Classification, Feature Scaling, Logarithmic Transformations, and One-Hot Encoding.
- **Libraries Used:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Environment:** Jupyter Notebook / JupyterLab

---

## 📊 Dataset Information

The dataset includes financial, demographic, and historical credit features for each applicant:

| Feature | Description |
| :--- | :--- |
| **person_age** | Age of the applicant |
| **person_income** | Annual income of the applicant |
| **person_home_ownership** | Home ownership status (`RENT`, `OWN`, `MORTGAGE`, `OTHER`) |
| **person_emp_length** | Employment length in years |
| **loan_intent** | Purpose of the loan (`DEBTCONSOLIDATION`, `MEDICAL`, `EDUCATION`, etc.) |
| **loan_grade** | Assigned credit grade (`A`, `B`, `C`, `D`, etc.) |
| **loan_amnt** | Total loan amount requested |
| **loan_int_rate** | Loan interest rate percentage |
| **loan_percent_income** | Loan amount relative to annual income |
| **cb_person_default_on_file** | Historical default on record (`Y` / `N`) |
| **cb_person_cred_hist_length** | Credit history length in years |
| **loan_status** | **Target Variable** (`0` = Approved/Low Risk, `1` = Default/High Risk) |

---

## 📁 Project Structure

```text
├── LoanDefaultRiskPrediction.ipynb   # Main Jupyter Notebook with EDA, preprocessing, and modeling
├── dataset.csv                       # Raw and processed dataset files                       
└──README.md                          # Project documentation                

---

## 📈 Exploratory Data Analysis (EDA)

Before building the predictive model, extensive EDA was conducted to understand feature distributions and correlations:

- **Correlation Heatmap:** Examined numerical relationships between income, loan amount, age, and loan default status.
- **Feature Distribution Histograms:** Analyzed skewness across financial features such as `person_income` and `loan_amnt`.
- **Key Insights:** Raw income and loan amounts exhibit right-skewed distributions, requiring normalization prior to model training.

---

## ⚙️ Data Preprocessing & Feature Engineering

To prepare the dataset for optimal model performance, the following preprocessing steps were applied:

1. **Log Transformations:**
   - Applied `np.log1p` to highly skewed numerical columns (`person_income` $\rightarrow$ `person_income_log`, `loan_amnt` $\rightarrow$ `loan_amnt_log`) to normalize distributions.
   - Dropped original raw income and loan amount columns post-transformation.
2. **Categorical Encoding:**
   - Applied One-Hot Encoding (`pd.get_dummies`) to convert categorical columns (`person_home_ownership`, `loan_intent`, `loan_grade`, `cb_person_default_on_file`) into numerical vectors.
3. **Feature Alignment:**
   - Used column reindexing to guarantee input alignment strictly against training features (`X_train.columns`).
4. **Feature Scaling:**
   - Applied standard scaling (`StandardScaler`) to normalize feature values to zero mean and unit variance.

---

## 🤖 Model Development

1. **Train/Test Split:** Partitioned data into training and testing sets to evaluate unseen risk profiles.
2. **Model Training:** Trained predictive classifiers using standardized applicant features.
3. **Inference Pipeline:** Built a robust input pipeline capable of parsing raw applicant data and predicting risk outcomes.

---

## 📊 Sample Inference Code

Below is the implementation used to evaluate new loan applicants using the trained pipeline:

```python
import numpy as np
import pandas as pd

# 1. Define raw applicant input
new_user = pd.DataFrame([{
    'person_age': 35,
    'person_income': 200000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 19.0,
    'loan_intent': 'DEBTCONSOLIDATION',
    'loan_grade': 'C',
    'loan_amnt': 2000,
    'loan_int_rate': 10.68,
    'loan_percent_income': 0.02,
    'cb_person_default_on_file': 'Y',
    'cb_person_cred_hist_length': 3
}])

# 2. Log transformations
new_user_processed = new_user.copy()
new_user_processed['person_income_log'] = np.log1p(new_user_processed['person_income'])
new_user_processed['loan_amnt_log'] = np.log1p(new_user_processed['loan_amnt'])
new_user_processed = new_user_processed.drop(columns=['person_income', 'loan_amnt'])

# 3. Categorical encoding & strict column reindexing
new_user_processed = pd.get_dummies(new_user_processed)
new_user_processed = new_user_processed.reindex(columns=X_train.columns, fill_value=0)

# 4. Feature scaling & probability prediction
new_user_scaled = scaler.transform(new_user_processed)
prediction = model.predict(new_user_scaled)[0]
default_probability = model.predict_proba(new_user_scaled)[0][1]

# 5. Output results
print(f"Loan Decision: {'DEFAULT / HIGH RISK' if prediction == 1 else 'APPROVED / LOW RISK'}")
print(f"Default Probability: {default_probability:.2%}")
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
- **Development Environment:** JupyterLab / Jupyter Notebook

---



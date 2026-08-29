# 💳 CreditWise — Loan Approval Prediction

A machine learning project that predicts whether a loan application will be approved or rejected, based on applicant financial and demographic data. Built as part of an ML course assignment.

---

## 📌 Project Overview

Banks and financial institutions process thousands of loan applications every day. Manually reviewing each one is time-consuming and prone to bias. This project automates the loan approval decision by training and comparing multiple classification models on real-world-style applicant data.

**Goal:** Predict the binary outcome — `Loan_Approved` (Yes / No) — from applicant features.

**Key metric focused on:** `Precision` — to minimize false positives (i.e., avoid approving loans that should be rejected).

---

## 📁 Repository Structure

```
CreditWise/
├── credit_wise_loan_approval.ipynb   # Main Jupyter notebook (EDA + Modelling)
├── loan_approval_data.csv            # Dataset (~1001 records)
└── README.md                         # Project documentation
```

---

## 📊 Dataset Description

**File:** `loan_approval_data.csv`  
**Records:** ~1001 rows | **Features:** 20 columns (including target)

| Column | Type | Description |
|---|---|---|
| `Applicant_ID` | Numeric | Unique identifier (dropped before training) |
| `Applicant_Income` | Numeric | Monthly income of the primary applicant |
| `Coapplicant_Income` | Numeric | Monthly income of the co-applicant |
| `Employment_Status` | Categorical | `Salaried`, `Self-employed`, `Contract`, `Unemployed` |
| `Age` | Numeric | Age of the applicant |
| `Marital_Status` | Categorical | `Married`, `Single` |
| `Dependents` | Numeric | Number of dependents |
| `Credit_Score` | Numeric | Credit score (range ~550–800) |
| `Existing_Loans` | Numeric | Number of existing active loans |
| `DTI_Ratio` | Numeric | Debt-to-Income ratio |
| `Savings` | Numeric | Total savings of the applicant |
| `Collateral_Value` | Numeric | Value of collateral offered |
| `Loan_Amount` | Numeric | Requested loan amount |
| `Loan_Term` | Numeric | Loan repayment term (in months) |
| `Loan_Purpose` | Categorical | `Home`, `Car`, `Personal`, `Business`, `Education` |
| `Property_Area` | Categorical | `Urban`, `Semiurban`, `Rural` |
| `Education_Level` | Categorical | `Graduate`, `Not Graduate` |
| `Gender` | Categorical | `Male`, `Female` |
| `Employer_Category` | Categorical | `Private`, `Government`, `MNC`, `Business`, `Unemployed` |
| `Loan_Approved` ✅ | Categorical | **Target** — `Yes` / `No` |

> **Note:** The dataset contains missing values across several columns, handled explicitly in the pipeline.

---

## 🔬 ML Pipeline

### 1. 🧹 Data Cleaning

- **Numerical columns** — Missing values imputed using **mean** strategy (`SimpleImputer`).
- **Categorical columns** — Missing values imputed using **most frequent** (mode) strategy.
- `Applicant_ID` column dropped as it carries no predictive signal.

### 2. 📈 Exploratory Data Analysis (EDA)

Visualizations generated in the notebook:

#### Target Variable Distribution
The dataset exhibits a ~70/30 class imbalance, highlighting the necessity to optimize for Precision over raw Accuracy.
![Loan Approval Imbalance](./Charts%20and%20Graphs/LoanApprovalPercentage_piechart.png)

#### Credit Score impact on Loan Approval
Higher credit scores heavily skew towards approved loans, as seen in the distribution below:
![Credit Score vs Approval](./Charts%20and%20Graphs/Applicant_count_CreditScore_LoanApproval_hue.png)

#### Feature Distribution & Outlier Analysis
Box plots generated for all continuous numeric features to identify skewness and outliers prior to StandardScaler normalization:
![Box Plots](./Charts%20and%20Graphs/Box_plots.png)

#### Feature Correlation Heatmap
Pearson correlation matrix identifying collinearity and top predictive signals (e.g. `Credit_Score_sq` and `DTI_Ratio_sq`).
![Correlation Heatmap](./Charts%20and%20Graphs/creditwise_heatmap.png)

*Other visualizations explored in the notebook include `Education_Level` bar charts, `Income` histograms, and stacked distributions.*

### 3. 🔢 Encoding

- **Label Encoding** — Applied to `Education_Level` (ordinal) and `Loan_Approved` (target: Yes=1, No=0).
- **One-Hot Encoding** — Applied to `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, and `Employer_Category` (first category dropped to avoid multicollinearity).

### 4. ✂️ Train-Test Split

- **80/20** split with `random_state = 42` for reproducibility.

### 5. ⚖️ Feature Scaling

- **StandardScaler** — Applied to all features after the split (fit on train, transform on test) to normalize the feature space.

### 6. 🛠️ Feature Engineering

Polynomial and log transformations applied to improve model performance:

| New Feature | Transformation |
|---|---|
| `DTI_Ratio_sq` | `DTI_Ratio ** 2` |
| `Credit_Score_sq` | `Credit_Score ** 2` |
| `Applicant_Income_log` | `log1p(Applicant_Income)` |

Original skewed columns (`Credit_Score`, `DTI_Ratio`, `Applicant_Income`) were **dropped** after transformation to avoid redundancy.

---

## 🤖 Models Trained & Evaluated

Three classification algorithms were trained and compared — **both before and after feature engineering**:

| Model | Library |
|---|---|
| Logistic Regression | `sklearn.linear_model` |
| K-Nearest Neighbors (k=5) | `sklearn.neighbors` |
| Gaussian Naive Bayes | `sklearn.naive_bayes` |

### Evaluation Metrics

Each model was evaluated on:

- **Precision** — Primary metric (minimize false approvals)
- **Recall**
- **F1 Score**
- **Accuracy**
- **Confusion Matrix**

> ✅ **Naive Bayes** was identified as the best-performing model based on precision.

---

## ⚙️ Tech Stack

| Tool | Notes |
|---|---|
| Python 3.14+ | Runtime |
| JupyterLab | Development environment (via Anaconda) |
| Anaconda | Python distribution & environment manager |
| pandas | Data loading & manipulation |
| NumPy | Numerical transformations |
| scikit-learn | Imputation, encoding, scaling, modelling |
| Matplotlib | Plotting |
| Seaborn | Statistical visualizations |

---

## 🚀 How to Run

> 💡 **Development environment:** This notebook was built using **JupyterLab** launched from the **Anaconda Prompt**. It is recommended to use the same setup for the best experience, though any Jupyter-compatible environment will work.

### Option A — Using Anaconda (Recommended)

1. Open **Anaconda Prompt**
2. Navigate to the project folder:
   ```bash
   cd path/to/CreditWise
   ```
3. Launch JupyterLab:
   ```bash
   jupyter lab
   ```
4. Open `credit_wise_loan_approval.ipynb` and run all cells top to bottom.

### Option B — Using pip

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/CreditWise.git
cd CreditWise
```

#### 2. Install dependencies

```bash
pip install -r requirements.txt
```

#### 3. Launch the notebook

```bash
jupyter lab credit_wise_loan_approval.ipynb
```

> Ensure `loan_approval_data.csv` is in the **same directory** as the notebook.

---

## 📌 Key Takeaways

- **Credit Score** and **DTI Ratio** are among the most correlated features with loan approval.
- **Precision** was chosen as the primary metric because falsely approving a risky loan (false positive) has higher real-world costs than missing a good applicant (false negative).
- **Gaussian Naive Bayes** outperformed Logistic Regression and KNN on precision for this dataset.
- **Feature engineering** (polynomial + log transforms) was explored to further boost model performance.

---

## 🙋 Author

Made as part of an ML course project.  
Feel free to fork, star ⭐, or open issues if you find this useful!

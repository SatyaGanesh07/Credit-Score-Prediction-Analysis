# Credit Risk & Credit Score Analysis | **Exploratory Data Analysis (EDA)**
#### **Tools: Python, Pandas, NumPy, Matplotlib, Seaborn**

--- 

## Project Overview

In today's financial landscape, accurate credit score assessment is vital for both financial institutions and customers. Paisabazaar, a leading financial services company, helps individuals find and apply for banking and credit products.

This project performs an **Exploratory Data Analysis (EDA)** on customer financial and behavioral data to understand patterns, detect anomalies, and extract data-driven insights to support credit risk assessment, reduce potential loan defaults, and improve decision-making efficiency.

---

## Problem Statement

Paisabazaar relies on accurate customer creditworthiness assessment to facilitate loan approvals and mitigate financial risks.The current methods of credit evaluation can be enhanced through advanced data analysis to better capture the combined impact of income, debt, credit utilization, and payment behavior on credit scores.

**Goal:** Accurately predict credit scores based on customer data, including income, credit card usage, and payment behavior, to assist Paisabazaar in:

* Reducing loan defaults
* Providing personalized financial product recommendations
* Optimizing credit assessment processes

---

## Business Objectives

1. **Enhance Credit Risk Management:** Improve the accuracy of credit score classification to reduce loan defaults.
2. **Personalize Financial Recommendations:** Offer tailored financial products based on predicted credit scores.
3. **Optimize Loan Approval Processes:** Streamline approvals through predictive modeling.
4. **Increase Customer Satisfaction:** Provide actionable insights and advice to customers based on their credit profile.

---

## Dataset Description

* **Size:** 100,000 records, 28 columns
* **Features include:**

  * **Demographic:** Age, Occupation, Name, SSN
  * **Financial:** Annual Income, Monthly In-hand Salary, Number of Bank Accounts, Credit Cards, Loans, Interest Rate
  * **Credit Information:** Credit Utilization Ratio, Outstanding Debt, Credit History Age, Credit Mix
  * **Behavioral Metrics:** Payment Behavior, Delayed Payments, Credit Inquiries, Payment of Minimum Amount
  * **Target Variable:** Credit Score (Good, Standard, Poor)

**Source:** Dataset provided as `dataset.csv`

---

## Methodology

### 1. Data Cleaning & Preparation

* Removed irrelevant columns (Customer_ID, Name, SSN, etc.)
* Handled missing values (imputed `Annual_Income` with mean)
* Converted data types for analysis (`Monthly_Balance` to numeric)
* Removed outliers in `Annual_Income` using IQR
* Created new features:

  * `Debt_to_Income_Ratio = Outstanding_Debt / Annual_Income`
  * `Income_Bracket` categorized into Low, Medium, High

### 2. Exploratory Data Analysis (EDA)

#### **Univariate Analysis**

* Age distribution histogram → identify dominant demographics
* Income and Monthly Salary histograms → understand financial spread
* Credit Score bar plot → check class distribution (Good, Standard, Poor)
* Credit Utilization box plot → detect over-leveraged customers
* Interest Rate histogram → analyze borrowing costs

#### **Bivariate Analysis**

* Credit Score vs Age (box plot) → correlation with financial stability
* Credit Score vs Income (violin plot) → income disparity across credit scores
* Number of Loans vs Credit Score (stacked bar) → effect of multiple loans
* Delay from Due Date vs Credit Score → payment punctuality correlation
* Occupation vs Credit Score (bar plot) → occupational influence on credit

#### **Multivariate Analysis**

* Pair plots for Annual_Income, Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age vs Credit Score
* Correlation heatmap → identify relationships among financial attributes
* Credit Score distribution across top 10 Loan Types (heatmap)

---

## Key Insights

* **Demographics:** Majority of customers aged 30-40, working population drives financial engagement
* **Income Distribution:** Skewed toward middle-income, enabling targeted loan and credit products
* **Credit Scores:** Standard category dominates, Good and Poor categories require attention for targeted interventions
* **Behavioral Trends:** Timely payments correlate with better credit scores; delayed payments are a key risk factor
* **Occupational Insights:** Healthcare and technical roles dominate Good and Standard credit scores

---

## Recommendations / Solution

* Implement a **predictive credit scoring model** using features such as income, loans, credit utilization, and payment behavior.
* Automate credit assessment to **streamline loan approvals**.
* Offer **personalized financial products** based on predicted credit scores.
* Monitor and analyze **payment behavior** to reduce defaults.

---

## Project Structure

```
Paisabazaar-Credit-EDA/
│
├── dataset.csv                 # Raw dataset
├── EDA_Notebook.ipynb          # Jupyter notebook with full analysis
├── images/                     # Folder containing all plots
├── README.md                   # Project overview and summary
├── requirements.txt            # Python libraries required
```

---

## Imported Libraries


### This project uses the following Python libraries for data manipulation, visualization, and analysis:

```python


# Data manipulation and numerical analysis
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Image and plot styling utilities
import matplotlib.image as mpimg
import matplotlib.lines as lines
import matplotlib.patches as patches

# Warnings handling
import warnings
warnings.filterwarnings('ignore')
```
---

# Tools & Technologies

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**

--- 
## Project Scope

This project focuses on **Exploratory Data Analysis (EDA)** to understand the key factors influencing credit scores.  
While no machine learning model is built in this phase, the insights derived from EDA are intended to support future development of predictive credit scoring models.
3. Open `EDA_Notebook.ipynb` and run all cells to reproduce the analysis.

---
## Data Limitations

- The dataset is static and does not capture real-time customer behavior.
- Some behavioral attributes are self-reported and may contain bias.
- Credit score labels are categorical rather than continuous, limiting fine-grained risk estimation.
- External factors such as economic conditions are not included.

--- 

## Evaluation Metrics (For Predictive Modeling Phase)

- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Score
--- 
## Contact

**Name:** Venkata Satya Ganesh  
**Role:** Aspiring Data Analyst  
**Skills:** Python, SQL, Power BI,Snowflake   
**LinkedIn:** <your-linkedin-link>

## Contact


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
The primary business objective is to improve Paisabazaar's credit assessment process by accurately predicting individual credit scores based on customer data. By leveraging features such as income, credit card usage, loan history, and payment behavior, the goal is to:

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

## Load Dataset 
```python
bank_df = pd.read_csv(r"C:/Users/satya/OneDrive/Desktop/New folder/Power BI/Project/Banking-Fraud-Analysis-Paisabazzar/Paisabazaar_dateset.csv")

# Dataset First Look
bank_df.head()

# Dataset Rows & Columns count
print(f'Number of Rows in {bank_df.shape[0]}')
print(f'Number of Columns in {bank_df.shape[1]}')

# Dataset Info
bank_df.info()

# Dataset Duplicate Value Count
bank_df.duplicated().sum()

# Missing Values/Null Values Count
bank_df.isnull().sum()

```

# Visualizing the missing 

```python 
def apply_chart_styling(ax,fig,title,subtitle,insight_text,logo_path = 'logo.png'):

    fig.patch.set_facecolor('#D3D3D3')
    ax.set_facecolor('#D3D3D3')

    fig.text(0.09,1.05 , title,fontsize = 18 , fontweight = 'bold', fontfamily = 'serif')
    fig.text(0.09,0.99 , subtitle,fontsize = 12,fontweight = 'bold',fontfamily = 'serif')

    fig.text(1.1, 1.01, 'Insight', fontsize = 12, fontweight = 'bold',fontfamily = 'serif')
    fig.text(1.1, 0.50, insight_text, fontsize = 12, fontweight = 'bold',fontfamily = 'serif')

    logo = mpimg.imread(logo_path)
    logo_ax = fig.add_axes([1.5,0.85,0.1,0.1])
    logo_ax.imshow(logo)
    logo_ax.axis('off')

    ax.grid(axis = 'y',linestyle = '-', alpha = 0.4)
    ax.set_axisbelow(True)

    for spine in ['top','right','left']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis = 'both',which = 'major', labelsize = 12)

    l1 = lines.Line2D([1, 1], [0, 1], transform=fig.transFigure, figure=fig, color='black', lw=0.2)
    fig.lines.extend([l1])

missing_data = bank_df.isnull().sum().sort_values(ascending = False)

fig,ax = plt.subplots(1,1,figsize = (18,10))

bars = ax.bar(missing_data.index,missing_data.values,color = 'black')

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')

apply_chart_styling(ax = ax, fig = fig, title = 'Missing Data' , subtitle = 'Analyzing missing data values', insight_text = '''Certainly looking upto the
dataset we can clearly see that there are no missing values''')

plt.tight_layout()
plt.show()

```
### Understanding Variables 

```python
 # Dataset Columns
bank_df.columns

# Dataset Describe
bank_df.describe()
```
1. ID: Unique identifier for each record.  
2. Customer_ID: Unique identifier for each customer.  
3. Month: Month of data collection or financial activity.  
4. Name: Customer’s name (likely anonymized).  
5. Age: Age of the customer.  
6. SSN: Social Security Number (likely anonymized).  
7. Occupation: Job title or role of the customer.  
8. Annual_Income: Total income earned by the customer in a year.  
9. Monthly_Inhand_Salary: Net salary received by the customer monthly.  
10. Num_Bank_Accounts: Number of bank accounts held by the customer.  
11. Num_Credit_Card: Number of credit cards owned by the customer.  
12. Interest_Rate: Interest rate applicable to the customer’s loans or credit.  
13. Num_of_Loan: Total number of loans taken by the customer.  
14. Type_of_Loan: Categories of loans taken (e.g., personal, home, auto).  
15. Delay_from_due_date: Average delay in payments from the due date.  
16. Num_of_Delayed_Payment: Count of payments that were delayed.  
17. Changed_Credit_Limit: Any changes made to the customer’s credit limit.  
18. Num_Credit_Inquiries: Number of inquiries made on the customer’s credit report.  
19. Credit_Mix: Composition of credit accounts (e.g., revolving, installment).  
20. Outstanding_Debt: Total amount of debt currently owed by the customer.  
21. Credit_Utilization_Ratio: Percentage of available credit being used.  
22. Credit_History_Age: Length of time the customer has had credit accounts.  
23. Payment_of_Min_Amount: Indicator of whether the customer pays the minimum amount due.  
24. Total_EMI_per_month: Total equated monthly installment payments.  
25. Amount_invested_monthly: Amount of money the customer invests each month.  
26. Payment_Behaviour: Customer’s general payment habits (e.g., on-time, late).  
27. Monthly_Balance: Average balance in the customer’s accounts monthly.  
28. Credit_Score: Classification of the customer’s creditworthy financial health and behavior.  

--- 
### Check Unique Values for each variable.
```python
for column in bank_df.columns:
    print(f'{column} : {len(bank_df[column].unique())}')
```
### 3.***Data Wrangling***
```python
bank_df.drop(['Customer_ID','Name','SSN','Month','Payment_of_Min_Amount','ID'],axis = 1, inplace = True)
Q1 = bank_df['Annual_Income'].quantile(0.25)
Q3 = bank_df['Annual_Income'].quantile(0.75)
IQR = Q3 - Q1
bank_df = bank_df[(bank_df['Annual_Income'] >= (Q1 - 1.5 * IQR)) & (bank_df['Annual_Income'] <= (Q3 + 1.5 * IQR))]
bank_df['Annual_Income'].fillna(bank_df['Annual_Income'].mean(), inplace=True)
bank_df['Monthly_Balance'] = pd.to_numeric(bank_df['Monthly_Balance'], errors='coerce')
bank_df['Debt_to_Income_Ratio'] = bank_df['Outstanding_Debt'] / bank_df['Annual_Income']
bank_df['Income_Bracket'] = pd.cut(bank_df['Annual_Income'], bins=[0, 20000, 50000, 100000], labels=['Low', 'Medium', 'High'])
```

### Data Manipulations:

1. **Removed Irrelevant Columns**: Dropped unnecessary columns such as customer identifiers and unrelated attributes to focus on relevant financial and behavioral features.

2. **Outlier Removal**: Filtered out extreme values in the **Annual_Income** column using the Interquartile Range (IQR) method. This step ensures that outliers do not skew the analysis.

3. **Handled Missing Values**: Filled any missing entries in the **Annual_Income** column with the mean value to maintain dataset integrity and ensure all records have valid income data.

4. **Data Type Conversion**: Converted the **Monthly_Balance** column to a numeric type, ensuring that calculations involving this column can be performed accurately.

5. **Calculated Debt-to-Income Ratio**: Introduced a new metric, **Debt_to_Income_Ratio**, which shows the proportion of a customer’s income that goes towards paying debt. This is an important measure of creditworthiness.

6. **Created Income Bracket**: Categorized customers' **Annual_Income** into three brackets (Low, Medium, High) to facilitate segmentation and targeted analysis.

### Insights Found:

- **Outlier Management**: Removing outliers allows for a more accurate representation of customer income, leading to more reliable insights.

- **Debt-to-Income Ratio**: This new metric provides a clearer understanding of customers' financial health, indicating potential credit risks.

- **Income Segmentation**: The creation of income brackets enables targeted analysis, making it easier to tailor financial product recommendations based on income levels.

- **Improved Data Quality**: Addressing missing values and ensuring appropriate data types enhances the reliability of the dataset for predictivd on financial attributes.
---

## **Credit Score Distribution**
### Chart - 1 visualization code
```python
fig,ax = plt.subplots(1,1,figsize = (18,10))

palette = ['#0066ff', '#000000', '#FFFFFF'] 
sns.countplot(x = 'Credit_Score', data = bank_df , palette = palette , ax = ax)

apply_chart_styling(ax = ax,fig = fig,title = 'Credit Score Distribution' , subtitle = 'Analyzing Credit Score Count',insight_text = '''

The Credit Score Distribution chart reveals that 
the Standard credit score category has the highest count, 
indicating that most individuals fall within this middle range. 
In contrast, the Good credit score 
category shows a lower count, 
while the Poor category has the fewest individuals. 
This distribution suggests opportunities for Paisabazaar 
to provide targeted 
financial education and products 
aimed at improving credit ratings, 
particularly for those in the Standard category. 
Additionally, 
developing tailored services, such as credit-building 
loans and educational resources, 
could help customers transition to a Good credit score. 
Overall, analyzing this distribution allows for better 
customer segmentation and 
more personalized marketing strategies, 
ultimately guiding strategic decisions 
related to product development and customer engagement.''')

plt.tight_layout()
plt.show()
```
--- 

## Age Demographics
### Chart - 2 visualization code
```python
fig,ax = plt.subplots(1,1,figsize = (18,10))

sns.histplot(x = 'Age' , data = bank_df , ax = ax ,kde = True, bins = 30, color = '#0066FF')
apply_chart_styling(ax = ax,fig = fig,title = 'Age Distribution' , subtitle = 'Analyzing Age Distribution',insight_text = '''
The Age Distribution chart provides insights into 
the demographic makeup of the customer base. 
It shows that the age group between 30 and 40 years has the highest count of customers, 
indicating that this demographic is most actively engaged with the services offered. 
The count gradually decreases for ages below 30 and above 40, suggesting a decline in 
participation among younger and older age groups. 

The overlayed trend line illustrates a 
generally declining pattern as age increases, 
with a slight uptick in the 50s, indicating a 
potential interest in financial products among older customers. 
This information can help Paisabazaar tailor their marketing strategies 
and product offerings to cater specifically to the age group with the 
highest engagement, while also exploring ways to attract younger customers and 
retain older clients. ''')
plt.tight_layout()
plt.show()
```
--- 

## Income Distrubution

### Chart - 3 visualization code
```python
fig, ax = plt.subplots(1, 2, figsize=(18, 10))

# Flatten the axes array for easier iteration
axes = ax.flatten()

variables = ['Annual_Income', 'Monthly_Inhand_Salary']

for i, var in enumerate(variables):
    sns.histplot(bank_df[var], bins=30, kde=True, ax=axes[i], color='#0066ff')
    
    apply_chart_styling(
        ax=axes[i], 
        fig=fig, 
        title=f'Distribution of Annual Income and Monthly in hand Salary ', 
        subtitle=f'Analyzing income to capture variations across individuals.', 
        insight_text='''The Distribution of Annual Income and Monthly 
In-Hand Salary** chart provides insights 
into the financial status of individuals within the dataset.
The left side of the chart shows the distribution of **Annual Income**, 
with a prominent peak around the 20,000 to 40,000 range, 
indicating that a significant number of individuals earn within this bracket. 
There’s a gradual decline in frequency as income increases, 
suggesting a right-skewed distribution.
        
On the right side, the **Monthly In-Hand Salary** distribution mirrors 
the annual income distribution but reveals a more pronounced peak between 2,000 to 4,000. 
This indicates that most individuals have a monthly salary that correlates with their annual income, 
further emphasizing the skewness towards lower incomes.
        
The analysis suggests that a large segment of the population earns modest incomes, 
and there may be potential opportunities for financial products aimed at individuals within this 
income range, particularly focusing on budgeting and savings. Additionally, the consistent patterns 
between annual income and monthly salary imply stability in financial planning among the demographic. '''
)

# Hide unused subplots if any
for j in range(len(variables), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
```
--- 

## Credit Utilization 
### Chart - 4 visualization code
```python
fig,ax = plt.subplots(1,1,figsize = (18,10))
palette = ['#0066ff', '#000000', '#FFFFFF']
sns.boxplot(x ='Credit_Utilization_Ratio', data = bank_df , palette = palette , ax = ax)

for line in ax.artists:
    if line.get_label() == 'median':
        line.set_color('red')  

apply_chart_styling(fig = fig, ax = ax ,title = 'Credit Utilization', subtitle = 'Analyzing Credit Utilization', insight_text = ''' The provided visualization displays credit utilization distribution. 
The chart seems to analyze the ratio, with values concentrated 
between approximately 28% to 35%. 
There is a moderate range of variation in credit utilization. 
This analysis might indicate that users in 
the sample have credit usage within this range, 
and deviations outside of it are minimal.''')
plt.tight_layout()
plt.show()
```
--- 
## Distribution Of Interest Rate

### Chart - 5 visualization code
```python
fig,ax = plt.subplots(1,1,figsize = (18,10))
palette = ['#0066ff', '#000000', '#FFFFFF']
sns.histplot(x ='Interest_Rate', data = bank_df , color = '#0066FF' , ax = ax,bins = 30 ,kde =True)

apply_chart_styling(fig = fig, ax = ax ,title = 'Distribution Of Interest Rate', subtitle = 'Analyzing Interest Rate', insight_text = '''The chart depicts the distribution of interest rates. It shows variability across a broad range, 
with a few notable peaks. 
The most significant spikes occur around the 5% and 20% interest rates. 
These peaks suggest that a large portion of the data points 
cluster at these rates. Additionally, the density curve overlaid highlights fluctuations, 
with rates predominantly falling between 0% and 35%. 
This suggests a diverse spread of interest rates with concentrations at specific points.''')
plt.tight_layout()
plt.show()
```

# **Bivariate Analysis**
## Age and impact on it's Credit Score
### Chart - 6 visualization 
```python
fig,ax = plt.subplots(1,1,figsize = (18,10))

sns.boxplot(x = 'Credit_Score' , y = 'Age' , data = bank_df , ax = ax, palette = palette)

for line in ax.artists:
    if line.get_label() == 'median':
        line.set_color('red')  
        
apply_chart_styling(fig = fig , ax = ax , title = 'Age and Its Impact on Credit Score', subtitle = 'Analyzing the Correlation Between Age Groups and Creditworthiness', insight_text = '''The chart shows a comparison 
between age groups and credit scores. 
The age distribution for good, standard, and poor credit scores spans 
a similar range, 
generally between 30 to 50 years. 
The median age for individuals with good and poor credit is around 40, 
while for those with standard credit, the median is closer to 35. 
This indicates that age does not strongly influence credit score differences in this dataset. ''')
plt.tight_layout()
plt.show()
```
### Chart - 7 visualization code
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,10))


sns.violinplot(x='Credit_Score', y='Annual_Income', data=bank_df, ax=ax1, palette=palette)
ax1.set_title('Annual Income and Credit Score')
ax1.set_ylabel('Annual Income')


sns.violinplot(x='Credit_Score', y='Monthly_Inhand_Salary', data=bank_df, ax=ax2, palette=palette)
ax2.set_title('Monthly Inhand Salary and Credit Score')
ax2.set_ylabel('Monthly Inhand Salary')

apply_chart_styling(fig=fig, ax=ax1, title='Income and Its Impact on Credit Score', subtitle='Analyzing the Correlation Between Income and Creditworthiness', insight_text=''' ''')
apply_chart_styling(fig=fig, ax=ax2, title='', subtitle='', insight_text='''The chart shows the correlation between income and credit scores using violin plots. 
The left plot compares annual income with credit scores, showing that individuals with higher 
credit scores tend to have higher annual incomes. The distribution narrows for those with standard and poor credit scores, 
indicating lower annual incomes in those groups.
The right plot focuses on monthly in-hand salary, reinforcing a similar trend. 
Those with good credit scores have higher monthly salaries, 
while the distribution for individuals with standard and poor credit scores shifts 
towards lower monthly incomes. The overall insight is that higher incomes are associated 
with better credit scores. ''')
plt.tight_layout()
plt.show()
```

### The Relationship Between Credit Inquiries and Credit Score

```python
fig, ax = plt.subplots(1, 1, figsize=(18, 10))


sns.countplot(data=bank_df, x='Num_Credit_Inquiries', hue='Credit_Score', ax=ax, palette= palette)

# Apply chart styling
apply_chart_styling(
    fig=fig,
    ax=ax,
    title='The Relationship Between Credit Inquiries and Credit Score',
    subtitle='Exploring How Credit Inquiries Impact Creditworthiness',
    insight_text='''
    The provided chart explores the relationship between credit inquiries and creditworthiness. 
    It reveals that while a 
    higher number of inquiries can generally lead to a lower credit score, 
    there are exceptions. 
    Individuals with credit scores between 700 and 800, 
    considered good to excellent, can have a higher number of inquiries without significantly 
    impacting their score. 
    However, for those with lower credit scores, 
    even a moderate number of inquiries can negatively 
    affect their creditworthiness'''
)

plt.tight_layout()
plt.show()
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
## Contact

For any questions or suggestions, please open an issue or contact me via LinkedIn:  
[Satya Ganesh LinkedIn](https://www.linkedin.com/in/satya-ganesh-5a89b2283/)


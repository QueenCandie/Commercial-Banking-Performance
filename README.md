Commercial Banking Performance Analytics & Deposit Forecasting

📊 Project Overview

This project analyzes commercial banking performance across six banking teams and uses machine learning to forecast monthly deposit performance.

The project demonstrates an end-to-end data analytics workflow using Python, including data cleaning, data quality validation, feature engineering, exploratory analysis, business KPI analysis, visualization, and predictive modeling.

---

🎯 Business Problem

Commercial banking teams are measured against deposit targets and other performance indicators.

The objective of this project is to:

- Evaluate deposit performance against targets
- Identify high- and low-performing teams
- Analyze deposit trends over time
- Identify variables associated with deposit performance
- Forecast future deposit performance
- Compare different machine learning models
- Generate actionable business recommendations

---

🛠️ Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning
- Business Intelligence
- Data Visualization

---

🔄 Analytical Workflow

1. Data Cleaning

The dataset was validated for:

- Missing values
- Duplicate records
- Negative values
- Data consistency
- Column structure

No missing values or duplicate records were identified during the initial validation.

---

2. Feature Engineering

New analytical features were created, including:

- Year
- Month
- NPL Ratio
- Total Income
- Deposit Gap Percentage
- Retention Status
- Lagged deposit variables
- Three-month rolling deposit average

---
📈 Key Business Findings

Overall Performance

Overall deposit target achievement was:

82.98%

This indicates that actual deposits were below the aggregate deposit target during the analysis period.

---

🏆 Best Performing Team

VI1

- Target Achievement: **84.09%**
- Average Deposit Retention: **90.43%**
- Average MoM Growth: **1.68%**

VI1 recorded the highest deposit target achievement among the six teams.

---

⚠️ Team Requiring Attention

Marina

- Target Achievement: **81.50%**
- Average Deposit Retention: **90.80%**
- Average MoM Growth: **1.63%**

Marina recorded the lowest target achievement and should be investigated for potential performance improvement opportunities.

---

💰 Largest Deposit Contributor

Lekki

Total deposit contribution:

₦170.47 Billion

Although VI1 had the highest target achievement percentage, Lekki generated the largest absolute deposit contribution.

---

🔎 Performance Driver Analysis

Correlation analysis was performed to identify variables with strong linear relationships with total deposits.

The strongest observed relationship was:

Performing Loans → Total Deposit**

Correlation:

0.915

Other notable relationships included:

- Net Interest Income: **0.792**
- Fee Income: **0.740**
- Non-Performing Loans: **0.552**

Correlation indicates association and does not establish causation.

---

🤖 Deposit Forecasting

Machine learning was used to forecast monthly commercial banking deposits.

Lag features and a three-month rolling average were created using historical deposit information.

A time-based train/test split was used to prevent future information from entering the training process.

 Models Evaluated

1. Linear Regression
2. Random Forest Regressor

---

🏆 Model Comparison

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Linear Regression | ₦1.504bn | ₦1.739bn | 3.05% |
| Random Forest | ₦2.541bn | ₦2.752bn | 5.16% |

 Selected Model

Linear Regression

Linear Regression achieved the lowest MAE, RMSE and MAPE on the holdout period.

---

 🚨 Data Leakage Prevention

An initial forecasting experiment produced an unrealistically perfect 0% error.

Further investigation identified data leakage caused by calculating the rolling average using the current month's deposit.

The feature engineering process was corrected so that the rolling average uses only historical observations.

This resulted in a more realistic:

3.05% MAPE

This demonstrates the importance of validating model results and preventing future information from entering predictive features.

---

💡 Business Recommendations

1. Investigate the deposit performance gap, particularly for Marina.

2. Identify successful strategies used by high-performing teams such as VI1 and Lekki and evaluate whether they can be replicated.

3. Monitor performing loans alongside deposit performance as an important indicator of commercial banking activity.

4. Use Linear Regression as the current baseline for short-term deposit forecasting.

5. Expand the historical dataset before deploying the forecasting model for production decision-making.

---

 ⚠️ Project Limitations

The forecasting dataset contains a relatively small number of monthly observations.

Therefore, the forecasting model should be treated as an analytical baseline rather than a production-ready forecasting system.

Additional historical data would improve model reliability and allow for more robust time-series validation.

---

 📌 Project Outcome

This project demonstrates an end-to-end approach to transforming commercial banking data into actionable business insights using Python and machine learning.

The workflow covers:

**Data → Cleaning → Analysis → Visualization → Feature Engineering → Machine Learning → Evaluation → Business Recommendations**

---

👩🏽‍💻 Author

Oluwaseyi Obarayo

Data Analyst | Business Intelligence | Python | SQL | Power BI

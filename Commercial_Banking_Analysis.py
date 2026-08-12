# ============================================================
# COMMERCIAL BANKING PERFORMANCE ANALYTICS
# AND DEPOSIT FORECASTING
# ============================================================
#
# Author: Oluwaseyi Obarayo
# Project: Commercial Banking Performance Analytics
#
# Description:
# This project analyzes commercial banking performance across
# six teams and develops a baseline machine learning model for
# monthly deposit forecasting.
#
# Tools:
# Python, Pandas, NumPy, Matplotlib, Scikit-learn
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 2. LOAD DATA
# ============================================================

# Update this path if running the script locally.
DATA_PATH = "commercial_banking_performance.csv"

df = pd.read_csv(DATA_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. DATA CLEANING
# ============================================================

# Standardize column names
df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
)

# Remove duplicate records
df = df.drop_duplicates()

# Convert Date column
if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"])

# Sort chronologically
if "Date" in df.columns:
    df = df.sort_values("Date")

print("\n========== DATA QUALITY CHECK ==========")

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

# NPL Ratio
if "Non_Performing_Loans" in df.columns and "Performing_Loans" in df.columns:

    total_loans = (
        df["Performing_Loans"] +
        df["Non_Performing_Loans"]
    )

    df["NPL_Ratio"] = (
        df["Non_Performing_Loans"] /
        total_loans
    ) * 100


# Deposit achievement
if "Total_Deposit" in df.columns and "Deposit_Target" in df.columns:

    df["Percent_Achieved"] = (
        df["Total_Deposit"] /
        df["Deposit_Target"]
    ) * 100


# Deposit gap
if "Total_Deposit" in df.columns and "Deposit_Target" in df.columns:

    df["Deposit_Gap"] = (
        df["Deposit_Target"] -
        df["Total_Deposit"]
    )


# Month and year
if "Date" in df.columns:

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month


# ============================================================
# 5. BASIC BUSINESS PERFORMANCE
# ============================================================

print("\n========== BUSINESS PERFORMANCE ==========")

total_deposit = df["Total_Deposit"].sum()
total_target = df["Deposit_Target"].sum()

overall_achievement = (
    total_deposit /
    total_target
) * 100

print(
    f"Overall deposit achievement: "
    f"{overall_achievement:.2f}%"
)


# ============================================================
# 6. TEAM PERFORMANCE ANALYSIS
# ============================================================

team_summary = (
    df.groupby("Team")
    .agg(
        Total_Deposit=("Total_Deposit", "sum"),
        Deposit_Target=("Deposit_Target", "sum"),
        Average_Retention=("Deposit_Retention", "mean"),
        Average_MoM_Growth=("MoM_Growth", "mean"),
        Average_NPL_Ratio=("NPL_Ratio", "mean")
    )
    .reset_index()
)

team_summary["Achievement"] = (
    team_summary["Total_Deposit"] /
    team_summary["Deposit_Target"]
) * 100

# Retention is stored as a decimal
team_summary["Retention_Percentage"] = (
    team_summary["Average_Retention"] * 100
)

# MoM Growth is already stored as a percentage
team_summary["MoM_Growth_Percentage"] = (
    team_summary["Average_MoM_Growth"]
)


print("\n========== TEAM PERFORMANCE ==========")

print(
    team_summary[
        [
            "Team",
            "Total_Deposit",
            "Deposit_Target",
            "Achievement",
            "Retention_Percentage",
            "MoM_Growth_Percentage",
            "Average_NPL_Ratio"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. IDENTIFY BEST AND WEAKEST TEAMS
# ============================================================

best_team = team_summary.loc[
    team_summary["Achievement"].idxmax()
]

weakest_team = team_summary.loc[
    team_summary["Achievement"].idxmin()
]

highest_deposit_team = team_summary.loc[
    team_summary["Total_Deposit"].idxmax()
]

print("\n========== KEY TEAM RESULTS ==========")

print(
    f"Best performing team: "
    f"{best_team['Team']}"
)

print(
    f"Achievement: "
    f"{best_team['Achievement']:.2f}%"
)

print(
    f"Average retention: "
    f"{best_team['Retention_Percentage']:.2f}%"
)

print(
    f"Average MoM growth: "
    f"{best_team['MoM_Growth_Percentage']:.2f}%"
)

print(
    f"\nTeam needing attention: "
    f"{weakest_team['Team']}"
)

print(
    f"Achievement: "
    f"{weakest_team['Achievement']:.2f}%"
)

print(
    f"\nLargest deposit contributor: "
    f"{highest_deposit_team['Team']}"
)

print(
    f"Total deposit: "
    f"₦{highest_deposit_team['Total_Deposit'] / 1e9:.2f} Billion"
)


# ============================================================
# 8. VISUALIZATION — TEAM ACHIEVEMENT
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    team_summary["Team"],
    team_summary["Achievement"]
)

plt.axhline(
    100,
    linestyle="--",
    label="Target"
)

plt.title("Deposit Target Achievement by Team")
plt.xlabel("Team")
plt.ylabel("Achievement (%)")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()


# ============================================================
# 9. VISUALIZATION — TOTAL DEPOSIT BY TEAM
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    team_summary["Team"],
    team_summary["Total_Deposit"] / 1e9
)

plt.title("Total Deposit by Team")
plt.xlabel("Team")
plt.ylabel("Total Deposit (₦ Billion)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 10. MONTHLY DEPOSIT TREND
# ============================================================

monthly_deposit = (
    df.groupby("Date")["Total_Deposit"]
    .sum()
    .reset_index()
)

monthly_deposit["Deposit_Billion"] = (
    monthly_deposit["Total_Deposit"] /
    1e9
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_deposit["Date"],
    monthly_deposit["Deposit_Billion"],
    marker="o"
)

plt.title("Monthly Commercial Banking Deposit Trend")
plt.xlabel("Date")
plt.ylabel("Deposit (₦ Billion)")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 11. CORRELATION ANALYSIS
# ============================================================

driver_columns = [
    "Total_Deposit",
    "Deposit_Retention",
    "Performing_Loans",
    "Non_Performing_Loans",
    "Customers",
    "New_Customers",
    "Net_Interest_Income",
    "Fee_Income",
    "NPL_Ratio",
    "MoM_Growth",
    "Percent_Achieved"
]

# Keep only columns that exist
driver_columns = [
    col for col in driver_columns
    if col in df.columns
]

correlation = df[driver_columns].corr()

deposit_correlation = (
    correlation["Total_Deposit"]
    .sort_values(ascending=False)
)

print("\n========== CORRELATION WITH TOTAL DEPOSIT ==========")

print(deposit_correlation)


# ============================================================
# 12. DEPOSIT FORECASTING DATASET
# ============================================================

forecast_df = (
    df.groupby("Date")["Total_Deposit"]
    .sum()
    .reset_index()
)

forecast_df = forecast_df.sort_values("Date")

forecast_df["Deposit_Billion"] = (
    forecast_df["Total_Deposit"] /
    1e9
)


# Lag features
forecast_df["Lag_1"] = (
    forecast_df["Deposit_Billion"].shift(1)
)

forecast_df["Lag_2"] = (
    forecast_df["Deposit_Billion"].shift(2)
)

forecast_df["Lag_3"] = (
    forecast_df["Deposit_Billion"].shift(3)
)


# IMPORTANT:
# Shift first, then calculate rolling average.
# This prevents current-month data leakage.

forecast_df["Rolling_3_Month"] = (
    forecast_df["Deposit_Billion"]
    .shift(1)
    .rolling(3)
    .mean()
)


model_df = forecast_df.dropna().copy()


# ============================================================
# 13. PREPARE TRAINING AND TEST DATA
# ============================================================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Rolling_3_Month"
]

X = model_df[features]

y = model_df["Deposit_Billion"]


# Time-based split
# The final four observations are kept for testing.

X_train = X.iloc[:-4]
X_test = X.iloc[-4:]

y_train = y.iloc[:-4]
y_test = y.iloc[-4:]


print("\n========== FORECASTING DATA ==========")

print(
    "Training observations:",
    len(X_train)
)

print(
    "Testing observations:",
    len(X_test)
)


# ============================================================
# 14. MODEL 1 — LINEAR REGRESSION
# ============================================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)


# ============================================================
# 15. LINEAR REGRESSION EVALUATION
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_mape = (
    np.mean(
        np.abs(
            (y_test - linear_pred) /
            y_test
        )
    ) * 100
)


print("\n========== LINEAR REGRESSION ==========")

print(
    f"MAE: ₦{linear_mae:.3f} Billion"
)

print(
    f"RMSE: ₦{linear_rmse:.3f} Billion"
)

print(
    f"MAPE: {linear_mape:.2f}%"
)


# ============================================================
# 16. MODEL 2 — RANDOM FOREST
# ============================================================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=4,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(
    X_test
)


# ============================================================
# 17. RANDOM FOREST EVALUATION
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)

rf_mape = (
    np.mean(
        np.abs(
            (y_test - rf_pred) /
            y_test
        )
    ) * 100
)


print("\n========== RANDOM FOREST ==========")

print(
    f"MAE: ₦{rf_mae:.3f} Billion"
)

print(
    f"RMSE: ₦{rf_rmse:.3f} Billion"
)

print(
    f"MAPE: {rf_mape:.2f}%"
)


# ============================================================
# 18. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        rf_rmse
    ],

    "MAPE": [
        linear_mape,
        rf_mape
    ]
})


print("\n========== MODEL COMPARISON ==========")

print(
    comparison.to_string(index=False)
)


# ============================================================
# 19. SELECT BEST MODEL
# ============================================================

if linear_mape < rf_mape:

    selected_model = "Linear Regression"

    selected_predictions = linear_pred

    selected_mape = linear_mape

else:

    selected_model = "Random Forest"

    selected_predictions = rf_pred

    selected_mape = rf_mape


print(
    f"\nSelected model: "
    f"{selected_model}"
)

print(
    f"Selected model MAPE: "
    f"{selected_mape:.2f}%"
)


# ============================================================
# 20. ACTUAL VS PREDICTED
# ============================================================

results = pd.DataFrame({

    "Date": model_df["Date"].iloc[-4:].values,

    "Actual_Deposit": y_test.values,

    "Predicted_Deposit": selected_predictions

})


print("\n========== ACTUAL VS PREDICTED ==========")

print(
    results.to_string(index=False)
)


# ============================================================
# 21. FORECAST VISUALIZATION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    results["Date"],
    results["Actual_Deposit"],
    marker="o",
    label="Actual Deposit"
)

plt.plot(
    results["Date"],
    results["Predicted_Deposit"],
    marker="o",
    linestyle="--",
    label=f"{selected_model} Prediction"
)

plt.title(
    "Actual vs Predicted Commercial Banking Deposits"
)

plt.xlabel("Month")

plt.ylabel("Deposit (₦ Billion)")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.show()


# ============================================================
# 22. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "Feature": features,

    "Importance": rf_model.feature_importances_

})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n========== RANDOM FOREST FEATURE IMPORTANCE ==========")

print(
    importance.to_string(index=False)
)


# ============================================================
# 23. EXECUTIVE BUSINESS INSIGHTS
# ============================================================

print("\n")
print("=" * 60)
print("EXECUTIVE BUSINESS INSIGHTS")
print("=" * 60)

print(
    f"\nOverall deposit achievement: "
    f"{overall_achievement:.2f}%"
)

print(
    f"\nBest performing team: "
    f"{best_team['Team']}"
)

print(
    f"Target achievement: "
    f"{best_team['Achievement']:.2f}%"
)

print(
    f"Average retention: "
    f"{best_team['Retention_Percentage']:.2f}%"
)

print(
    f"Average MoM growth: "
    f"{best_team['MoM_Growth_Percentage']:.2f}%"
)

print(
    f"\nTeam requiring attention: "
    f"{weakest_team['Team']}"
)

print(
    f"Target achievement: "
    f"{weakest_team['Achievement']:.2f}%"
)

print(
    f"\nLargest deposit contributor: "
    f"{highest_deposit_team['Team']}"
)

print(
    f"Total deposit: "
    f"₦{highest_deposit_team['Total_Deposit'] / 1e9:.2f} Billion"
)

print(
    "\nStrongest observed correlation: "
    "Performing Loans vs Total Deposit"
)

print(
    f"Correlation: "
    f"{df['Performing_Loans'].corr(df['Total_Deposit']):.3f}"
)

print(
    f"\nSelected forecasting model: "
    f"{selected_model}"
)

print(
    f"Forecast MAPE: "
    f"{selected_mape:.2f}%"
)


# ============================================================
# 24. BUSINESS RECOMMENDATIONS
# ============================================================

print("\n========== BUSINESS RECOMMENDATIONS ==========")

print(
    "\n1. Investigate the deposit performance gap, "
    "particularly for Marina."
)

print(
    "\n2. Identify successful practices from high-performing "
    "teams and evaluate opportunities to replicate them."
)

print(
    "\n3. Monitor performing loans alongside deposit growth "
    "as an important indicator of commercial banking activity."
)

print(
    "\n4. Use Linear Regression as the current baseline "
    "for short-term deposit forecasting."
)

print(
    "\n5. Expand the historical dataset before deploying "
    "the forecasting model for production decision-making."
)


# ============================================================
# END OF PROJECT
# ============================================================

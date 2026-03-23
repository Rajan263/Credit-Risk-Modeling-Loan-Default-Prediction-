
# CREDIT RISK ANALYSIS USING PYTHON 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("dataset.csv")
print("\n--- Dataset Preview ---")
print(data.head())

print(data.info())

print(data.describe())

data.fillna(data.mean(numeric_only=True), inplace=True)

plt.figure()
sns.countplot(x="LoanStatus", data=data)
plt.title("Loan Status Distribution (0 = Default, 1 = Paid)")
plt.show()
plt.figure()
sns.countplot(x="EmploymentStatus", hue="LoanStatus", data=data)
plt.title("Employment Status vs Loan Status")
plt.show()

plt.figure()
sns.barplot(x="LoanStatus", y="Income", data=data)
plt.title("Income vs Loan Status")
plt.show()

plt.figure()
sns.boxplot(x="LoanStatus", y="LoanAmount", data=data)
plt.title("Loan Amount vs Loan Status")
plt.show()


plt.figure()
sns.countplot(x="CreditHistory", hue="LoanStatus", data=data)
plt.title("Credit History vs Loan Status")
plt.show()


def risk_category(row):
    if row["CreditHistory"] == 0 and row["Income"] < 400000:
        return "High Risk"
    elif row["LoanAmount"] > (row["Income"] * 0.6):
        return "Medium Risk"
    else:
        return "Low Risk"

data["RiskCategory"] = data.apply(risk_category, axis=1)

plt.figure()
sns.countplot(x="RiskCategory", data=data)
plt.title("Customer Risk Category Distribution")
plt.show()

print("\n--- Risk Category Count ---")
print(data["RiskCategory"].value_counts())

print("\n--- Average Income by Risk Category ---")
print(data.groupby("RiskCategory")["Income"].mean())

print("\n--- Average Loan Amount by Risk Category ---")
print(data.groupby("RiskCategory")["LoanAmount"].mean())

print("\n--- Final Dataset with Risk Category ---")
print(data.head())

data.to_csv("credit_risk_final_output.csv", index=False)

print("\nAnalysis completed successfully!")
print("Final file saved as 'credit_risk_final_output.csv'")


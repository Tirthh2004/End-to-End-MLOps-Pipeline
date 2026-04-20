"""
Dataset: Telecom Customer Churn (Synthetic but Realistic)
---------------------------------------------------------
Generates a realistic churn dataset with 5000 rows and saves it as CSV.
Features mirror what real telecom companies use to predict churn.
Run this once: python data/generate_dataset.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

def generate_churn_dataset(n=N):
    tenure          = np.random.randint(1, 73, n)                     # months with company
    monthly_charges = np.round(np.random.uniform(20, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    total_charges   = np.clip(total_charges, 0, None)

    contract        = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                       p=[0.55, 0.25, 0.20])
    internet        = np.random.choice(["DSL", "Fiber optic", "No"], n,
                                       p=[0.34, 0.44, 0.22])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n,
        p=[0.34, 0.23, 0.22, 0.21]
    )
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    dependents      = np.random.choice([0, 1], n, p=[0.70, 0.30])
    tech_support    = np.random.choice([0, 1], n, p=[0.50, 0.50])
    online_security = np.random.choice([0, 1], n, p=[0.50, 0.50])
    num_services    = np.random.randint(1, 9, n)

    # Realistic churn probability based on known drivers
    churn_prob = (
        0.05
        + 0.30 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.15 * (payment_method == "Electronic check")
        + 0.08 * (senior_citizen == 1)
        - 0.15 * (tenure > 36)
        - 0.10 * (num_services > 4)
        + 0.05 * (monthly_charges > 80)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges,
        "senior_citizen":   senior_citizen,
        "dependents":       dependents,
        "tech_support":     tech_support,
        "online_security":  online_security,
        "num_services":     num_services,
        "contract":         contract,
        "internet_service": internet,
        "payment_method":   payment_method,
        "churn":            churn,
    })
    return df

if __name__ == "__main__":
    df = generate_churn_dataset()
    df.to_csv("data/churn.csv", index=False)
    print(f"Dataset saved → data/churn.csv")
    print(f"Shape        : {df.shape}")
    print(f"Churn rate   : {df['churn'].mean():.2%}")
    print(df.head(3).to_string())

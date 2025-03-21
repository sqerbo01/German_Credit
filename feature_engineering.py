import pandas as pd


# still in progress

def engineer_features(df):
    """
    Adds engineered features to the dataframe, focusing on financial strain, risk proxies, 
    and stability metrics. Handles categorical encoding where needed.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # --- 1. Encode Ordinal Features First (Critical for Derived Features) ---
    # Credit History (Risk Level: 0=worst, 2=best)

    credit_history_map = {
    'prior_payments_delayed': 0,      # Highest risk (history of late payments)
    'outstanding_credit': 1,          # Moderate risk (current unpaid debts)
    'no_credits': 2,                  # Neutral/unknown risk (no credit history)
    'credits_paid_to_date': 3,        # Low risk (current credits managed well)
    'all_credits_paid_back': 4        # Lowest risk (proven repayment ability)
}
    df['CreditHistoryRisk'] = df['CreditHistory'].map(credit_history_map)
    
    # ExistingSavings (Convert categories to numeric midpoints)
    savings_map = {
        'less_100': 50,
        '100_to_500': 300,
        '500_to_1000': 750,
        'greater_1000': 1500,   # Assumed upper bound
        'unknown': pd.NA
    }
    df['ExistingSavingsNumeric'] = df['ExistingSavings'].map(savings_map)
    
    # EmploymentDuration (Convert to approximate years)
    employment_duration_map = {
        'less_1': 0.5,
        '1_to_4': 2.5,
        '4_to_7': 5.5,
        'greater_7': 8.0
    }
    df['EmploymentDurationYears'] = df['EmploymentDuration'].map(employment_duration_map)
    
    # --- 2. Create New Features ---
    # Financial Strain
    df['loan_to_income_proxy'] = df['LoanAmount'] / (df['InstallmentPercent'] + 1e-5)
    df['debt_to_savings'] = df['LoanAmount'] / (df['ExistingSavingsNumeric'] + 1e-5)
    
    # Risk Interaction Terms
    df['loan_amount_x_credit_risk'] = df['LoanAmount'] * (3 - df['CreditHistoryRisk'])  # Higher = riskier
    
    # Stability Metrics
    df['employment_stability'] = df['EmploymentDurationYears'] / (df['LoanDuration'] / 12 + 1e-5)  # Years employement per year loan duration
    df['residence_to_employment_ratio'] = df['CurrentResidenceDuration'] / (df['EmploymentDurationYears'] + 1e-5)
    
    # Demographic Ratios
    df['dependents_per_age'] = df['Dependents'] / (df['Age'] + 1e-5)
    
    # --- 3. Binary Flags ---
    df['prior_default_flag'] = (df['CreditHistory'] == 'prior_payments_delayed').astype(int)
    
    return df
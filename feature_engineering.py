import pandas as pd
import numpy as np

# still in progress

def engineer_features(df):
    """
    Adds engineered features to the dataframe, focusing on financial strain, risk proxies, 
    and stability metrics. Handles categorical encoding where needed.
    """
    # Create a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # --- 1. Encode Ordinal Features First (Critical for Derived Features) ---

    # Add these to your existing engineer_features() function:

    # CheckingStatus - Higher values = better financial situation
    checking_status_map = {
        'no_checking': 0,
        'less_0': 1,
        '0_to_200': 2,
        'greater_200': 3
    }
    df['CheckingStatus_Ordinal'] = df['CheckingStatus'].map(checking_status_map)

    # InstallmentPlans - Higher values = more formal payment plans
    installment_plans_map = {
        'none': 0,
        'stores': 1,
        'bank': 2
    }
    df['InstallmentPlans_Ordinal'] = df['InstallmentPlans'].map(installment_plans_map)

    # Housing - Rough stability hierarchy
    housing_map = {
        'rent': 0,
        'free': 1,  # Free housing often less stable than owned
        'own': 2
    }
    df['Housing_Ordinal'] = df['Housing'].map(housing_map)

    # Job - Employment stability scale
    job_map = {
        'unemployed': 0,
        'unskilled': 1,
        'skilled': 2,
        'management_self-employed': 3
    }
    df['Job_Ordinal'] = df['Job'].map(job_map)

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
        'unknown': np.nan
    }
    df['ExistingSavingsNumeric'] = df['ExistingSavings'].map(savings_map)
    
    # EmploymentDuration (Convert to approximate years)
    employment_duration_map = {
        'unemployed': 0.0,
        'less_1': 0.5,
        '1_to_4': 2.5,
        '4_to_7': 5.5,
        'greater_7': 8.0
    }

    df['EmploymentDurationYears'] = df['EmploymentDuration'].map(employment_duration_map)
    
    # --- 2. Create New Features ---
    # Financial Strain
    df['loan_to_income_proxy'] = df['LoanAmount'] / (df['InstallmentPercent'] + 1e-5)
    df['debt_to_savings'] = np.where(
        df['ExistingSavingsNumeric'].isna(),  # If savings is unknown
        np.nan,  # Keep as NA
        df['LoanAmount'] / (df['ExistingSavingsNumeric'] + 1e-5)  # Else calculate ratio
    )

    # Risk Interaction Terms
    df['loan_amount_x_credit_risk'] = df['LoanAmount'] * (5 - df['CreditHistoryRisk'])  # Higher = riskier
    
    # Stability Metrics
    df['employment_stability'] = df['EmploymentDurationYears'] / (df['LoanDuration'] / 12 + 1e-5)  # Years employement per year loan duration
    df['residence_to_employment_ratio'] = df['CurrentResidenceDuration'] / (df['EmploymentDurationYears'] + 1e-5)
    
    # Demographic Ratios
    df['dependents_per_age'] = df['Dependents'] / (df['Age'] + 1e-5)
    
    # --- 3. Binary Flags ---
    df['prior_default_flag'] = (df['CreditHistory'] == 'prior_payments_delayed').astype(int)

    # Loan amount per age
    df['loan_amount_per_age'] = df['LoanAmount'] / (df['Age'] + 1e-5)

    # Residence duration per age
    df['residence_duration_per_age'] = df['CurrentResidenceDuration'] / (df['Age'] + 1e-5)

    # Employment years per age
    df['employment_years_per_age'] = df['EmploymentDurationYears'] / (df['Age'] + 1e-5)

    # Flag for large loans (threshold can be adjusted)
    large_loan_threshold = 3000
    df['is_large_loan'] = (df['LoanAmount'] > large_loan_threshold).astype(int)

    df['LoanTier'] = pd.cut(df['LoanAmount'], bins=[0, 3000, 6000, np.inf], labels=['small', 'medium', 'large'])

    if 'Risk' in df.columns:
        df['Risk_Numeric'] = df['Risk'].map({'Risk': 1, 'No Risk': 0})


    return df


def feature_selection(df):
    """
    Selects a subset of features to use in the model, dropping:
    - Original categorical columns that were encoded
    - Low-importance engineered features
    - Redundant features
    - Direct identifiers of the target
    """
    # Columns to KEEP (original numerical + engineered features)
    selected_columns = [
        # Original numerical features
        'LoanDuration',
        'LoanAmount',
        'InstallmentPercent',
        'CurrentResidenceDuration',
        'Age',
        'ExistingCreditsCount',
        'Dependents',

        # Original categorical features
        "LoanPurpose",
        "Sex",
        "OthersOnLoan", 
        "OwnsProperty",
        "Telephone",
        "ForeignWorker",
        
        # Engineered numerical features
        "CheckingStatus_Ordinal",
        "InstallmentPlans_Ordinal",
        "Housing_Ordinal",
        "Job_Ordinal",
        'CreditHistoryRisk',
        'ExistingSavingsNumeric',
        'EmploymentDurationYears',
        'loan_to_income_proxy',
        'debt_to_savings',
        'loan_amount_x_credit_risk',
        'employment_stability',
        'dependents_per_age',
        'prior_default_flag',
        'loan_amount_per_age',
        'is_large_loan',
        'savings_missing',
        
        # Stratification/target columns (keep temporarily)
        'LoanTier',
        'Risk_Numeric'
    ]
    
    # Columns to DROP
    drop_columns = [
        # Original categorical columns (now encoded numerically)
        'CheckingStatus',
        'CreditHistory',
        'ExistingSavings',
        'EmploymentDuration',
        'InstallmentPlans',
        'Housing',
        'Job',
        
        # Redundant engineered features
        'residence_to_employment_ratio',  # Covered by employment_stability
        'residence_duration_per_age',     # Similar to loan_amount_per_age
        'employment_years_per_age',       # Similar to dependents_per_age
        
        # Target column (keep the numeric version)
        'Risk'
    ]
    
    # Select and return the final features
    final_columns = [col for col in selected_columns if col in df.columns]
    return df[final_columns]


def preprocessing(df, is_test=False):
    """Handles both training and test sets"""
    cat_cols = ['LoanPurpose', 'Sex', 'OthersOnLoan', 'OwnsProperty', 'Telephone', 'ForeignWorker']
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Handle Missing Values
    df['ExistingSavingsNumeric'] = df['ExistingSavingsNumeric'].fillna(-1)
    df['debt_to_savings'] = df['debt_to_savings'].fillna(-1)

    # Define Features
    FEATURES = [col for col in df.columns if col not in ['Risk_Numeric', 'LoanTier']]
    
    if is_test:
        # Test set won't have target
        return df[FEATURES], df['LoanAmount'] # X, amount
    else:
        # Training set has target
        return df[FEATURES], df['Risk_Numeric'], df['LoanAmount'] # X, y, amount
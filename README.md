import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Set random seed for reproducibility
np.random.seed(42)

# Number of data points
num_samples = 1500

# Generate synthetic data for features
features = {
    'CreditScore': np.random.randint(300, 850, size=num_samples),
    'AnnualIncome': np.random.randint(20000, 150000, size=num_samples),
    'LoanAmount': np.random.randint(5000, 200000, size=num_samples),
    'YearsEmployed': np.random.randint(0, 40, size=num_samples),
    'Age': np.random.randint(18, 80, size=num_samples)
}

# Generate synthetic target variable (CreditRisk) based on features
weights = np.array([0.3, 0.00003, 0.0001, -0.05, -0.1])
bias = 0.5

X = pd.DataFrame(features)
y = X.dot(weights) + bias + np.random.normal(0, 0.1, size=num_samples)
y = (y > np.median(y)).astype(int)  # Convert to binary outcome for classification (0 = Low risk, 1 = High risk)

# Combine features and target into a single DataFrame
data = pd.concat([X, pd.Series(y, name='CreditRisk')], axis=1)

# Scale features using Min-Max normalization
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['CreditRisk']))
scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
scaled_data['CreditRisk'] = data['CreditRisk']

# Save the dataset to an Excel file
scaled_data.to_excel('credit_risk_data.xlsx', index=False)



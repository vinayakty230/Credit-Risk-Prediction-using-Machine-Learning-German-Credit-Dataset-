
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("german_credit_data.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
df.drop(columns=['Unnamed:_0'], inplace=True)

# Handle missing values
df['Saving_accounts'].fillna('no_info', inplace=True)
df['Checking_account'].fillna('no_info', inplace=True)

# Encode categorical features
categorical_cols = ['Sex', 'Housing', 'Saving_accounts', 'Checking_account', 'Purpose']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Target variable
df['Risk'] = df['Credit_amount'].apply(lambda x: 1 if x >= 2000 else 0)

# Prepare features and scale
X = df.drop(columns=['Credit_amount', 'Risk'])
y = df['Risk']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save processed objects
import joblib
joblib.dump(scaler, "credit_scaler.pkl")
joblib.dump(label_encoders, "credit_label_encoders.pkl")
pd.DataFrame(X_scaled).to_csv("X_scaled.csv", index=False)
pd.DataFrame(y).to_csv("y.csv", index=False)

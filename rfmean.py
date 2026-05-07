import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("medical_students_dataset.csv")

target = "Smoking"
sample_size = 20000
random_state = 42

# Fjern rækker hvor target mangler
df = df.dropna(subset=[target])

# Tag sample på 20.000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=random_state)

X = df.drop(columns=[target])
y = df[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=random_state,
    stratify=y_encoded
)

categorical_cols = X_train.select_dtypes(include=["object"]).columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
    encoded_missing_value=np.nan
)

X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

X_train_encoded[categorical_cols] = encoder.fit_transform(
    X_train_encoded[categorical_cols]
)

X_test_encoded[categorical_cols] = encoder.transform(
    X_test_encoded[categorical_cols]
)

mean_imputer = SimpleImputer(strategy="mean")

X_train_imputed = mean_imputer.fit_transform(X_train_encoded)
X_test_imputed = mean_imputer.transform(X_test_encoded)

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=random_state
)

rf.fit(X_train_imputed, y_train)

y_pred = rf.predict(X_test_imputed)
accuracy = accuracy_score(y_test, y_pred)

print("Method: mean imputation")
print(f"Rows used: {len(df)}")
print(f"Sample size: {sample_size}")
print(f"Accuracy: {accuracy:.4f}")
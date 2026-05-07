import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("medical_students_dataset.csv")

target = "Smoking"
sample_size = 20000
random_state = 42

# Fjern rækker hvor target mangler
df = df.dropna(subset=[target])

# Tag sample på 20.000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=random_state)

# Drop alle resterende rækker med missing values
df_dropna = df.dropna()

X = df_dropna.drop(columns=[target])
y = df_dropna[target]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

categorical_cols = X.select_dtypes(include=["object"]).columns
numeric_cols = X.select_dtypes(exclude=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=random_state,
    stratify=y_encoded
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Method: dropna")
print(f"Initial sample size: {sample_size}")
print(f"Rows after dropna: {len(df_dropna)}")
print(f"Accuracy: {accuracy:.4f}")
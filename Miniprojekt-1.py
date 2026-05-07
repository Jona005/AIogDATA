import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import KNNImputer

# Indlæs data
df = pd.read_csv('/Users/gustavbrondum/Desktop/DAKI/AI og Data/Miniprojekt Ai og Data/medical_students_dataset.csv')

# OMDØB KOLONNER HER (så de matcher din SQL-kode)
df.columns = [col.replace(' ', '_') for col in df.columns]

# Identificer typer af datamangel

df.info()
df.describe()
df.isna().sum()
print("procentdel")
df.isna().mean()*100
import matplotlib.pyplot as plt
import missingno as msno

msno.matrix(df)
msno.heatmap(df)

plt.show()




# Load dataset

# Vælg antal samples
sample_size = 20000
random_state = 42

# Tag sample først, så original, mean og KNN bruger præcis samme rækker
if len(df) > sample_size:
    df_sample = df.sample(n=sample_size, random_state=random_state)
else:
    df_sample = df.copy()

# Kopi af sample
df_encoded = df_sample.copy()

# Konverter kategoriske kolonner til numeriske koder
for col in df_encoded.columns:
    if not pd.api.types.is_numeric_dtype(df_encoded[col]):
        df_encoded[col] = df_encoded[col].astype("category").cat.codes

        # cat.codes laver NaN om til -1, så vi sætter dem tilbage til NaN
        df_encoded.loc[df_sample[col].isna(), col] = np.nan

# -----------------------------
# Mean imputation på sample
# -----------------------------
df_mean_imputed = df_encoded.copy()

for col in df_mean_imputed.columns:
    mean_value = df_mean_imputed[col].mean()
    df_mean_imputed[col] = df_mean_imputed[col].fillna(mean_value)

# -----------------------------
# KNN imputation på samme sample
# -----------------------------
knn_imputer = KNNImputer(n_neighbors=5)

df_knn_imputed = pd.DataFrame(
    knn_imputer.fit_transform(df_encoded),
    columns=df_encoded.columns,
    index=df_encoded.index
)

# -----------------------------
# Plot alle features
# -----------------------------
for feature in df_encoded.columns:
    original = df_encoded[feature]
    mean_imputed = df_mean_imputed[feature]
    knn_imputed = df_knn_imputed[feature]

    print(f"Feature: {feature}")
    print(f"Sample size: {len(df_sample)}")
    print(f"Original observations without missing values: {len(original)}")
    print(f"Missing values before imputation: {df_encoded[feature].isna().sum()}")
    print(f"Mean used for imputation: {df_encoded[feature].mean()}")
    print("-" * 50)

    combined_data = pd.concat([
        original,
        mean_imputed,
        knn_imputed
    ])

    # Hvis kolonnen kun har én unik værdi, undgå fejl i bins
    if combined_data.nunique() <= 1:
        bins = 5
    else:
        bins = np.linspace(
            combined_data.min(),
            combined_data.max(),
            25
        )

    fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Original sample
    axs[0].hist(
        original,
        bins=bins,
        edgecolor="white",
        color="#5DA5DA"
    )
    axs[0].set_title(f"Original\nn={len(original)} obs.")
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel("Number of observations")

    # Mean imputation sample
    axs[1].hist(
        mean_imputed,
        bins=bins,
        edgecolor="white",
        color="#60BD68"
    )
    axs[1].set_title(f"After mean imputation\nn={len(mean_imputed)} obs.")
    axs[1].set_xlabel(feature)

    # KNN imputation sample
    axs[2].hist(
        knn_imputed,
        bins=bins,
        edgecolor="white",
        color="#F17C3A"
    )
    axs[2].set_title(f"After KNN imputation\nn={len(knn_imputed)} obs.")
    axs[2].set_xlabel(feature)

    plt.tight_layout()
    plt.show()

# visuelisering af missing data på alle features

# =================================================================
# DATABASE SEKTION: OPPRETTELSE AF 2 SEPARATE DATABASER
# =================================================================
import sqlite3

# --- DATABASE 1: Administrative Data (student_admin.db) ---
# Denne database indeholder stamdata om den studerende
conn_admin = sqlite3.connect('student_admin.db')

admin_df = df_knn_imputed[['Student_ID', 'Age', 'Gender', 'Blood_Type']]
admin_df.to_sql('Admin_Info', conn_admin, index=False, if_exists='replace')

print("Database 1 (student_admin.db) oprettet med tabellen 'Admin_Info'.")
conn_admin.close()


# --- DATABASE 2: Medicinske Data (medical_research.db) ---
# Denne database indeholder de kliniske målinger og resultater
conn_research = sqlite3.connect('medical_research.db')

research_df = df_knn_imputed[['Student_ID', 'Height', 'Weight', 'BMI', 'Temperature', 
                             'Heart_Rate', 'Blood_Pressure', 'Cholesterol', 
                             'Diabetes', 'Smoking']]
research_df.to_sql('Research_Data', conn_research, index=False, if_exists='replace')

print("Database 2 (medical_research.db) oprettet med tabellen 'Research_Data'.")
conn_research.close()


# --- EKSEMPEL PÅ UDTRÆK FRA EN AF DATABASERNE ---
# Vi åbner research-databasen for at tjekke at data er gemt korrekt
conn = sqlite3.connect('medical_research.db')
query = "SELECT Student_ID, BMI, Heart_Rate, Smoking FROM Research_Data WHERE BMI > 25 LIMIT 5"
df_check = pd.read_sql(query, conn)
conn.close()

print("\nTjek af data fra medical_research.db (Studerende med BMI > 25):")
print(df_check)
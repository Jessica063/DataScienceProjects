import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Load the data
df = pd.read_csv("input.csv")

# Step 2: Identify numerical and categorical columns
num_cols = ["Age", "Salary"]
cat_cols = ["Department"]

# Step 3: Handle Missing Values
num_imputer = SimpleImputer(strategy="mean")  # Replace missing numbers with mean
cat_imputer = SimpleImputer(strategy="most_frequent")  # Replace missing categories with mode

# Step 4: Encoding and Scaling
encoder = OneHotEncoder(handle_unknown='ignore')  # Convert categorical to one-hot encoding
scaler = StandardScaler()  # Standardize numerical features

# Step 5: Create a preprocessing pipeline
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler)
    ]), num_cols),
    
    ("cat", Pipeline([
        ("imputer", cat_imputer),
        ("encoder", encoder)
    ]), cat_cols)
])

# Step 6: Apply transformations
transformed_data = preprocessor.fit_transform(df)

# Step 7: Convert processed data back to DataFrame
processed_df = pd.DataFrame(transformed_data)

# Step 8: Save the transformed data
processed_df.to_csv("output.csv", index=False)

print("Data Preprocessing and Transformation Completed Successfully!")

df1 = pd.read_csv("output.csv")
print(df1.head())
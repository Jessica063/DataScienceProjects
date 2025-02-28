# Data Preprocessing, Transformation, and Loading Pipeline  

## Project Description  
This project demonstrates an end-to-end **data preprocessing pipeline** using **Pandas** and **Scikit-Learn**. It focuses on handling missing values, encoding categorical features, scaling numerical features, and transforming raw data into a clean format suitable for machine learning models.  

## Features Implemented  
- **Handling Missing Values**:  
  - Numerical columns: Replaced missing values with the mean.  
  - Categorical columns: Filled missing values with the most frequent category.  
- **Feature Encoding**:  
  - Applied **One-Hot Encoding** to categorical variables.  
- **Feature Scaling**:  
  - Standardized numerical columns using **StandardScaler**.  
- **Pipeline Integration**:  
  - Used **Scikit-Learn Pipelines** and **ColumnTransformer** to streamline the transformation process.  
- **Data Export**:  
  - Processed data is saved as `output.csv` for further analysis.  

## Technologies Used  
- Python  
- Pandas  
- Scikit-Learn  
- NumPy  

## Output
- The script reads input.csv, processes it, and saves the transformed version as output.csv.
- The first few rows of the transformed data are displayed in the console.

import pandas as pd
import numpy as np

# --- 1. SETUP: Create a Dummy CSV File if it doesn't exist ---
# This ensures the script runs even if you don't have 'data.csv' yet.
file_name = "01_sica_evolucao-mensal-das-consultas-medicas-hospitalares.csv"
try:
    df = pd.read_csv(file_name, sep=',')
    print("Loaded existing 'data.csv'.")
except FileNotFoundError:
    print("No 'data.csv' found. Creating a dummy DataFrame to proceed.")
    
    # Define placeholder data
    data = {
        'ID': [101, 102, 103, 104, 105, 106, 107],
        'Feature_A': [15.5, 22.1, 19.0, 11.2, 18.8, 25.0, 14.7],
        'Feature_B': ['Red', 'Blue', 'Green', 'Red', 'Blue', np.nan, 'Green'],
        'Notes_Column': ['OK', 'Skip', 'Good', 'OK', 'Skip', 'Good', 'OK'],
        'Target_Value': [5, np.nan, 8, 5, 9, 7, 6]
    }
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    # Save the dummy data so you can inspect the file later
    df.to_csv('data.csv', index=False)
    print("Dummy 'data.csv' created and loaded.")


# --- 2. EXPLORATION & INSPECTION ---
print("\n--- DataFrame Inspection ---")
# Display the first 5 rows
print("Head (First 5 Rows):\n", df.head())

# Get summary information (data types, non-null counts, memory usage)
print("\nInfo (Data Types & Null Counts):")
df.info()

# Get statistical summary for numerical columns
print("\nDescribe (Statistics):\n", df.describe())

# Check the shape (rows, columns)
print(f"\nDataFrame Shape (Rows, Columns): {df.shape}")


# --- 3. POLISHING & CLEANING (Your key tasks) ---
print("\n--- Data Polishing ---")

# Task A: Deleting unnecessary columns
# We'll drop the 'Notes_Column' because it doesn't seem useful for modeling.
# Use 'axis=1' to specify column, and 'inplace=True' to modify the df directly.
df.drop(columns=['Periodo_format_2'], inplace=True)
print(f"1. Dropped 'Grupo Hospitalar'. New columns: {list(df.columns)}")

# Task B: Handling Missing/Unnecessary Data (Rows)
# We check for the count of missing values (NaN) per column
print("\nMissing values before cleaning:\n", df.isnull().sum())

# Option 1 (Simple): Drop rows where ANY value is missing.
# df_cleaned = df.dropna(inplace=False) 
# print(f"2a. Dropped all rows with NaNs. New Shape: {df_cleaned.shape}")

# Option 2 (Recommended for this base): Fill missing numerical data (Target_Value) with the column mean.
# This keeps the rows that only have a missing value in the Target column (like row 102 in the dummy data).
# mean_target = df['Target_Value'].mean()
# df['Target_Value'].fillna(mean_target, inplace=True)
# print("2b. Filled 'Target_Value' NaNs with the mean.")

# # Task C: Filtering Data (Removing rows based on a condition)
# # Example: Only keep rows where 'Feature_A' is greater than 15.0
# df = df[df['Feature_A'] > 15.0]
# print(f"3. Filtered rows where Feature_A > 15.0. New Shape: {df.shape}")


# --- 4. FINAL OUTPUT & SAVE ---
print("\n--- Final Cleaned DataFrame ---")
print(df)

# Save the polished data to a new CSV file
output_file_name = file_name
df.to_csv(output_file_name, index=False)
print(f"\nSuccessfully saved the cleaned data to '{output_file_name}'.")
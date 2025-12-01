import pandas as pd
import glob
import os

# 1. LOAD DATA (Robust Load)
print("--- Loading Data ---")
try:
    df_fem = pd.read_csv('7210_1.csv', low_memory=False, on_bad_lines='skip')
    print(f"Women's Data: {len(df_fem)} rows")
except:
    print("Error loading Women's data.")
    df_fem = pd.DataFrame()

try:
    # Check for file in folder or root
    men_files = glob.glob('./mens_data/*.csv')
    if men_files:
        df_mal = pd.read_csv(men_files[0], low_memory=False, on_bad_lines='skip')
    elif os.path.exists('mens_data.csv'):
        df_mal = pd.read_csv('mens_data.csv', low_memory=False, on_bad_lines='skip')
    else:
        df_mal = pd.DataFrame()
    print(f"Men's Data: {len(df_mal)} rows")
except:
    print("Error loading Men's data.")
    df_mal = pd.DataFrame()

# 2. ANALYSIS FUNCTION
def analyze_brands(df, dataset_name):
    if df.empty: return

    # Force lowercase string
    brands = df['brand'].astype(str).str.lower().str.strip()

    # Filter for 'a' and 'n'
    brands_a = brands[brands.str.startswith('a')]
    brands_n = brands[brands.str.startswith('n')]

    print(f"\n=== {dataset_name}: Brands starting with 'A' ===")
    print(brands_a.value_counts().head(20).to_string())

    print(f"\n=== {dataset_name}: Brands starting with 'N' ===")
    print(brands_n.value_counts().head(20).to_string())

# 3. RUN ANALYSIS
analyze_brands(df_fem, "WOMEN'S DATA")
analyze_brands(df_mal, "MEN'S DATA")

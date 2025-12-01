import pandas as pd
import numpy as np
from scipy import stats
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import requests
import glob
import os

# ==========================================
# 1. ROBUST DATA LOADING
# ==========================================
print("--- 1. Loading Data ---")

# Women's Data
try:
    df_fem = pd.read_csv('7210_1.csv', low_memory=False, on_bad_lines='skip')
    print(f"✅ Women's Data Loaded: {len(df_fem)} rows")
except:
    print("❌ Error: Women's CSV not found.")
    df_fem = pd.DataFrame()

# Men's Data
try:
    men_files = glob.glob('./mens_data/*.csv')
    if men_files:
        df_mal = pd.read_csv(men_files[0], low_memory=False, on_bad_lines='skip')
    elif os.path.exists('mens_data.csv'):
        df_mal = pd.read_csv('mens_data.csv', low_memory=False, on_bad_lines='skip')
    else:
        df_mal = pd.DataFrame()
    print(f"✅ Men's Data Loaded: {len(df_mal)} rows")
except:
    print("❌ Error loading Men's data.")
    df_mal = pd.DataFrame()

# ==========================================
# 2. ADVANCED CLEANING & MAPPING
# ==========================================
print("\n--- 2. Cleaning & Normalizing ---")

# Live Currency
rates = {'CAD': 1.35, 'AUD': 1.50, 'EUR': 0.92, 'GBP': 0.79}
try:
    r = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()['rates']
    rates.update(r)
except: pass

def clean_dataset(df):
    if df.empty: return df

    # A. Currency
    def convert_price(row):
        curr = str(row['prices.currency']).upper()
        try: price = float(row['prices.amountMin'])
        except: return np.nan

        if curr != 'USD' and curr in rates:
            return price / rates[curr]
        return price

    df['price_usd'] = df.apply(convert_price, axis=1)
    df = df.dropna(subset=['price_usd'])

    # B. Deep Brand Imputation
    def get_brand(row):
        # 1. Check existing brand column
        if pd.notna(row['brand']) and str(row['brand']).lower() != 'nan':
            b = str(row['brand']).lower()
            if 'nike' in b: return 'nike'
            if 'adidas' in b or 'adi' in b: return 'adidas'

        # 2. Check Name column
        if pd.notna(row['name']):
            n = str(row['name']).lower()
            if 'nike' in n: return 'nike'
            if 'adidas' in n: return 'adidas'

        return 'other'

    df['brand_final'] = df.apply(get_brand, axis=1)

    # C. Name Cleaning (Remove "Men's", "Women's" for better matching)
    def clean_name(name):
        n = str(name).lower()
        n = n.replace("women's", "").replace("men's", "")
        n = n.replace("womens", "").replace("mens", "")
        return n.strip()

    df['name_clean'] = df['name'].apply(clean_name)

    return df

df_fem = clean_dataset(df_fem)
df_mal = clean_dataset(df_mal)

# Filter Target Brands
target_brands = ['nike', 'adidas']
df_fem = df_fem[df_fem['brand_final'].isin(target_brands)].copy()
df_mal = df_mal[df_mal['brand_final'].isin(target_brands)].copy()

print(f"Final Count -> Women: {len(df_fem)}, Men: {len(df_mal)}")

# ==========================================
# 3. FUZZY MATCHING (Pink Tax Linkage)
# ==========================================
print("\n--- 3. Matching Shoes (Threshold: 80%) ---")

matches = []

for brand in target_brands:
    fem_subset = df_fem[df_fem['brand_final'] == brand]
    mal_subset = df_mal[df_mal['brand_final'] == brand]

    # Lookup: {clean_name: price}
    # We use groupby to handle duplicates (take mean price of duplicates)
    mal_lookup = mal_subset.groupby('name_clean')['price_usd'].mean().to_dict()
    mal_names = list(mal_lookup.keys())

    if not mal_names: continue

    for _, row in fem_subset.iterrows():
        fem_name = row['name_clean']

        # Match
        match, score = process.extractOne(fem_name, mal_names, scorer=fuzz.token_sort_ratio)

        # Lowered threshold to 80 to catch more variations
        if score >= 80:
            men_price = mal_lookup[match]
            women_price = row['price_usd']
            pink_tax = women_price - men_price

            matches.append({
                'brand': brand,
                'women_shoe': fem_name,
                'men_shoe': match,
                'women_price': women_price,
                'men_price': men_price,
                'pink_tax': pink_tax,
                'score': score
            })

df_pink = pd.DataFrame(matches)
print(f"✅ Matched Pairs Found: {len(df_pink)}")

# ==========================================
# 4. STATISTICAL ANALYSIS
# ==========================================
if len(df_pink) > 5:
    print("\n--- 4. Results ---")
    print(df_pink[['brand', 'women_shoe', 'men_shoe', 'pink_tax']].head())

    # Test 1: Is Pink Tax > 0?
    t_stat, p_val = stats.ttest_1samp(df_pink['pink_tax'], 0, alternative='greater')
    print(f"\nOverall Mean Pink Tax: ${df_pink['pink_tax'].mean():.2f}")
    print(f"P-Value (Is it > $0?): {p_val:.5f}")

    # Test 2: Nike vs Adidas
    nike_pt = df_pink[df_pink['brand']=='nike']['pink_tax']
    adi_pt = df_pink[df_pink['brand']=='adidas']['pink_tax']

    print(f"\nNike Avg Pink Tax:   ${nike_pt.mean():.2f} (n={len(nike_pt)})")
    print(f"Adidas Avg Pink Tax: ${adi_pt.mean():.2f} (n={len(adi_pt)})")

    if len(nike_pt) > 1 and len(adi_pt) > 1:
        t2, p2 = stats.ttest_ind(nike_pt, adi_pt, equal_var=False)
        print(f"Comparison P-Value: {p2:.5f}")
        if p2 < 0.05:
            print("Verdict: Significant difference between brands.")
        else:
            print("Verdict: No significant difference between brands.")
else:
    print("⚠️ Still not enough matches for statistics.")

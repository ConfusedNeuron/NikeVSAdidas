import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport

# 1. LOAD DATA
# Replace with your actual file path
#url = "https://www.kaggle.com/datasets/datafiniti/womens-shoes-prices/download"
# Assuming you have the file locally, e.g., '7210_1.csv' is a common name for this dataset
df = pd.read_csv('7210_1.csv', low_memory=False)
print(df.describe())

profile = ProfileReport(df, title="Women's Shoe Data Profile", minimal=True)

# DISABLE word clouds for Text variables (This is the one you were missing)
profile.config.vars.text.words = False

# Keep this one just in case other columns are seen as Categorical
profile.config.vars.cat.words = False

profile.to_file("womens_shoe_report.html")
# ---------------------------------------------------------
# STEP 1: DATA CLEANING (The "Precursor")
# ---------------------------------------------------------

# Normalize the brand names to lowercase to catch "nike" vs "Nike"
df['brand_clean'] = df['brand'].str.lower().str.strip()

# Fix the Adidas variants as per your project instructions
# We map all weird variations to a single standard 'adidas'
adidas_variants = ['adi', 'addidas', 'addidas outdoor', 'adidas']
df.loc[df['brand_clean'].isin(adidas_variants), 'brand_clean'] = 'adidas'

# Filter only for the two brands we care about
df_filtered = df[df['brand_clean'].isin(['nike', 'adidas'])].copy()

# Ensure 'prices.amountMin' is numeric (sometimes currency symbols mess this up)
# Note: This dataset sometimes has multiple price columns, usually 'prices.amountMin' is best
df_filtered['price'] = pd.to_numeric(df_filtered['prices.amountMin'], errors='coerce')
df_filtered = df_filtered.dropna(subset=['price']) # Drop rows with no price

# Separate into two groups
nike_prices = df_filtered[df_filtered['brand_clean'] == 'nike']['price']
adidas_prices = df_filtered[df_filtered['brand_clean'] == 'adidas']['price']

print(f"Sample Sizes -> Nike: {len(nike_prices)}, Adidas: {len(adidas_prices)}")

# ---------------------------------------------------------
# STEP 2: DIAGNOSTIC TESTS (Checking Assumptions)
# ---------------------------------------------------------

print("\n--- Diagnostic Tests ---")

# Visual Check
plt.figure(figsize=(10, 5))
sns.histplot(df_filtered, x='price', hue='brand_clean', kde=True)
plt.title("Price Distribution: Nike vs Adidas")
plt.show()

# Statistical Check for Normality (Shapiro-Wilk)
# Note: Shapiro can be sensitive with large data; Visuals are often better.
shapiro_nike = stats.shapiro(nike_prices)
shapiro_adidas = stats.shapiro(adidas_prices)

print(f"Nike Normality p-value: {shapiro_nike.pvalue:.5f}")
print(f"Adidas Normality p-value: {shapiro_adidas.pvalue:.5f}")

# Check for Equal Variance (Levene's Test)
levene_test = stats.levene(nike_prices, adidas_prices)
print(f"Equal Variance (Levene) p-value: {levene_test.pvalue:.5f}")

# ---------------------------------------------------------
# STEP 3: RESEARCH HYPOTHESIS TEST
# ---------------------------------------------------------

print("\n--- Research Hypothesis Result ---")

# LOGIC: Choose the test based on diagnostics
alpha = 0.05

if shapiro_nike.pvalue > alpha and shapiro_adidas.pvalue > alpha:
    print("Decision: Data looks Normal. Using Independent T-Test.")
    # equal_var=False performs Welch's T-test (safer if Levene failed)
    stat, p_val = stats.ttest_ind(nike_prices, adidas_prices, equal_var=(levene_test.pvalue > alpha))
    test_name = "T-Test"
else:
    print("Decision: Data is NOT Normal (likely skewed). Using Mann-Whitney U Test.")
    stat, p_val = stats.mannwhitneyu(nike_prices, adidas_prices, alternative='two-sided')
    test_name = "Mann-Whitney U"

print(f"\n{test_name} p-value: {p_val:.10f}")

if p_val < alpha:
    print("CONCLUSION: REJECT H0.")
    print("There IS a statistically significant difference in price.")
    # check which mean/median is higher to say WHO is more expensive
    if nike_prices.median() > adidas_prices.median():
        print("-> Nike is more expensive.")
    else:
        print("-> Adidas is more expensive.")
else:
    print("CONCLUSION: FAIL TO REJECT H0.")
    print("We cannot say there is a difference in price.")

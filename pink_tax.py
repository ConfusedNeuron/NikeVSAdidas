from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from rapidfuzz import process, fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
import statsmodels.stats.api as sms

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# -------------------------
# Configurable utilities
# -------------------------
DEFAULT_RATE_TABLE = {'CAD': 1.35, 'AUD': 1.50, 'EUR': 0.92, 'GBP': 0.79, 'USD': 1.0}

def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Load CSV with defensive settings and minimal memory usage."""
    if not path.exists():
        logger.warning("File %s not found. Returning empty DataFrame.", path)
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False, on_bad_lines='skip', **kwargs)
    logger.info("Loaded %s (%d rows, %d cols)", path.name, df.shape[0], df.shape[1])
    return df

# -------------------------
# Cleaning & recoding
# -------------------------
def standardize_brand_column(df: pd.DataFrame, brand_col: str = 'brand', name_col: str = 'name') -> pd.DataFrame:
    """Create brand_final using vectorized rules."""
    df = df.copy()
    df['brand_src'] = df.get(brand_col, '').fillna('').astype(str).str.lower()
    df['name_src'] = df.get(name_col, '').fillna('').astype(str).str.lower()

    df['brand_final'] = 'other'
    mask_nike = df['brand_src'].str.contains(r'\bnike\b', na=False) | df['name_src'].str.contains(r'\bnike\b', na=False)
    mask_adidas = df['brand_src'].str.contains(r'\badidas\b|^adi\b', na=False) | df['name_src'].str.contains(r'\badidas\b|^adi\b', na=False)

    df.loc[mask_nike, 'brand_final'] = 'nike'
    df.loc[mask_adidas, 'brand_final'] = 'adidas'

    return df

def extract_price_usd(df: pd.DataFrame, price_col: str = 'prices.amountMin',
                      currency_col: str = 'prices.currency', rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Vectorized conversion to USD-based price_usd."""
    df = df.copy()
    if rates is None:
        rates = DEFAULT_RATE_TABLE

    if price_col not in df.columns:
        raise KeyError(f"{price_col} not found in dataframe")

    df['price_raw'] = pd.to_numeric(df[price_col], errors='coerce')
    df[currency_col] = df.get(currency_col, 'USD').fillna('USD').astype(str).str.upper()

    df['rate'] = df[currency_col].map(rates).astype(float)
    df['price_usd'] = df['price_raw'] / df['rate']

    pre_drop = len(df)
    df = df.dropna(subset=['price_usd'])
    logger.info("Converted prices to USD and dropped %d rows with missing prices", pre_drop - len(df))
    return df

def clean_name_col(df: pd.DataFrame, name_col: str = 'name') -> pd.DataFrame:
    df = df.copy()
    df['name_clean'] = df.get(name_col, '').fillna('').astype(str).str.lower()

    patterns = ["women's", "women", "womens", "women’s", "men's", "mens", "men", "men’s"]
    for p in patterns:
        df['name_clean'] = df['name_clean'].str.replace(p, '', regex=False)

    df['name_clean'] = df['name_clean'].str.replace(r'\s+', ' ', regex=True).str.strip()
    return df

# -------------------------
# Fuzzy matching
# -------------------------
def build_match_index(df_f: pd.DataFrame, df_m: pd.DataFrame,
                      f_name_col: str = 'name_clean', m_name_col: str = 'name_clean',
                      scorer=fuzz.token_sort_ratio, cutoff: int = 85, top_n: int = 1) -> pd.DataFrame:
    logger.info("Building fuzzy match index: cutoff=%s", cutoff)
    m_choices = df_m[m_name_col].fillna('').astype(str).tolist()
    m_indices = df_m.index.tolist()

    matches = []
    for i, fname in zip(df_f.index, df_f[f_name_col].fillna('').astype(str)):
        if not fname:
            continue
        res = process.extract(fname, m_choices, scorer=scorer, limit=top_n)
        for (matched_name, score, pos) in res:
            if score >= cutoff:
                m_idx = m_indices[pos]
                matches.append({
                    'f_index': i,
                    'm_index': m_idx,
                    'f_name': fname,
                    'm_name': matched_name,
                    'score': score
                })

    match_df = pd.DataFrame(matches)
    logger.info("Found %d candidate matches (score >= %d)", len(match_df), cutoff)
    return match_df

# -------------------------
# Pink-tax difference dataset
# -------------------------
def construct_pink_tax_df(df_f: pd.DataFrame, df_m: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in matches_df.iterrows():
        f_idx = row['f_index']
        m_idx = row['m_index']
        pf = df_f.at[f_idx, 'price_usd']
        pm = df_m.at[m_idx, 'price_usd']
        brand = df_f.at[f_idx, 'brand_final']

        rows.append({
            'f_index': f_idx,
            'm_index': m_idx,
            'brand': brand,
            'price_f': pf,
            'price_m': pm,
            'diff': pf - pm,
            'score': row['score']
        })
    return pd.DataFrame(rows)

# -------------------------
# Transformations
# -------------------------
def try_boxcox(series: pd.Series) -> Tuple[np.ndarray, float]:
    arr = series.dropna().astype(float).values
    if np.any(arr <= 0):
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        transformed = pt.fit_transform(arr.reshape(-1, 1)).flatten()
        logger.info("Used Yeo-Johnson transform (non-positive values present)")
        return transformed, np.nan
    else:
        transformed, fitted_lambda = stats.boxcox(arr)
        logger.info("Box-Cox lambda = %.4f", fitted_lambda)
        return transformed, float(fitted_lambda)

# -------------------------
# Hypothesis tests
# -------------------------
def mann_whitney_test(group1: pd.Series, group2: pd.Series) -> dict:
    g1 = group1.dropna().astype(float)
    g2 = group2.dropna().astype(float)

    u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    n1, n2 = len(g1), len(g2)

    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    return {
        'U': float(u_stat),
        'p': float(p_val),
        'n1': n1,
        'n2': n2,
        'rank_biserial': float(r_rb)
    }

def paired_test(diff_series: pd.Series) -> dict:
    diffs = diff_series.dropna()

    # Shapiro test on sample (large N)
    try:
        stat_shapiro = stats.shapiro(diffs.sample(n=min(5000, len(diffs))))
    except Exception:
        stat_shapiro = (np.nan, np.nan)

    # If normal
    if stat_shapiro[1] is not None and stat_shapiro[1] > 0.05:
        t_stat, p = stats.ttest_1samp(diffs, popmean=0.0)
        effect = diffs.mean() / diffs.std(ddof=1)
        return {'method': 'paired_t', 'stat': float(t_stat), 'p': float(p), 'effect': float(effect)}

    # Else: Wilcoxon
    w_stat, p = stats.wilcoxon(diffs)
    return {'method': 'wilcoxon', 'stat': float(w_stat), 'p': float(p), 'effect': np.nan}

# -------------------------
# Plot helpers
# -------------------------
def save_hist_and_qq(series: pd.Series, out_path: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.histplot(series.dropna(), ax=axes[0], kde=True)
    axes[0].set_title(f"{title} - Histogram")

    stats.probplot(series.dropna(), dist="norm", plot=axes[1])
    axes[1].set_title(f"{title} - Q-Q")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved histogram & QQ to %s", out_path)

# -------------------------
# CLI main
# -------------------------
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--women', type=str, required=True)
    parser.add_argument('--men', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='outputs')
    parsed = parser.parse_args(args=args)

    outdir = Path(parsed.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_w = load_csv(Path(parsed.women))
    df_m = load_csv(Path(parsed.men))

    df_w = extract_price_usd(df_w)
    df_m = extract_price_usd(df_m)

    df_w = standardize_brand_column(df_w)
    df_m = standardize_brand_column(df_m)

    df_w = clean_name_col(df_w)
    df_m = clean_name_col(df_m)

    # Save histogram
    save_hist_and_qq(df_w['price_usd'], outdir / 'women_price_hist_qq.png', 'Women Price USD')

    # Mann-Whitney example (Nike vs Adidas)
    nike_prices = df_w.loc[df_w['brand_final'] == 'nike', 'price_usd']
    adidas_prices = df_w.loc[df_w['brand_final'] == 'adidas', 'price_usd']
    mw = mann_whitney_test(nike_prices, adidas_prices)
    logger.info("Mann-Whitney (Nike vs Adidas): %s", mw)

if __name__ == "__main__":
    main()

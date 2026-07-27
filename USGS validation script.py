# %% [markdown]
# # GLOW-S Master Validation Notebook
# 
# End-to-end validation and figure generation for all 8 algorithm variants against USGS in-situ field widths.
# 
# **Algorithms tested:**
# | # | Algorithm | CSV key |
# |---|-----------|----------|
# | 1 | Zou 2018 (base) | `Zou_base` |
# | 2 | CREST-S(Zou) (**best**) | `Zou_SS` |
# | 3 | Ultra-Blue (base) | `UltraB_base` |
# | 4 | CREST-S(Ultra-Blue) | `UltraB_SS` |
# | 5 | MuWI (base) | `MuWI_base` |
# | 6 | CREST-S(MuWI) | `MuWI_SS` |
# | 7 | WI2015 (base) | `WI2015_base` |
# | 8 | CREST-S(WI2015) | `WI2015_SS` |

# %% [markdown]
# ## 1. Imports and Global Configuration

# %%
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import geopandas as gpd
from shapely.geometry import Point
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

warnings.filterwarnings('ignore')

# ── Global plot style (matches original notebook) ────────────────────────────
plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.linestyle'    : '--',
    'grid.alpha'        : 0.45,
    'figure.dpi'        : 120,
})

# ── Output directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "/N/lustre/project/proj-212/Suraj/River_data_data/Output_Data_Master/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print("Output directory:", OUTPUT_DIR)

# %% [markdown]
# ## 2. File Paths

# %%
# ── In-situ ───────────────────────────────────────────────────────────────────
FIELD_DIR    = '/N/lustre/project/proj-212/Suraj/River_data_data/USGS_field_data/'
FIELD_CSV    = os.path.join(FIELD_DIR, 'discharge_stage_width_measurements.csv')
MAPPING_CSV  = '/N/lustre/project/proj-212/Suraj/River_data_data/USGS_Ortho/Orthogonal_preliminary_updated_1to1_mapped.csv'

# ── Model CSVs (8 SS algorithms + 4 Snow intermediate) ───────────────────────
VAL_DIR = '/N/lustre/project/proj-212/Suraj/River_data_data/USGS_Validation_Data/'

ALGO_PATHS = {
    # Base algorithms
    'Zou_base'      : os.path.join(VAL_DIR, 'model_data_MASTER_ALL.csv'),
    'UltraB_base'   : os.path.join(VAL_DIR, 'UltraB_merged.csv'),
    'MuWI_base'     : os.path.join(VAL_DIR, 'MuWI_merged.csv'),
    'WI2015_base'   : os.path.join(VAL_DIR, 'WI2015_merged.csv'),
    # Intermediate: Snow-only guard (used ONLY in Base→Snow→SS trajectory plots)
    'Zou_snow'      : os.path.join(VAL_DIR, 'Zou2018SnowRefined_merged.csv'),
    'WI2015_snow'   : os.path.join(VAL_DIR, 'WISnowRefined_merged.csv'),
    'MuWI_snow'     : os.path.join(VAL_DIR, 'MuWISnowRefined_merged.csv'),
    'UltraB_snow'   : os.path.join(VAL_DIR, 'UltraBSnowRefined_merged.csv'),
    # Final SS algorithms (base + snow + sediment guard)
    'Zou_SS'        : os.path.join(VAL_DIR, 'Zou2018SSRefined_merged.csv'),
    'UltraB_SS'     : os.path.join(VAL_DIR, 'UltraBSSRefined_merged.csv'),
    'MuWI_SS'       : os.path.join(VAL_DIR, 'MuWISSRefined_merged.csv'),
    'WI2015_SS'     : os.path.join(VAL_DIR, 'WI2015SSRefined_merged.csv'),
}

# ── Display labels ────────────────────────────────────────────────────────────
ALGO_LABELS = {
    'Zou_base'    : 'Zou 2018',
    'Zou_snow'    : 'Zou 2018 + Snow Guard',
    'Zou_SS'      : 'CREST-S(Zou)',
    'UltraB_base' : 'Ultra-Blue',
    'UltraB_snow' : 'Ultra-Blue + Snow Guard',
    'UltraB_SS'   : 'CREST-S(Ultra-Blue)',
    'MuWI_base'   : 'MuWI',
    'MuWI_snow'   : 'MuWI + Snow Guard',
    'MuWI_SS'     : 'CREST-S(MuWI)',
    'WI2015_base' : 'WI2015',
    'WI2015_snow' : 'WI2015 + Snow Guard',
    'WI2015_SS'   : 'CREST-S(WI2015)',
}

# ── Colors: base=muted, snow=mid, SS=vivid ───────────────────────────────────
ALGO_COLORS = {
    'Zou_base'    : '#B0C4DE',  'Zou_snow'    : '#6BAED6',  'Zou_SS'    : '#1F78B4',
    'UltraB_base' : '#B2DF8A',  'UltraB_snow' : '#74C476',  'UltraB_SS' : '#33A02C',
    'MuWI_base'   : '#FB9A99',  'MuWI_snow'   : '#F4606C',  'MuWI_SS'   : '#E31A1C',
    'WI2015_base' : '#CAB2D6',  'WI2015_snow' : '#9E7DC5',  'WI2015_SS' : '#6A3D9A',
}

# ── Key groupings ─────────────────────────────────────────────────────────────
# ── Short base-algorithm names, used inside CREST-S(...) labels ─────────────
BASE_SHORT_NAMES = {
    'Zou_base'    : 'Zou',
    'UltraB_base' : 'Ultra-Blue',
    'MuWI_base'   : 'MuWI',
    'WI2015_base' : 'WI2015',
}
def crest_s_label(base_key):
    """Return 'CREST-S(<base>)' given a *_base key (e.g. 'Zou_base')."""
    return f"CREST-S({BASE_SHORT_NAMES[base_key]})"

BASE_KEYS  = ['Zou_base',  'UltraB_base', 'MuWI_base',  'WI2015_base']
SNOW_KEYS  = ['Zou_snow',  'UltraB_snow', 'MuWI_snow',  'WI2015_snow']   # intermediate
SS_KEYS    = ['Zou_SS',    'UltraB_SS',   'MuWI_SS',    'WI2015_SS']     # final

# ALL_KEYS for existing plots = base + SS only (unchanged behaviour)
ALL_KEYS   = BASE_KEYS + SS_KEYS

# 3-step trajectory tuples: (base_key, snow_key, ss_key)
TRAJ_PAIRS = [
    ('Zou_base',    'Zou_snow',    'Zou_SS'),
    ('UltraB_base', 'UltraB_snow', 'UltraB_SS'),
    ('MuWI_base',   'MuWI_snow',   'MuWI_SS'),
    ('WI2015_base', 'WI2015_snow', 'WI2015_SS'),
]

# ── Width threshold (m) for small vs large river split ───────────────────────
W_THRESH = 30

print("Paths configured.")
print(f"  Base algorithms : {BASE_KEYS}")
print(f"  Snow algorithms : {SNOW_KEYS}")
print(f"  SS   algorithms : {SS_KEYS}")


# %% [markdown]
# ## 3. Shared Helper Functions

# %%
# ── Metric computation ────────────────────────────────────────────────────────
def compute_metrics(obs, pred):
    """Returns (MSE, R2_pearson, NSE) for paired arrays."""
    obs  = np.asarray(obs,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[mask], pred[mask]
    n = len(obs)
    if n < 2 or np.std(obs) == 0:
        return np.nan, np.nan, np.nan
    mse = np.mean((obs - pred) ** 2)
    r2  = np.corrcoef(obs, pred)[0, 1] ** 2
    sse = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1.0 - sse / sst if sst > 0 else np.nan
    return mse, r2, nse

def compute_nse_1to1(obs, pred):
    """NSE computed against the 1:1 line (same as Nash-Sutcliffe)."""
    _, _, nse = compute_metrics(obs, pred)
    return nse

def relative_bias(obs, pred):
    """Per-point relative bias (Sim - Obs) / Obs."""
    obs  = np.asarray(obs,  dtype=float)
    pred = np.asarray(pred, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        rb = np.where(obs != 0, (pred - obs) / obs, np.nan)
    return rb

# ── Date parsing (identical to original notebook) ────────────────────────────
def robust_date_parser(date_str):
    for fmt in ('%m/%d/%Y %H:%M', '%m/%d/%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(date_str, errors='coerce')

# ── Standard boxplot styling ──────────────────────────────────────────────────
MEAN_PROPS   = {'marker': 'D', 'markerfacecolor': 'white',
                'markeredgecolor': 'black', 'markersize': 7}
MEDIAN_PROPS = dict(color='black', linewidth=2)

def style_box(bp, facecolor='#AED6F1', alpha=0.7):
    for patch in bp['boxes']:
        patch.set_facecolor(facecolor)
        patch.set_alpha(alpha)

# ── Add metric annotation to axes ────────────────────────────────────────────
def annotate_metrics(ax, nse, r2, n, x=0.05, y=0.88):
    fontsize = 13
    txt = f'$NSE = {nse:.3f}$\n$R^2 = {r2:.3f}$\n$n = {n:,}$'
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=fontsize,
            fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.35', fc='white', alpha=0.75))

# ── 1:1 reference line ────────────────────────────────────────────────────────
def plot_1to1(ax, obs, pred, **kwargs):
    lo = min(np.nanmin(obs), np.nanmin(pred))
    hi = max(np.nanmax(obs), np.nanmax(pred))
    kw = dict(color='black', linestyle='--', linewidth=1.2, label='1:1 Line')
    kw.update(kwargs)
    ax.plot([lo, hi], [lo, hi], **kw)
    return lo, hi

print("Helper functions defined.")

# %% [markdown]
# ## 4. Load and Pre-process USGS Field Data

# %%
print("─" * 60)
print("Loading USGS in-situ field data ...")
df_field = pd.read_csv(FIELD_CSV)

# Unit conversion: ft → m
df_field['chan_width'] = df_field['chan_width'] * 0.3048

# Standardize station ID
df_field['site_no'] = pd.to_numeric(df_field['site_no'], errors='coerce')
df_field = df_field.dropna(subset=['site_no']).copy()
df_field['site_no'] = df_field['site_no'].astype(int)

# Standardize date
df_field['date_std'] = df_field['measurement_dt'].apply(robust_date_parser).dt.normalize()

print(f"Field records loaded  : {len(df_field):,}")
print(f"Unique stations       : {df_field['site_no'].nunique():,}")
print(f"Date range            : {df_field['date_std'].min().date()} → {df_field['date_std'].max().date()}")

# ── Load distance-filter mapping ─────────────────────────────────────────────
df_mapping = pd.read_csv(MAPPING_CSV)
df_mapping['site_no'] = pd.to_numeric(df_mapping['site_no'], errors='coerce')
SITES_TO_EXCLUDE = set(
    df_mapping[df_mapping['dist_to_comid_m'] > 500]['site_no'].dropna().astype(int)
)
print(f"Stations excluded (>500 m from COMID): {len(SITES_TO_EXCLUDE):,}")

# %% [markdown]
# ## 5. Load All 8 Algorithm Model CSVs

# %%
def load_model_csv(path, key):
    """Load one algorithm CSV and standardise columns."""
    df = pd.read_csv(path)
    df['riverID'] = pd.to_numeric(df['riverID'], errors='coerce')
    df = df.dropna(subset=['riverID']).copy()
    df['riverID']  = df['riverID'].astype(int)
    df['date_std'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    df['algo_key'] = key
    return df

raw_model = {}
for key, path in ALGO_PATHS.items():
    try:
        raw_model[key] = load_model_csv(path, key)
        print(f"  {key:<20}: {len(raw_model[key]):>8,} records")
    except FileNotFoundError:
        print(f"  {key:<20}: FILE NOT FOUND — {path}")

print(f"\nLoaded {len(raw_model)} algorithm datasets.")

# %% [markdown]
# ## 6. Merge Model Data with Field Observations (All Algorithms)

# %%
# ── Width filter threshold ────────────────────────────────────────────────────
# All validation is restricted to in-situ widths >= W_THRESH (30 m).
# Rivers below this threshold are excluded globally because spectral indices
# are unreliable at sub-pixel scales for narrow channels, and their inclusion
# distorts aggregate metrics (e.g. NSE collapses when small-river predictions
# are physically invalid). This single filter propagates to every downstream
# cell automatically via the VAL dict.
MIN_WIDTH_M = W_THRESH   # 30 m — defined in the paths cell

def merge_and_filter(df_field, df_model, sites_to_exclude):
    """
    Inner-join field and model on (site_no / riverID, date_std).
    Apply 500 m distance filter, deduplicate, and exclude in-situ
    widths below MIN_WIDTH_M (30 m).
    """
    merged = pd.merge(
        df_field, df_model,
        left_on=['site_no', 'date_std'],
        right_on=['riverID', 'date_std'],
        how='inner'
    ).dropna(subset=['chan_width', 'width'])

    merged = merged[~merged['site_no'].isin(sites_to_exclude)]
    merged = merged.drop_duplicates(subset=['site_no', 'date_std'], keep='first')

    # ── Exclude narrow rivers (<30 m in-situ width) ───────────────────────────
    n_before = len(merged)
    merged = merged[merged['chan_width'] >= MIN_WIDTH_M].copy()
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        print(f"    [width filter] dropped {n_dropped:,} records with "
              f"in-situ width < {MIN_WIDTH_M} m")

    return merged.copy()

# Build a dict of validation DataFrames, one per algorithm
VAL = {}
for key, df_m in raw_model.items():
    VAL[key] = merge_and_filter(df_field, df_m, SITES_TO_EXCLUDE)
    # Compute relative bias
    VAL[key]['Relative_Bias'] = relative_bias(VAL[key]['chan_width'], VAL[key]['width'])
    VAL[key]['bias_abs']      = VAL[key]['Relative_Bias'].abs()
    print(f"  {key:<20}: {len(VAL[key]):>6,} matched pairs | "
          f"{VAL[key]['site_no'].nunique():>4} stations")

print(f"\nMerge complete. All VAL records have in-situ width >= {MIN_WIDTH_M} m.")

# %% [markdown]
# ## 7. Global Metric Summary Table (All 8 Algorithms)

# %%
rows = []
for key in ALL_KEYS:
    df = VAL[key]
    obs, pred = df['chan_width'].values, df['width'].values
    mse, r2, nse = compute_metrics(obs, pred)
    rb = df['Relative_Bias'].dropna()
    rows.append({
        'Algorithm'   : ALGO_LABELS[key],
        'Key'         : key,
        'n'           : len(df),
        'NSE'         : round(nse, 4),
        'R2'          : round(r2,  4),
        'RMSE (m)'    : round(np.sqrt(mse), 3),
        'Mean Bias'   : round(rb.mean(), 4),
        'Median Bias' : round(rb.median(), 4),
        'Type'        : 'Guard' if 'snow' in key else 'Base',
    })

metrics_df = pd.DataFrame(rows).sort_values('NSE', ascending=False).reset_index(drop=True)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'master_metric_summary.csv'), index=False)

# Pretty print
print("\n══ GLOBAL METRIC SUMMARY (sorted by NSE) ══")
print(metrics_df[['Algorithm','n','NSE','R2','RMSE (m)','Mean Bias','Median Bias','Type']].to_string(index=False))

# %% [markdown]
# ## 8. Figure 2 — Algorithm Comparison Scatter Plots (2 × 4 Panel)
# 
# Top row = 5 worst algorithms by NSE. Bottom row = 5 best. Ordered **worst → best** left to right.

# %%
# ── Build ordered metric arrays ─────────────────────────────────────────────
nse_vals  = {k: compute_metrics(VAL[k]['chan_width'].values, VAL[k]['width'].values)[2] for k in ALL_KEYS}
r2_vals   = {k: compute_metrics(VAL[k]['chan_width'].values, VAL[k]['width'].values)[1] for k in ALL_KEYS}
bias_vals = {k: VAL[k]['Relative_Bias'].median() for k in ALL_KEYS}
rmse_vals = {k: np.sqrt(compute_metrics(VAL[k]['chan_width'].values, VAL[k]['width'].values)[0]) for k in ALL_KEYS}

PAIR_ORDER = list(zip(BASE_KEYS, SS_KEYS))  # 5 pairs
pair_labels = [ALGO_LABELS[b].replace(' (base)', '').replace('Zou 2018', 'Zou')
               for b, _ in PAIR_ORDER]

# Figure 2 — Scatterplots: Zou → WI2015 → Ultra-Blue → MuWI (left to right)
# Row 0 = Base variants, Row 1 = SS Filter variants
SCATTER_ORDER = ['Zou_base', 'WI2015_base', 'UltraB_base', 'MuWI_base',
                 'Zou_SS', 'WI2015_SS', 'UltraB_SS', 'MuWI_SS']
sorted_keys = SCATTER_ORDER

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False, sharey=False)
fig.suptitle('Algorithm Comparison: Predicted vs In-situ Width\n(left to right: Zou → WI2015 → Ultra-Blue → MuWI)',
             fontsize=15, fontweight='bold', y=1.01)

for ax_idx, key in enumerate(sorted_keys):
    row_idx = ax_idx // 4
    col_idx = ax_idx  % 4
    ax  = axes[row_idx, col_idx]
    df  = VAL[key]
    obs  = df['chan_width'].values
    pred = df['width'].values
    mse, r2, nse = compute_metrics(obs, pred)
    n = len(df)

    ax.scatter(obs, pred,
               alpha=0.35, s=8,
               color=ALGO_COLORS[key], rasterized=True)

    lo, hi = plot_1to1(ax, obs, pred)
    annotate_metrics(ax, nse, r2, n)

    ax.set_title(ALGO_LABELS[key], fontsize=13, fontweight='bold', pad=6)
    ax.set_xlabel('In-situ Width (m)')
    ax.set_ylabel('Predicted Width (m)')
    ax.tick_params(labelsize=14.4)

# Row labels
axes[0, 0].annotate('Base', xy=(-0.30, 0.5), xycoords='axes fraction',
                    fontsize=13, fontweight='bold', rotation=90, va='center',
                    annotation_clip=False)
axes[1, 0].annotate('CREST-S', xy=(-0.30, 0.5), xycoords='axes fraction',
                    fontsize=13, fontweight='bold', rotation=90, va='center',
                    annotation_clip=False)

plt.tight_layout()
plt.subplots_adjust(left=0.10)  # extra left margin for row labels
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Algorithm_Scatter_ZouWIUBMuWI.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Algorithm_Scatter_ZouWIUBMuWI.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 2 saved (ordered worst→best).")


# %% [markdown]
# ## 9. Figure 3 — Metric Comparison Across All 8 Algorithms
# 
# Four panels in a **2×2 layout** (NSE, R², RMSE, Mean Relative Bias). Algorithms ordered worst→best NSE. Base = blue bar, Guard = red bar.

# %%
# Figure 3 — Metric Comparison: All 8 Variants (2×2 panel)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Figure 3 — Metric Comparison: All 8 Algorithm Variants')

metrics_to_plot = [
    ('NSE',            nse_vals,  'Nash–Sutcliffe Efficiency'),
    ('R²',             r2_vals,   'Coefficient of Determination R²'),
    ('RMSE (m)',       rmse_vals, 'RMSE (m)'),
    ('Median Rel. Bias', bias_vals, 'Median Relative Bias [(Sim-Obs)/Obs]'),
]

x = np.arange(len(PAIR_ORDER))
width_bar = 0.32

for ax_i, (metric_name, mdict, ylabel) in enumerate(metrics_to_plot):
    ax = axes[ax_i // 2, ax_i % 2]

    base_vals_arr = np.array([mdict[b] for b, _ in PAIR_ORDER])
    snow_vals_arr = np.array([mdict[s] for _, s in PAIR_ORDER])

    c_base, c_SS = '#2c7bb6', '#d7191c'
    bars_b = ax.bar(x - width_bar/2, base_vals_arr, width_bar,
                    color=c_base, alpha=0.55, edgecolor='black', linewidth=0.6,
                    label='Base')
    bars_s = ax.bar(x + width_bar/2, snow_vals_arr, width_bar,
                    color=c_SS, alpha=0.85, edgecolor='black', linewidth=0.6,
                    label='+ Guard')


    if metric_name == 'Median Rel. Bias':
        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)
    elif metric_name == 'RMSE (m)':
        pass  # no reference line needed
    else:
        ax.axhline(1, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)

    ax.set_ylabel(ylabel)
    ax.legend(fontsize=13.2, loc='lower right')
    ax.tick_params(axis='y', labelsize=14.4)
    ax.set_title(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=14.4, fontweight='bold', rotation=15)
    ax.set_xlabel('Algorithm Family')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Metric_Comparison_2x2.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Metric_Comparison_2x2.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 3 (2x2 panel) saved.")


# %% [markdown]
# ### 9b. Figure 3 (Alternative) — Dot-Line Metric Comparison (2×2)
# 
# Each metric as a horizontal dot plot with paired base/guard dots connected by lines. **RMSE added as 4th panel.** Mean Relative Bias panel additionally shows **min and max** per-observation extent as shaded extent bars.

# %%
# Figure 3 (Alt) — Dot-Line Metric Comparison  (2×2 panel)
from matplotlib.lines import Line2D

fig, axes_2x2 = plt.subplots(2, 2, figsize=(16, 12))
axes = axes_2x2.flatten()  # flatten to 1-D for same loop logic
fig.suptitle('Per-Algorithm Metric Improvement from Snow/Sediment Filter',
             fontsize=15, fontweight='bold')

metric_configs = [
    ('NSE',       nse_vals,  'Nash–Sutcliffe Efficiency (NSE)'),
    ('R²',        r2_vals,   'Coefficient of Determination (R²)'),
    ('RMSE (m)',  rmse_vals, 'RMSE (m)'),
    ('Median Bias', bias_vals, 'Median Relative Bias'),
]

y_pos = np.arange(len(PAIR_ORDER))

# Pre-compute bias normalisation denominator for the bias panel:
# max of |mean| and |median| across ALL algorithms (base + guard combined)
_bias_all_vals = []
for bk, sk in PAIR_ORDER:
    for key in (bk, sk):
        rb = VAL[key]['Relative_Bias'].dropna()
        _bias_all_vals.extend([abs(rb.mean()), abs(rb.median())])
_bias_norm_denom = max(_bias_all_vals) if _bias_all_vals else 1.0

for ax_idx, (label, mdict, xlabel) in enumerate(metric_configs):
    ax = axes[ax_idx]

    for i, (bk, sk) in enumerate(PAIR_ORDER):
        bv = mdict[bk]
        sv = mdict[sk]

        if label == 'Median Bias':
            # ── Bias panel: normalise both mean and median to avoid scale conflict ──
            # Denominator = max(|value|) across all base+guard mean & median values
            # computed once before the loop via _bias_norm_denom (see below).
            rb_b_all = VAL[bk]['Relative_Bias'].dropna()
            rb_s_all = VAL[sk]['Relative_Bias'].dropna()

            bv_med_raw = rb_b_all.median()
            bv_mn_raw  = rb_b_all.mean()
            sv_med_raw = rb_s_all.median()
            sv_mn_raw  = rb_s_all.mean()

            bv_med = bv_med_raw / _bias_norm_denom
            bv_mn  = bv_mn_raw  / _bias_norm_denom
            sv_med = sv_med_raw / _bias_norm_denom
            sv_mn  = sv_mn_raw  / _bias_norm_denom

            lo_b, hi_b = (np.percentile(rb_b_all, [5, 95]) / _bias_norm_denom)
            lo_s, hi_s = (np.percentile(rb_s_all, [5, 95]) / _bias_norm_denom)

            # P5-P95 extent lines on same row
            ax.plot([lo_b, hi_b], [i, i],
                    color=ALGO_COLORS[bk], linewidth=4.0, alpha=0.30,
                    solid_capstyle='round', zorder=1)
            ax.plot([lo_s, hi_s], [i, i],
                    color=ALGO_COLORS[sk], linewidth=2.5, alpha=0.45,
                    solid_capstyle='round', zorder=2)
            for val, col in [(lo_b, ALGO_COLORS[bk]), (hi_b, ALGO_COLORS[bk]),
                             (lo_s, ALGO_COLORS[sk]), (hi_s, ALGO_COLORS[sk])]:
                ax.plot([val, val], [i - 0.18, i + 0.18],
                        color=col, linewidth=1.5, zorder=4)

            # Connecting line between base and guard medians
            ax.plot([bv_med, sv_med], [i, i], color='grey', linewidth=2.0, zorder=1)

            # Median — filled diamond (base), open circle (guard)
            ax.scatter(bv_med, i, color=ALGO_COLORS[bk], s=180, zorder=6,
                       edgecolors='black', linewidths=1.0, marker='D')
            ax.scatter(sv_med, i, color=ALGO_COLORS[sk], s=140, zorder=6,
                       edgecolors=ALGO_COLORS[sk], linewidths=1.5,
                       facecolors='none', marker='o')

            # Mean — hollow square (base), hollow triangle (guard)
            ax.scatter(bv_mn, i, color=ALGO_COLORS[bk], s=120, zorder=6,
                       edgecolors=ALGO_COLORS[bk], linewidths=1.5,
                       facecolors='none', marker='s')
            ax.scatter(sv_mn, i, color=ALGO_COLORS[sk], s=100, zorder=6,
                       edgecolors=ALGO_COLORS[sk], linewidths=1.5,
                       facecolors='none', marker='^')

            # Raw value annotation on right margin — tight, small font
            x_right = ax.get_xlim()[1] if ax.get_xlim()[1] != 1.0 else 1.05
            ax.text(1.02, i - 0.18,
                    f'Base  M̄={bv_mn_raw:.3f} M̃={bv_med_raw:.3f}',
                    va='center', fontsize=6.5, color=ALGO_COLORS[bk],
                    transform=ax.get_yaxis_transform(), clip_on=False)
            ax.text(1.02, i + 0.18,
                    f'Guard M̄={sv_mn_raw:.3f} M̃={sv_med_raw:.3f}',
                    va='center', fontsize=6.5, color=ALGO_COLORS[sk],
                    transform=ax.get_yaxis_transform(), clip_on=False)

        else:
            # Non-bias panels: original dot-line logic unchanged
            ax.plot([bv, sv], [i, i], color='grey', linewidth=2.0, zorder=1)
            ax.scatter(bv, i, color=ALGO_COLORS[bk], s=180, zorder=3,
                       edgecolors='black', linewidths=1.0, marker='D')
            ax.scatter(sv, i, color=ALGO_COLORS[sk], s=140, zorder=3,
                       edgecolors=ALGO_COLORS[sk], linewidths=1.5,
                       facecolors='none', marker='o')

    if label == 'Median Bias':
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    elif label in ('NSE', 'R²'):
        ax.axvline(1, color='grey', linewidth=0.6, linestyle=':', alpha=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels)
    ax.set_xlabel(xlabel, fontsize=14.4, fontweight='bold')
    ax.set_title(label, fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', labelsize=14.4)
    ax.invert_yaxis()

# Shared legend
legend_elems = [
    Line2D([0],[0], marker='D', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=12, label='Base (median, diamond)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='grey', markersize=12, label='Guard (median, circle)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='grey', markersize=10, label='Base (mean, square)'),
    Line2D([0],[0], marker='^', color='w', markerfacecolor='none',
           markeredgecolor='grey', markersize=10, label='Guard (mean, triangle)'),
    Line2D([0],[0], linewidth=4.0, color='grey', alpha=0.35,
           label='P5–P95 range (bias panel only)'),
]
# Legend placed outside below all panels (see fig.legend below)

fig.legend(
    handles=legend_elems,
    fontsize=14.4,
    loc='lower center',
    ncol=5,
    bbox_to_anchor=(0.5, -0.06),
    framealpha=0.9,
    edgecolor='grey',
    title='Symbol guide',
    title_fontsize=14.4,
)
plt.tight_layout()
plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.35)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Alt_DotLine_Metrics_2x2.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Alt_DotLine_Metrics_2x2.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 3 (Alt) 2×2 dot-line saved.")


# %% [markdown]
# ## 13. Figure — Spatial Distribution of Relative Bias (CREST-S(Zou))

# %%
# 1. Prepare Data
df_map_data = VAL['Zou_SS'].dropna(subset=['lon', 'lat', 'Relative_Bias']).copy()
geometry = [Point(xy) for xy in zip(df_map_data['lon'], df_map_data['lat'])]
gdf = gpd.GeoDataFrame(df_map_data, geometry=geometry, crs='EPSG:4326')
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# 2. Setup Figure and Grid
fig = plt.figure(figsize=(18, 9))
gs  = fig.add_gridspec(1, 2,
                        width_ratios=[1.0, 0.025],
                        left=0.06, right=0.93, wspace=0.05)
ax_map  = fig.add_subplot(gs[0])
ax_cbar = fig.add_subplot(gs[1])

fig.suptitle('Spatial Distribution of Relative Bias — CREST-S(Zou)',
             fontsize=16, fontweight='bold', y=1.01)

# 3. Create Custom Colormap with an Extended White Center
rdbu = cm.get_cmap('RdBu_r', 512)

# Linear positions 0 → 1
t = np.linspace(0.0, 1.0, 512)

# Calculate normalized distance from the center (0.0 at center, 1.0 at edges)
distance = np.abs(t - 0.5) * 2

# Exponent > 1 stretches out the white region in the middle.
# 2.5 provides a very gentle transition. Increase to 3.0+ for an even wider white area.
exponent = 2.5 

# Reconstruct the compressed trajectory around the center
t_compressed = 0.5 + 0.5 * np.sign(t - 0.5) * (distance ** exponent)

colors_compressed = rdbu(t_compressed)
cmap_gentle = LinearSegmentedColormap.from_list(
    'RdBu_gentle', colors_compressed, N=512
)

# 4. Normalization and Plotting Base Map
CBAR_VMIN, CBAR_VMAX = -1.0, 1.0
norm = mcolors.TwoSlopeNorm(vmin=CBAR_VMIN, vcenter=0.0, vmax=CBAR_VMAX)

world.plot(ax=ax_map, color='#E8E8E8', edgecolor='#AAAAAA', linewidth=0.4)

# 5. Plot Data Points
gdf_plot = gdf.copy()
gdf_plot['bias_clipped'] = gdf_plot['Relative_Bias'].clip(CBAR_VMIN, CBAR_VMAX)

ax_map.scatter(
    gdf_plot['lon'], gdf_plot['lat'],
    c=gdf_plot['bias_clipped'],
    cmap=cmap_gentle,
    norm=norm,
    s=40,
    edgecolors='none', linewidths=0,
    alpha=0.85, zorder=4
)

# 6. Map Formatting
ax_map.set_xlim(-130, -60)
ax_map.set_ylim(22, 52)
ax_map.set_aspect('auto')
ax_map.set_title('CONUS — USGS Validation Stations', fontsize=14, fontweight='bold')
ax_map.tick_params(labelsize=14.4)
ax_map.set_xlabel('Longitude', fontsize=16.8, fontweight='bold')
ax_map.set_ylabel('Latitude',  fontsize=16.8, fontweight='bold', labelpad=6)

# 7. Colorbar Formatting
sm = cm.ScalarMappable(cmap=cmap_gentle, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_label('Relative Bias [(Sim−Obs)/Obs]',
                fontsize=14.4, fontweight='bold', labelpad=10)
cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
cbar.set_ticklabels(['≤-1.0', '-0.5', '0', '+0.5', '≥+1.0'])
cbar.ax.tick_params(labelsize=13.2)

ax_cbar.yaxis.set_label_position('right')
ax_cbar.yaxis.tick_right()

# 8. Save and Display
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Bias_Map_Zou_SSFilter.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Bias_Map_Zou_SSFilter.png'),
            bbox_inches='tight', dpi=150)
plt.show()

# 9. Summary Statistics
n_clipped = (gdf['Relative_Bias'].abs() > CBAR_VMAX).sum()
print(f"Bias map saved.")
print(f"  Points clipped (|bias| > {CBAR_VMAX}): {n_clipped:,} ({100*n_clipped/len(gdf):.1f}%)")

# %% [markdown]
# ## 15. RMSE by Width Bin — All 8 Algorithms
# 
# **Changes:** Removed `<30 m` bin, switched to **Normalised RMSE** (RMSE ÷ mean observed width per bin), added **R² bar chart** panel beneath each Norm. RMSE panel. 4-row × 5-col layout.

# %%
# ── Bins: remove <30 m, use 30–60, 60–100, 100–200, >200 ────────────────────
BIN_EDGES_4  = [30, 60, 100, 200, np.inf]
BIN_LABELS_4 = ['30–60 m', '60–100 m', '100–200 m', '>200 m']

def nrmse_r2_by_bin(df, bin_edges, bin_labels):
    """Normalised RMSE (RMSE / mean_obs) and R² for each width bin."""
    obs     = df['chan_width'].values
    pred    = df['width'].values
    bin_ids = np.clip(np.digitize(obs, bin_edges) - 1, 0, len(bin_labels)-1)
    nrmse_list, r2_list, n_list = [], [], []
    for b in range(len(bin_labels)):
        mask = bin_ids == b
        if mask.sum() > 1:
            o_b, p_b = obs[mask], pred[mask]
            mean_obs = o_b.mean()
            rmse     = np.sqrt(np.mean((p_b - o_b)**2))
            nrmse_list.append(rmse / mean_obs if mean_obs > 0 else np.nan)
            r2_list.append(np.corrcoef(o_b, p_b)[0, 1]**2)
            n_list.append(mask.sum())
        else:
            nrmse_list.append(np.nan)
            r2_list.append(np.nan)
            n_list.append(mask.sum())
    return nrmse_list, r2_list, n_list

# Collect metrics for all algorithms
nrmse_data, r2_bin_data = {}, {}
for key in ALL_KEYS:
    nrmse_data[key], r2_bin_data[key], _ = nrmse_r2_by_bin(VAL[key], BIN_EDGES_4, BIN_LABELS_4)

# ── Interleaved keys: Base|SS pairs ──────────────────────────────────────────
INTERLEAVED_KEYS = []
for bk, sk in zip(BASE_KEYS, SS_KEYS):
    INTERLEAVED_KEYS.append(bk)
    INTERLEAVED_KEYS.append(sk)

n_bins  = len(BIN_LABELS_4)
n_algos = len(INTERLEAVED_KEYS)   # 8
bar_w   = 0.09
x       = np.arange(n_bins)

fig, ax_nrmse = plt.subplots(1, 1, figsize=(18, 7))
fig.suptitle(
    'Normalised RMSE by In-situ Width Bin — All Algorithm Variants'
    'Base | CREST-S pairs  (hatched = CREST-S)',
    fontsize=15, fontweight='bold'
)

HATCH_MAP = {k: ('///' if 'SS' in k else '') for k in INTERLEAVED_KEYS}

for algo_idx, key in enumerate(INTERLEAVED_KEYS):
    offset  = (algo_idx - n_algos / 2 + 0.5) * bar_w
    nrmse_v = nrmse_data[key]
    col     = ALGO_COLORS[key]
    hatch   = HATCH_MAP[key]
    ax_nrmse.bar(x + offset, nrmse_v, bar_w,
                 color=col, edgecolor='black', linewidth=0.7,
                 hatch=hatch, alpha=0.88, label=ALGO_LABELS[key])

# Pair dividers
for bin_x in x:
    for pair_end in [1, 3, 5]:
        sep_x = bin_x + (pair_end - n_algos / 2 + 0.5) * bar_w + bar_w / 2
        ax_nrmse.axvline(sep_x, color='#888888', linewidth=0.6,
                         linestyle=':', alpha=0.5)

ax_nrmse.set_xticks(x)
ax_nrmse.set_xticklabels(BIN_LABELS_4, fontsize=15.6)
ax_nrmse.set_xlabel('In-situ Width Bin', fontsize=15.6, fontweight='bold')
ax_nrmse.set_ylabel('Normalised RMSE  (RMSE / mean observed width)',
                     fontsize=14.4, fontweight='bold')
ax_nrmse.set_title('Normalised RMSE by Width Bin', fontsize=13, fontweight='bold')
ax_nrmse.tick_params(axis='y', labelsize=14.4)

import matplotlib.patches as mpatches
legend_patches = []
for bk, sk in zip(BASE_KEYS, SS_KEYS):
    fam = ALGO_LABELS[bk]
    legend_patches.append(
        mpatches.Patch(facecolor=ALGO_COLORS[bk], edgecolor='black',
                       linewidth=0.8, label=f'{fam} — Base'))
    legend_patches.append(
        mpatches.Patch(facecolor=ALGO_COLORS[sk], edgecolor='black',
                       hatch='///', linewidth=0.8, label=crest_s_label(bk)))

ax_nrmse.legend(handles=legend_patches, fontsize=13.2, loc='upper right',
                ncol=2, title='Algorithm  (hatched = CREST-S)',
                title_fontsize=14.4, framealpha=0.9, edgecolor='grey')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_NRMSE_ByBin_AllAlgos.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_NRMSE_ByBin_AllAlgos.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Normalised RMSE by-bin figure saved.")


# %% [markdown]
# ## 20. Seasonal Bias Analysis — Zou 2018 Base vs SS Filter
# 
# **Changed to a single combined plot** with two panels:
# - Left: side-by-side grouped boxplots (Base solid, Guard hatched) per season; DJF winter band highlighted
# - Right: horizontal mean ± 1σ bars per season; DJF highlighted
# 
# Checks whether the snow guard most effectively reduces bias in DJF (winter), consistent with the snow-contamination hypothesis.

# %%
# Seasonal Analysis — Zou 2018 Base vs Zou 2018 + SS Filter
# Expanded to show Relative Bias, RMSE, NSE, and R² per season.
# This gives a fuller picture than bias alone — an algorithm can have low
# bias but poor RMSE/NSE if errors are symmetric but large.

SEASON_MAP   = {12:'DJF', 1:'DJF', 2:'DJF',
                3:'MAM', 4:'MAM', 5:'MAM',
                6:'JJA', 7:'JJA', 8:'JJA',
                9:'SON', 10:'SON', 11:'SON'}
SEASON_ORDER       = ['DJF', 'MAM', 'JJA', 'SON']
SEASON_COLORS      = ['#5B9BD5', '#A9D18E', '#FFC000', '#ED7D31']
SEASON_LABELS_FULL = ['DJF (Winter)', 'MAM (Spring)', 'JJA (Summer)', 'SON (Autumn)']

IQR_FLOOR = 0.01

def _std_rbias(series):
    s = series.dropna()
    iqr = max(np.percentile(s, 75) - np.percentile(s, 25), IQR_FLOOR)
    return s.mean() / iqr, s.median() / iqr, iqr, s.mean(), s.median()

def _season_metrics(df, season_col='season'):
    """Compute NSE, RMSE, R² and bias statistics per season."""
    rows = []
    for s in SEASON_ORDER:
        sub = df[df[season_col] == s].dropna(subset=['chan_width','width','Relative_Bias'])
        if len(sub) < 5:
            rows.append({'season': s, 'n': len(sub),
                         'nse': np.nan, 'rmse': np.nan, 'r2': np.nan,
                         'bias_mean': np.nan, 'bias_median': np.nan})
            continue
        obs  = sub['chan_width'].values
        pred = sub['width'].values
        mse, r2, nse = compute_metrics(obs, pred)
        rows.append({
            'season'     : s,
            'n'          : len(sub),
            'nse'        : nse,
            'rmse'       : np.sqrt(mse),
            'r2'         : r2,
            'bias_mean'  : sub['Relative_Bias'].mean(),
            'bias_median': sub['Relative_Bias'].median(),
        })
    return pd.DataFrame(rows).set_index('season')

COMPARE = [
    ('Zou_base', 'Zou 2018 Base',        '#B0C4DE', ALGO_COLORS['Zou_base'], ''),
    ('Zou_SS',   'CREST-S(Zou)', '#1F78B4', ALGO_COLORS['Zou_SS'],  '///'),
]

# Pre-compute seasonal dataframes and metric tables
seasonal_dfs, metric_tables = {}, {}
for key, label, _, _, _ in COMPARE:
    df_s = VAL[key].copy()
    df_s['season'] = df_s['date_std'].dt.month.map(SEASON_MAP)
    seasonal_dfs[key]   = df_s
    metric_tables[key]  = _season_metrics(df_s)

# ─── Figure layout: 4 rows × 2 panels ────────────────────────────────────────
# Row 0: Relative Bias boxplots (left) + standardised bar (right)
# Row 1: RMSE grouped bar
# Row 2: NSE grouped bar
# Row 3: R² grouped bar
from matplotlib.patches import Patch

fig = plt.figure(figsize=(22, 24))
gs  = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.28,
                        height_ratios=[1.6, 1, 1, 1],
                        left=0.07, right=0.97)

ax_box   = fig.add_subplot(gs[0, 0])   # bias boxplots
ax_bias  = fig.add_subplot(gs[0, 1])   # standardised bias bars
ax_rmse  = fig.add_subplot(gs[1, :])   # RMSE (full width)
ax_nse   = fig.add_subplot(gs[2, :])   # NSE  (full width)
ax_r2    = fig.add_subplot(gs[3, :])   # R²   (full width)

fig.suptitle('Seasonal Performance — Zou 2018 Base vs CREST-S(Zou)'
             'Relative Bias | RMSE | NSE | R²',
             fontsize=15, fontweight='bold')

# ── Row 0 Left: Bias boxplots (existing style) ────────────────────────────────
for key_idx, (key, label, bar_col, algo_col, hatch) in enumerate(COMPARE):
    df_s = seasonal_dfs[key]
    offset = -0.2 if key_idx == 0 else 0.2
    positions = [i + 1 + offset for i in range(len(SEASON_ORDER))]
    plot_data = [df_s[df_s['season'] == s]['Relative_Bias'].dropna().values
                 for s in SEASON_ORDER]
    bp = ax_box.boxplot(plot_data, positions=positions, widths=0.32,
                        patch_artist=True, showmeans=True, showfliers=False,
                        meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS)
    for patch, col in zip(bp['boxes'], SEASON_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.7 if key_idx == 0 else 0.9)
        patch.set_hatch(hatch)
        patch.set_linewidth(1.2)
    for pos, d in zip(positions, plot_data):
        if not len(d): continue
        mn, med = np.mean(d), np.median(d)
        q75     = np.percentile(d, 75)
        col_ann = 'navy' if key_idx == 0 else 'darkred'
        ax_box.scatter([pos], [mn], marker='D', s=50, color=col_ann,
                       edgecolors='black', linewidths=0.5, zorder=5)
        ax_box.text(pos, q75 + 0.05, f'M̄={mn:.2f}M̃={med:.2f}',
                    ha='center', fontsize=6.0, color=col_ann, linespacing=1.3)

ax_box.axvspan(0.55, 1.45, color='lightblue', alpha=0.18, zorder=0)
ax_box.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
ax_box.set_xticks(range(1, len(SEASON_ORDER)+1))
ax_box.set_xticklabels(SEASON_LABELS_FULL, fontsize=12)
ax_box.set_ylabel('Relative Bias [(Sim−Obs)/Obs]', fontweight='bold')
ax_box.set_title('Seasonal Bias Distribution', fontweight='bold')
legend_elems = [
    Patch(facecolor='grey', alpha=0.7,              label='Zou 2018 Base'),
    Patch(facecolor='grey', alpha=0.9, hatch='///', label='CREST-S(Zou)'),
    Patch(facecolor='lightblue', alpha=0.35,         label='DJF winter band'),
]
ax_box.legend(handles=legend_elems, fontsize=12, loc='upper right')

# ── Row 0 Right: Standardised bias bars ───────────────────────────────────────
for key_idx, (key, label, bar_col, algo_col, hatch) in enumerate(COMPARE):
    df_s = seasonal_dfs[key]
    s_means, s_meds, raw_means, raw_meds, s_stds = [], [], [], [], []
    for s in SEASON_ORDER:
        rb = df_s[df_s['season'] == s]['Relative_Bias']
        mn_std, med_std, iqr, mn_r, med_r = _std_rbias(rb)
        s_means.append(mn_std); s_meds.append(med_std)
        raw_means.append(mn_r); raw_meds.append(med_r)
        s_stds.append(rb.dropna().std() / max(iqr, IQR_FLOOR))
    y_pos = np.arange(len(SEASON_ORDER)) + key_idx * 0.35
    ax_bias.barh(y_pos, s_means, height=0.30, color=bar_col,
                 edgecolor='black', linewidth=0.7,
                 xerr=s_stds, ecolor='black', capsize=3,
                 label=f'{label} (mean)', alpha=0.85)
    ax_bias.scatter(s_meds, y_pos, marker='o', s=55,
                    color='white', edgecolors=bar_col, linewidths=1.8, zorder=5)
    for yp, mn_r, med_r in zip(y_pos, raw_means, raw_meds):
        ax_bias.text(0.01, yp - 0.10, f'M̄={mn_r:.3f} | M̃={med_r:.3f}',
                     va='top', fontsize=6.0, color='#444444',
                     transform=ax_bias.get_yaxis_transform())

ax_bias.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
djf_idx = SEASON_ORDER.index('DJF')
ax_bias.axhspan(djf_idx - 0.1, djf_idx + 0.75, color='lightblue', alpha=0.25)
ax_bias.set_yticks(np.arange(len(SEASON_ORDER)) + 0.175)
ax_bias.set_yticklabels(SEASON_LABELS_FULL, fontsize=12)
ax_bias.set_xlabel('Standardised Relative Bias (÷ IQR)  —  bar=mean, dot=median',
                    fontweight='bold')
ax_bias.set_title('Standardised Seasonal Bias(Mean & Median, Base vs CREST-S(Zou))',
                   fontweight='bold')
handles_b, labels_b = ax_bias.get_legend_handles_labels()
seen = {}
for h, l in zip(handles_b, labels_b):
    if l not in seen: seen[l] = h
ax_bias.legend(seen.values(), seen.keys(), fontsize=10.8, loc='lower right')
ax_bias.invert_yaxis()

# ── Rows 1-3: RMSE, NSE, R² grouped bar charts ───────────────────────────────
x      = np.arange(len(SEASON_ORDER))
bar_w2 = 0.32
offset_map = {0: -bar_w2/2, 1: bar_w2/2}

metric_configs = [
    (ax_rmse, 'rmse',  'RMSE (m)',                              False, None),
    (ax_nse,  'nse',   'NSE',                                   True,  1.0),
    (ax_r2,   'r2',    'R²',                                    True,  1.0),
]

for ax_m, mkey, ylabel, add_ref, ref_val in metric_configs:
    for key_idx, (key, label, bar_col, algo_col, hatch) in enumerate(COMPARE):
        mt = metric_tables[key]
        vals = [mt.loc[s, mkey] if s in mt.index else np.nan for s in SEASON_ORDER]
        ns   = [mt.loc[s, 'n']  if s in mt.index else 0      for s in SEASON_ORDER]
        bars = ax_m.bar(x + offset_map[key_idx], vals, bar_w2,
                        color=bar_col, edgecolor='black', linewidth=0.7,
                        hatch=hatch, alpha=0.88, label=label)
        # Annotate value + n above each bar
        for bar, v, n in zip(bars, vals, ns):
            if np.isfinite(v):
                yann = bar.get_height() + 0.01 if v >= 0 else bar.get_height() - 0.05
                ax_m.text(bar.get_x() + bar.get_width()/2, yann,
                          f'{v:.3f}(n={n})', ha='center', va='bottom',
                          fontsize=6.5, rotation=0)

    # DJF highlight band
    ax_m.axvspan(-0.5, 0.5, color='lightblue', alpha=0.15, zorder=0)
    if add_ref:
        ax_m.axhline(ref_val, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)

    ax_m.set_xticks(x)
    ax_m.set_xticklabels(SEASON_LABELS_FULL, fontsize=13.2)
    ax_m.set_ylabel(ylabel, fontsize=14.4, fontweight='bold')
    ax_m.set_title(f'Seasonal {ylabel} — Base vs CREST-S(Zou)', fontsize=12, fontweight='bold')
    ax_m.tick_params(axis='y', labelsize=13.2)
    ax_m.legend(fontsize=12, loc='lower right')

fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Seasonal_Expanded.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Seasonal_Expanded.png'),
            bbox_inches='tight', dpi=150)
plt.show()

# ── Print DJF stats across all metrics ───────────────────────────────────────
print("\n══ DJF vs Non-DJF Summary ══")
for key, label in [('Zou_base', 'Base'), ('Zou_SS', 'CREST-S(Zou)')]:
    df_s = seasonal_dfs[key]
    mt   = metric_tables[key]
    djf  = df_s[df_s['season'] == 'DJF']['Relative_Bias'].dropna()
    oth  = df_s[df_s['season'] != 'DJF']['Relative_Bias'].dropna()
    print(f"  {label:<18}  DJF bias median={djf.median():.3f} | "
          f"non-DJF median={oth.median():.3f} | Δ={djf.median()-oth.median():.3f}")
    print(f"    DJF: NSE={mt.loc['DJF','nse']:.3f}  RMSE={mt.loc['DJF','rmse']:.2f}m  "
          f"R²={mt.loc['DJF','r2']:.3f}  (n={mt.loc['DJF','n']})")
    non_djf_nse = mt.loc[['MAM','JJA','SON'],'nse'].mean()
    non_djf_rmse = mt.loc[['MAM','JJA','SON'],'rmse'].mean()
    print(f"    non-DJF avg: NSE={non_djf_nse:.3f}  RMSE={non_djf_rmse:.2f}m\n")

print("Seasonal expanded figure saved.")


# %% [markdown]
# ### 20B. Seasonal Comparison but with every record

# %%
# ── Seasonal Bias — ALL records including in-situ width < 30 m ───────────────
# This block mirrors Section 20 exactly but uses an unfiltered validation set
# (no chan_width >= 30 m restriction). Purpose: to see whether the seasonal
# bias pattern — particularly the DJF winter signal — is driven by narrow
# rivers that were excluded from the main analysis, or is consistent across
# the full width distribution.
#
# Rebuilt independently from raw_model so the global VAL dict is not affected.

def _merge_unfiltered(df_field, df_model, sites_to_exclude):
    """Same as merge_and_filter but WITHOUT the chan_width >= 30 m cut."""
    merged = pd.merge(
        df_field, df_model,
        left_on=['site_no', 'date_std'],
        right_on=['riverID', 'date_std'],
        how='inner'
    ).dropna(subset=['chan_width', 'width'])
    merged = merged[~merged['site_no'].isin(sites_to_exclude)]
    merged = merged.drop_duplicates(subset=['site_no', 'date_std'], keep='first')
    merged['Relative_Bias'] = relative_bias(merged['chan_width'], merged['width'])
    merged['bias_abs']      = merged['Relative_Bias'].abs()
    return merged.copy()

VAL_ALL = {}
for key in ('Zou_base', 'Zou_SS'):
    VAL_ALL[key] = _merge_unfiltered(df_field, raw_model[key], SITES_TO_EXCLUDE)
    n_all      = len(VAL_ALL[key])
    n_filtered = len(VAL[key])
    n_narrow   = n_all - n_filtered
    print(f"  {key:<20}: {n_all:>6,} total  |  "
          f"{n_narrow:>5,} narrow (<30 m)  |  "
          f"{n_filtered:>6,} in main analysis")

# ── Plot: identical layout to Section 20 ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                         gridspec_kw={'width_ratios': [2, 1]})
fig.suptitle('Seasonal Relative Bias — ALL Records (including in-situ width < 30 m)\n'
             'Zou 2018 Base vs CREST-S(Zou)',
             fontsize=13, fontweight='bold')

ax = axes[0]

for key_idx, (key, lbl_suffix) in enumerate([
        ('Zou_base', 'Base'),
        ('Zou_SS',   'CREST-S(Zou)'),
]):
    df_s = VAL_ALL[key].copy()
    df_s['season'] = df_s['date_std'].dt.month.map(SEASON_MAP)

    offset = -0.2 if key_idx == 0 else 0.2
    positions = [i + 1 + offset for i in range(len(SEASON_ORDER))]
    plot_data = [df_s[df_s['season'] == s]['Relative_Bias'].dropna().values
                 for s in SEASON_ORDER]

    bp = ax.boxplot(
        plot_data,
        positions=positions,
        widths=0.32,
        patch_artist=True,
        showmeans=True, showfliers=False,
        meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS
    )

    hatch = '' if key_idx == 0 else '///'
    for patch, col in zip(bp['boxes'], SEASON_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.7 if key_idx == 0 else 0.9)
        patch.set_hatch(hatch)
        patch.set_linewidth(1.2)

    for pos, d in zip(positions, plot_data):
        if len(d) == 0:
            continue
        mn  = np.mean(d)
        med = np.median(d)
        q75 = np.percentile(d, 75)
        col = 'navy' if key_idx == 0 else 'darkred'
        ax.scatter([pos], [mn], marker='D', s=60, color=col,
                   edgecolors='black', linewidths=0.6, zorder=5)
        ax.text(pos, q75 + 0.05, f'M̄={mn:.2f}\nM̃={med:.2f}',
                ha='center', fontsize=6.5, color=col, linespacing=1.3)

ax.axvspan(0.55, 1.45, color='lightblue', alpha=0.18, zorder=0, label='DJF (winter)')
ax.axhline(0, color='red', linestyle='--', linewidth=1.3, alpha=0.7)
ax.set_xticks(range(1, len(SEASON_ORDER) + 1))
ax.set_xticklabels(SEASON_LABELS_FULL)
ax.set_ylabel('Relative Bias [(Sim−Obs)/Obs]', fontweight='bold')
ax.set_xlabel('Season', fontweight='bold')
ax.set_title('Seasonal Bias Distribution — All widths (side-by-side)')

from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor='grey', alpha=0.7,              label='Zou 2018 Base'),
    Patch(facecolor='grey', alpha=0.9, hatch='////', label='CREST-S(Zou)'),
    Patch(facecolor='lightblue', alpha=0.35,         label='DJF winter band'),
]
ax.legend(handles=legend_elems, fontsize=14.4, loc='upper right')

# ── Panel B ───────────────────────────────────────────────────────────────────
ax2 = axes[1]

for key_idx, (key, color, lbl) in enumerate([
    ('Zou_base', '#B0C4DE', 'Base'),
    ('Zou_SS',   '#1F78B4', 'CREST-S(Zou)'),
]):
    df_s = VAL_ALL[key].copy()
    df_s['season'] = df_s['date_std'].dt.month.map(SEASON_MAP)

    s_means, s_medians, raw_means, raw_medians, season_stds_raw = [], [], [], [], []
    for s in SEASON_ORDER:
        rb_s = df_s[df_s['season'] == s]['Relative_Bias']
        mn_std, med_std, iqr, mn_raw, med_raw = _std_rbias(rb_s)
        s_means.append(mn_std)
        s_medians.append(med_std)
        raw_means.append(mn_raw)
        raw_medians.append(med_raw)
        season_stds_raw.append(rb_s.dropna().std() / max(iqr, IQR_FLOOR))

    y_pos = np.arange(len(SEASON_ORDER)) + key_idx * 0.35

    ax2.barh(y_pos, s_means, height=0.3,
             color=color, edgecolor='black', linewidth=0.7,
             xerr=season_stds_raw, ecolor='black', capsize=3,
             label=(f'Zou {lbl} (mean)' if lbl == 'Base' else f'{lbl} (mean)'),
             alpha=0.85)
    ax2.scatter(s_medians, y_pos, marker='o', s=60,
                color='white', edgecolors=color, linewidths=1.8, zorder=5)

    for yp, mn_r, med_r in zip(y_pos, raw_means, raw_medians):
        flag = ' *' if abs(mn_r) > IQR_FLOOR * 50 else ''
        ax2.text(0.01, yp - 0.10,
                 f'M̄={mn_r:.3f} | M̃={med_r:.3f}{flag}',
                 va='top', fontsize=6.5, color='#444444',
                 transform=ax2.get_yaxis_transform())

ax2.axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)
djf_idx = SEASON_ORDER.index('DJF')
ax2.axhspan(djf_idx - 0.1, djf_idx + 0.75, color='lightblue', alpha=0.25, zorder=0)
ax2.set_yticks(np.arange(len(SEASON_ORDER)) + 0.175)
ax2.set_yticklabels(SEASON_LABELS_FULL)
ax2.set_xlabel('Standardised Relative Bias (÷ IQR)  —  bar=mean, dot=median',
               fontweight='bold')
ax2.set_title('Standardised Seasonal Bias — All widths\n(Mean & Median, Base vs Guard)')
handles, labels = ax2.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = h
ax2.legend(seen.values(), seen.keys(), fontsize=12, loc='lower right')
ax2.invert_yaxis()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Seasonal_Bias_AllWidths.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Seasonal_Bias_AllWidths.png'),
            bbox_inches='tight', dpi=150)
plt.show()

# ── Print DJF stats with narrow-river context ─────────────────────────────────
print("\n── DJF stats (ALL records, including <30 m) ──")
for key, label in [('Zou_base', 'Base'), ('Zou_SS', '+ Guard')]:
    df_s = VAL_ALL[key].copy()
    df_s['season'] = df_s['date_std'].dt.month.map(SEASON_MAP)
    djf   = df_s[df_s['season'] == 'DJF']['Relative_Bias'].dropna()
    other = df_s[df_s['season'] != 'DJF']['Relative_Bias'].dropna()
    djf_narrow = df_s[(df_s['season'] == 'DJF') &
                      (df_s['chan_width'] < 30)]['Relative_Bias'].dropna()
    print(f"  {label:<14}  DJF median={djf.median():.3f}  non-DJF median={other.median():.3f}"
          f"  Δ={djf.median()-other.median():.3f}"
          f"  n_DJF={len(djf)}  (of which <30m: {len(djf_narrow)})")

# %% [markdown]
# ## 21. Head-to-Head: CREST-S(Zou) vs WI2015 Base (Matched Pairs)

# %%
# Merge on (site_no, date_std) to compare the two on the same observations
df_base = VAL['Zou_base'][['site_no','date_std','chan_width','width','Relative_Bias']].copy()
df_SS = VAL['Zou_SS'][['site_no','date_std','chan_width','width','Relative_Bias']].copy()

df_base = df_base.rename(columns={'width': 'width_base', 'Relative_Bias': 'rb_base'})
df_SS = df_SS.rename(columns={'width': 'width_SS', 'Relative_Bias': 'rb_SS'})

paired = pd.merge(df_base, df_SS, on=['site_no', 'date_std', 'chan_width'], how='inner')
print(f"Matched pairs for head-to-head comparison: {len(paired):,}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Head-to-Head Comparison: Zou 2018 Base vs CREST-S(Zou)\n(same observations)')

for ax, (w_col, rb_col, label, color) in zip(axes, [
    ('width_base', 'rb_base', 'Zou 2018 Base',         '#B0C4DE'),
    ('width_SS', 'rb_SS', 'CREST-S(Zou)', '#1F78B4'),
]):
    obs  = paired['chan_width'].values
    pred = paired[w_col].values
    _, r2, nse = compute_metrics(obs, pred)
    n = len(paired)

    ax.scatter(obs, pred, alpha=0.4, s=10, color=color, rasterized=True)
    lo, hi = plot_1to1(ax, obs, pred)
    annotate_metrics(ax, nse, r2, n)
    ax.set_title(label)
    ax.set_xlabel('In-situ width (m)')
    ax.set_ylabel('Predicted width (m)')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_HeadToHead_Zou_SSFilter.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_HeadToHead_Zou_SSFilter.png'),
            bbox_inches='tight', dpi=150)
plt.show()

# Bias improvement per point
paired['rb_improvement'] = paired['rb_base'].abs() - paired['rb_SS'].abs()
improved = (paired['rb_improvement'] > 0).sum()
print(f"Guard reduced |bias| for {improved:,} / {len(paired):,} pairs "
      f"({100*improved/len(paired):.1f}%)")

# %% [markdown]
# ## 23. Export Final Metric Summary CSV

# %%
# Re-compute full table for all algorithms (all sizes)
summary_rows = []
for key in ALL_KEYS:
    df = VAL[key]
    obs, pred = df['chan_width'].values, df['width'].values
    mse, r2, nse = compute_metrics(obs, pred)
    rb = df['Relative_Bias'].dropna()

    # Small / large split
    df_sm = df[df['chan_width'] <  W_THRESH]
    df_lg = df[df['chan_width'] >= W_THRESH]
    _, _, nse_sm = compute_metrics(df_sm['chan_width'], df_sm['width'])
    _, _, nse_lg = compute_metrics(df_lg['chan_width'], df_lg['width'])

    summary_rows.append({
        'Algorithm'     : ALGO_LABELS[key],
        'Key'           : key,
        'Type'          : 'Guard' if 'snow' in key else 'Base',
        'n_total'       : len(df),
        'n_small'       : len(df_sm),
        'n_large'       : len(df_lg),
        'NSE_all'       : round(nse,    4),
        'NSE_small'     : round(nse_sm, 4) if not np.isnan(nse_sm) else np.nan,
        'NSE_large'     : round(nse_lg, 4) if not np.isnan(nse_lg) else np.nan,
        'R2'            : round(r2,     4),
        'RMSE_m'        : round(np.sqrt(mse), 3),
        'Mean_Bias'     : round(rb.mean(),   4),
        'Median_Bias'   : round(rb.median(), 4),
    })

final_summary = pd.DataFrame(summary_rows).sort_values('NSE_all', ascending=False)
out_path = os.path.join(OUTPUT_DIR, 'master_final_metric_summary.csv')
final_summary.to_csv(out_path, index=False)

print("══ FINAL METRIC SUMMARY ══")
print(final_summary.to_string(index=False))
print(f"\nSaved to: {out_path}")

# %% [markdown]
# ## 24. Sediment Concentration Comparison

# %%
# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 24: Sediment Concentration Bias Analysis
# Hypothesis: Rivers in high-SSC regimes show greater width bias in base 
# algorithms; the snow/sediment guard partially corrects this.
#
# Data flow:
#   site_no → COMID        (MAPPING_CSV, already loaded as df_mapping)
#   COMID   → sword_id     (mb_to_sword NetCDF files)
#   sword_id → SSC (mg/L)  (GloRivSed CSVs, median over all scenes per reach)
# ─────────────────────────────────────────────────────────────────────────────

import xarray as xr
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── N.1 Build COMID → SWORD lookup (best match by partial overlap length) ────

MB_TO_SWORD_DIR = "/N/lustre/project/proj-212/Suraj/globalQ/data/gauge/ms_translate/ms_translate/mb_to_sword/"
GLORIVS_DIR     = "/N/lustre/project/proj-212/Suraj/globalQ/data/gauge/GloRivSed/"

print("Building COMID → SWORD lookup ...")
_records = []
for _fpath in sorted(glob.glob(MB_TO_SWORD_DIR + "*.nc")):
    _ds = xr.open_dataset(_fpath)
    _comids     = _ds["mb"].values
    _sword_cols = sorted([v for v in _ds.data_vars if v.startswith("sword_")])
    _len_cols   = sorted([v for v in _ds.data_vars if v.startswith("part_len_")])
    for _sc, _lc in zip(_sword_cols, _len_cols):
        _sids = _ds[_sc].values
        _lens = _ds[_lc].values
        _mask = (_comids > 0) & (_sids > 0) & np.isfinite(_lens) & (_lens > 0)
        if _mask.sum() == 0:
            continue
        _records.append(pd.DataFrame({
            "COMID"    : _comids[_mask],
            "sword_id" : _sids[_mask],
            "part_len" : _lens[_mask],
        }))
    _ds.close()

_mb_sword_long = pd.concat(_records, ignore_index=True)

comid_to_sword_sed = (
    _mb_sword_long
    .sort_values("part_len", ascending=False)
    .drop_duplicates(subset="COMID", keep="first")
    [["COMID", "sword_id"]]
    .reset_index(drop=True)
)
print(f"  COMID→SWORD lookup: {len(comid_to_sword_sed):,} entries")

# ── N.2 Load and aggregate GloRivSed → median SSC per SWORD reach ────────────

print("Loading GloRivSed CSVs ...")
_csv_files = glob.glob(GLORIVS_DIR + "*.csv")
if not _csv_files:
    raise FileNotFoundError(f"No CSVs found in {GLORIVS_DIR}")

_sed_chunks = []
for _f in _csv_files:
    try:
        # Read only the two columns we need; avoids loading full wide table
        _tmp = pd.read_csv(_f, usecols=["reach_ID", "y_pred"])
        _sed_chunks.append(_tmp)
    except ValueError:
        # Some files may use slightly different capitalisation — handle gracefully
        _tmp = pd.read_csv(_f)
        _col_map = {c: c.lower() for c in _tmp.columns}
        _tmp = _tmp.rename(columns=_col_map)
        _tmp = _tmp.rename(columns={"reach_id": "reach_ID", "y_pred": "y_pred"})
        _sed_chunks.append(_tmp[["reach_ID", "y_pred"]])

glorivs_raw = pd.concat(_sed_chunks, ignore_index=True)
glorivs_raw["y_pred"]   = pd.to_numeric(glorivs_raw["y_pred"],   errors="coerce")
glorivs_raw["reach_ID"] = pd.to_numeric(glorivs_raw["reach_ID"], errors="coerce")
glorivs_raw = glorivs_raw.dropna(subset=["reach_ID", "y_pred"])
glorivs_raw = glorivs_raw[glorivs_raw["y_pred"] > 0]  # drop physically invalid values

# Collapse to one representative SSC value per reach (median is robust to outliers)
glorivs_median = (
    glorivs_raw
    .groupby("reach_ID")["y_pred"]
    .median()
    .reset_index()
    .rename(columns={"reach_ID": "sword_id", "y_pred": "ssc_median_mgl"})
)
print(f"  GloRivSed: {len(glorivs_median):,} unique SWORD reaches with SSC data")
print(f"  SSC range: {glorivs_median['ssc_median_mgl'].min():.1f} – "
      f"{glorivs_median['ssc_median_mgl'].max():.1f} mg/L")

# ── N.3 Build site_no → sword_id chain ───────────────────────────────────────

# Use the same df_mapping already loaded in Block 4 (site_no + COMID columns)
# Apply the same 500 m distance filter used throughout the notebook
_mapping_clean = df_mapping[df_mapping["dist_to_comid_m"] <= 500][["site_no", "COMID"]].copy()
_mapping_clean["COMID"]   = pd.to_numeric(_mapping_clean["COMID"],   errors="coerce")
_mapping_clean["site_no"] = pd.to_numeric(_mapping_clean["site_no"], errors="coerce")
_mapping_clean = _mapping_clean.dropna().astype(int)

site_to_sword_sed = (
    _mapping_clean
    .merge(comid_to_sword_sed, on="COMID", how="inner")
    .merge(glorivs_median,     on="sword_id", how="inner")
)
print(f"\n  Sites with full chain (site_no→COMID→SWORD→SSC): "
      f"{site_to_sword_sed['site_no'].nunique():,}")

# ── N.4 Attach SSC to validation DataFrames (WI2015 base and guard) ──────────

COMPARE_PAIRS = [
    ("Zou_base",    "Zou_SS"),
    ("UltraB_base", "UltraB_SS"),
    ("MuWI_base",   "MuWI_SS"),
    ("WI2015_base", "WI2015_SS"),
]

def _attach_ssc(val_df, site_to_sword_df):
    return (
        val_df
        .merge(site_to_sword_df[["site_no", "ssc_median_mgl"]],
               on="site_no", how="inner")
        .dropna(subset=["Relative_Bias", "ssc_median_mgl"])
    )

# SSC bins: log-spaced thresholds in mg/L (adjust if your data range differs)
SSC_BINS   = [0, 25, 100, 500, np.inf]
SSC_LABELS = ["Low\n(<25)", "Medium\n(25–100)", "High\n(100–500)", "Very High\n(>500)"]
SSC_COLORS = ["#A8D5A2", "#F9E07A", "#F4A460", "#C0392B"]

# ── N.5 Plot: SSC-binned relative bias, base vs guard, per algorithm pair ─────

fig, axes = plt.subplots(
    len(COMPARE_PAIRS), 2,
    figsize=(20, 7 * len(COMPARE_PAIRS)),
    gridspec_kw={"width_ratios": [2, 1]}
)
# Ensure axes is always 2-D even for a single pair
if len(COMPARE_PAIRS) == 1:
    axes = axes[np.newaxis, :]

fig.suptitle(
    "Sediment Concentration vs Width Bias — All Algorithms\n"
    "(Base vs Snow/Sediment Filter; median SSC per SWORD reach from GloRivSed)",
    fontsize=16, fontweight="bold"
)

for row_idx, (base_key, snow_key) in enumerate(COMPARE_PAIRS):

    ax_box  = axes[row_idx, 0]   # left:  grouped boxplots per SSC bin
    ax_mean = axes[row_idx, 1]   # right: mean ± 1σ horizontal bars

    pair_label = ALGO_LABELS[base_key].replace(" (base)", "")

    # Attach SSC to each algorithm
    df_base_s = _attach_ssc(VAL[base_key], site_to_sword_sed)
    df_SS_s = _attach_ssc(VAL[snow_key], site_to_sword_sed)

    # Bin by SSC
    for _df in (df_base_s, df_SS_s):
        _df["ssc_bin"] = pd.cut(
            _df["ssc_median_mgl"], bins=SSC_BINS, labels=SSC_LABELS, right=True
        )

    # ── Panel A: grouped boxplots ─────────────────────────────────────────────
    offset = 0.22
    for key_idx, (df_s, key, hatch, lbl_suffix) in enumerate([
        (df_base_s, base_key, "",    "Base"),
        (df_SS_s, snow_key, "///", "+ Guard"),
    ]):
        positions  = [i + 1 + (key_idx - 0.5) * offset * 2
                      for i in range(len(SSC_LABELS))]
        plot_data  = [df_s[df_s["ssc_bin"] == lbl]["Relative_Bias"].dropna().values
                      for lbl in SSC_LABELS]

        bp = ax_box.boxplot(
            plot_data,
            positions=positions,
            widths=0.30,
            patch_artist=True,
            showmeans=True, showfliers=False,
            meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS,
        )
        for patch, col in zip(bp["boxes"], SSC_COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.75 if key_idx == 0 else 0.95)
            patch.set_hatch(hatch)
            patch.set_linewidth(1.2)

        # Annotate both mean (diamond) and median above each box
        for pos, d in zip(positions, plot_data):
            if len(d) == 0:
                continue
            mn  = np.mean(d)
            med = np.median(d)
            q75 = np.percentile(d, 75)
            col = "navy" if key_idx == 0 else "darkred"
            # Mean diamond marker on the plot
            ax_box.scatter([pos], [mn], marker='D', s=55, color=col,
                           edgecolors='black', linewidths=0.6, zorder=5)
            # Dual annotation stacked: mean over median
            ax_box.text(pos, q75 + 0.04, f"M̄={mn:.2f}\nM̃={med:.2f}",
                        ha="center", fontsize=6.5, color=col, linespacing=1.3)
            ax_box.text(pos, ax_box.get_ylim()[0] + 0.02, f"n={len(d)}",
                        ha="center", fontsize=6, color="grey")

    ax_box.axhline(0, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
    ax_box.set_xticks(range(1, len(SSC_LABELS) + 1))
    ax_box.set_xticklabels(SSC_LABELS)
    ax_box.set_xlabel("Suspended Sediment Concentration Regime", fontweight="bold")
    ax_box.set_ylabel("Relative Bias [(Sim−Obs)/Obs]", fontweight="bold")
    ax_box.set_title(f"{pair_label}: Bias by SSC Bin (Base vs {crest_s_label(base_key)})")

    legend_elems = [
        mpatches.Patch(facecolor="grey", alpha=0.75,            label=f"{ALGO_LABELS[base_key]} (Base)"),
        mpatches.Patch(facecolor="grey", alpha=0.95, hatch="///", label=f"{ALGO_LABELS[snow_key]}"),
    ]
    ax_box.legend(handles=legend_elems, fontsize=14.4, loc="upper left")

    # ── Panel B: standardised mean (bar) + median (dot), IQR-normalised ─────────
    y_pos = np.arange(len(SSC_LABELS))
    IQR_FLOOR_SED = 0.01

    for key_idx, (df_s, key, color, lbl_suffix) in enumerate([
        (df_base_s, base_key, ALGO_COLORS[base_key], "Base"),
        (df_SS_s, snow_key, ALGO_COLORS[snow_key], "+ Guard"),
    ]):
        s_means, s_meds, raw_means, raw_meds, s_stds = [], [], [], [], []
        for lbl in SSC_LABELS:
            rb_bin = df_s[df_s["ssc_bin"] == lbl]["Relative_Bias"].dropna()
            if len(rb_bin) > 1:
                iqr = max(np.percentile(rb_bin, 75) - np.percentile(rb_bin, 25), IQR_FLOOR_SED)
                s_means.append(rb_bin.mean()   / iqr)
                s_meds.append( rb_bin.median() / iqr)
                raw_means.append(rb_bin.mean())
                raw_meds.append( rb_bin.median())
                s_stds.append(  rb_bin.std()   / iqr)
            else:
                s_means.append(np.nan); s_meds.append(np.nan)
                raw_means.append(np.nan); raw_meds.append(np.nan)
                s_stds.append(np.nan)

        yy = y_pos + key_idx * 0.35

        # Mean as filled bar
        ax_mean.barh(yy, s_means, height=0.28,
                     color=color, edgecolor="black", linewidth=0.7,
                     xerr=s_stds, ecolor="black", capsize=3,
                     label=f"{ALGO_LABELS[key]} (mean)", alpha=0.85)

        # Median as white dot with coloured edge on top of bar
        ax_mean.scatter(s_meds, yy, marker="o", s=55,
                        color="white", edgecolors=color, linewidths=1.8, zorder=5)

        # Raw value text annotation below each bar
        for yp, mn_r, med_r in zip(yy, raw_means, raw_meds):
            if np.isfinite(mn_r):
                ax_mean.text(0.01, yp - 0.11,
                             f"M̄={mn_r:.3f} | M̃={med_r:.3f}",
                             va="top", fontsize=6.0, color="#444444",
                             transform=ax_mean.get_yaxis_transform())

    ax_mean.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_mean.set_yticks(y_pos + 0.175)
    ax_mean.set_yticklabels(SSC_LABELS)
    ax_mean.set_xlabel("Standardised Relative Bias (÷ IQR)  —  bar=mean, dot=median",
                        fontweight="bold")
    ax_mean.set_title(f"Standardised Bias per SSC Bin\n(Mean & Median, Base vs {crest_s_label(base_key)})")
    handles_m, labels_m = ax_mean.get_legend_handles_labels()
    ax_mean.legend(handles_m, labels_m, fontsize=12, loc="lower right")
    ax_mean.invert_yaxis()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "Fig_Sediment_Bias_SSC_Bins.pdf"),
            bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, "Fig_Sediment_Bias_SSC_Bins.png"),
            bbox_inches="tight", dpi=150)
plt.show()
print("Sediment bias figure saved.")

# ── N.6 Diagnostic summary ───────────────────────────────────────────────────
print("\n══ SSC BIN COVERAGE SUMMARY ══")
for base_key, snow_key in COMPARE_PAIRS:
    df_base_s = _attach_ssc(VAL[base_key], site_to_sword_sed)
    df_base_s["ssc_bin"] = pd.cut(df_base_s["ssc_median_mgl"],
                                   bins=SSC_BINS, labels=SSC_LABELS)
    print(f"\n{ALGO_LABELS[base_key]}:")
    print(df_base_s.groupby("ssc_bin", observed=True)["Relative_Bias"]
          .agg(n="count", mean="mean", median="median", std="std")
          .round(3).to_string())

# %%
# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 24B: Date-Matched Sediment Concentration Bias Analysis
# Unlike Block 24 (median SSC per reach, all-time), this version matches each
# validation observation to the GloRivSed scene closest in time, within a
# tolerance window. This answers: "was the river actually turbid on the day
# of this specific width measurement?" rather than "is this reach generally
# a high-sediment river?"
# ─────────────────────────────────────────────────────────────────────────────

DATE_TOLERANCE_DAYS = 8  # +/- window around the validation date to search for
                          # a GloRivSed scene. Increase if coverage is too thin.

# ── B.1 Reload GloRivSed WITH date column (Block 24 only kept reach_ID + y_pred) ──
print("Loading GloRivSed CSVs with date ...")
_sed_chunks_dt = []
for _f in glob.glob(GLORIVS_DIR + "*.csv"):
    try:
        _tmp = pd.read_csv(_f, usecols=["reach_ID", "y_pred", "date"])
    except ValueError:
        _tmp = pd.read_csv(_f)
        _tmp.columns = [c.lower() for c in _tmp.columns]
        _tmp = _tmp.rename(columns={"reach_id": "reach_ID"})
        _tmp = _tmp[["reach_ID", "y_pred", "date"]]
    _sed_chunks_dt.append(_tmp)

glorivs_dated = pd.concat(_sed_chunks_dt, ignore_index=True)
glorivs_dated["reach_ID"] = pd.to_numeric(glorivs_dated["reach_ID"], errors="coerce")
glorivs_dated["y_pred"]   = pd.to_numeric(glorivs_dated["y_pred"],   errors="coerce")
glorivs_dated["date"]     = pd.to_datetime(glorivs_dated["date"], format="%Y-%m-%d", errors="coerce")
glorivs_dated = glorivs_dated.dropna(subset=["reach_ID", "y_pred", "date"])
glorivs_dated = glorivs_dated[glorivs_dated["y_pred"] > 0]
glorivs_dated = glorivs_dated.rename(columns={"reach_ID": "sword_id"})

print(f"  GloRivSed dated records: {len(glorivs_dated):,}")
print(f"  Date range: {glorivs_dated['date'].min().date()} – {glorivs_dated['date'].max().date()}")

# ── B.2 site_no -> sword_id chain (reuse the lookup already built in Block 24) ──
# site_to_sword_sed already exists from Block 24 (site_no, COMID, sword_id, ssc_median_mgl)
site_to_sword_only = site_to_sword_sed[["site_no", "sword_id"]].drop_duplicates()

# ── B.3 Nearest-date merge per algorithm ──────────────────────────────────────
def attach_dated_ssc(val_df, site_to_sword_df, glorivs_df, tol_days):
    """
    For each validation row, find the GloRivSed scene for the SAME sword_id
    that is closest in time, within +/- tol_days. Uses merge_asof per-group
    since merge_asof requires sorted, single-key matching.
    """
    df = val_df.merge(site_to_sword_df, on="site_no", how="inner").copy()
    df["date_std"] = pd.to_datetime(df["date_std"], errors="coerce")
    df = df.dropna(subset=["date_std", "sword_id", "Relative_Bias"])

    matched_chunks = []
    # Group by sword_id so merge_asof's nearest-match logic is scoped correctly
    for sid, df_grp in df.groupby("sword_id"):
        sed_grp = glorivs_df[glorivs_df["sword_id"] == sid]
        if sed_grp.empty:
            continue
        df_grp   = df_grp.sort_values("date_std")
        sed_grp  = sed_grp.sort_values("date")
        merged = pd.merge_asof(
            df_grp, sed_grp[["date", "y_pred"]],
            left_on="date_std", right_on="date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tol_days),
        )
        matched_chunks.append(merged)

    if not matched_chunks:
        return pd.DataFrame(columns=list(df.columns) + ["date", "y_pred"])

    out = pd.concat(matched_chunks, ignore_index=True)
    out = out.dropna(subset=["y_pred"])  # drop rows where no scene fell within tolerance
    out = out.rename(columns={"y_pred": "ssc_matched_mgl"})
    return out

print(f"\nMatching validation observations to GloRivSed scenes "
      f"within +/-{DATE_TOLERANCE_DAYS} days ...")

dated_data = {}
for base_key, snow_key in COMPARE_PAIRS:
    for key in (base_key, snow_key):
        dated_data[key] = attach_dated_ssc(
            VAL[key], site_to_sword_only, glorivs_dated, DATE_TOLERANCE_DAYS
        )
        n_total   = len(VAL[key])
        n_matched = len(dated_data[key])
        print(f"  {ALGO_LABELS[key]:30s}  matched {n_matched:6,} / {n_total:6,} "
              f"({100*n_matched/n_total:5.1f}%)")

# ── B.4 Same SSC bins as Block 24, applied to the date-matched data ──────────
for key, df in dated_data.items():
    if len(df) == 0:
        continue
    df["ssc_bin"] = pd.cut(df["ssc_matched_mgl"], bins=SSC_BINS, labels=SSC_LABELS, right=True)

# ── B.5 Plot: same layout as Block 24, but date-matched SSC ──────────────────
fig, axes = plt.subplots(
    len(COMPARE_PAIRS), 2,
    figsize=(20, 7 * len(COMPARE_PAIRS)),
    gridspec_kw={"width_ratios": [2, 1]}
)
if len(COMPARE_PAIRS) == 1:
    axes = axes[np.newaxis, :]

fig.suptitle(
    f"Sediment Concentration vs Width Bias — All Algorithms (Date-Matched, ±{DATE_TOLERANCE_DAYS}d)\n"
    "(SSC matched to nearest GloRivSed scene per observation, not all-time reach median)",
    fontsize=16, fontweight="bold"
)

for row_idx, (base_key, snow_key) in enumerate(COMPARE_PAIRS):
    ax_box  = axes[row_idx, 0]
    ax_mean = axes[row_idx, 1]
    pair_label = ALGO_LABELS[base_key].replace(" (base)", "")

    df_base_d = dated_data[base_key]
    df_SS_d = dated_data[snow_key]

    offset = 0.22
    for key_idx, (df_d, key, hatch) in enumerate([
        (df_base_d, base_key, ""),
        (df_SS_d, snow_key, "///"),
    ]):
        positions = [i + 1 + (key_idx - 0.5) * offset * 2 for i in range(len(SSC_LABELS))]
        plot_data = [df_d[df_d["ssc_bin"] == lbl]["Relative_Bias"].dropna().values
                     if len(df_d) > 0 else np.array([])
                     for lbl in SSC_LABELS]

        bp = ax_box.boxplot(
            plot_data, positions=positions, widths=0.30, patch_artist=True,
            showmeans=True, showfliers=False,
            meanprops=MEAN_PROPS, medianprops=MEDIAN_PROPS,
        )
        for patch, col in zip(bp["boxes"], SSC_COLORS):
            patch.set_facecolor(col)
            patch.set_alpha(0.75 if key_idx == 0 else 0.95)
            patch.set_hatch(hatch)
            patch.set_linewidth(1.2)

        for pos, d in zip(positions, plot_data):
            if len(d) == 0:
                continue
            ax_box.text(pos, ax_box.get_ylim()[0] + 0.02, f"n={len(d)}",
                        ha="center", fontsize=7, color="grey")

    ax_box.axhline(0, color="red", linestyle="--", linewidth=1.3, alpha=0.7)
    ax_box.set_xticks(range(1, len(SSC_LABELS) + 1))
    ax_box.set_xticklabels(SSC_LABELS, fontsize=14.4, fontweight="bold")
    ax_box.set_xlabel("Suspended Sediment Concentration Regime (date-matched)",
                       fontweight="bold", fontsize=15.6)
    ax_box.set_ylabel("Relative Bias [(Sim−Obs)/Obs]", fontweight="bold", fontsize=15.6)
    ax_box.set_title(f"{pair_label}: Bias by SSC Bin — Date-Matched", fontsize=14, fontweight="bold")

    legend_elems = [
        mpatches.Patch(facecolor="grey", alpha=0.75, label=f"{ALGO_LABELS[base_key]} (Base)"),
        mpatches.Patch(facecolor="grey", alpha=0.95, hatch="///", label=f"{ALGO_LABELS[snow_key]}"),
    ]
    ax_box.legend(handles=legend_elems, fontsize=13.2, loc="upper left")

    y_pos = np.arange(len(SSC_LABELS))
    for key_idx, (df_d, key, color) in enumerate([
        (df_base_d, base_key, ALGO_COLORS[base_key]),
        (df_SS_d, snow_key, ALGO_COLORS[snow_key]),
    ]):
        s_means, s_meds, raw_means, raw_meds, s_stds = [], [], [], [], []
        IQR_FLOOR_D = 0.01
        for lbl in SSC_LABELS:
            rb_bin = df_d[df_d["ssc_bin"] == lbl]["Relative_Bias"].dropna() if len(df_d) > 0 else pd.Series([], dtype=float)
            if len(rb_bin) > 1:
                iqr = max(np.percentile(rb_bin, 75) - np.percentile(rb_bin, 25), IQR_FLOOR_D)
                s_means.append(rb_bin.mean()   / iqr)
                s_meds.append( rb_bin.median() / iqr)
                raw_means.append(rb_bin.mean())
                raw_meds.append( rb_bin.median())
                s_stds.append(  rb_bin.std()   / iqr)
            else:
                s_means.append(np.nan); s_meds.append(np.nan)
                raw_means.append(np.nan); raw_meds.append(np.nan)
                s_stds.append(np.nan)

        yy = y_pos + key_idx * 0.35
        ax_mean.barh(yy, s_means, height=0.28, color=color, edgecolor="black",
                     linewidth=0.7, xerr=s_stds, ecolor="black", capsize=3,
                     label=f"{ALGO_LABELS[key]} (mean)", alpha=0.85)
        ax_mean.scatter(s_meds, yy, marker="o", s=55,
                        color="white", edgecolors=color, linewidths=1.8, zorder=5)
        for yp, mn_r, med_r in zip(yy, raw_means, raw_meds):
            if np.isfinite(mn_r):
                ax_mean.text(0.01, yp - 0.11,
                             f"M̄={mn_r:.3f} | M̃={med_r:.3f}",
                             va="top", fontsize=6.0, color="#444444",
                             transform=ax_mean.get_yaxis_transform())

    ax_mean.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_mean.set_yticks(y_pos + 0.175)
    ax_mean.set_yticklabels(SSC_LABELS, fontsize=14.4, fontweight="bold")
    ax_mean.set_xlabel("Standardised Relative Bias (÷ IQR)  —  bar=mean, dot=median",
                        fontweight="bold", fontsize=14.4)
    ax_mean.set_title("Standardised Bias per SSC Bin\n(Mean & Median, Date-Matched, Bin\n(Date-Matched)", fontsize=14, fontweight="bold")
    ax_mean.legend(fontsize=13.2, loc="lower right")
    ax_mean.invert_yaxis()

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, f"Fig_Sediment_Bias_DateMatched_{DATE_TOLERANCE_DAYS}d.pdf"),
            bbox_inches="tight", dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, f"Fig_Sediment_Bias_DateMatched_{DATE_TOLERANCE_DAYS}d.png"),
            bbox_inches="tight", dpi=150)
plt.show()
print("Date-matched sediment bias figure saved.")

# ── B.6 Coverage comparison: all-time median vs date-matched ────────────────
print("\n══ COVERAGE COMPARISON: All-Time Median vs Date-Matched ══")
for base_key, _ in COMPARE_PAIRS:
    n_alltime = len(_attach_ssc(VAL[base_key], site_to_sword_sed))
    n_dated   = len(dated_data[base_key])
    print(f"{ALGO_LABELS[base_key]:25s}  all-time: {n_alltime:6,}   "
          f"date-matched (±{DATE_TOLERANCE_DAYS}d): {n_dated:6,}   "
          f"retained: {100*n_dated/max(n_alltime,1):5.1f}%")

# %%
# ─────────────────────────────────────────────────────────────────────────────
# BLOCK 24C (revised): COMID / Date pairs where Base vs Guard width predictions
# are most similar and most different, restricted to HIGH SSC zones only.
#
# NOTE: SSC here uses the all-time median per reach (same lookup as Block 24),
# NOT date-matched SSC — date-matching retained only ~20-30% of observations,
# too sparse to support a reliable "most similar / most different" ranking.
# The date attached to each row below is the real USGS observation date;
# the SSC label is a long-term regime classification of that reach, not a
# same-day reading.
# ─────────────────────────────────────────────────────────────────────────────

REF_BASE_KEY = "Zou_base"
REF_SS_KEY = "Zou_SS"
N_PAIRS_EACH = 15   # how many "most similar" and "most different" examples to return

# Attach all-time median SSC (same helper + lookup as Block 24)
df_base_ref = _attach_ssc(VAL[REF_BASE_KEY], site_to_sword_sed).copy()
df_SS_ref = _attach_ssc(VAL[REF_SS_KEY], site_to_sword_sed).copy()

df_base_ref["ssc_bin"] = pd.cut(df_base_ref["ssc_median_mgl"], bins=SSC_BINS, labels=SSC_LABELS, right=True)
df_SS_ref["ssc_bin"] = pd.cut(df_SS_ref["ssc_median_mgl"], bins=SSC_BINS, labels=SSC_LABELS, right=True)

# Restrict to High + Very High SSC bins only, as requested
HIGH_SSC_LABELS = ["High\n(100–500)", "Very High\n(>500)"]
df_base_high = df_base_ref[df_base_ref["ssc_bin"].isin(HIGH_SSC_LABELS)]
df_SS_high = df_SS_ref[df_SS_ref["ssc_bin"].isin(HIGH_SSC_LABELS)]

# Match base and guard on the SAME observation (site_no + date_std), so we are
# comparing the algorithm's behaviour on identical river/time instances.
# NOTE: sword_id is NOT included as a merge key here — _attach_ssc() only
# carries through site_no and ssc_median_mgl, so sword_id isn't present on
# either side. site_no already uniquely determines the reach, so this is
# not a loss of specificity.
matched = df_base_high.merge(
    df_SS_high,
    on=["site_no", "date_std"],
    suffixes=("_base", "_guard"),
    how="inner"
)

if matched.empty:
    print("No matched High/Very-High SSC observations found.\n"
          "Check that SSC_BINS thresholds and site_to_sword_sed coverage are as expected.")
else:
    # Pull COMID (and sword_id) back in for reporting
    matched = matched.merge(
        site_to_sword_sed[["site_no", "COMID", "sword_id"]].drop_duplicates(),
        on="site_no", how="left"
    )

    # Sanity check: same physical observation should have identical observed width
    # regardless of which algorithm pair it came from
    assert (matched["chan_width_base"] == matched["chan_width_guard"]).all(), \
        "chan_width mismatch between base and guard merge — check merge keys."

    # Definition: difference in algorithm behaviour = |Relative_Bias_base - Relative_Bias_guard|
    # Small value -> guard barely changed the prediction (guard "neutral" here)
    # Large value -> guard substantially changed the prediction (guard "active" here)
    matched["abs_diff_rbias"] = (matched["Relative_Bias_base"] - matched["Relative_Bias_guard"]).abs()

    cols_report = [
        "COMID", "sword_id", "date_std",
        "ssc_median_mgl_base",
        "chan_width_base", "width_base", "Relative_Bias_base",
        "width_guard", "Relative_Bias_guard",
        "abs_diff_rbias",
    ]
    matched_report = matched[cols_report].rename(columns={
        "ssc_median_mgl_base": "ssc_median_mgl",
        "chan_width_base": "observed_width_m",
        "width_base": "pred_width_base_m",
        "width_guard": "pred_width_guard_m",
    })

    most_similar   = matched_report.nsmallest(N_PAIRS_EACH, "abs_diff_rbias").reset_index(drop=True)
    most_different = matched_report.nlargest(N_PAIRS_EACH, "abs_diff_rbias").reset_index(drop=True)

    print(f"High-SSC matched observations available: {len(matched_report):,}\n")

    print(f"══ MOST SIMILAR (Base ≈ Guard) — Top {N_PAIRS_EACH}, High SSC only ══")
    print(most_similar.round(3).to_string(index=False))

    print(f"\n\n══ MOST DIFFERENT (Base ≠ Guard) — Top {N_PAIRS_EACH}, High SSC only ══")
    print(most_different.round(3).to_string(index=False))

    most_similar.to_csv(os.path.join(OUTPUT_DIR, "HighSSC_MostSimilar_BaseGuard.csv"), index=False)
    most_different.to_csv(os.path.join(OUTPUT_DIR, "HighSSC_MostDifferent_BaseGuard.csv"), index=False)
    print(f"\nSaved both lists to {OUTPUT_DIR}")

# %% [markdown]
# ## 25. Base → Snow → SS Trajectory Analysis
# 
# These cells use the intermediate **Snow Guard** CSVs to trace how each algorithm
# improves (or changes) across three stages:
# 
# | Stage | Description |
# |---|---|
# | **Base** | Original algorithm, no correction |
# | **Snow** | Snow contamination guard applied |
# | **SS** | Snow + Sediment guard applied (final) |
# 
# > **Important:** Snow Guard data is used **only** in this section and the COMID
# > trajectory segments below. All earlier figures compare Base vs SS only.

# %% [markdown]
# ### 25a. Metric Comparison — Base vs Snow vs SS (Bar Chart)
# 
# Side-by-side bars for all three stages, one panel per metric.

# %%
# ── Pre-compute metrics for all 12 keys ──────────────────────────────────────
def get_metrics_dict(keys):
    out = {}
    for k in keys:
        if k not in VAL:
            continue
        obs  = VAL[k]['chan_width'].values
        pred = VAL[k]['width'].values
        mse, r2, nse = compute_metrics(obs, pred)
        out[k] = {
            'NSE' : nse,
            'R2'  : r2,
            'RMSE': np.sqrt(mse),
            'MeanBias': VAL[k]['Relative_Bias'].mean(),
        }
    return out

metrics_all12 = get_metrics_dict(BASE_KEYS + SNOW_KEYS + SS_KEYS)

fam_labels = ['Zou 2018', 'Ultra-Blue', 'MuWI', 'WI2015']
metric_list = [('NSE', 'NSE'), ('R2', 'R²'), ('RMSE', 'RMSE (m)'), ('MeanBias', 'Mean Relative Bias')]

x    = np.arange(len(TRAJ_PAIRS))
w    = 0.22
offsets = [-w, 0, w]
stage_labels = ['Base', '+ Snow Guard', 'CREST-S']
stage_colors = ['#AAAAAA', '#6BAED6', '#1F78B4']   # grey → mid-blue → vivid

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Figure — Metric Trajectory: Base → Snow Guard → CREST-S\n'
             'All 4 Algorithm Families', fontsize=13, fontweight='bold')

for ax_i, (mkey, mlabel) in enumerate(metric_list):
    ax = axes[ax_i // 2, ax_i % 2]

    for stage_i, (keys, color, slabel) in enumerate(
            zip([BASE_KEYS, SNOW_KEYS, SS_KEYS], stage_colors, stage_labels)):
        vals = [metrics_all12.get(k, {}).get(mkey, np.nan) for k in keys]
        bars = ax.bar(x + offsets[stage_i], vals, w,
                      color=color, edgecolor='black', linewidth=0.6,
                      label=slabel, alpha=0.88)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.005 if v >= 0 else -0.03),
                        f'{v:.3f}', ha='center', va='bottom',
                        fontsize=6.5, rotation=90)

    if mkey == 'MeanBias':
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    elif mkey in ('NSE', 'R2'):
        ax.axhline(1, color='grey', linewidth=0.6, linestyle=':', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(fam_labels, fontsize=10.8, fontweight='bold')
    ax.set_ylabel(mlabel, fontsize=10.8, fontweight='bold')
    ax.set_title(mlabel, fontsize=10, fontweight='bold')
    ax.legend(fontsize=9.6)

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig25a_Trajectory_Metrics_Bar.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig25a_Trajectory_Metrics_Bar.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 25a saved.")


# %% [markdown]
# ### 25b. Metric Trajectory — Dot-Line (Base → Snow → SS)
# 
# Each row = one algorithm family. Three dots connected by lines show direction and magnitude of change at each stage.

# %%
from matplotlib.lines import Line2D

fig, axes = plt.subplots(1, 4, figsize=(20, 7), sharey=True)
fig.suptitle('Figure — Metric Trajectory per Algorithm Family\n'
             '(Base → Snow Guard → CREST-S)',
             fontsize=13, fontweight='bold')

metric_list_traj = [
    ('NSE',      'NSE',                   True),
    ('R2',       'R²',                    True),
    ('RMSE',     'RMSE (m)',              False),
    ('MeanBias', 'Mean Relative Bias',    False),
]

y_ticks = np.arange(len(TRAJ_PAIRS))
stage_markers = ['s', 'o', 'D']   # base=square, snow=circle, SS=diamond
stage_sizes   = [90, 80, 110]
stage_labels  = ['Base', '+ Snow Guard', 'CREST-S']

for ax_i, (mkey, mlabel, ref1) in enumerate(metric_list_traj):
    ax = axes[ax_i]

    for row_i, (bk, sk, gk) in enumerate(TRAJ_PAIRS):
        vals = [
            metrics_all12.get(bk, {}).get(mkey, np.nan),
            metrics_all12.get(sk, {}).get(mkey, np.nan),
            metrics_all12.get(gk, {}).get(mkey, np.nan),
        ]
        # Connecting line
        ax.plot(vals, [row_i] * 3, color=ALGO_COLORS[gk],
                linewidth=1.8, zorder=1, alpha=0.7)

        for stage_i, (v, mk, sz) in enumerate(zip(vals, stage_markers, stage_sizes)):
            key_for_color = [bk, sk, gk][stage_i]
            ax.scatter(v, row_i, marker=mk, s=sz, zorder=3,
                       color=ALGO_COLORS[key_for_color],
                       edgecolors='black', linewidths=0.8)

            # Annotate value next to dot
            ax.text(v, row_i + 0.15, f'{v:.3f}', ha='center',
                    fontsize=6.5, color=ALGO_COLORS[key_for_color],
                    fontweight='bold')

    if ref1:
        ax.axvline(1, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)
    ax.axvline(0, color='black', linewidth=0.6, linestyle='--', alpha=0.4)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(fam_labels if ax_i == 0 else [], fontsize=10.8)
    ax.set_xlabel(mlabel, fontsize=10.8, fontweight='bold')
    ax.set_title(mlabel, fontsize=10, fontweight='bold')
    ax.invert_yaxis()

# Legend
legend_elems = [
    Line2D([0],[0], marker='s', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=9, label='Base'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=9, label='+ Snow Guard'),
    Line2D([0],[0], marker='D', color='w', markerfacecolor='grey',
           markeredgecolor='black', markersize=9, label='CREST-S'),
]
axes[-1].legend(handles=legend_elems, fontsize=10.8, loc='lower right',
                bbox_to_anchor=(1.0, 0.0))

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig25b_Trajectory_DotLine.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig25b_Trajectory_DotLine.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 25b saved.")


# %% [markdown]
# ## 26. COMID Trajectory Records — Base → Snow → SS
# 
# Three lookup tables showing individual (COMID, date) pairs where the
# relative bias follows a specific trajectory across the three algorithm stages.
# 
# | Segment | Trajectory | Interpretation |
# |---|---|---|
# | **26A** | Base → Snow ↓ → SS ↓ | Consistently improving |
# | **26B** | Base → Snow ↓ → SS ↑ | Snow helps, sediment guard hurts |
# | **26C** | Base → Snow ↑ → SS ↓ | Snow hurts, sediment guard recovers |
# 
# **Definition of improvement:** `|Relative_Bias|` decreases (prediction gets
# closer to observed). A decrease in `|RBias|` is improvement regardless of sign.
# 
# > Uses Zou 2018 as the reference algorithm family. Change `REF_FAM` below
# > to switch to a different family.

# %%
# ── Configuration ─────────────────────────────────────────────────────────────
REF_FAM      = ('Zou_base', 'Zou_snow', 'Zou_SS')   # change family here if needed
N_SHOW       = 20    # rows to print per segment
ABS_IMPROVE_THRESH = 0.01  # minimum |ΔRBias| to count as a real change (avoids noise)

bk, sk, gk = REF_FAM

# ── Build matched-triplet dataframe ───────────────────────────────────────────
# Each row = one (site_no, date_std) observation present in ALL THREE stages.
# We keep observed width from base (all three should be identical — same field obs).

df_b = VAL[bk][['site_no', 'date_std', 'chan_width', 'width', 'Relative_Bias']].copy()
df_s = VAL[sk][['site_no', 'date_std', 'width', 'Relative_Bias']].copy()
df_g = VAL[gk][['site_no', 'date_std', 'width', 'Relative_Bias']].copy()

df_b = df_b.rename(columns={'width':'pred_base', 'Relative_Bias':'rb_base'})
df_s = df_s.rename(columns={'width':'pred_snow', 'Relative_Bias':'rb_snow'})
df_g = df_g.rename(columns={'width':'pred_ss',   'Relative_Bias':'rb_ss'})

triplets = (
    df_b
    .merge(df_s, on=['site_no','date_std'], how='inner')
    .merge(df_g, on=['site_no','date_std'], how='inner')
    .dropna(subset=['rb_base','rb_snow','rb_ss'])
)

# Add COMID via the mapping table already loaded
_map = df_mapping[['site_no','COMID']].copy()
_map['site_no'] = pd.to_numeric(_map['site_no'], errors='coerce')
_map['COMID']   = pd.to_numeric(_map['COMID'],   errors='coerce')
_map = _map.dropna().astype(int).drop_duplicates('site_no')
triplets = triplets.merge(_map, on='site_no', how='left')

# Absolute bias columns
triplets['abs_base'] = triplets['rb_base'].abs()
triplets['abs_snow'] = triplets['rb_snow'].abs()
triplets['abs_ss']   = triplets['rb_ss'].abs()

# Stage-to-stage deltas (negative = improvement, positive = worsening)
triplets['d_base_snow'] = triplets['abs_snow'] - triplets['abs_base']  # Base→Snow
triplets['d_snow_ss']   = triplets['abs_ss']   - triplets['abs_snow']  # Snow→SS

print(f"Matched triplet observations: {len(triplets):,}")
print(f"Unique sites (COMIDs): {triplets['site_no'].nunique()}")

# ── Segment masks ─────────────────────────────────────────────────────────────
# Use threshold to avoid flagging noise as a real change
T = ABS_IMPROVE_THRESH

seg_A = triplets[(triplets['d_base_snow'] < -T) & (triplets['d_snow_ss'] < -T)]   # both improve
seg_B = triplets[(triplets['d_base_snow'] < -T) & (triplets['d_snow_ss'] >  T)]   # snow↑ SS↓
seg_C = triplets[(triplets['d_base_snow'] >  T) & (triplets['d_snow_ss'] < -T)]   # snow↓ SS↑

print(f"\nSegment A (consistent improvement Base→Snow→SS) : {len(seg_A):>6,} records")
print(f"Segment B (Snow improves, SS reverts)            : {len(seg_B):>6,} records")
print(f"Segment C (Snow worsens,  SS recovers)           : {len(seg_C):>6,} records")

REPORT_COLS = [
    'COMID', 'site_no', 'date_std', 'chan_width',
    'pred_base', 'rb_base',
    'pred_snow', 'rb_snow',
    'pred_ss',   'rb_ss',
    'd_base_snow', 'd_snow_ss',
]

def _fmt_segment(df, label, sort_col, ascending):
    """Sort by magnitude of change, show top N_SHOW, save CSV."""
    out = df[REPORT_COLS].copy()
    out = out.sort_values(sort_col, ascending=ascending).head(N_SHOW).reset_index(drop=True)
    out = out.round(4)
    fname = os.path.join(OUTPUT_DIR, f'COMID_Traj_{label}.csv')
    out.to_csv(fname, index=False)
    print(f"\n══ {label} — top {N_SHOW} by |Δ| ══")
    print(out.to_string(index=False))
    print(f"  → saved to {fname}")
    return out

# Sort by: largest total improvement for A, largest Base→Snow gain for B and C
seg_A_report = _fmt_segment(
    seg_A.assign(_total=lambda d: d['abs_base'] - d['abs_ss']),
    label     = 'A_Consistent_Improvement',
    sort_col  = 'd_snow_ss',   # largest SS improvement
    ascending = True
)

seg_B_report = _fmt_segment(
    seg_B,
    label     = 'B_SnowHelps_SSHurts',
    sort_col  = 'd_snow_ss',   # largest SS reversion (most positive)
    ascending = False
)

seg_C_report = _fmt_segment(
    seg_C,
    label     = 'C_SnowHurts_SSRecovers',
    sort_col  = 'd_snow_ss',   # largest SS recovery (most negative)
    ascending = True
)


# %% [markdown]
# ### 26D. Trajectory Visualisation — |RBias| Bump Chart
# 
# Sample of records from each segment plotted as connected dots across the three algorithm stages, coloured by segment. Makes the trajectory pattern immediately visible.

# %%
# ── Sample up to 30 records per segment for readability ──────────────────────
N_SAMPLE = 30
np.random.seed(42)

def _sample(df, n):
    return df.sample(min(n, len(df)), random_state=42) if len(df) > 0 else df

sA = _sample(seg_A, N_SAMPLE)
sB = _sample(seg_B, N_SAMPLE)
sC = _sample(seg_C, N_SAMPLE)

seg_info = [
    (sA, '#2CA02C', 'A: Base→Snow↓ → SS↓ (consistent improvement)'),
    (sB, '#FF7F0E', 'B: Base→Snow↓ → SS↑ (Snow helps, SS reverts)'),
    (sC, '#D62728', 'C: Base→Snow↑ → SS↓ (Snow hurts, SS recovers)'),
]

stage_x = [0, 1, 2]
stage_names = ['Base', '+ Snow Guard', 'CREST-S']

fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=False)
fig.suptitle('COMID Trajectory: |Relative Bias| across Algorithm Stages\n'
             f'(Sample of {N_SAMPLE} records per segment)',
             fontsize=13, fontweight='bold')

for ax, (df_seg, color, title) in zip(axes, seg_info):
    for _, row in df_seg.iterrows():
        vals = [row['abs_base'], row['abs_snow'], row['abs_ss']]
        ax.plot(stage_x, vals, color=color, alpha=0.25, linewidth=1.0)

    # Median trajectory
    med_vals = [
        df_seg['abs_base'].median(),
        df_seg['abs_snow'].median(),
        df_seg['abs_ss'].median(),
    ]
    ax.plot(stage_x, med_vals, color=color, linewidth=3.0,
            marker='D', markersize=9, markeredgecolor='black',
            label='Median trajectory', zorder=5)

    # Annotate medians
    for xi, mv in zip(stage_x, med_vals):
        ax.text(xi, mv + 0.01, f'{mv:.3f}',
                ha='center', fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(stage_x)
    ax.set_xticklabels(stage_names, fontsize=12, fontweight='bold')
    ax.set_ylabel('|Relative Bias|', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=9, fontweight='bold', wrap=True)
    ax.set_xlim(-0.3, 2.3)
    ax.legend(fontsize=10.8)
    n = len(df_seg)
    ax.text(0.98, 0.98, f'n shown = {n}', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color='grey')

plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig26_COMID_Trajectory_BumpChart.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig26_COMID_Trajectory_BumpChart.png'),
            bbox_inches='tight', dpi=150)
plt.show()
print("Figure 26 (bump chart) saved.")


# %% [markdown]
# ### 27. Coverage Differences
# 
# 

# %%
# ── Coverage Comparison: Records present in Base but not Snow/SS, and vice versa ──
# Uses the same REF_FAM defined in the previous cell.
# "Record" = a unique (site_no, date_std) pair.

bk, sk, gk = REF_FAM

def _key_set(val_key):
    return set(
        zip(VAL[val_key]['site_no'].astype(str),
            VAL[val_key]['date_std'].astype(str))
    )

keys_base = _key_set(bk)
keys_snow = _key_set(sk)
keys_ss   = _key_set(gk)

# Records in Snow/SS but absent from Base (new coverage gained by modifications)
in_snow_not_base = keys_snow - keys_base
in_ss_not_base   = keys_ss   - keys_base
in_either_not_base = (keys_snow | keys_ss) - keys_base

# Records in Base but absent from Snow or SS (coverage lost after modification)
in_base_not_snow = keys_base - keys_snow
in_base_not_ss   = keys_base - keys_ss
in_base_not_either = keys_base - (keys_snow | keys_ss)

print("══ COVERAGE DIFFERENCES ══")
print(f"\nAlgorithm family : {ALGO_LABELS[bk].replace(' (base)', '')} "
      f"(Base={bk}, Snow={sk}, SS={gk})")
print(f"\nTotal unique (site_no, date) records:")
print(f"  Base : {len(keys_base):>7,}")
print(f"  Snow : {len(keys_snow):>7,}")
print(f"  SS   : {len(keys_ss):>7,}")

print(f"\n── Records IN Snow/SS but NOT in Base (new records after modification) ──")
print(f"  In Snow, not Base          : {len(in_snow_not_base):>6,}")
print(f"  In SS,   not Base          : {len(in_ss_not_base):>6,}")
print(f"  In Snow or SS, not Base    : {len(in_either_not_base):>6,}")

print(f"\n── Records IN Base but NOT in Snow/SS (records dropped after modification) ──")
print(f"  In Base, not Snow          : {len(in_base_not_snow):>6,}")
print(f"  In Base, not SS            : {len(in_base_not_ss):>6,}")
print(f"  In Base, not Snow nor SS   : {len(in_base_not_either):>6,}")

# ── Convert to DataFrames for inspection ──────────────────────────────────────
def _pairs_to_df(pair_set, label):
    if not pair_set:
        print(f"\n  {label}: no records")
        return pd.DataFrame()
    df = pd.DataFrame(list(pair_set), columns=['site_no', 'date_std'])
    df['site_no']  = pd.to_numeric(df['site_no'],  errors='coerce')
    df['date_std'] = pd.to_datetime(df['date_std'], errors='coerce')
    df = df.sort_values(['site_no', 'date_std']).reset_index(drop=True)
    # Attach COMID for context
    _map = df_mapping[['site_no','COMID']].copy()
    _map['site_no'] = pd.to_numeric(_map['site_no'], errors='coerce')
    _map['COMID']   = pd.to_numeric(_map['COMID'],   errors='coerce')
    _map = _map.dropna().astype(int).drop_duplicates('site_no')
    df = df.merge(_map, on='site_no', how='left')
    fname = os.path.join(OUTPUT_DIR, f'Coverage_{label}.csv')
    df.to_csv(fname, index=False)
    print(f"\n══ {label} — first 20 rows ══")
    print(df.head(20).to_string(index=False))
    print(f"  → full list saved to {fname}")
    return df

new_in_snow   = _pairs_to_df(in_snow_not_base,   'InSnow_NotBase')
new_in_ss     = _pairs_to_df(in_ss_not_base,     'InSS_NotBase')
lost_in_snow  = _pairs_to_df(in_base_not_snow,   'InBase_NotSnow')
lost_in_ss    = _pairs_to_df(in_base_not_ss,     'InBase_NotSS')
lost_in_both  = _pairs_to_df(in_base_not_either, 'InBase_NotSnowNorSS')

# ── Per-site summary: how many dates each site gained/lost ───────────────────
print("\n══ PER-SITE COVERAGE CHANGE SUMMARY (top 15 by records gained/lost) ══")
for label, df in [('Gained in Snow vs Base', new_in_snow),
                   ('Gained in SS vs Base',   new_in_ss),
                   ('Lost in Snow vs Base',   lost_in_snow),
                   ('Lost in SS vs Base',     lost_in_ss)]:
    if df.empty:
        continue
    site_counts = df.groupby('site_no').size().sort_values(ascending=False).head(15)
    print(f"\n  {label}:")
    print(site_counts.to_string())

# %% [markdown]
# ### 28. Segment statistics as percentages of total validation records

# %%
# ── Trajectory Segment Statistics — % of total validation records ─────────────
# Depends on: triplets, seg_A, seg_B, seg_C, T (from the triplet-building cell)
# Also computes a complete 4-way classification so all % sum to 100%.

N_total_val  = len(VAL[bk])          # total records in base validation set
N_triplet    = len(triplets)          # subset matched across all 3 stages
T = ABS_IMPROVE_THRESH

# ── Full 4-way classification of all triplet records ─────────────────────────
# Every triplet falls into exactly one of these buckets:
seg_A_mask = (triplets['d_base_snow'] < -T) & (triplets['d_snow_ss'] < -T)
seg_B_mask = (triplets['d_base_snow'] < -T) & (triplets['d_snow_ss'] >  T)
seg_C_mask = (triplets['d_base_snow'] >  T) & (triplets['d_snow_ss'] < -T)
seg_D_mask = (triplets['d_base_snow'] >  T) & (triplets['d_snow_ss'] >  T)  # both worsen
seg_E_mask = ~(seg_A_mask | seg_B_mask | seg_C_mask | seg_D_mask)           # within threshold (negligible change)

n_A = seg_A_mask.sum()
n_B = seg_B_mask.sum()
n_C = seg_C_mask.sum()
n_D = seg_D_mask.sum()
n_E = seg_E_mask.sum()

assert n_A + n_B + n_C + n_D + n_E == N_triplet, "Buckets don't sum to triplet total — check logic"

# ── Also compute Base→SS direct improvement on the triplet set ───────────────
# (net effect, ignoring the snow intermediate)
triplets['d_base_ss'] = triplets['abs_ss'] - triplets['abs_base']
n_net_improve = (triplets['d_base_ss'] < -T).sum()
n_net_worsen  = (triplets['d_base_ss'] >  T).sum()
n_net_neutral = N_triplet - n_net_improve - n_net_worsen

def pct_triplet(n): return f"{100 * n / N_triplet:.1f}%" if N_triplet > 0 else "—"
def pct_total(n):   return f"{100 * n / N_total_val:.1f}%" if N_total_val > 0 else "—"

print("══ TRAJECTORY SEGMENT STATISTICS ══")
print(f"\nReference algorithm family : {ALGO_LABELS[bk].replace(' (base)', '')}")
print(f"Improvement threshold (|ΔRBias|) : {T}")
print(f"\nBase validation records (total)  : {N_total_val:>7,}  (100.0%)")
print(f"Matched triplet records          : {N_triplet:>7,}  ({pct_total(N_triplet)} of base total)")

print(f"""
── 4-Way Classification of Triplet Records ──────────────────────────────────

  Segment  │ Base→Snow │ Snow→SS │  n records  │ % of triplets │ % of base total
  ─────────┼───────────┼─────────┼─────────────┼───────────────┼────────────────
  A        │  improves │improves │ {n_A:>10,}  │ {pct_triplet(n_A):>13} │ {pct_total(n_A):>15}
  B        │  improves │ worsens │ {n_B:>10,}  │ {pct_triplet(n_B):>13} │ {pct_total(n_B):>15}
  C        │  worsens  │improves │ {n_C:>10,}  │ {pct_triplet(n_C):>13} │ {pct_total(n_C):>15}
  D        │  worsens  │ worsens │ {n_D:>10,}  │ {pct_triplet(n_D):>13} │ {pct_total(n_D):>15}
  E (neut.)│  ~neutral │~neutral │ {n_E:>10,}  │ {pct_triplet(n_E):>13} │ {pct_total(n_E):>15}
  ─────────┼───────────┼─────────┼─────────────┼───────────────┼────────────────
  Total    │           │         │ {N_triplet:>10,}  │ {"100.0%":>13} │ {pct_total(N_triplet):>15}
""")

print(f"""── Net Base→SS Effect (ignoring Snow intermediate) ─────────────────────────

  Net improved  (Base → SS better) : {n_net_improve:>7,}  │ {pct_triplet(n_net_improve):>6} of triplets │ {pct_total(n_net_improve):>6} of base total
  Net worsened  (Base → SS worse)  : {n_net_worsen:>7,}  │ {pct_triplet(n_net_worsen):>6} of triplets │ {pct_total(n_net_worsen):>6} of base total
  Net neutral   (within threshold) : {n_net_neutral:>7,}  │ {pct_triplet(n_net_neutral):>6} of triplets │ {pct_total(n_net_neutral):>6} of base total
""")

# ── Segment A breakdown: how much of the improvement came from Snow vs SS? ───
seg_A_df = triplets[seg_A_mask]
mean_gain_snow = (-seg_A_df['d_base_snow']).mean()   # positive = improvement
mean_gain_ss   = (-seg_A_df['d_snow_ss']).mean()
total_gain     = (seg_A_df['abs_base'] - seg_A_df['abs_ss']).mean()

print(f"── Segment A breakdown: where does the improvement come from? ───────────────")
print(f"  Mean |RBias| reduction Base→Snow : {mean_gain_snow:.4f}  "
      f"({100*mean_gain_snow/total_gain:.1f}% of total gain)")
print(f"  Mean |RBias| reduction Snow→SS   : {mean_gain_ss:.4f}  "
      f"({100*mean_gain_ss/total_gain:.1f}% of total gain)")
print(f"  Mean total |RBias| reduction     : {total_gain:.4f}")

# ── Save summary to CSV ───────────────────────────────────────────────────────
summary_df = pd.DataFrame({
    'Segment': ['A', 'B', 'C', 'D', 'E (neutral)', 'TOTAL'],
    'Base→Snow': ['improves','improves','worsens','worsens','~neutral','—'],
    'Snow→SS':   ['improves','worsens','improves','worsens','~neutral','—'],
    'n_records': [n_A, n_B, n_C, n_D, n_E, N_triplet],
    'pct_of_triplets': [round(100*n/N_triplet,1) for n in [n_A,n_B,n_C,n_D,n_E,N_triplet]],
    'pct_of_base_total': [round(100*n/N_total_val,1) for n in [n_A,n_B,n_C,n_D,n_E,N_triplet]],
})
fname = os.path.join(OUTPUT_DIR, 'Trajectory_Segment_Statistics.csv')
summary_df.to_csv(fname, index=False)
print(f"\n  Summary saved to {fname}")

# ── Segment B: 30 records with sufficient context ─────────────────────────────
# "Sufficient" = both |rb_base| and |rb_snow| >= 0.05 (not near-zero noise),
# and chan_width is available. Sorted by magnitude of SS worsening (d_snow_ss
# descending) so the most informative cases appear first.

SEG_B_COLS = [
    'COMID', 'site_no', 'date_std', 'chan_width',
    'pred_base', 'rb_base',
    'pred_snow', 'rb_snow',
    'pred_ss',   'rb_ss',
    'd_base_snow', 'd_snow_ss',
]

seg_B_df = triplets[seg_B_mask].copy()

# Filter to records where both base and snow biases are non-trivial
# so the listed records are genuinely informative, not near-threshold noise
seg_B_informative = seg_B_df[
    (seg_B_df['abs_base'].abs() >= 0.05) &
    (seg_B_df['abs_snow'].abs() >= 0.05) &
    (seg_B_df['chan_width'].notna())
].copy()

# Sort by d_snow_ss descending: largest SS worsening at the top
seg_B_informative = seg_B_informative.sort_values('d_snow_ss', ascending=False)

seg_B_out = seg_B_informative[SEG_B_COLS].head(30).reset_index(drop=True).round(4)

fname_b = os.path.join(OUTPUT_DIR, 'COMID_SegmentB_DetailedRecords.csv')
seg_B_out.to_csv(fname_b, index=False)

print(f"\n══ SEGMENT B — 30 Records: Base→Snow Improves, Snow→SS Worsens ══")
print(f"   (filtered to |rb_base| ≥ 0.05 and |rb_snow| ≥ 0.05 for informativeness)")
print(f"   Sorted by d_snow_ss descending (largest SS worsening first)")
print(f"   Total Segment B records: {len(seg_B_df):,}  |  informative subset: {len(seg_B_informative):,}\n")
print(seg_B_out.to_string(index=True))
print(f"\n  → Saved to {fname_b}")

# ── Summary statistics for the shown records ──────────────────────────────────
print(f"\n── Statistics for these 30 records ──")
print(f"  chan_width  : mean={seg_B_out['chan_width'].mean():.1f} m  "
      f"| median={seg_B_out['chan_width'].median():.1f} m  "
      f"| range={seg_B_out['chan_width'].min():.1f}–{seg_B_out['chan_width'].max():.1f} m")
print(f"  rb_base     : mean={seg_B_out['rb_base'].mean():.3f}  "
      f"| median={seg_B_out['rb_base'].median():.3f}")
print(f"  rb_snow     : mean={seg_B_out['rb_snow'].mean():.3f}  "
      f"| median={seg_B_out['rb_snow'].median():.3f}  "
      f"(Δ vs base: {(seg_B_out['rb_snow']-seg_B_out['rb_base']).mean():.3f})")
print(f"  rb_ss       : mean={seg_B_out['rb_ss'].mean():.3f}  "
      f"| median={seg_B_out['rb_ss'].median():.3f}  "
      f"(Δ vs snow: {(seg_B_out['rb_ss']-seg_B_out['rb_snow']).mean():.3f})")
print(f"  d_base_snow : mean={seg_B_out['d_base_snow'].mean():.3f}  (negative = improvement)")
print(f"  d_snow_ss   : mean={seg_B_out['d_snow_ss'].mean():.3f}  (positive = worsening)")


# %%
# ── Segment E: Sample records showing the neutral/borderline cases ─────────────
# Shows records where at least one transition (Base→Snow or Snow→SS) fell within
# the noise threshold T. Presented in the same format as the trajectory segments
# in the COMID block above.

seg_E_df = triplets[seg_E_mask].copy()

print(f"\n══ SEGMENT E — Sample Neutral Records (n={len(seg_E_df):,}) ══")
print(f"   (at least one stage-to-stage |ΔRBias| ≤ {T})\n")

# Sub-classify E so the reader understands WHY each record is neutral
seg_E_df['e_reason'] = 'both transitions near-zero'
seg_E_df.loc[
    (seg_E_df['d_base_snow'].abs() <= T) & (seg_E_df['d_snow_ss'].abs() > T),
    'e_reason'
] = 'Snow neutral, SS changed'
seg_E_df.loc[
    (seg_E_df['d_base_snow'].abs() > T)  & (seg_E_df['d_snow_ss'].abs() <= T),
    'e_reason'
] = 'Snow changed, SS neutral'

# Show a representative sample from each sub-reason
E_REPORT_COLS = [
    'COMID', 'site_no', 'date_std', 'chan_width',
    'pred_base', 'rb_base',
    'pred_snow', 'rb_snow',
    'pred_ss',   'rb_ss',
    'd_base_snow', 'd_snow_ss', 'e_reason',
]

N_SHOW_E = 10   # rows per sub-reason group

for reason, grp in seg_E_df.groupby('e_reason'):
    print(f"  ── {reason} (n={len(grp):,}) ──")
    sample = grp[E_REPORT_COLS].head(N_SHOW_E).reset_index(drop=True).round(4)
    print(sample.to_string(index=False))
    print()

# Save full Segment E to CSV
fname_e = os.path.join(OUTPUT_DIR, 'COMID_Traj_E_Neutral.csv')
seg_E_df[E_REPORT_COLS].round(4).to_csv(fname_e, index=False)
print(f"  → Full Segment E records saved to {fname_e}")

# %%
# ── Validation Site Map: Data Availability Check ──────────────────────────────
# The USGS field CSV (df_field) contains measurement records but no coordinates.
# Coordinates are carried by the model CSVs and survive into VAL after the merge.
# This block verifies what coordinate information is available and from which source,
# and flags how many unique validation sites can be placed on a map.

print("══ COORDINATE AVAILABILITY CHECK ══\n")

# ── 1. Check df_field columns ────────────────────────────────────────────────
coord_candidates = [c for c in df_field.columns
                    if any(x in c.lower() for x in ['lat', 'lon', 'lng', 'x', 'y', 'coord', 'geo'])]
print(f"df_field columns ({len(df_field.columns)} total):")
print(f"  All columns : {list(df_field.columns)}")
print(f"  Coord-like  : {coord_candidates if coord_candidates else 'NONE — no coordinate columns in field data'}")

# ── 2. Check df_mapping columns ───────────────────────────────────────────────
map_coord_cols = [c for c in df_mapping.columns
                  if any(x in c.lower() for x in ['lat', 'lon', 'lng', 'x', 'y', 'coord', 'geo'])]
print(f"\ndf_mapping columns ({len(df_mapping.columns)} total):")
print(f"  All columns : {list(df_mapping.columns)}")
print(f"  Coord-like  : {map_coord_cols if map_coord_cols else 'NONE'}")

# ── 3. Check VAL columns (model CSVs carry coordinates after merge) ───────────
ref_val = VAL['Zou_SS']
val_coord_cols = [c for c in ref_val.columns
                  if any(x in c.lower() for x in ['lat', 'lon', 'lng', 'x', 'y', 'coord', 'geo'])]
print(f"\nVAL['Zou_SS'] columns ({len(ref_val.columns)} total):")
print(f"  All columns : {list(ref_val.columns)}")
print(f"  Coord-like  : {val_coord_cols if val_coord_cols else 'NONE'}")

# ── 4. Assess site-level coordinate coverage ──────────────────────────────────
print("\n══ SITE-LEVEL COORDINATE COVERAGE ══")

if 'lat' in ref_val.columns and 'lon' in ref_val.columns:
    site_coords = (
        ref_val[['site_no', 'lat', 'lon']]
        .dropna(subset=['lat', 'lon'])
        .drop_duplicates('site_no')
    )
    total_sites    = ref_val['site_no'].nunique()
    sites_with_coords = len(site_coords)
    sites_missing  = total_sites - sites_with_coords

    print(f"  Source       : VAL['Zou_SS'] (lat/lon from model CSV)")
    print(f"  Total unique sites in validation : {total_sites:,}")
    print(f"  Sites WITH coordinates           : {sites_with_coords:,}  "
          f"({100*sites_with_coords/total_sites:.1f}%)")
    print(f"  Sites WITHOUT coordinates        : {sites_missing:,}  "
          f"({100*sites_missing/total_sites:.1f}%)")

    # Coordinate range sanity check
    print(f"\n  Latitude  range : {site_coords['lat'].min():.3f} → {site_coords['lat'].max():.3f}")
    print(f"  Longitude range : {site_coords['lon'].min():.3f} → {site_coords['lon'].max():.3f}")

    # Flag any implausible coordinates (outside CONUS bounding box)
    outside_conus = site_coords[
        (site_coords['lat'] < 24)  | (site_coords['lat'] > 50) |
        (site_coords['lon'] < -125)| (site_coords['lon'] > -66)
    ]
    if len(outside_conus) > 0:
        print(f"\n  ⚠ {len(outside_conus)} site(s) have coordinates outside the CONUS bounding box:")
        print(outside_conus[['site_no','lat','lon']].to_string(index=False))
    else:
        print(f"\n  ✓ All coordinates fall within the CONUS bounding box.")

    print(f"\n  → Map can be drawn using VAL['Zou_SS'] lat/lon. Proceed to next block.")

else:
    print("  ✗ No 'lat'/'lon' columns found in VAL['Zou_SS'].")
    if map_coord_cols:
        print(f"  → Potential coordinate columns in df_mapping: {map_coord_cols}")
        print(f"  → Consider joining df_mapping to df_field on site_no to recover coordinates.")
    else:
        print("  → No coordinate columns found in df_field or df_mapping.")
        print("  → Will need to fetch USGS station coordinates via dataretrieval or nwis API.")

# %%
# ── Validation Site Location Map ──────────────────────────────────────────────
# Coordinates confirmed available from VAL['Zou_SS'] (lat/lon from model CSV).
# 27 sites flagged outside CONUS: 25 Alaska, 2 Puerto Rico — shown in separate side panels.

import geopandas as gpd
from shapely.geometry import Point
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# ── Build one-row-per-site summary ───────────────────────────────────────────
df_sites = (
    VAL['Zou_SS']
    .dropna(subset=['lat', 'lon'])
    .groupby('site_no')
    .agg(
        lat          = ('lat',          'first'),
        lon          = ('lon',          'first'),
        n_obs        = ('chan_width',    'count'),
        median_width = ('chan_width',    'median'),
        mean_rbias   = ('Relative_Bias', 'mean'),
    )
    .reset_index()
)

# Attach COMID
_map_clean = (
    df_mapping[['site_no', 'COMID']]
    .assign(site_no=lambda d: pd.to_numeric(d['site_no'], errors='coerce'),
            COMID  =lambda d: pd.to_numeric(d['COMID'],   errors='coerce'))
    .dropna().astype(int).drop_duplicates('site_no')
)
df_sites = df_sites.merge(_map_clean, on='site_no', how='left')

# ── Region classification ─────────────────────────────────────────────────────
def classify_region(row):
    if row['lat'] > 50 or row['lon'] < -140:
        return 'Alaska'
    elif row['lat'] < 25:
        return 'Puerto Rico'
    else:
        return 'CONUS'

df_sites['region'] = df_sites.apply(classify_region, axis=1)

n_conus = (df_sites['region'] == 'CONUS').sum()
n_ak    = (df_sites['region'] == 'Alaska').sum()
n_pr    = (df_sites['region'] == 'Puerto Rico').sum()

print(f"Site breakdown:")
print(f"  CONUS       : {n_conus:,}")
print(f"  Alaska      : {n_ak:,}")
print(f"  Puerto Rico : {n_pr:,}")
print(f"  Total       : {len(df_sites):,}")

# ── Shared colour/size scaling (computed across ALL sites) ────────────────────
log_n  = np.log1p(df_sites['n_obs'].values)
n_norm = (log_n - log_n.min()) / (log_n.max() - log_n.min() + 1e-9)

w_vals = df_sites['median_width'].values
w_norm = (w_vals - w_vals.min()) / (w_vals.max() - w_vals.min() + 1e-9)
sizes  = 20 + w_norm * 160

def _scatter(ax, mask):
    """Plot sites matching boolean mask onto ax."""
    idx = np.where(mask)[0]
    return ax.scatter(
        df_sites.loc[mask, 'lon'],
        df_sites.loc[mask, 'lat'],
        c=n_norm[idx], cmap='YlOrRd',
        s=sizes[idx],
        edgecolors='#333333', linewidths=0.4,
        alpha=0.88, zorder=4, vmin=0, vmax=1,
    )

# ── Load basemap ──────────────────────────────────────────────────────────────
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# ── Figure: main CONUS map + two separate, non-overlapping side panels ──────
fig = plt.figure(figsize=(26, 10))

# Main axes: CONUS (left portion of the figure)
ax_main = fig.add_axes([0.03, 0.08, 0.54, 0.86])   # [left, bottom, width, height]

# Colorbar axis: thin strip immediately to the right of the CONUS map
ax_cb = fig.add_axes([0.585, 0.15, 0.014, 0.65])

# Side panels: Alaska (top-right) and Puerto Rico (bottom-right), stacked in
# their own column so neither overlaps each other or the CONUS map.
ax_ak = fig.add_axes([0.66, 0.50, 0.32, 0.44])
ax_pr = fig.add_axes([0.66, 0.06, 0.32, 0.36])

fig.suptitle(
    f'USGS Validation Site Locations — {len(df_sites):,} Unique Stations  '
    f'(CONUS: {n_conus} | Alaska: {n_ak} | Puerto Rico: {n_pr})\n'
    f'In-situ width ≥ {W_THRESH} m | Within 500 m of COMID centroid',
    fontsize=13, fontweight='bold', y=0.98
)

# ── Helper: draw basemap + sites on any axes ──────────────────────────────────
def _draw_panel(ax, xlim, ylim, mask, title=None, fontsize=8):
    world.plot(ax=ax, color='#F0F0F0', edgecolor='#BBBBBB', linewidth=0.3)
    sc = _scatter(ax, mask)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    if title:
        ax.set_title(title, fontsize=fontsize, fontweight='bold', pad=3)
    return sc

mask_conus = df_sites['region'] == 'CONUS'
mask_ak    = df_sites['region'] == 'Alaska'
mask_pr    = df_sites['region'] == 'Puerto Rico'

# Main CONUS panel
sc = _draw_panel(ax_main,
                 xlim=(-125, -65), ylim=(24, 50),
                 mask=mask_conus)
ax_main.set_xlabel('Longitude', fontsize=13.2, fontweight='bold')
ax_main.set_ylabel('Latitude',  fontsize=13.2, fontweight='bold')
ax_main.tick_params(labelsize=12)
ax_main.set_xticks(range(-120, -60, 10))
ax_main.set_yticks(range(25, 51, 5))
ax_main.xaxis.set_tick_params(labelbottom=True)
ax_main.yaxis.set_tick_params(labelleft=True)
ax_main.grid(linestyle=':', linewidth=0.4, alpha=0.5)

# Alaska and Puerto Rico are now shown as fully separate panels to the right
# of the CONUS map (see ax_ak / ax_pr below) rather than as overlapping insets.

# Alaska panel
_draw_panel(ax_ak,
            xlim=(-170, -128), ylim=(54, 72),
            mask=mask_ak,
            title=f'Alaska ({n_ak} sites)')

# Puerto Rico panel
_draw_panel(ax_pr,
            xlim=(-68, -65), ylim=(17.8, 18.8),
            mask=mask_pr,
            title=f'Puerto Rico ({n_pr} sites)')

# ── Colorbar ──────────────────────────────────────────────────────────────────
norm_cb = mcolors.Normalize(vmin=df_sites['n_obs'].min(),
                             vmax=df_sites['n_obs'].max())
sm = cm.ScalarMappable(cmap='YlOrRd', norm=norm_cb)
sm.set_array([])
cbar = fig.colorbar(sm, cax=ax_cb)
cbar.set_label('Validation Records per Site\n(log-scaled colour)',
               fontsize=12, fontweight='bold', labelpad=8)
cbar.ax.tick_params(labelsize=10.8)

# ── Size legend ───────────────────────────────────────────────────────────────
width_breaks = [30, 100, 300, 1000]
size_elems = []
for wb in width_breaks:
    wb_n = (wb - w_vals.min()) / (w_vals.max() - w_vals.min() + 1e-9)
    wb_s = 20 + wb_n * 160
    size_elems.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#AAAAAA', markeredgecolor='#333333',
               markersize=np.sqrt(wb_s),
               label=f'{wb} m')
    )

ax_main.legend(
    handles=size_elems,
    title='Median In-situ Width',
    title_fontsize=12,
    fontsize=10.8,
    loc='lower left',
    framealpha=0.92,
    edgecolor='grey'
)

fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Validation_Site_Map.pdf'),
            bbox_inches='tight', dpi=150)
fig.savefig(os.path.join(OUTPUT_DIR, 'Fig_Validation_Site_Map.png'),
            bbox_inches='tight', dpi=150)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n══ SITE SUMMARY BY REGION ══")
print(df_sites.groupby('region').agg(
    n_sites      = ('site_no',      'count'),
    n_obs_total  = ('n_obs',        'sum'),
    median_width = ('median_width', 'median'),
    lat_range    = ('lat', lambda x: f"{x.min():.1f}–{x.max():.1f}°N"),
    lon_range    = ('lon', lambda x: f"{x.min():.1f}–{x.max():.1f}°E"),
).to_string())
print(f"\nMap saved.")

# %%




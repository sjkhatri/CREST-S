# %% [markdown]
# # Multi-Algorithm River Width Validation – Wisconsin
# Validates four satellite-derived width algorithms against RiverScope in-situ measurements:
# - **WI** (GLOW-S)
# - **MuWI**
# - **UltraB**
# - **Zou**
# 

# %% [markdown]
# ## 1. Imports

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# %% [markdown]
# ## 2. File Paths

# %%
BASE = "/N/lustre/project/proj-212/Suraj/River_data_data/Data_for_riverscope"

INSITU_PATH = f"{BASE}/combined_river_widths_output_test_new.csv"

# Algorithm sim-data files
SIM_PATHS = {
    "WI":     f"{BASE}/filtered_combined_widths_WI2.csv",
    "MuWI":   f"{BASE}/filtered_combined_widths_MuWI2.csv",
    "UltraB": f"{BASE}/filtered_combined_widths_UltraB2.csv",
    "Zou":    f"{BASE}/filtered_combined_widths_Zou2.csv",
}

# Colour palette – one colour per algorithm (used consistently throughout)
ALGO_COLORS = {
    "WI":     "#1f77b4",   # blue
    "MuWI":   "#ff7f0e",   # orange
    "UltraB": "#2ca02c",   # green
    "Zou":    "#d62728",   # red
}


# %% [markdown]
# ## 3. Load In-Situ Data

# %%
print(f"Loading in-situ data from: {INSITU_PATH}")
insitu_data = pd.read_csv(INSITU_PATH)
print("Successfully loaded 'insitu_data'.")
print(insitu_data.head())
insitu_data.info()


# %%
print("Number of unique COMIDs:", insitu_data['COMID'].nunique())


# %% [markdown]
# ### Standardise In-Situ Dates & IDs

# %%
insitu_data['Date_std'] = pd.to_datetime(insitu_data['Date'], errors='coerce').dt.normalize()
insitu_data['ID_unique_std'] = insitu_data['ID_unique'].astype(str).str.strip()

null_dates = insitu_data['Date_std'].isna().sum()
if null_dates:
    print(f"Warning: {null_dates} rows had unparseable dates → NaT")
print("In-situ date standardisation complete.")
print(insitu_data[['Date', 'Date_std', 'ID_unique', 'ID_unique_std']].head())


# %% [markdown]
# ## 4. Load All Algorithm Sim Files

# %%
sim_data_dict = {}

for algo, path in SIM_PATHS.items():
    print(f"\nLoading [{algo}] from: {path}")
    df = pd.read_csv(path)
    
    # Standardise date
    df['date_std'] = pd.to_datetime(df['date'], errors='coerce').dt.normalize()
    
    # Standardise river ID
    df['riverID_std'] = df['riverID'].astype(str).str.strip()
    
    null_dates = df['date_std'].isna().sum()
    if null_dates:
        print(f"  Warning: {null_dates} rows had unparseable dates → NaT")
    
    unique_comids = df['riverID'].str.extract(r'R(\d+)X')[0].nunique()
    print(f"  Shape: {df.shape}  |  Unique COMIDs: {unique_comids}")
    
    sim_data_dict[algo] = df
    
print("\nAll sim files loaded.")


# %% [markdown]
# ## 5. Merge Each Algorithm with In-Situ Data

# %%
merged_dict = {}

for algo, sim_df in sim_data_dict.items():
    merged = pd.merge(
        insitu_data,
        sim_df,
        left_on=['ID_unique_std', 'Date_std'],
        right_on=['riverID_std', 'date_std'],
        suffixes=('', f'_{algo}')
    )
    merged_dict[algo] = merged
    print(f"[{algo}]  insitu={insitu_data.shape} | sim={sim_df.shape} | merged={merged.shape}")


# %% [markdown]
# ## 6. Filtering & Median Computation

# %%
def filteredComidTo60pcMedian(one_comid):
    """
    For a single COMID subset:
      - Keeps only the most frequent riverIDs that cumulatively cover
        the full dataset (min_data_taken_from_high_freq_stn * total).
      - Requires ≥4 observations per day before computing a daily median.
    Returns: (median_days2000, sd_days2000) indexed by days-since-2000.
    """
    one_comid_copy = one_comid.copy()
    min_data_taken_from_high_freq_stn = 1

    one_comid_copy['date'] = pd.to_datetime(one_comid_copy['date_std'])
    base_date = pd.Timestamp('2000-01-01')
    one_comid_copy['days2000'] = (one_comid_copy['date_std'] - base_date).dt.days
    one_comid_copy.drop('date_std', axis=1, inplace=True)

    river_id_counts = one_comid_copy['riverID'].value_counts()
    cumulative_sum  = river_id_counts.cumsum()
    total_sum       = river_id_counts.sum()
    threshold       = min_data_taken_from_high_freq_stn * total_sum
    n_keep          = cumulative_sum[cumulative_sum < threshold].count() + 1
    filtered_data   = one_comid_copy[one_comid_copy['riverID'].isin(river_id_counts.index[:n_keep])]

    grp = filtered_data.groupby('days2000')['width']
    valid_grp = filtered_data.groupby('days2000').filter(lambda x: len(x) >= 4)
    median_days2000 = valid_grp.groupby('days2000')['width'].median()
    sd_days2000     = valid_grp.groupby('days2000')['width'].std()

    return median_days2000, sd_days2000


# %% [markdown]
# ## 7. Compute Median Width per COMID (all algorithms)

# %%
def process_comid_for_algo(args):
    comid, merged_df = args
    df_tmp = merged_df[merged_df['COMID'] == comid].copy()
    median_days2000, sd_days2000 = filteredComidTo60pcMedian(df_tmp)

    date = pd.Timestamp('2000-01-01') + pd.to_timedelta(median_days2000.index, unit='D')
    df_median = pd.DataFrame({
        'date_std':    date,
        'width_median': median_days2000.values,
        'width_cv':     sd_days2000.values / median_days2000.values
    })
    df_tmp = pd.merge(df_tmp, df_median, on='date_std', how='right')
    df_tmp = df_tmp.drop_duplicates(subset='date_std', keep='first')
    return df_tmp


bar_fmt = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'

df_cat_dict = {}   # stores processed results per algorithm

for algo, merged_df in merged_dict.items():
    comid_list = merged_df['COMID'].unique()
    print(f"\nProcessing [{algo}] – {len(comid_list)} COMIDs …")
    
    args = [(c, merged_df) for c in comid_list]
    with Pool(processes=min(50, cpu_count())) as pool:
        results = list(tqdm(
            pool.imap(process_comid_for_algo, args),
            total=len(comid_list),
            bar_format=bar_fmt,
            dynamic_ncols=False
        ))
    
    df_cat = pd.concat([r for r in results if not r.empty], ignore_index=True)
    df_cat_dict[algo] = df_cat
    print(f"  → combined shape: {df_cat.shape}")

print("\nAll algorithms processed.")


# %% [markdown]
# ## 8. Performance Metrics

# %%
def computePerform(obs, pred):
    mse = np.mean((obs - pred) ** 2)
    R2  = np.corrcoef(obs, pred)[0, 1] ** 2
    sse = np.sum((obs - pred) ** 2)
    sst = np.sum((obs - np.mean(obs)) ** 2)
    nse = 1 - sse / sst
    return mse, R2, nse


gof_dict      = {}
obs_avg_dict  = {}
pred_avg_dict = {}

for algo, df_cat in df_cat_dict.items():
    df_cat = df_cat.dropna(subset=['width_m', 'width'])

    gof_rows, obs_avg_rows, pred_avg_rows = [], [], []

    for comid in df_cat['COMID'].unique():
        df_part = df_cat[df_cat['COMID'] == comid]
        obs  = df_part['width_m'].values
        pred = df_part['width_median'].values
        lat  = df_part['lat'].values[0]
        lon  = df_part['lon'].values[0]

        if obs.shape[0] > 1 and pred.shape[0] > 1:
            model = LinearRegression()
            model.fit(obs.reshape(-1, 1), pred.reshape(-1, 1))
            r2_lr = model.score(obs.reshape(-1, 1), pred.reshape(-1, 1))
            mse, r2, nse = computePerform(obs, pred)
            
            gof_rows.append([comid, mse, r2_lr, nse, obs.shape[0], lat, lon])
            obs_avg_rows.append(np.mean(obs))
            pred_avg_rows.append(np.mean(pred))

    gof_dict[algo]      = pd.DataFrame(gof_rows, columns=['COMID', 'MSE', 'R2', 'NSE', 'num_obs', 'lat', 'lon'])
    obs_avg_dict[algo]  = np.array(obs_avg_rows)
    pred_avg_dict[algo] = np.array(pred_avg_rows)
    print(f"[{algo}]  COMIDs with valid metrics: {len(gof_rows)}")


# %%
# Quick summary table
summary_rows = []
for algo, gof in gof_dict.items():
    summary_rows.append({
        'Algorithm': algo,
        'N_COMIDs':  len(gof),
        'Median R²': round(gof['R2'].median(), 3),
        'Median NSE': round(gof['NSE'].median(), 3),
        'Median MSE': round(gof['MSE'].median(), 1),
    })
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))


# %% [markdown]
# ## 9. Individual Validation Scatter Plots (per algorithm)

# %%
plt.rcParams['font.size']   = 13
plt.rcParams['font.family'] = 'sans-serif'

for algo, df_cat in df_cat_dict.items():
    df_plot = df_cat.dropna(subset=['width_m', 'width_median', 'width_cv']).copy()
    df_plot = df_plot[(df_plot['width_cv'] > 0.00) & (df_plot['width_cv'] < 0.25)]
    
    obs_w  = df_plot['width_m'].values
    pred_w = df_plot['width_median'].values
    cv_w   = df_plot['width_cv'].values
    
    mask   = (obs_w > 10) & (pred_w > 10)
    obs_w, pred_w, cv_w = obs_w[mask], pred_w[mask], cv_w[mask]
    
    if len(obs_w) == 0:
        print(f"[{algo}] No valid data after filtering – skipping plot.")
        continue

    R2 = np.corrcoef(obs_w, pred_w)[0, 1] ** 2
    
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(obs_w, pred_w, c=cv_w, s=50, cmap='plasma', alpha=0.8)
    ax.plot([10, 1000], [10, 1000], color='black', lw=1.2, label='1:1')
    ax.set_xlim(10, 1000); ax.set_ylim(10, 1000)
    ax.set_xscale('log');   ax.set_yscale('log')
    ax.set_xlabel('PlanetScope Width (m)', fontweight='bold')
    ax.set_ylabel('Sentinel-2 Width (m)', fontweight='bold')
    ax.set_title(f'{algo} vs In-Situ', fontweight='bold')
    ax.grid(linestyle='--', alpha=0.6)
    ax.text(0.05, 0.92, f'R² = {R2:.2f}  |  N = {len(obs_w)}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Width CV', fontweight='bold')
    
    # Inset
    ax_inset = inset_axes(ax, width='35%', height='35%', loc='lower right', borderpad=2)
    ax_inset.scatter(obs_w, pred_w, c=cv_w, s=20, cmap='plasma', alpha=0.8)
    ax_inset.plot([0, 200], [0, 200], color='black', lw=1)
    ax_inset.set_xlim(10, 200); ax_inset.set_ylim(10, 200)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f'scatter_{algo}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[{algo}]  R² = {R2:.3f}  |  N points = {len(obs_w)}")


# %% [markdown]
# ## 10. Per-COMID Average Width – Individual Plots

# %%
plt.rcParams['font.size']   = 13
plt.rcParams['font.family'] = 'sans-serif'

for algo in gof_dict:
    obs_a  = obs_avg_dict[algo]
    pred_a = pred_avg_dict[algo]
    
    if len(obs_a) == 0:
        continue
    
    R2 = np.corrcoef(obs_a, pred_a)[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(obs_a, pred_a, color=ALGO_COLORS[algo], alpha=0.75, s=40, edgecolors='k', lw=0.4)
    ax.plot([0, 1000], [0, 1000], color='black', lw=1.2)
    ax.set_xlabel('Average PlanetScope Width (m)', fontweight='bold')
    ax.set_ylabel('Average Sentinel-2 Width (m)', fontweight='bold')
    ax.set_title(f'{algo} – COMID averages', fontweight='bold')
    ax.grid(linestyle='--', alpha=0.6)
    ax.text(0.05, 0.92, f'R² = {R2:.2f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(f'avg_scatter_{algo}.png', dpi=150, bbox_inches='tight')
    plt.show()


# %% [markdown]
# ## 11. Combined 2×2 Scatter Plot (all algorithms)

# %%
plt.rcParams['font.size']   = 12
plt.rcParams['font.family'] = 'sans-serif'

algos = list(df_cat_dict.keys())
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, algo in enumerate(algos):
    ax = axes[idx]
    df_plot = df_cat_dict[algo].dropna(subset=['width_m', 'width_median', 'width_cv']).copy()
    df_plot = df_plot[(df_plot['width_cv'] > 0.00) & (df_plot['width_cv'] < 0.25)]
    
    obs_w  = df_plot['width_m'].values
    pred_w = df_plot['width_median'].values
    cv_w   = df_plot['width_cv'].values
    mask   = (obs_w > 10) & (pred_w > 10)
    obs_w, pred_w, cv_w = obs_w[mask], pred_w[mask], cv_w[mask]
    
    if len(obs_w) == 0:
        ax.set_title(f'{algo} – No data')
        continue
    
    R2 = np.corrcoef(obs_w, pred_w)[0, 1] ** 2

    sc = ax.scatter(obs_w, pred_w, c=cv_w, s=35, cmap='plasma', alpha=0.85)
    ax.plot([10, 1000], [10, 1000], color='black', lw=1.2)
    ax.set_xlim(10, 1000); ax.set_ylim(10, 1000)
    ax.set_xscale('log');   ax.set_yscale('log')
    ax.set_xlabel('PlanetScope Width (m)', fontweight='bold')
    ax.set_ylabel('Sentinel-2 Width (m)', fontweight='bold')
    ax.set_title(algo, fontweight='bold', color=ALGO_COLORS[algo])
    ax.grid(linestyle='--', alpha=0.5)
    ax.text(0.05, 0.92, f'R² = {R2:.2f}  |  N = {len(obs_w)}',
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    plt.colorbar(sc, ax=ax, label='Width CV')

plt.suptitle('All Algorithms – Width Validation (WI)', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('combined_scatter_2x2.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 12. Combined Overlay Scatter (all algorithms on one axes)

# %%
plt.rcParams['font.size']   = 13
plt.rcParams['font.family'] = 'sans-serif'

fig, ax = plt.subplots(figsize=(10, 9))

for algo in algos:
    df_plot = df_cat_dict[algo].dropna(subset=['width_m', 'width_median', 'width_cv']).copy()
    df_plot = df_plot[(df_plot['width_cv'] > 0.00) & (df_plot['width_cv'] < 0.25)]
    obs_w   = df_plot['width_m'].values
    pred_w  = df_plot['width_median'].values
    mask    = (obs_w > 10) & (pred_w > 10)
    obs_w, pred_w = obs_w[mask], pred_w[mask]
    
    if len(obs_w) == 0:
        continue
    
    R2 = np.corrcoef(obs_w, pred_w)[0, 1] ** 2
    ax.scatter(obs_w, pred_w, s=35, alpha=0.6, color=ALGO_COLORS[algo],
               label=f'{algo} (R²={R2:.2f})')

ax.plot([10, 1000], [10, 1000], color='black', lw=1.4, label='1:1')
ax.set_xlim(10, 1000); ax.set_ylim(10, 1000)
ax.set_xscale('log');   ax.set_yscale('log')
ax.set_xlabel('PlanetScope Width (m)', fontweight='bold')
ax.set_ylabel('Sentinel-2 Width (m)', fontweight='bold')
ax.set_title('All Algorithms vs In-Situ (overlay)', fontweight='bold')
ax.grid(linestyle='--', alpha=0.5)
ax.legend(framealpha=0.9, fontsize=10)
plt.tight_layout()
plt.savefig('combined_overlay_scatter.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## 13. Global Map of Validation Sites

# %%
# ── 16. Global Map of Validation Sites ─────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- 1. Build combined table: one row per (COMID, algo) ---
gof_rows_all = []
for algo, gof in gof_dict.items():
    tmp = gof[['COMID', 'lat', 'lon', 'R2', 'num_obs']].copy()
    tmp['algo'] = algo
    gof_rows_all.append(tmp)

gof_all = pd.concat(gof_rows_all, ignore_index=True)

# --- 2. Per-COMID summary ---
comid_summary = (
    gof_all.groupby('COMID')
    .agg(
        lat    = ('lat',     'first'),
        lon    = ('lon',     'first'),
        med_R2 = ('R2',      'median'),
        tot_obs= ('num_obs', 'sum'),
        n_algos= ('algo',    'nunique'),
    )
    .reset_index()
)
print(f"Unique validation COMIDs: {len(comid_summary)}")

# --- 3. Map extents: auto-pad around the data points ---
lon_min, lon_max = comid_summary['lon'].min(), comid_summary['lon'].max()
lat_min, lat_max = comid_summary['lat'].min(), comid_summary['lat'].max()
pad_x = max((lon_max - lon_min) * 0.15, 2.0)
pad_y = max((lat_max - lat_min) * 0.15, 2.0)

# --- 4. Plot on a cartopy map ---
fig = plt.figure(figsize=(20, 12))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(
    [lon_min - pad_x, lon_max + pad_x,
     lat_min - pad_y, lat_max + pad_y],
    crs=ccrs.PlateCarree()
)

# Basemap layers
ax.add_feature(cfeature.LAND,       facecolor='#f5f5f0', zorder=0)
ax.add_feature(cfeature.OCEAN,      facecolor='#cde6f5', zorder=0)
ax.add_feature(cfeature.LAKES,      facecolor='#cde6f5', alpha=0.8, zorder=1)
ax.add_feature(cfeature.RIVERS,     edgecolor='#7ab8d4', linewidth=0.5, zorder=1)
ax.add_feature(cfeature.STATES,     edgecolor='#aaaaaa', linewidth=0.6, zorder=2)
ax.add_feature(cfeature.BORDERS,    edgecolor='#555555', linewidth=1.0, zorder=2)
ax.add_feature(cfeature.COASTLINE,  edgecolor='#333333', linewidth=0.8, zorder=2)
ax.gridlines(draw_labels=True, linewidth=0.4, color='grey',
             alpha=0.6, linestyle='--', zorder=3)

# Validation points
sizes = np.clip(comid_summary['tot_obs'].values, 5, 200) * 0.8
sc = ax.scatter(
    comid_summary['lon'],
    comid_summary['lat'],
    c          = comid_summary['med_R2'],
    s          = sizes,
    cmap       = 'RdYlGn',
    vmin       = 0, vmax = 1,
    alpha      = 0.88,
    edgecolors = 'k',
    linewidths = 0.4,
    transform  = ccrs.PlateCarree(),
    zorder     = 4,
)

#cbar = plt.colorbar(sc, ax=ax, pad=0.06, shrink=0.75, aspect=25)
#cbar.set_label('Median R² (across algorithms)', fontweight='bold')

# Size legend
for n_obs, label in [(10, '10 obs'), (50, '50 obs'), (150, '150+ obs')]:
    ax.scatter([], [], s=np.clip(n_obs, 5, 200)*0.8,
               c='grey', alpha=0.7, edgecolors='k', lw=0.3,
               label=label, transform=ccrs.PlateCarree())

ax.legend(title='Total obs', title_fontsize=9, fontsize=8,
          loc='lower left', framealpha=0.9, scatterpoints=1)

ax.set_title(
    f'Validation Site Locations – Wisconsin\n'
    f'({len(comid_summary)} unique COMIDs | {len(algos)} algorithms)',
    fontweight='bold', fontsize=13, pad=12
)

plt.tight_layout()
plt.savefig('validation_sites_map.png', dpi=150, bbox_inches='tight')
plt.show()
print("Map saved → validation_sites_map.png")


# %%




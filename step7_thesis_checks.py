"""Step 7: recount the thesis data set, score the simple references on its test month, and measure spatial correlation (Moran's I).

Usage: python step7_thesis_checks.py CRIME_CSV TRACT_FILE
  CRIME_CSV  the Chicago crimes file from the Chicago Data Portal (Crimes - 2001 to Present, CSV export)
  TRACT_FILE the Chicago census tract boundaries (Boundaries - Census Tracts - 2000, GeoJSON export)

Rebuilds the daily tract panel the way the thesis notebook does: crimes dated 2021-01-01 up to and including 2023-12-31 (a bound of
midnight), rows with a latitude and longitude, joined to the tract that contains the point, counted per tract per day with zeros filled in.
Training days are before 2023-11-01, validation is November 2023, the test month is December 2023. Writes out/step7_counts.csv,
out/step7_references.csv and out/step7_moran.csv.
"""
import sys, time, warnings
import numpy as np, pandas as pd
from common import *

if len(sys.argv) != 3:
    raise SystemExit(__doc__)
CSV, GEO = sys.argv[1], sys.argv[2]
START, END = pd.Timestamp('2021-01-01'), pd.Timestamp('2023-12-31')
VAL_FROM, TEST_FROM = pd.Timestamp('2023-11-01'), pd.Timestamp('2023-12-01')
t0 = time.time()
warnings.filterwarnings('ignore')
import geopandas as gpd
from libpysal.weights import KNN
from esda.moran import Moran

# ---- 1. Tracts, crimes, join, daily panel
tracts = gpd.read_file(GEO)
tracts = tracts.set_crs(epsg=4326) if tracts.crs is None else tracts.to_crs(epsg=4326)
tracts = tracts.reset_index(drop=True)
tracts['GEOID'] = [str(i) for i in range(len(tracts))]   # tract id is the row number, as in the notebook
tracts['lon'], tracts['lat'] = tracts.geometry.centroid.x, tracts.geometry.centroid.y   # centroids of unprojected shapes, as in the notebook
crimes = pd.read_csv(CSV, usecols=['ID', 'Date', 'Latitude', 'Longitude'], low_memory=False)
crimes['Date'] = pd.to_datetime(crimes['Date'])
crimes = crimes[(crimes['Date'] >= START) & (crimes['Date'] <= END)]
n_window = len(crimes)
crimes = crimes.dropna(subset=['Latitude', 'Longitude'])
pts = gpd.GeoDataFrame(crimes, geometry=gpd.points_from_xy(crimes['Longitude'], crimes['Latitude']), crs='EPSG:4326')
joined = gpd.sjoin(pts, tracts[['GEOID', 'geometry']], how='inner', predicate='within')
joined['day'] = joined['Date'].dt.normalize()
ids = sorted(tracts['GEOID'])                              # sorted as text, the notebook's tract order
days = pd.date_range(START, END, freq='D')
panel = (joined.groupby(['GEOID', 'day']).size().unstack(fill_value=0).reindex(index=ids, columns=days, fill_value=0))
Y = panel.values.astype(float)                             # tracts x days
n_tr, n_days = Y.shape
tr_end = int(days.searchsorted(VAL_FROM))
te_start = int(days.searchsorted(TEST_FROM))
test_days = list(range(te_start, n_days))
n_pred = n_tr * len(test_days)
n_dec31 = int(Y[:, -1].sum()); n_dec30 = int(Y[:, -2].sum())
counts = pd.DataFrame([
    ('tracts in the boundary file', n_tr),
    ('crimes dated 2021-01-01 to 2023-12-31 (midnight bound)', n_window),
    ('crimes kept after the join to a tract', len(joined)),
    ('days in the panel', n_days),
    ('test predictions (tracts x December days)', n_pred),
    ('tracts with at least one crime', int((Y.sum(1) > 0).sum())),
    ('crimes on 2023-12-30', n_dec30),
    ('crimes on 2023-12-31', n_dec31)], columns=['quantity', 'value'])
counts.to_csv(f'{OUT}/step7_counts.csv', index=False)
print(f'1. data rebuilt ({time.time() - t0:.0f} s)\n')
print(counts.to_string(index=False))
print(f'\nMeaning: the thesis panel has {n_tr} tracts, {len(joined):,} crimes and {n_days:,} days, and its test month holds {n_pred:,} tract-day predictions. '
      f'Date filter: the crime dates are timestamps and the notebook keeps dates up to 2023-12-31 00:00:00, so 31 December 2023 holds only the {n_dec31} crimes stamped exactly midnight '
      f'against {n_dec30} on 30 December; that day is in the test set and its effect is shown below. Limit: a count that matches shows the same rows were joined, not that the portal rows are correct.')

# ---- 2. Simple references on the same test days (pooled over tract-days)
train_mean = Y[:, :tr_end].mean(1)
refs = {'Each tract\'s training average': lambda d: train_mean,
        'Average of the previous 30 days': lambda d: Y[:, d - 30:d].mean(1),
        'Average of the previous 7 days': lambda d: Y[:, d - 7:d].mean(1),
        'Same weekday, mean of the previous 4 weeks': lambda d: np.mean([Y[:, d - 7 * k] for k in (1, 2, 3, 4)], 0),
        'Yesterday\'s count': lambda d: Y[:, d - 1],
        'Zero': lambda d: np.zeros(n_tr)}


def score(fn, ds):
    p = np.concatenate([fn(d) for d in ds]); a = np.concatenate([Y[:, d] for d in ds])
    return len(a), float(np.abs(a - p).mean()), float(np.sqrt(((a - p) ** 2).mean())), float(1 - ((a - p) ** 2).sum() / ((a - a.mean()) ** 2).sum())


rows = []
for name, fn in refs.items():
    n, mae, rmse, r2 = score(fn, test_days)
    n2, mae2, _, r22 = score(fn, test_days[:-1])
    rows.append(dict(reference=name, predictions=n, MAE=mae, RMSE=rmse, R2=r2, MAE_without_dec31=mae2, R2_without_dec31=r22, predictions_without_dec31=n2))
ref = pd.DataFrame(rows)
ref.round(4).to_csv(f'{OUT}/step7_references.csv', index=False)
print(f'\n2. simple references on December 2023 ({time.time() - t0:.0f} s). MAE is crimes per tract per day, R squared is over all tract-days:\n')
print(ref.round({'MAE': 3, 'RMSE': 3, 'R2': 3, 'MAE_without_dec31': 3, 'R2_without_dec31': 3}).drop(columns='predictions_without_dec31').to_string(index=False))
a = ref.iloc[0]
print(f"\nMeaning: predicting each tract's own training average gives R squared {a['R2']:.3f} and MAE {a['MAE']:.3f} on the {int(a['predictions']):,} test predictions, "
      f"the natural reference for a daily crime model, because hot spots are persistent. Without 31 December the same reference gives R squared {a['R2_without_dec31']:.3f} and MAE {a['MAE_without_dec31']:.3f} "
      '(the nearly empty last day lowers this reference\'s R squared). Limit: one test month, and R squared is measured against the test month\'s own average.')

# ---- 3. Moran's I on the five-nearest-neighbour graph of tract centroids (row-standardised weights)
coords = tracts.set_index('GEOID').loc[ids, ['lon', 'lat']].values
w = KNN.from_array(coords, k=5)
w.transform = 'r'
np.random.seed(0)
tm = Y[:, :tr_end].mean(1)
m = Moran(tm, w, permutations=999)
I_mean, p_mean, expected = float(m.I), float(m.p_sim), float(m.EI)
daily = [(Moran(Y[:, d], w, permutations=0).I, Moran(Y[:, d] - tm, w, permutations=0).I) for d in range(tr_end)]
dd = pd.DataFrame(daily, columns=['raw', 'minus_tract_mean'])
mor = pd.DataFrame([
    ('Tract means over the training days', 1, I_mean),
    ('Each training day, median', len(dd), float(dd['raw'].median())),
    ('Each training day after removing the tract mean, median', len(dd), float(dd['minus_tract_mean'].median()))], columns=['measure', 'n', 'morans_I'])
mor['expected_if_no_pattern'] = expected
mor.round(4).to_csv(f'{OUT}/step7_moran.csv', index=False)
print(f"\n3. Moran's I ({time.time() - t0:.0f} s). Positive means neighbouring tracts have alike values; {expected:.4f} is expected with no pattern:\n")
print(mor.round(3).to_string(index=False))
print(f"\nMeaning: busy tracts sit next to busy tracts (tract means {I_mean:.3f}, permutation p {p_mean:.3f}), and a single day shows a weaker pattern ({mor.iloc[1]['morans_I']:.3f}). "
      f"After each tract's own mean is removed the daily value is {mor.iloc[2]['morans_I']:.3f}, close to zero, so most of the neighbour similarity is which tracts are busy all the time, "
      'not what happens on a given day. Limits: neighbours are the five nearest centroids; centroids are taken from unprojected shapes, as in the notebook; Moran\'s I shows similarity, not that neighbours influence each other.')
print(f'Step 7 total {time.time() - t0:.0f} s')

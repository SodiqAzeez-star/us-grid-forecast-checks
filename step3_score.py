"""Step 3: score LightGBM, with and without a day-ahead temperature forecast, against each operator's own day-ahead forecast.

Setup. One sample per operating day and operator (24 hourly targets). Inputs stop at 07:00 New York time on the previous day, which is
when the forecast is treated as issued. Train on operating days before 2025-08-01, choose the number of trees on 2025-08-01 to 2025-10-31
(validation), score once on 2025-11-01 to 2026-10-31 (test). The score is hourly MAPE (mean absolute percentage error) over hours where
demand, the operator's forecast and both LightGBM forecasts all exist. PJM's stored forecast is also scored moved one hour later
("operator aligned", see step 2). CISO's stored forecast runs far below its reported demand at midday for a reason not yet confirmed, so it is labelled not comparable.
Writes out/scores.csv (two decimals), out/scores_4dp.csv and out/scores.json.
"""
import json, os, time
import numpy as np, pandas as pd
import lightgbm as lgb
import weather
from common import *

NJOBS = int(os.environ.get('OMP_NUM_THREADS', 4))
t_start = time.time()

# ---- 1. Read, clean and arrange demand and operator forecasts into operating-day by hour-number matrices
b = read_balance({'Balancing Authority': 'ba', 'Data Date': 'date', 'Hour Number': 'hour', 'UTC Time at End of Hour': 'utc',
                  'Demand Forecast (MW)': 'fc', 'Demand (MW)': 'dem', 'Region': 'region'})
b['utc'] = pd.to_datetime(b['utc'], format=TS_FMT)
b['date'] = pd.to_datetime(b['date'], format='%m/%d/%Y')
b = b.drop_duplicates(['ba', 'utc'], keep='last')   # six-month files can repeat boundary hours; the later file wins
b = b[(b['utc'] > pd.Timestamp('2023-07-01')) & (b['date'] <= PERIOD_END)]
b = b[b['region'].isin(LOWER48)]
full_idx = pd.date_range(b['utc'].min(), b['utc'].max(), freq='h')
b = b[b['ba'].isin(SCORED)]
dem = b.pivot(index='utc', columns='ba', values='dem').reindex(full_idx)[SCORED]
fc = b.pivot(index='utc', columns='ba', values='fc').reindex(full_idx)[SCORED]
span = (dem.index > pd.Timestamp('2023-11-01 05:00')) & (dem.index <= pd.Timestamp('2026-11-01 08:00'))
med_all = dem[span].median()
lo, hi = med_all * LOW_MULT, med_all * HIGH_MULT


def clean(df):
    """Demand or forecast at or below zero, below 0.2 or above 2.5 times the operator's median becomes missing."""
    return df.mask((df <= 0) | df.lt(lo, axis=1) | df.gt(hi, axis=1))


dem_c, fc_c = clean(dem), clean(fc)
DAYS = pd.date_range(PERIOD_START, PERIOD_END, freq='D')
ND = len(DAYS)
TEST_MASK = DAYS >= TEST_START
VAL_MASK = (DAYS >= VAL_START) & (DAYS < TEST_START)
TRAIN_MASK = DAYS < VAL_START

A = np.full((ND, len(SCORED), 24), np.nan); F = A.copy(); U = np.zeros((ND, len(SCORED), 24), dtype='int64')
for oi, ba in enumerate(SCORED):
    s = b[b['ba'] == ba].set_index('utc')
    t = pd.DataFrame({'date': s['date'].values, 'hour': s['hour'].values, 'a': dem_c[ba].reindex(s.index).values,
                      'f': fc_c[ba].reindex(s.index).values, 'utc': s.index})
    t = t[(t['hour'] <= 24) & (t['date'] >= PERIOD_START)]
    piv = lambda k: t.pivot(index='date', columns='hour', values=k).reindex(index=DAYS, columns=range(1, 25))
    A[:, oi], F[:, oi] = piv('a').values, piv('f').values
    U[:, oi] = piv('utc').values.astype('datetime64[s]').astype('int64')   # hour-end instants, seconds
print(f'1. data prepared: {ND} operating days, {int(np.isfinite(A).sum())} demand hours ({time.time() - t_start:.0f} s)', flush=True)

# ---- 2. Temperature inputs (measured history and the day-ahead forecast), three cities per operator
print('2. temperature downloads or cache:', flush=True)
TEMP, TEMPFC = weather.load('measured'), weather.load('forecast')
temp_idx = TEMP.index.values.astype('datetime64[s]').astype('int64')
fc_idx = TEMPFC.index.values.astype('datetime64[s]').astype('int64')


def at(idx, vals, t):
    """Value at instant t (seconds) from a sorted hourly index, NaN if absent."""
    pos = np.clip(np.searchsorted(idx, t), 0, len(idx) - 1)
    return np.where(idx[pos] == t, vals[pos], np.nan)


# ---- 3. Features. Demand lags are at least 48 h before the target; every other input ends at the issue instant, except t_fc.
iss_sec = pd.DatetimeIndex([issue_utc(d) for d in DAYS]).values.astype('datetime64[s]').astype('int64')
demand = dem_c
dem_idx = demand.index.values.astype('datetime64[s]').astype('int64')
med = demand.loc[(demand.index >= '2023-11-01') & (demand.index < VAL_START)].median()   # training-period scale per operator
dow, doy = DAYS.day_of_week.values, DAYS.day_of_year.values
from pandas.tseries.holiday import USFederalHolidayCalendar
hol = DAYS.isin(USFederalHolidayCalendar().holidays(DAYS.min() - pd.Timedelta(days=3), DAYS.max() + pd.Timedelta(days=3))).astype(float)
cal = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), hol], 1)

rows = []
for oi, o in enumerate(SCORED):
    v = demand[o].values
    last = pd.Series(v, index=demand.index).ffill(limit=24).values
    scale = float(med[o])
    tv, fv = TEMP[o].values, TEMPFC[o].values
    tpos = np.searchsorted(temp_idx, iss_sec, side='right') - 1
    assert (temp_idx[tpos] <= iss_sec).all()
    tcs = np.concatenate([[0], np.cumsum(tv)])
    t_last = tv[tpos]
    t_mean24 = (tcs[tpos + 1] - tcs[tpos - 23]) / 24
    t_mean168 = (tcs[tpos + 1] - tcs[tpos - 167]) / 168
    pos = np.searchsorted(dem_idx, iss_sec, side='right') - 1
    csum = np.concatenate([[0], np.nancumsum(v)]); ccnt = np.concatenate([[0], np.cumsum(~np.isnan(v))])
    for h in range(24):
        t = U[:, oi, h]
        feats = {}
        for lag in (48, 72, 168, 336):
            feats[f'lag{lag}'] = at(dem_idx, v, t - lag * 3600) / scale
        feats['last'] = last[pos] / scale
        for w in (24, 168):
            lo_ = np.clip(pos + 1 - w, 0, None)
            feats[f'mean{w}'] = ((csum[pos + 1] - csum[lo_]) / np.maximum(ccnt[pos + 1] - ccnt[lo_], 1)) / scale
        feats['horizon'] = (t - iss_sec) / 3600.0
        feats['t_lag48'] = at(temp_idx, tv, t - 48 * 3600)
        feats['t_lag168'] = at(temp_idx, tv, t - 168 * 3600)
        feats['t_last'] = t_last
        feats['t_mean24'] = t_mean24
        feats['t_mean168'] = t_mean168
        feats['t_fc'] = at(fc_idx, fv, t)             # forecast made one day before, for the target hour
        df = pd.DataFrame(feats)
        df['hour'] = h; df['op'] = oi
        for ci in range(cal.shape[1]):
            df[f'cal{ci}'] = cal[:, ci]
        df['day'] = np.arange(ND)
        df['y'] = A[:, oi, h] / scale
        df['scale'] = scale
        rows.append(df)
D = pd.concat(rows, ignore_index=True)
assert (D['horizon'].dropna() > 0).all(), 'a target hour would be at or before the issue instant'
base = [c for c in D.columns if c not in ('y', 'day', 'scale') and not c.startswith('t_')]
VARIANTS = {'no weather': base,
            'temperature forecast': base + ['t_lag48', 't_lag168', 't_last', 't_mean24', 't_mean168', 't_fc']}
tr = D[TRAIN_MASK[D['day']] & D['y'].notna()]
va = D[VAL_MASK[D['day']] & D['y'].notna()]
assert D.loc[D['day'].isin(np.where(TRAIN_MASK)[0]), 'day'].max() < np.where(VAL_MASK)[0].min()
print(f'3. features built: {len(tr)} training rows, {len(va)} validation rows ({time.time() - t_start:.0f} s)', flush=True)

# ---- 4. Fit one pooled model per variant; the number of trees is chosen on validation only
preds, info = {}, {}
for name, cols in VARIANTS.items():
    t0 = time.time()
    m = lgb.LGBMRegressor(objective='l1', n_estimators=2000, learning_rate=0.03, num_leaves=31, min_child_samples=40, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_jobs=NJOBS, verbose=-1)
    m.fit(tr[cols], tr['y'], eval_X=va[cols], eval_y=va['y'], categorical_feature=['op'], callbacks=[lgb.early_stopping(100, verbose=False)])
    P = np.full_like(A, np.nan)
    P[D['day'].values, D['op'].values, D['hour'].values] = m.predict(D[cols]) * D['scale'].values
    preds[name] = P
    info[name] = dict(best_iteration=int(m.best_iteration_), fit_seconds=round(time.time() - t0, 1), features=cols)
    print(f'4. {name}: {m.best_iteration_} trees ({time.time() - t0:.0f} s)', flush=True)

# ---- 5. Score on the test year
Fa = F.copy()
pj = SCORED.index('PJM')
flat = F[:, pj, :].ravel()
Fa[:, pj, :] = np.concatenate([[np.nan], flat[:-1]]).reshape(F[:, pj, :].shape)   # PJM's forecast moved one hour later
models = {'LightGBM, no weather': preds['no weather'], 'LightGBM, temperature forecast': preds['temperature forecast'],
          'Operator as stored': F, 'Operator aligned (PJM moved one hour)': Fa}
ok = np.isfinite(A) & np.isfinite(F) & np.isfinite(Fa) & TEST_MASK[:, None, None]
for P in preds.values():
    ok &= np.isfinite(P)
ci = SCORED.index('CISO')
table = []
for name, P in models.items():
    e = np.where(ok, np.abs(P - A) / A, np.nan)
    row = {'model': name}
    for oi, o in enumerate(SCORED):
        row[o] = 100 * float(np.nanmean(e[:, oi]))
    row['pooled'] = 100 * float(np.nanmean(e))
    row['pooled_excl_CISO'] = 100 * float(np.nanmean(np.delete(e, ci, axis=1)))
    table.append(row)
precise = pd.DataFrame(table)
precise.round(4).to_csv(f'{OUT}/scores_4dp.csv', index=False)   # unrounded check values for borderline cases
tab = precise.round({c: 2 for c in precise.columns if c != 'model'})
tab.to_csv(f'{OUT}/scores.csv', index=False)
test_hours = U[TEST_MASK][np.broadcast_to(ok[TEST_MASK], U[TEST_MASK].shape)]
meta = dict(test_start=str(TEST_START.date()), test_end=str(PERIOD_END.date()), scored_hours=int(ok.sum()),
            data_first_utc=str(pd.Timestamp(int(U[U > 0].min()), unit='s')), data_last_utc=str(pd.Timestamp(int(test_hours.max()), unit='s')),
            train_end=str((VAL_START - pd.Timedelta(days=1)).date()), validation=f'{VAL_START.date()} to {(TEST_START - pd.Timedelta(days=1)).date()}',
            threads=NJOBS, models=info, ciso_note='CISO is not comparable: its stored forecast runs far below its reported demand at midday for a reason not yet confirmed')
json.dump(meta, open(f'{OUT}/scores.json', 'w'), indent=1)

show = tab.copy()
show['CISO'] = show['CISO'].astype(str) + ' (not comparable)'
print(f'\nTest year {TEST_START.date()} to {PERIOD_END.date()}, {int(ok.sum())} operator-hours. Hourly MAPE, percent (lower is better):\n')
print(show.to_string(index=False))
lg, op = tab.iloc[1], tab.iloc[3]
beat = [o for o in SCORED if o != 'CISO' and lg[o] < op[o]]
print(f"\nMeaning: with the temperature forecast, LightGBM's error across the four comparable operators is {lg['pooled_excl_CISO']} against {op['pooled_excl_CISO']} for the aligned operators. "
      f"It is below the operator's own forecast for: {', '.join(beat) if beat else 'none of PJM, ERCO, MISO, NYIS'} (CISO is not comparable). "
      "PJM aligned is the fair PJM comparison. Limits are in RUN_STEPS.md.")
print(f'Step 3 total {time.time() - t_start:.0f} s')

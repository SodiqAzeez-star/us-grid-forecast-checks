"""Step 2: offset audit of every balancing authority in the EIA-930 BALANCE files.

For each authority and each calendar month the stored day-ahead forecast is shifted by -3 to +3 hours against demand. A month is flagged
when the best shift is not zero and cuts the mean absolute percentage error (MAPE) by at least 10 percent. An authority is reported as
shifted when it has at least FLAG_MONTHS flagged months. flagged_mape_* are hour-weighted MAPE over the flagged months only, as stored and at each month's best shift. Writes out/audit_table.csv, out/audit_by_month.csv, out/audit_flagged.csv.
"""
import numpy as np, pandas as pd
from common import *

END = pd.Timestamp('2026-11-01')    # the last full month in the files is October 2026
OFFS = range(-3, 4)
MIN_HOURS_MONTH = 300
MONTH_GAIN = 0.10
FLAG_MONTHS = 5
HIGHLIGHT = ['PJM', 'CISO']

d = read_balance({'Balancing Authority': 'ba', 'UTC Time at End of Hour': 'utc', 'Demand Forecast (MW)': 'fc', 'Demand (MW)': 'dem'})
d['utc'] = pd.to_datetime(d['utc'], format=TS_FMT)
d = d.drop_duplicates(['ba', 'utc'], keep='last')   # six-month files can repeat boundary hours; the later file wins
d = d[d.utc < END]

rows, monthly = [], []
for ba, g in d.groupby('ba'):
    if g['fc'].notna().sum() < 1000:
        continue
    g = g.set_index('utc').reindex(pd.date_range(g.utc.min(), END - pd.Timedelta(hours=1), freq='h'))
    g = g[g.index >= g['fc'].first_valid_index()]
    x = pd.DataFrame({'dem': g['dem'].where(g['dem'] > 0)})
    for k in OFFS:
        x[f'f{k}'] = g['fc'].shift(-k).where(lambda s: s > 0)
    x = x[x.notna().all(axis=1)]
    ape = {k: (x[f'f{k}'] - x['dem']).abs() / x['dem'] for k in OFFS}
    for per, idx in x.groupby(x.index.to_period('M')).groups.items():
        if len(idx) < MIN_HOURS_MONTH:
            continue
        mp = {k: 100 * ape[k].loc[idx].mean() for k in OFFS}
        b = min(mp, key=mp.get)
        gain = 1 - mp[b] / mp[0]
        monthly.append(dict(ba=ba, month=str(per), hours=len(idx), best_offset=b, mape_stored=mp[0], mape_best=mp[b], gain=gain,
                            flagged=bool(b != 0 and gain >= MONTH_GAIN)))
    mp = {k: 100 * ape[k].mean() for k in OFFS}
    b = min(mp, key=mp.get)
    rows.append(dict(ba=ba, hours_compared=len(x), mean_demand_mw=x['dem'].mean(), mape_stored=mp[0], best_offset=b, mape_best=mp[b]))

table = pd.DataFrame(rows)
month = pd.DataFrame(monthly)
n_flag = month.groupby('ba')['flagged'].sum().rename('months_flagged')
n_mon = month.groupby('ba')['flagged'].size().rename('months_checked')
table = table.merge(n_flag, on='ba', how='left').merge(n_mon, on='ba', how='left').fillna({'months_flagged': 0, 'months_checked': 0})
fl = month[month['flagged']].assign(ws=lambda x: x.hours * x.mape_stored, wb=lambda x: x.hours * x.mape_best).groupby('ba')
fl = pd.DataFrame({'flagged_mape_stored': fl['ws'].sum() / fl['hours'].sum(), 'flagged_mape_best': fl['wb'].sum() / fl['hours'].sum()})
table = table.merge(fl, on='ba', how='left')
fm = month[month['flagged']].groupby('ba')['month']
table = table.merge(fm.min().rename('first_flagged'), on='ba', how='left').merge(fm.max().rename('last_flagged'), on='ba', how='left')
table['shifted'] = table['months_flagged'] >= FLAG_MONTHS
table['highlight'] = table['ba'].isin(HIGHLIGHT)
table.to_csv(f'{OUT}/audit_table.csv', index=False)
month.to_csv(f'{OUT}/audit_by_month.csv', index=False)
flagged = table[table['shifted'] | table['highlight']].sort_values('months_flagged', ascending=False)
flagged.to_csv(f'{OUT}/audit_flagged.csv', index=False)

print(f'{len(table)} balancing authorities with a day-ahead forecast; {int(table["shifted"].sum())} show a timing shift (at least {FLAG_MONTHS} flagged months).\n')
show = flagged[['ba', 'months_flagged', 'months_checked', 'first_flagged', 'last_flagged', 'best_offset', 'flagged_mape_stored', 'flagged_mape_best']].copy()
show[['flagged_mape_stored', 'flagged_mape_best']] = show[['flagged_mape_stored', 'flagged_mape_best']].round(2)
print(show.to_string(index=False))
print('\nMeaning: for these authorities the stored forecast lines up with demand better after moving it by best_offset hours, in the months counted. '
      'The first_flagged and last_flagged columns show when. The limits of this check are in RUN_STEPS.md.')

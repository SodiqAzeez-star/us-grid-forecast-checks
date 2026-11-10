"""Hourly 2 m air temperature from Open-Meteo for three cities per operator, averaged per operator (UTC hours, degrees Celsius).

measured : archive API, variable temperature_2m (what the weather was).
forecast : Previous Runs API, variable temperature_2m_previous_day1, model gfs_seamless (the forecast made one day before).
Each result is cached as a CSV in raw/ and read from there on later runs. Open-Meteo's terms must be checked before any use beyond private reruns.
"""
import hashlib, json, os, time, urllib.request
import pandas as pd
from common import RAW

POINTS = {
    'PJM': {'Philadelphia': (39.95, -75.17), 'Washington DC': (38.91, -77.04), 'Chicago': (41.88, -87.63)},
    'ERCO': {'Houston': (29.76, -95.37), 'Dallas': (32.78, -96.80), 'San Antonio': (29.42, -98.49)},
    'MISO': {'Minneapolis': (44.98, -93.27), 'Detroit': (42.33, -83.05), 'New Orleans': (29.95, -90.07)},
    'NYIS': {'New York City': (40.71, -74.01), 'Buffalo': (42.89, -78.88), 'Albany': (42.65, -73.76)},
    'CISO': {'Los Angeles': (34.05, -118.24), 'San Jose': (37.34, -121.89), 'San Diego': (32.72, -117.16)},
}
START, END = '2023-10-01', '2026-10-31'
KINDS = {
    'measured': ('https://archive-api.open-meteo.com/v1/archive?latitude=%s&longitude=%s&start_date=%s&end_date=%s'
                 '&hourly=temperature_2m&timezone=UTC&temperature_unit=celsius', 'temperature_2m', 5),
    'forecast': ('https://previous-runs-api.open-meteo.com/v1/forecast?latitude=%s&longitude=%s&start_date=%s&end_date=%s'
                 '&hourly=temperature_2m_previous_day1&models=gfs_seamless&timezone=UTC&temperature_unit=celsius', 'temperature_2m_previous_day1', 3),
}


def _get(url):
    for attempt in range(6):
        try:
            return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e:
            print('  retry', e, flush=True)
            time.sleep(20 * (attempt + 1))
    raise SystemExit('Open-Meteo request failed: ' + url)


def load(kind):
    """Return the per-operator temperature frame for `kind`, downloading it first if raw/ has no copy."""
    path = os.path.join(RAW, f'temperature_{kind}.csv')
    if not os.path.exists(path):
        url, var, pause = KINDS[kind]
        out = {}
        for op, cities in POINTS.items():
            cols = []
            for name, (la, lo) in cities.items():
                j = _get(url % (la, lo, START, END))
                s = pd.Series(j['hourly'][var], index=pd.to_datetime(j['hourly']['time']), dtype=float)
                cols.append(s)
                print(f'  {kind} {op} {name}: {len(s)} hours, {int(s.isna().sum())} missing', flush=True)
                time.sleep(pause)
            out[op] = pd.concat(cols, axis=1).mean(axis=1)
        pd.DataFrame(out).to_csv(path, index_label='utc')
    print(f'  {os.path.basename(path)} sha256 {hashlib.sha256(open(path, "rb").read()).hexdigest()}')
    df = pd.read_csv(path, index_col='utc')
    df.index = pd.to_datetime(df.index)
    return df

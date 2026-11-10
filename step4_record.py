"""Step 4: PJM's own forecasts of its summer peak, by report year (vintage), against the peak that followed.

Reads data/pjm_forecast_by_vintage.csv (each forecast with its report file, URL, PDF page and table) and data/pjm_actual_peaks.csv
(each actual with its workbook URL and sheet row). Error = 100 x (forecast - actual) / actual, positive when the forecast was too high.
Horizon = target year minus report year. Two actuals are used: PJM's weather-normalised peak and the unrestricted (as-metered) peak.
Writes out/pjm_record.csv (one row per forecast and actual pair, sources kept) and out/pjm_record_summary.csv.
"""
import numpy as np, pandas as pd
from common import *

fc = pd.read_csv(f'{DATA}/pjm_forecast_by_vintage.csv')
act = pd.read_csv(f'{DATA}/pjm_actual_peaks.csv').rename(columns={'year': 'target_year', 'source_url': 'actual_source_url', 'table': 'actual_table'})
fc = fc.rename(columns={'forecast_mw': 'forecast_mw', 'source_url': 'forecast_source_url', 'table': 'forecast_table', 'pdf_page': 'forecast_pdf_page'})
rec = fc.merge(act, on='target_year', how='inner')
rec['forecast_pdf_page'] = rec['forecast_pdf_page'].astype('Int64')
rec['horizon_years'] = rec['target_year'] - rec['vintage']
rec['err_vs_weather_normalised_pct'] = (100 * (rec['forecast_mw'] - rec['weather_normalized_mw']) / rec['weather_normalized_mw']).round(2)
rec['err_vs_unrestricted_pct'] = (100 * (rec['forecast_mw'] - rec['unrestricted_actual_mw']) / rec['unrestricted_actual_mw']).round(2)
cols = ['vintage', 'target_year', 'horizon_years', 'forecast_mw', 'weather_normalized_mw', 'unrestricted_actual_mw', 'err_vs_weather_normalised_pct',
        'err_vs_unrestricted_pct', 'source_file', 'forecast_pdf_page', 'forecast_table', 'forecast_source_url', 'actual_table', 'actual_source_url']
rec = rec[cols].sort_values(['vintage', 'target_year'])
rec.to_csv(f'{OUT}/pjm_record.csv', index=False)

out = []
for h in (1, 3, 5):
    r = rec[rec['horizon_years'] == h]
    for label, c in (('weather-normalised peak', 'err_vs_weather_normalised_pct'), ('unrestricted (as-metered) peak', 'err_vs_unrestricted_pct')):
        out.append(dict(horizon_years=h, actual=label, n=len(r), mean_error_pct=round(r[c].mean(), 2), mean_abs_error_pct=round(r[c].abs().mean(), 2),
                        min_pct=r[c].min(), max_pct=r[c].max()))
summary = pd.DataFrame(out)
summary.to_csv(f'{OUT}/pjm_record_summary.csv', index=False)
print(f'{len(rec)} forecast and actual pairs from {rec["vintage"].nunique()} PJM reports ({rec["vintage"].min()} to {rec["vintage"].max()}), targets {rec["target_year"].min()} to {rec["target_year"].max()}.\n')
print(summary.to_string(index=False))
print('\nOne pair, so the sources can be opened by hand:')
r = rec.iloc[0]
print(f'  {r.source_file} page {r.forecast_pdf_page}, {r.forecast_table}: forecast {r.forecast_mw:,} MW for {r.target_year}; actual {r.weather_normalized_mw:,} MW ({r.actual_table}).')
print('\nMeaning: one year ahead PJM\'s own forecast peak ran above the weather-normalised peak on average and in some years below it. '
      'Positive means too high. It tests PJM against PJM\'s own reports, not against PJM\'s files of hourly data.')

# ---- The wider record: PJM from 2014 (a separate series), NYISO and ISO-NE, in out/record_summary_all.csv.
# Same error and horizon definitions as above; only the weather-normalised actual is used. Each operator's own files are in data/.
def pairs(forecast_files, actual_file):
    f = pd.concat([pd.read_csv(f'{DATA}/{n}') for n in forecast_files], ignore_index=True)
    a = pd.read_csv(f'{DATA}/{actual_file}').rename(columns={'year': 'target_year'})
    p = f.merge(a[['target_year', 'weather_normalized_mw']], on='target_year', how='inner')
    p['horizon_years'] = p['target_year'] - p['vintage']
    p['err'] = (100 * (p['forecast_mw'] - p['weather_normalized_mw']) / p['weather_normalized_mw']).round(2)
    return p

SERIES = [
    ('PJM from 2017', ['pjm_forecast_by_vintage.csv'], 'pjm_actual_peaks.csv', 'Reports 2017 to 2025.'),
    ('PJM from 2014', ['pjm_early_forecast_by_vintage.csv', 'pjm_forecast_by_vintage.csv'], 'pjm_actual_peaks.csv',
     'Reports 2014 to 2025. PJM\'s weather-normalised actual changed over time (its 2015 report gives 156,140 MW for 2014, its 2026 workbook 148,880 MW, the value used); '
     'the 2014 and 2015 reports forecast before load management and efficiency and the 2016 report after solar.'),
    ('NYISO', ['nyiso_forecast_by_vintage.csv'], 'nyiso_actual_peaks.csv', 'Gold Books 2014 to 2024; the 2018, 2020 and 2022 books were not usable.'),
    ('ISO-NE', ['isone_forecast_by_vintage.csv'], 'isone_actual_peaks.csv', 'CELT reports 2013 to 2025; the net-load definition changes in the 2025 report.'),
]
wide = []
for name, files, actual_file, caveat in SERIES:
    p = pairs(files, actual_file)
    for h in (1, 3, 5):
        r = p[p['horizon_years'] == h]['err']
        wide.append(dict(series=name, horizon_years=h, n=len(r), mean_error_pct=round(r.mean(), 2), min_pct=r.min(), max_pct=r.max(), caveat=caveat))
wide = pd.DataFrame(wide)
wide.to_csv(f'{OUT}/record_summary_all.csv', index=False)
print('\nForecast minus weather-normalised actual, percent of actual, by series:\n')
print(wide.drop(columns='caveat').to_string(index=False))
print('\nMeaning: in these four series the operators\' own forecasts ran above the weather-normalised peak on average, and by more at longer horizons. '
      'Limits: PJM\'s normal changed over time, ISO-NE\'s definition changes in its 2025 report, NYISO\'s 2018, 2020 and 2022 books were not usable, '
      'and the two PJM series share most of their pairs.')

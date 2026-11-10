"""Step 6: write one static results page, out/results.html, from the run's own output files (no number is typed into this script).

Reads out/scores.csv, out/scores.json, out/audit_table.csv, out/audit_by_month.csv, out/price_lines.csv, out/record_summary_all.csv, out/step1_files.csv.
When out/neural_scores.csv exists (step 8) the page also shows the neural scores; without it the page is unchanged.
Usage: python step6_screen.py [output_path]
"""
import html, json, os, sys
import pandas as pd
from common import *

dest = sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/results.html'
scores = pd.read_csv(f'{OUT}/scores.csv')
meta = json.load(open(f'{OUT}/scores.json'))
audit = pd.read_csv(f'{OUT}/audit_table.csv')
months = pd.read_csv(f'{OUT}/audit_by_month.csv')
price = pd.read_csv(f'{OUT}/price_lines.csv')
record = pd.read_csv(f'{OUT}/record_summary_all.csv')
files = pd.read_csv(f'{OUT}/step1_files.csv')
e = html.escape

run_date = files['checked_utc'].max()[:10]
OPS = [o for o in SCORED]
row = {r['model']: r for _, r in scores.iterrows()}
nw, tf, oa = (row['LightGBM, no weather'], row['LightGBM, temperature forecast'], row['Operator aligned (PJM moved one hour)'])
beat = [OPERATOR_NAMES[o] for o in OPS if o != 'CISO' and tf[o] < oa[o]]

# ---- 1. Scores
head_cols = ''.join(f'<th>{e(OPERATOR_NAMES[o])}{"<br>(not comparable)" if o == "CISO" else ""}</th>' for o in OPS) + '<th>The four comparable</th>'
body = ''
for _, r in scores.iterrows():
    cells = ''.join(f'<td>{r[o]:.2f}</td>' for o in OPS) + f'<td>{r["pooled_excl_CISO"]:.2f}</td>'
    body += f'<tr><th scope="row">{e(r["model"].replace("LightGBM", "Simple model"))}</th>{cells}</tr>'
vmax = max(float(scores.loc[scores['model'].isin([nw['model'], tf['model'], oa['model']]), OPS].max().max()), 1)
bars = ''
for o in OPS:
    items = ''
    for label, cls, r in (('Simple model, no weather', 'nw', nw), ('Simple model, temperature forecast', 'tf', tf), ('Operator forecast (aligned)', 'op', oa)):
        items += (f'<div class="bar"><span class="bl">{e(label)}</span><span class="track"><span class="fill f{cls}" '
                  f'style="width:{100 * r[o] / vmax:.1f}%"></span></span><span class="bv">{r[o]:.2f}</span></div>')
    bars += f'<div class="op"><h3>{e(OPERATOR_NAMES[o])}{" (not comparable)" if o == "CISO" else ""}</h3>{items}</div>'
sentence_scores = (f'With a temperature forecast, a simple model beat the operator\'s own forecast for {", ".join(beat)}.' if beat else
                   'The grid operators\' own forecasts beat a simple model on PJM, ERCOT, MISO and NYISO.')
detail_scores = (f'On the same test year, a simple model (LightGBM, a standard tree model) with a temperature forecast missed by {tf["pooled_excl_CISO"]:.2f} percent on average across those four grids, '
                 f'against {oa["pooled_excl_CISO"]:.2f} percent for the operators\' own forecasts. Without weather it missed by {nw["pooled_excl_CISO"]:.2f} percent. '
                 'CAISO is shown but not compared, because its stored forecast runs far below its reported demand at midday for a reason not yet confirmed.')

# ---- 2. Audit
n_ba, n_shift = len(audit), int(audit['shifted'].sum())
shown = audit[audit['shifted'] | audit['highlight']].sort_values('months_flagged', ascending=False)
arow = ''.join(f'<tr><th scope="row">{e(r["ba"])}</th><td>{int(r["months_flagged"])} of {int(r["months_checked"])}</td><td>{int(r["best_offset"]):+d} h</td>'
               f'<td>{r["flagged_mape_stored"]:.2f}</td><td>{r["flagged_mape_best"]:.2f}</td></tr>' for _, r in shown.iterrows())
NAMES = {'CISO': 'California ISO (CISO)', 'AZPS': 'Arizona Public Service (AZPS)', 'SC': 'South Carolina Public Service Authority (SC)'}
sentence_audit = f'{n_shift} of {n_ba} balancing authorities show a timing mismatch between their stored forecast and their demand.'
detail_audit = (f'Balancing authorities are the bodies that balance each grid area. Each was checked month by month: the stored forecast was moved by up to three hours either way, and a month counts when the best move cuts the error by at least 10 percent. '
                f'Found: {", ".join(NAMES.get(b, b) for b in shown.loc[shown["shifted"], "ba"])}. This shows the two columns disagree in time, not which one is mislabelled. The check uses the public federal EIA file; the Arizona and South Carolina cases are smaller and unconfirmed.')

# ---- 3. Price
last = price.iloc[-1]
prow = ''.join(f'<tr><th scope="row">{e(r["delivery_year"])}</th><td class="eq">{e(r["arithmetic"].split(": ", 1)[1])}</td></tr>' for _, r in price.iterrows())
sentence_price = f'One percent of error in PJM\'s forecast peak is worth about ${last["one_point_usd_million"]:,.1f} million in the {last["delivery_year"]} delivery year.'
detail_price = 'Each line multiplies one percent of the capacity requirement by the auction price and 365 days. It counts quantity only and does not say anyone saved this amount.'

# ---- 4. Forecast record
rec_rows = ''
for name, g in record.groupby('series', sort=False):
    g = g.set_index('horizon_years')
    cells = ''.join(f'<td data-h="{h} year{"" if h == 1 else "s"} ahead">{g.loc[h, "mean_error_pct"]:+.1f} ({g.loc[h, "min_pct"]:+.1f} to {g.loc[h, "max_pct"]:+.1f}), {int(g.loc[h, "n"])}</td>' for h in (1, 3, 5))
    rec_rows += f'<tr><th scope="row">{e(name)}</th>{cells}</tr>'
rec_notes = ''.join(f'<li>{e(n)}: {e(c)}</li>' for n, c in record.drop_duplicates('series')[['series', 'caveat']].itertuples(index=False))
first, last5 = record[record['horizon_years'] == 1].set_index('series'), record[record['horizon_years'] == 5].set_index('series')
sentence_record = ('Mean forecast error against the weather-normalised peak, one year and five years ahead: ' +
                   '; '.join(f'{n} {first.loc[n, "mean_error_pct"]:+.1f} and {last5.loc[n, "mean_error_pct"]:+.1f} percent' for n in first.index) + '.')
detail_record = ('Each cell is the mean of forecast minus weather-normalised actual as a percent of actual (lowest to highest), then the number of forecast and actual pairs. '
                 'Positive means the forecast was too high. The two PJM rows share most of their pairs.')

# ---- 5. Neural scores (only when step 8 has written them)
neural_html = ''
if os.path.exists(f'{OUT}/neural_scores.csv'):
    neural = pd.read_csv(f'{OUT}/neural_scores.csv')
    nrows = ''.join(f'<tr><th scope="row">{e(r["model"])}, seed {int(r["seed"])}</th>' + ''.join(f'<td>{r[o]:.2f}</td>' for o in OPS) + f'<td>{r["pooled_excl_CISO"]:.2f}</td></tr>'
                    for _, r in neural.iterrows())
    ref_rows = ''.join(f'<tr><th scope="row">{e(r["model"])}</th>' + ''.join(f'<td>{r[o]:.2f}</td>' for o in OPS) + f'<td>{r["pooled_excl_CISO"]:.2f}</td></tr>' for r in (nw, oa))
    ranges = []
    for m, g in neural.groupby('model', sort=False):
        lo, hi = g['pooled_excl_CISO'].min(), g['pooled_excl_CISO'].max()
        ranges.append(f'{m} {lo:.2f}' if len(g) == 1 else f'{m} {lo:.2f} to {hi:.2f}')
    sentence_neural = f'Without weather, across the four comparable grids, the neural models averaged {" and ".join(ranges)} percent error, against {nw["pooled_excl_CISO"]:.2f} for LightGBM and {oa["pooled_excl_CISO"]:.2f} for the operators\' own forecasts.'
    detail_neural = ('Same test year and same average percent error per hour as the table above, over the same four grids. Each neural row is one training run at the seed shown; another seed gives a slightly '
                     'different score, so a gap between two neural rows smaller than that spread says nothing. CAISO is shown but is not comparable.')
    neural_html = (f'<h2>{e(sentence_neural)}</h2>\n<p>{e(detail_neural)}</p>\n<table class="scores"><thead><tr><th>Average percent error per hour (lower is better)</th>{head_cols}</tr></thead>'
                   f'<tbody>{nrows}{ref_rows}</tbody></table>\n\n')

page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Electricity load forecast checks: results of this run</title>
<style>
:root {{ --ink:#1b1f24; --muted:#4a5560; --line:#cfd6dd; --fill:#f3f5f7; --c1:#8a949e; --c2:#1f6f8b; --c3:#c9822b; --r:6px; --s1:8px; --s2:16px; --s3:24px; --s4:40px; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; padding:var(--s3) var(--s4); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }}
h1 {{ font-size:28px; line-height:1.2; margin:0 0 var(--s1); text-wrap:balance; }}
h2 {{ font-size:20px; line-height:1.3; margin:var(--s4) 0 var(--s2); text-wrap:balance; }}
h3 {{ font-size:15px; margin:0 0 var(--s1); }}
p, li {{ text-wrap:pretty; margin:0 0 var(--s2); }}
.when {{ color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; margin:0 0 var(--s2); }}
th, td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:right; vertical-align:top; }}
thead th {{ background:var(--fill); font-weight:600; }}
thead th:first-child, th.eqh {{ text-align:left; }}
tbody th {{ text-align:left; font-weight:600; }}
td.eq {{ text-align:left; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:var(--s2); }}
.op {{ background:var(--fill); border-radius:var(--r); padding:var(--s2); }}
.bar {{ display:grid; grid-template-columns:1fr; gap:2px; margin-bottom:var(--s1); }}
.bl {{ font-size:14px; color:var(--muted); }}
.track {{ display:block; background:#fff; border-radius:3px; height:14px; }}
.fill {{ display:block; height:14px; border-radius:3px; }}
.fnw {{ background:var(--c1); }} .ftf {{ background:var(--c2); }} .fop {{ background:var(--c3); }}
.bv {{ font-size:14px; font-weight:600; font-variant-numeric:tabular-nums; }}
.foot {{ margin-top:var(--s4); }}
@media (max-width:640px) {{ body {{ padding:var(--s2); }} table {{ font-size:13px; }} th, td {{ padding:6px 3px; }} table.scores {{ display:none; }} table.stack thead {{ display:none; }} table.stack tr, table.stack tbody th, table.stack td {{ display:block; text-align:left; border:0; padding:2px 0; }} table.stack tr {{ border-bottom:1px solid var(--line); padding:var(--s1) 0; }} table.stack td::before {{ content:attr(data-h) ": "; font-weight:600; }} }}
</style></head><body>
<h1>Electricity load forecast checks: results of this run</h1>
<p class="when">Run on {e(run_date)} (UTC). Test data {e(meta["test_start"])} to {e(meta["test_end"])}; audit data {e(months["month"].min())} to {e(months["month"].max())}. Public files from the US Energy Information Administration (Form EIA-930).</p>

<h2>{e(sentence_scores)}</h2>
<p>{e(detail_scores)}</p>
<table class="scores"><thead><tr><th>Average percent error per hour (lower is better)</th>{head_cols}</tr></thead><tbody>{body}</tbody></table>
<div class="grid">{bars}</div>
<p class="when">Bars: grey is the simple model without weather, blue is the simple model with the temperature forecast, orange is the operator's own forecast with PJM moved one hour. All bars share one scale.</p>

<h2>{e(sentence_audit)}</h2>
<p>{e(detail_audit)}</p>
<table><thead><tr><th>Balancing authority</th><th>Months with a shift</th><th>Best shift</th><th>Error as stored (%, months with a shift)</th><th>Error after shift (%)</th></tr></thead><tbody>{arow}</tbody></table>

<h2>{e(sentence_price)}</h2>
<p>{e(detail_price)}</p>
<table><thead><tr><th>Delivery year</th><th class="eqh">Arithmetic</th></tr></thead><tbody>{prow}</tbody></table>

<h2>{e(sentence_record)}</h2>
<p>{e(detail_record)}</p>
<table class="stack"><thead><tr><th>Series</th><th>1 year ahead</th><th>3 years ahead</th><th>5 years ahead</th></tr></thead><tbody>{rec_rows}</tbody></table>
<ul>{rec_notes}</ul>

{neural_html}<p class="foot">Weather data by Open-Meteo.com, licensed under CC BY 4.0; electricity data from the US Energy Information Administration, Form EIA-930. Every number on this page was written by the run's own scripts into the out folder; this page contains no typed-in results. Limits are listed in RUN_STEPS.md.</p>
</body></html>'''
open(dest, 'w').write(page)
print('Wrote', dest)

"""Step 5: what one percentage point of PJM peak forecast error is worth in the capacity auction, quantity effect only.

For each delivery year: one point = 0.01 x Reliability Requirement adjusted for FRR (UCAP MW) x RTO clearing price ($/MW-day) x 365.
The requirement comes from PJM's Base Residual Auction planning parameters workbook for that year (URL and row in data/pjm_one_point_price.csv);
the price is the RTO resource clearing price from PJM's 2028/2029 BRA Results Report, Table 2, PDF page 8. Writes out/price_lines.csv.
Not included: the demand curve (price moves with quantity), the price cap from 2026/27, leap days, and who pays.
"""
import pandas as pd
from common import *

src = pd.read_csv(f'{DATA}/pjm_one_point_price.csv')
rows = []
for r in src.itertuples():
    point_mw = 0.01 * r.rr_adjusted_for_frr_ucap_mw
    usd = point_mw * r.rto_clearing_price_usd_mw_day * 365
    line = f'{r.delivery_year}: 0.01 x {r.rr_adjusted_for_frr_ucap_mw:,.1f} MW x ${r.rto_clearing_price_usd_mw_day:,.2f}/MW-day x 365 = ${usd / 1e6:,.1f} million'
    rows.append(dict(delivery_year=r.delivery_year, requirement_mw=r.rr_adjusted_for_frr_ucap_mw, price_usd_mw_day=r.rto_clearing_price_usd_mw_day,
                     one_point_mw=round(point_mw, 1), one_point_usd_million=round(usd / 1e6, 1), arithmetic=line,
                     requirement_source=r.params_source, requirement_cell=r.params_cell))
    print(line)
out = pd.DataFrame(rows)
out.to_csv(f'{OUT}/price_lines.csv', index=False)
print('\nMeaning: each line is the yearly capacity cost of PJM\'s forecast peak being one percent off, holding the clearing price fixed. '
      'It states worth under those assumptions, not a saving anyone made.')

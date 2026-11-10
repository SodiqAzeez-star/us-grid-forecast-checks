"""Shared constants and helpers for the run kit. Every step imports this file and nothing else from the kit."""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, 'raw')
OUT = os.path.join(HERE, 'out')
DATA = os.path.join(HERE, 'data')
for d in (RAW, OUT):
    os.makedirs(d, exist_ok=True)

EIA_BASE = 'https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/'
BAL_FILES = ['2023_Jul_Dec', '2024_Jan_Jun', '2024_Jul_Dec', '2025_Jan_Jun', '2025_Jul_Dec', '2026_Jan_Jun', '2026_Jul_Dec']


def bal_path(tag):
    return os.path.join(RAW, f'EIA930_BALANCE_{tag}.csv')


SCORED = ['PJM', 'ERCO', 'MISO', 'NYIS', 'CISO']
OPERATOR_NAMES = {'PJM': 'PJM', 'ERCO': 'ERCOT', 'MISO': 'MISO', 'NYIS': 'NYISO', 'CISO': 'CAISO'}
PERIOD_START = pd.Timestamp('2023-11-01')   # first operating day used
PERIOD_END = pd.Timestamp('2026-10-31')     # last operating day used (inclusive)
TEST_START = pd.Timestamp('2025-11-01')
VAL_START = pd.Timestamp('2025-08-01')
ISSUE_TZ = 'America/New_York'               # a forecast for day D is issued at 07:00 here on day D-1
ISSUE_HOUR = 7
LOW_MULT, HIGH_MULT = 0.2, 2.5              # outlier bounds as multiples of the operator's median demand
LOWER48 = {'CAL', 'CAR', 'CENT', 'FLA', 'MIDA', 'MIDW', 'NE', 'NY', 'NW', 'SE', 'SW', 'TEN', 'TEX'}
TS_FMT = '%m/%d/%Y %I:%M:%S %p'


def read_balance(cols):
    """Read the listed columns of every BALANCE file into one frame (files must exist, see step1_download.py)."""
    missing = [bal_path(t) for t in BAL_FILES if not os.path.exists(bal_path(t))]
    if missing:
        raise SystemExit('Missing input files, run step1_download.py first:\n  ' + '\n  '.join(missing))
    frames = [pd.read_csv(bal_path(t), usecols=list(cols), thousands=',').rename(columns=cols) for t in BAL_FILES]
    return pd.concat(frames, ignore_index=True)


def issue_utc(day):
    """UTC instant (naive) at which the forecast for operating day `day` is issued."""
    t = pd.Timestamp(day) - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    return t.tz_localize(ISSUE_TZ, ambiguous=False, nonexistent='shift_forward').tz_convert('UTC').tz_localize(None)

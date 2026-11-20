# Run kit: electricity load forecast checks

Six short Python scripts that rerun the checks on a normal computer (no GPU, about 4 minutes, 0.5 GB of memory), plus two separate steps: `step7_thesis_checks.py` (thesis crime data, extra packages in `requirements-neural.txt`) and `step8_neural.py` (trains two neural networks, up to about two hours on four cores). Start with `RUN_STEPS.md`, which gives each command, the time it takes, the numbers to expect and what each result means and does not mean.

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python step1_download.py   # then step2_audit.py ... step6_screen.py
```

- `common.py`, `weather.py`: shared settings and the temperature download.
- `data/`: eight small input files (past forecasts, outcomes and auction figures) with every value's source page or URL.
- `raw/` (downloads) and `out/` (results, including `out/results.html`) are created by the steps.

## Data sources

- Electricity demand and day-ahead forecasts: US Energy Information Administration, Form EIA-930, public domain.
- Temperature: weather data by Open-Meteo.com (https://open-meteo.com/), licensed under CC BY 4.0. Results that use it carry this credit.
- The files in `data/`: each value's source page or URL is in the file.

## Licence and citation

MIT licence, see `LICENSE`. To cite this work, use `CITATION.cff`.

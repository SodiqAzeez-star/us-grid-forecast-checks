# Run steps

Steps 1 to 6 rerun the checks on a normal computer: no GPU, no neural networks, about 4 minutes and 0.5 GB of memory on four cores once the packages are installed. Steps 7 and 8 are separate: step 7 needs the thesis crime file and extra packages, and step 8 trains neural networks and takes up to about two hours on four cores. Each step prints a table and one plain sentence saying what it means. The numbers below are reference results; EIA revises its files, so a later run can differ in the last digit, and the difference is reported, not adjusted.

## Before the first step

You need Python 3.12 or newer, an internet connection, and about 700 MB of free disk (310 MB of downloads, 375 MB of packages). To check, type `python --version` in a command window. If it shows an older version, or Windows opens the Microsoft Store instead, install Python 3.13 from https://www.python.org/downloads/ and tick "Add python.exe to PATH" on the first screen of the installer.

Open a command window inside this folder. On Windows, open the folder in File Explorer, click the address bar, type `cmd` and press Enter. On a Mac, right-click the folder and choose New Terminal at Folder.

Then set up once. On Windows:

```
python -m venv venv
venv\Scripts\pip install -r requirements.txt
```

On a Mac or Linux:

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

Run every step from the same window. On Windows: `venv\Scripts\python step1_download.py`, then `step2_audit.py` and so on. On a Mac or Linux: `venv/bin/python step1_download.py`. The commands below are written the short way, `python step1_download.py`; put `venv\Scripts\` (Windows) or `venv/bin/` (Mac) in front of `python` each time.

To see whether a step ran, look in `out/`. Nothing outside `raw/` and `out/` is written.

## Step 1: download the data (about 10 seconds, 310 MB)

`python step1_download.py` downloads seven public files from the US Energy Information Administration (EIA), the Form EIA-930 "Balance" files, July 2023 to the latest day, into `raw/`. It prints each file's size and SHA-256 checksum and saves them in `out/step1_files.csv`.

Expect seven lines of checksums, for example `EIA930_BALANCE_2023_Jul_Dec.csv 42.6 MB c719fc17...`. Anyone downloading on the same day should see the same checksums, unless EIA revises a file in between; the newest file grows every day, so its checksum changes.

What it means: these are the exact inputs every other step uses, and the checksums let a second person show they used the same ones. Limit: the checksum proves which file was read, not that EIA's numbers are right.

## Step 2: check every balancing authority's timing (about 30 seconds)

`python step2_audit.py` takes each of the 55 balancing authorities that publish a day-ahead forecast and, for each month, slides the stored forecast by up to three hours either way against actual demand. A month is flagged if the best slide is not zero and lowers the average error by at least 10 percent. An authority is reported as shifted if five or more months are flagged. It writes `out/audit_table.csv`, `out/audit_by_month.csv` and `out/audit_flagged.csv`.

Expect: `55 balancing authorities ..., 4 show a timing shift`, and this table (error is the average percent error per hour, over the flagged months):

| Authority | Months flagged of 40 | Best shift | Error as stored | After shift |
|---|---|---|---|---|
| PJM | 40 | -1 h | 3.54 | 2.50 |
| CISO | 29 | -1 h | 7.09 | 5.66 |
| AZPS | 17 | -1 h | 8.21 | 6.34 |
| SC | 9 | +1 h | 3.72 | 3.08 |

What it means: PJM's stored forecast lines up with PJM's demand one hour better than the stored timing in every month, so comparing the two columns as stored makes PJM look worse than its forecast is. Limits: the check shows the files disagree in time, not which column is mislabelled; PJM's own files are not part of this kit; CISO's gap at midday is partly a level difference, not only a timing difference, so CISO is not comparable; AZPS and SC are unconfirmed. The threshold of five flagged months is a choice made for this kit; the next most-flagged authority is FMPP with 4 (`out/audit_table.csv`).

## Step 3: score a simple model against each operator's forecast (about 3 minutes the first time, 30 seconds after)

`python step3_score.py` builds one LightGBM model (a standard gradient-boosted tree model, run on the CPU) pooled across five operators: PJM, ERCOT, MISO, NYISO and CISO. It predicts each hour of the next day using demand history up to 07:00 New York time the day before, the calendar, and optionally temperature. It trains on days before 1 August 2025, picks the number of trees on August to October 2025, and is scored once on 1 November 2025 to 31 October 2026. The first time, it downloads temperature from Open-Meteo, three cities per operator, about 2.7 minutes of waiting by design (the kit pauses between requests); the files are kept in `raw/`. It writes `out/scores.csv`, `out/scores_4dp.csv` and `out/scores.json`.

Expect (average percent error per hour, 43,483 operator-hours):

| | PJM | ERCOT | MISO | NYISO | CISO | All five | Four comparable |
|---|---|---|---|---|---|---|---|
| LightGBM, no weather | 5.80 | 5.54 | 4.58 | 6.72 | 4.53 | 5.44 | 5.66 |
| LightGBM, temperature forecast | 3.35 | 3.71 | 3.15 | 4.64 | 4.02 | 3.78 | 3.71 |
| Operator as stored | 3.50 | 1.91 | 2.57 | 2.70 | 8.84 | 3.90 | 2.67 |
| Operator aligned (PJM moved one hour) | 2.60 | 1.91 | 2.57 | 2.70 | 8.84 | 3.71 | 2.45 |

ERCOT with the temperature forecast is 3.7146 unrounded, so it prints 3.71 here and can be written 3.72 if rounded from 3.715; `out/scores_4dp.csv` holds the unrounded values.

What it means: the model with a temperature forecast does not beat the operator's own forecast on PJM, ERCOT, MISO or NYISO: across those four it misses by 3.71 percent against 2.45. The 3.78 against 3.71 across all five is not a fair comparison, because it includes CISO. The operators' own forecasts are the better ones there. Limits: CISO is not comparable, because its stored forecast runs far below its reported demand at midday for a reason not yet confirmed, so CISO's 8.84 is not a fair target and the model's 4.02 is not a win; the temperature uses three cities per operator, not the operator's full area; the "forecast" temperature is the previous-day run of one weather model (GFS), and its lead time was not verified for every hour; one test year only; the model is one fixed setup, not tuned further.

## Step 4: PJM's own forecasts against what happened (about 1 second)

`python step4_record.py` reads `data/pjm_forecast_by_vintage.csv` and `data/pjm_actual_peaks.csv`. Every forecast row there carries its source file, PDF page and table; every actual carries its workbook URL and sheet row. It compares PJM's summer peak forecast in each report from 2017 to 2025 with the peak that followed, and writes `out/pjm_record.csv` (45 pairs, sources kept) and `out/pjm_record_summary.csv`.

Expect (error = forecast minus actual, as a percent of actual; positive means too high):

| Years ahead | Pairs | Average error vs weather-normalised peak | Range |
|---|---|---|---|
| 1 | 8 | +0.94 | -0.85 to +2.83 |
| 3 | 6 | +1.52 | -1.52 to +4.42 |
| 5 | 4 | +1.75 | -0.12 to +2.97 |

To check by hand, open the PJM report named in any row at the stated page: the first row is the 2017 report, page 59, Table B-1, RTO summer peak 152,999 MW for 2017.

What it means: PJM's own forecasts ran a little high on average and moved by a few percent year to year. Limits: this tests PJM against PJM's own reports, not against PJM's hourly files; weather-normalised actuals come from PJM's own Table F-1 and are a different construction from the forecast; the forecast values were extracted from the PDFs by a script and 24 of 24 sampled values matched the rendered pages, which tests the extraction, not the sources.

The same step also prints a wider table and writes `out/record_summary_all.csv` (mean, lowest, highest and pairs at 1, 3 and 5 years ahead, weather-normalised actual only, with a caveat column), from each operator's own files in `data/`. The table above and its files are unchanged. Expect mean errors at 1, 3 and 5 years ahead of:

- PJM from 2017: +0.9, +1.5, +1.8.
- PJM from 2014 (a separate series; PJM's weather-normalised actual for the same year changed between its reports, and the 2014 to 2016 reports differ in what load is netted): +2.2, +3.5, +5.2.
- NYISO (Gold Books 2014 to 2024; the 2018, 2020 and 2022 books were not usable): +1.9, +3.2, +5.8.
- ISO-NE (CELT reports 2013 to 2025; its net-load definition changes in the 2025 report): +1.4, +3.4, +6.0.

## Step 5: price one point of error (under 1 second)

`python step5_price.py` prints one arithmetic line per delivery year: 0.01 times PJM's reliability requirement times the auction clearing price times 365 days. Inputs are in `data/pjm_one_point_price.csv` with the planning-parameter workbook URL and row for each year; prices are from PJM's 2028/2029 Base Residual Auction Results Report, Table 2, PDF page 8. It writes `out/price_lines.csv`.

Expect nine lines, from `2020/21: 0.01 x 154,355.3 MW x $76.53/MW-day x 365 = $43.1 million` to `2028/29: 0.01 x 145,149.1 MW x $325.00/MW-day x 365 = $172.2 million`. The 2025/26 line is $131.6 million, 2026/27 $161.6 million, 2027/28 $171.7 million.

What it means: a one percent miss in the forecast peak changes the capacity PJM buys by about 1,300 to 1,500 MW, worth the stated dollars at that year's price. Limit: quantity only. It leaves out the demand curve (price moves with quantity), the price cap in force from 2026/27, leap days and who pays, and it states a worth, not a saving.

## Step 6: make the results page (under 1 second)

`python step6_screen.py` writes `out/results.html` from the files in `out/` alone and prints its path. Open it in any browser. It shows the date of the run and the date range of the data, the score table with one bar comparison per operator, the audit count, and the priced lines, each under a plain sentence. It is built to read at 1024 pixels wide.

## Step 7: the thesis data checks (about 1 to 2 minutes, 4 GB of memory)

Install the extra packages first: `venv\Scripts\pip install -r requirements-neural.txt` on Windows, or `venv/bin/pip install -r requirements-neural.txt` on a Mac. Download two files from the Chicago Data Portal: the crimes file (Crimes - 2001 to Present, CSV export) and the census tract boundaries (Boundaries - Census Tracts - 2000, GeoJSON export). Then `python step7_thesis_checks.py CRIMES.csv TRACTS.geojson` rebuilds the thesis panel the way the notebook does (crimes dated 1 January 2021 up to and including 31 December 2023, each joined to the tract that contains it, counted per tract per day), scores six simple references on December 2023, and measures Moran's I. It writes `out/step7_counts.csv`, `out/step7_references.csv` and `out/step7_moran.csv`.

Expect:

| Count | Value |
|---|---|
| Tracts | 878 |
| Crimes kept after the join | 696,262 |
| Days | 1,095 |
| Test predictions (878 tracts x 31 days) | 27,218 |
| Crimes on 30 December 2023, on 31 December 2023 | 740, 30 |

Each tract's own training average (days before 1 November 2023) scores R squared 0.335 and MAE 0.676 crimes per tract per day on the test month; without 31 December it scores 0.348 and 0.675. Moran's I is 0.387 on the tract means, 0.141 as the median over the 1,034 training days, and 0.018 as that median after each tract's own mean is removed.

What it means: each tract's own training average is the natural reference for a model of daily crime counts, because crime hot spots are persistent, and neighbouring tracts are alike mostly because the same tracts are busy every day. Limits: the notebook keeps dates up to 31 December 2023 at 00:00:00, so that day holds only the 30 crimes stamped exactly midnight and sits in the test set; the portal revises its crimes file, so a later download can change the counts; neighbours are the five nearest tract centroids, computed from unprojected shapes as the notebook does; matching counts show the same rows were read, not that the portal rows are right.

## Step 8: train the neural models (on four cores, about 9 minutes for the LSTM and 1.5 to 2 hours for the graph model)

`python step8_neural.py` trains two models on the same EIA files and scores them on the step 3 test year, with no weather. It downloads the seven matching INTERCHANGE files into `raw/` (685 MB, a few seconds to a minute) and needs step 1 first. One argument picks the model: `python step8_neural.py lstm` or `gat`; with none it trains both one after the other. `--seed N` sets the random seed (default 0, printed at the start). It uses a GPU when one is visible and the CPU otherwise, writes a checkpoint after every epoch into `out/neural/`, and if it is stopped, running the same command again resumes after the last finished epoch. `--fresh` starts over. Set `OMP_NUM_THREADS` to the number of cores to use. `python step8_neural.py gat --bench` trains 2 epochs and prints the seconds per epoch without scoring.

The two models are an LSTM that reads each operator's own last 168 hours of demand, and a graph attention BiLSTM that first passes the demand of all 50 well-covered balancing authorities through three graph attention layers at each hour, with a link wherever EIA reports interchange between two of them in the training days, then reads each of the five scored operators with a bidirectional LSTM. Both predict the next day's 24 hours from the information available at 07:00 New York time the day before. Settings are the pilot's validation winners (LSTM: 128 hidden units, learning rate 0.003, dropout 0.3; graph: 64 and 128 units, learning rate 0.001, dropout 0.2), 40 epochs at most, stop after 10 epochs without a better validation error (August to October 2025), and the best-validation checkpoint is scored once on the test year. It writes `out/neural_scores.csv` (two decimals), `out/neural_scores_4dp.csv` and `out/neural_scores.json`, and step 6 adds a neural section to the page when `out/neural_scores.csv` exists.

Expect (seed 0, hourly percent error): plain LSTM 5.44 pooled (PJM 5.47, ERCOT 5.34, MISO 4.00, NYISO 6.36, CISO 6.05); graph model 6.11 pooled (PJM 6.20, ERCOT 5.91, MISO 4.57, NYISO 7.45, CISO 6.41). Over the seeds run: LSTM 5.44, 5.44 and 5.45 at seeds 0, 1 and 2; graph model 6.11 and 6.15 at seeds 0 and 1. Your own numbers can differ a little with the machine. Timing comes from two epochs on four threads: 12.8 seconds per epoch for the LSTM and 167.8 for the graph model, with the LSTM running all 40 epochs and the graph model 33 to 40; a free cloud notebook has not been measured.

What it means: the plain LSTM beat the graph model on all five operators at every seed run, by more than the spread between seeds pooled, so on this data and these links the graph structure did not help. Both models trail LightGBM with a temperature forecast and the operators' own forecasts on the comparable grids. Limits: neural scores move with the seed and with the machine, so only differences larger than the spread between seeds mean anything; no weather input; one test year; the graph is built from interchange links without testing other graphs; CISO is not comparable. The scored hours are those where demand, the operator's forecast and its PJM-aligned copy all exist, which can differ by a few hours from step 3's count because step 3 also requires its LightGBM forecasts.

## What the checks do not show

- CISO is not comparable: its stored forecast runs far below its reported demand at midday for a reason not yet confirmed.
- The one-hour shift is shown inside the federal EIA file; PJM's own files are not part of this kit.
- Temperature uses three cities per operator.
- Pricing is quantity only.
- These are one fixed setup and one test year.
- Step 8's scores depend on the seed; Step 7 is the thesis data only and says nothing about electricity.
- Open-Meteo's terms of use must be read before any use beyond private reruns.

## Publish the thesis code on GitHub and get a Zenodo DOI

The thesis notebook is separate from this kit.

1. Make a folder containing the notebook file exactly as it is now. Do not rerun it or clear its outputs: the saved results stay unchanged. Add a `README.md`, a licence file of your choice, and a `CITATION.cff` with the title, your name and the year.
2. Do not add the Chicago crime data. Link to the Chicago Data Portal in the README and name the file used.
3. Put this line in the README: `Known issue: the notebook keeps crime records with date <= 2023-12-31, which is midnight at the start of that day, so 31 December 2023 holds only the 30 records stamped exactly 00:00:00 (740 on 30 December). The saved results are unchanged.`
4. Create a free account on github.com, choose New repository, name it, set it Public, and upload the folder's files with Add file, Upload files. Commit.
5. Create a free account on zenodo.org by choosing Log in with GitHub. In Zenodo, open the GitHub settings page and switch the new repository on.
6. On GitHub open the repository, choose Releases, Create a new release, give it a tag such as `v1.0`, and publish it.
7. Within a few minutes Zenodo shows the release with a DOI. Copy the DOI into the README and into `CITATION.cff`, and commit that change (it does not alter the notebook).
8. Check the DOI link opens the archived copy.

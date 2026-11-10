"""Step 8: train a plain LSTM and a graph attention BiLSTM on the EIA-930 data and score them on the same test year as step 3.

Usage: python step8_neural.py [lstm|gat|both] [--seed 0] [--epochs 40] [--bench] [--fresh]
  lstm, gat, both  which model to train (default both, one after the other)
  --seed N         random seed for weight start and batch order (default 0, printed at the start)
  --epochs N       training epochs (default 40; the cosine learning-rate schedule is set by this number)
  --bench          train for --epochs (default 2) and print seconds per epoch, write no scores
  --fresh          ignore a saved checkpoint and start from epoch 1

Needs the BALANCE files from step 1 and downloads the seven matching INTERCHANGE files into raw/ (about 685 MB). Inputs: each operator's own
demand history over the 168 hours up to 07:00 New York time the day before, the hour and weekday, and the calendar. Output: 24 hourly demands
for each of the five operators. Both models use the settings that scored best on validation (August to October 2025) in the pilot's four-setting
search; no setting is tuned here. Training stops after 10 epochs without a better validation error or at --epochs. The checkpoint with the best
validation error is the one scored, once, on 2025-11-01 to 2026-10-31.

The LSTM reads each scored operator's own series. The graph model builds a graph of every balancing authority with at least 95 percent clean
demand hours, linked where EIA reports interchange between two of them in the training days, passes the 50 demand series through three
graph attention layers at every hour, then reads each scored node with a bidirectional LSTM.
The device is a GPU when one is visible and the CPU otherwise. A checkpoint is written after every epoch to out/neural/ and a rerun with the same
model, seed and epochs resumes from it. Writes out/neural_scores.csv, out/neural_scores_4dp.csv and out/neural_scores.json.
"""
import argparse, glob, hashlib, json, os, sys, time, urllib.request
import numpy as np, pandas as pd
from common import *

p = argparse.ArgumentParser()
p.add_argument('model', nargs='?', default='both', choices=['lstm', 'gat', 'both'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=None)
p.add_argument('--bench', action='store_true')
p.add_argument('--fresh', action='store_true')
args = p.parse_args()
EPOCHS = args.epochs or (2 if args.bench else 40)
args.fresh = args.fresh or args.bench
NEURAL = os.path.join(OUT, 'neural')
os.makedirs(NEURAL, exist_ok=True)
LOOKBACK, PATIENCE = 168, 10
LABELS = {'lstm': 'LSTM', 'gat': 'Graph attention BiLSTM (interchange links)'}
CFG = {'lstm': dict(hidden=128, lr=3e-3, drop=0.3, batch=32), 'gat': dict(gcn=64, hidden=128, lr=1e-3, drop=0.2, batch=16)}
UA = {'User-Agent': 'runkit-eia930/1.0 (research rerun)'}



def np_state_out():
    """NumPy's random state as plain values and a tensor, so the checkpoint loads with weights_only=True."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return [name, torch.from_numpy(keys.astype(np.int64)), int(pos), int(has_gauss), float(cached)]


def np_state_in(v):
    """Rebuilds NumPy's random state from np_state_out's values."""
    return (v[0], v[1].numpy().astype(np.uint32), v[2], v[3], v[4])

def interchange_path(tag):
    return os.path.join(RAW, f'EIA930_INTERCHANGE_{tag}.csv')


def download_interchange():
    """Fetch the INTERCHANGE files that are missing or differ in size from the server's copy; record their checksums in out/step8_files.csv."""
    rows = []
    for tag in BAL_FILES:
        name = f'EIA930_INTERCHANGE_{tag}.csv'; url = EIA_BASE + name; dest = interchange_path(tag)
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method='HEAD'), timeout=60) as r:
            size = int(r.headers.get('Content-Length') or -1)   # -1 when the server does not report a size: keep a file already present
        if not (os.path.exists(dest) and size in (-1, os.path.getsize(dest))):
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r, open(dest + '.part', 'wb') as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                    os.replace(dest + '.part', dest)
                    break
                except Exception as e:
                    print(f'  retry {attempt + 1}: {e}'); time.sleep(10 * (attempt + 1))
            else:
                raise SystemExit(f'Could not download {url}')
        h = hashlib.sha256()
        with open(dest, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        rows.append(dict(file=name, url=url, bytes=os.path.getsize(dest), sha256=h.hexdigest()))
        print(f'  {name:38s} {os.path.getsize(dest) / 1e6:7.1f} MB  {h.hexdigest()[:16]}...', flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'step8_files.csv'), index=False)


def prepare():
    """Demand matrices for the scored operators (as in step 3), the hourly cleaned demand of every well-covered authority, and the graph edges."""
    b = read_balance({'Balancing Authority': 'ba', 'Data Date': 'date', 'Hour Number': 'hour', 'UTC Time at End of Hour': 'utc',
                      'Demand Forecast (MW)': 'fc', 'Demand (MW)': 'dem', 'Region': 'region'})
    b['utc'] = pd.to_datetime(b['utc'], format=TS_FMT)
    b['date'] = pd.to_datetime(b['date'], format='%m/%d/%Y')
    b = b.drop_duplicates(['ba', 'utc'], keep='last')
    b = b[(b['utc'] > pd.Timestamp('2023-07-01')) & (b['date'] <= PERIOD_END)]
    b = b[b['region'].isin(LOWER48)]
    dem = b.pivot(index='utc', columns='ba', values='dem')
    fc = b.pivot(index='utc', columns='ba', values='fc')
    full_idx = pd.date_range(dem.index.min(), dem.index.max(), freq='h')
    dem, fc = dem.reindex(full_idx), fc.reindex(full_idx)
    span = (dem.index > pd.Timestamp('2023-11-01 05:00')) & (dem.index <= pd.Timestamp('2026-11-01 08:00'))
    med = dem[span].median()
    lo, hi = med * LOW_MULT, med * HIGH_MULT
    clean = lambda df: df.mask((df <= 0) | df.lt(lo, axis=1) | df.gt(hi, axis=1))
    dem_c, fc_c = clean(dem), clean(fc)
    nodes = sorted(dem_c.columns[dem_c[span].notna().mean() >= 0.95])
    days = pd.date_range(PERIOD_START, PERIOD_END, freq='D')
    A = np.full((len(days), len(SCORED), 24), np.nan); F = A.copy()
    for oi, ba in enumerate(SCORED):
        s = b[b['ba'] == ba].set_index('utc')
        t = pd.DataFrame({'date': s['date'].values, 'hour': s['hour'].values, 'a': dem_c[ba].reindex(s.index).values, 'f': fc_c[ba].reindex(s.index).values})
        t = t[(t['hour'] <= 24) & (t['date'] >= PERIOD_START)]
        for k, M in (('a', A), ('f', F)):
            M[:, oi] = t.pivot(index='date', columns='hour', values=k).reindex(index=days, columns=range(1, 25)).values
    edges = set()
    for tag in BAL_FILES:
        i = pd.read_csv(interchange_path(tag), usecols=['Balancing Authority', 'Data Date', 'Directly Interconnected Balancing Authority', 'Interchange (MW)'])
        i.columns = ['a', 'date', 'b', 'mw']
        i['date'] = pd.to_datetime(i['date'], format='%m/%d/%Y')
        i = i[(i['date'] >= PERIOD_START) & (i['date'] < VAL_START) & i['mw'].notna()]   # training days only
        edges |= set(map(tuple, i[['a', 'b']].drop_duplicates().values))
    ns = set(nodes)
    und = sorted({tuple(sorted(e)) for e in edges if e[0] in ns and e[1] in ns and e[0] != e[1]})
    return dem_c[nodes], A, F, days, nodes, und


def calendar(days):
    from pandas.tseries.holiday import USFederalHolidayCalendar
    dow, doy = days.day_of_week.values, days.day_of_year.values
    hol = days.isin(USFederalHolidayCalendar().holidays(days.min() - pd.Timedelta(days=3), days.max() + pd.Timedelta(days=3))).astype(float)
    return np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25), hol], 1).astype('float32')


def score_row(P, A, F, Fa, test_mask, label, seed, info):
    """Hourly percent error over the test hours where demand, the operator forecast and its PJM-aligned copy exist (step 3's rule, without LightGBM)."""
    ok = np.isfinite(A) & np.isfinite(F) & np.isfinite(Fa) & np.isfinite(P) & test_mask[:, None, None]
    e = np.where(ok, np.abs(P - A) / A, np.nan)
    row = {'model': label, 'seed': seed}
    for oi, o in enumerate(SCORED):
        row[o] = 100 * float(np.nanmean(e[:, oi]))
    row['pooled'] = 100 * float(np.nanmean(e))
    row['pooled_excl_CISO'] = 100 * float(np.nanmean(np.delete(e, SCORED.index('CISO'), axis=1)))
    return {**row, 'scored_hours': int(ok.sum()), **info}


def collect():
    """Rebuild the three score files from every finished run in out/neural/."""
    res = [json.load(open(f)) for f in glob.glob(os.path.join(NEURAL, '*_result.json'))]
    if not res:
        return
    res.sort(key=lambda r: (list(LABELS).index(r['kind']), r['seed']))
    cols = ['model', 'seed'] + SCORED + ['pooled', 'pooled_excl_CISO']
    precise = pd.DataFrame([r['row'] for r in res])[cols]
    precise.round(4).to_csv(f'{OUT}/neural_scores_4dp.csv', index=False)
    precise.round({c: 2 for c in cols if c not in ('model', 'seed')}).to_csv(f'{OUT}/neural_scores.csv', index=False)
    json.dump(dict(test_start=str(TEST_START.date()), test_end=str(PERIOD_END.date()), ciso_note='CISO is not comparable: its forecast is a different quantity from its reported demand',
                   runs=[{k: v for k, v in r['row'].items() if k not in SCORED and k not in ('pooled', 'pooled_excl_CISO', 'model')} | {'model': r['row']['model']} for r in res]),
              open(f'{OUT}/neural_scores.json', 'w'), indent=1)


def train(kind, data):
    import torch, torch.nn as nn, torch.nn.functional as Fn
    dem_c, A, F, days, nodes, und = data
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nthreads = int(os.environ.get('OMP_NUM_THREADS', 4))
    torch.set_num_threads(nthreads)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = CFG[kind]; BATCH = cfg['batch']; N = len(nodes)
    ND = len(days)
    TEST_MASK, VAL_MASK, TRAIN_MASK = days >= TEST_START, (days >= VAL_START) & (days < TEST_START), days < VAL_START
    hrs = dem_c.index
    hr_sec = hrs.values.astype('datetime64[s]').astype('int64')
    fit = hrs < VAL_START                                         # the scaler sees training-period hours only
    mu, sd = dem_c[fit].mean(), dem_c[fit].std()
    Z = ((dem_c - mu) / sd).ffill(limit=24).fillna(0.0).values.astype('float32')   # fill forward at most 24 h, then the training mean
    et = hrs.tz_localize('UTC').tz_convert(ISSUE_TZ)
    tfeat = np.stack([np.sin(2 * np.pi * et.hour / 24), np.cos(2 * np.pi * et.hour / 24), np.sin(2 * np.pi * et.day_of_week / 7), np.cos(2 * np.pi * et.day_of_week / 7)], 1).astype('float32')
    iss_sec = pd.DatetimeIndex([issue_utc(d) for d in days]).values.astype('datetime64[s]').astype('int64')
    POS = np.searchsorted(hr_sec, iss_sec, side='right') - 1
    assert (hr_sec[POS] <= iss_sec).all() and POS.min() - LOOKBACK + 1 >= 0
    sidx = [nodes.index(o) for o in SCORED]
    tmu, tsd = mu[SCORED].values.astype('float32'), sd[SCORED].values.astype('float32')
    Y = ((A - tmu[None, :, None]) / tsd[None, :, None]).astype('float32')
    CAL = calendar(days)

    def batch(idx):
        w = np.stack([np.arange(POS[i] - LOOKBACK + 1, POS[i] + 1) for i in idx])
        assert (hr_sec[w] <= iss_sec[idx][:, None]).all()          # no input hour after the issue time
        return [torch.from_numpy(a).to(dev) for a in (Z[w], tfeat[w], CAL[idx], Y[idx])]

    class Head(nn.Module):
        def __init__(self, din, drop):
            super().__init__()
            self.op = nn.Embedding(len(SCORED), 8)
            self.fc1 = nn.Linear(din + 5 + 8, 64); self.fc2 = nn.Linear(64, 24); self.d = nn.Dropout(drop)

        def forward(self, h, cal):
            e = self.op.weight[None].expand(h.size(0), -1, -1)
            c = cal[:, None].expand(-1, h.size(1), -1)
            return self.fc2(Fn.relu(self.fc1(self.d(torch.cat([h, c, e], -1)))))

    class LSTMRegressor(nn.Module):
        def __init__(self, hidden, drop):
            super().__init__()
            self.lstm = nn.LSTM(5, hidden, num_layers=2, batch_first=True, dropout=drop)
            self.head = Head(hidden, drop)

        def forward(self, x, t, cal):
            B, T, _ = x.shape
            z = x[:, :, sidx].permute(0, 2, 1)
            inp = torch.cat([z.unsqueeze(-1), t[:, None].expand(-1, len(SCORED), -1, -1)], -1).reshape(B * len(SCORED), T, 5)
            _, (hn, _) = self.lstm(inp)
            return self.head(hn[-1].reshape(B, len(SCORED), -1), cal)

    class GATBiLSTM(nn.Module):
        def __init__(self, gcn, hidden, drop, edge_index):
            super().__init__()
            from torch_geometric.nn import GATConv
            self.g1 = GATConv(5, gcn, heads=2, concat=True, dropout=drop)
            self.g2 = GATConv(gcn * 2, gcn, heads=2, concat=True, dropout=drop)
            self.g3 = GATConv(gcn * 2, gcn, heads=1, concat=False, dropout=drop)
            self.ln1, self.ln2, self.ln3 = nn.LayerNorm(gcn * 2), nn.LayerNorm(gcn * 2), nn.LayerNorm(gcn)
            self.d = nn.Dropout(drop)
            self.lstm = nn.LSTM(gcn, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=drop)
            self.ln_out = nn.LayerNorm(hidden * 2)
            self.head = Head(hidden * 2, drop)
            self.register_buffer('ei', edge_index)
            self._cache = {}

        def big_edges(self, copies):
            """The same edge list repeated for every (sample, hour) graph in the batch."""
            if copies not in self._cache:
                off = (torch.arange(copies, device=self.ei.device) * N).view(-1, 1, 1)
                self._cache[copies] = (self.ei[None] + off).permute(1, 0, 2).reshape(2, -1)
            return self._cache[copies]

        def forward(self, x, t, cal):
            B, T, _ = x.shape
            h = torch.cat([x.unsqueeze(-1), t[:, :, None].expand(-1, -1, N, -1)], -1).reshape(B * T * N, 5)
            ei = self.big_edges(B * T)
            h = self.d(Fn.elu(self.ln1(self.g1(h, ei))))
            h = self.d(Fn.elu(self.ln2(self.g2(h, ei))))
            h = Fn.elu(self.ln3(self.g3(h, ei)))
            h = h.view(B, T, N, -1)[:, :, sidx].permute(0, 2, 1, 3).reshape(B * len(SCORED), T, -1)
            _, (hn, _) = self.lstm(h)
            fh = self.ln_out(torch.cat([hn[-2], hn[-1]], 1)).reshape(B, len(SCORED), -1)
            return self.head(fh, cal)

    if kind == 'lstm':
        model = LSTMRegressor(cfg['hidden'], cfg['drop'])
    else:
        ix = {n: i for i, n in enumerate(nodes)}
        s = [(ix[a], ix[b]) for a, b in und]
        ei = torch.tensor(sorted(set(s + [(b, a) for a, b in s])), dtype=torch.long).T.contiguous()
        model = GATBiLSTM(cfg['gcn'], cfg['hidden'], cfg['drop'], ei)
    model.to(dev)

    def predict(day_idx):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(day_idx), BATCH):
                x, t, c, _ = batch(day_idx[i:i + BATCH])
                out.append(model(x, t, c).cpu().numpy())
        return np.concatenate(out) * tsd[None, :, None] + tmu[None, :, None]

    def val_mape():
        d = np.where(VAL_MASK)[0]
        P = predict(d)
        ok = np.isfinite(A[d]) & np.isfinite(F[d]) & np.isfinite(P)
        return float(np.nanmean(np.where(ok, np.abs(P - A[d]) / A[d], np.nan)))

    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=cfg['lr'] / 100)
    tr_days = np.where(TRAIN_MASK)[0]
    tr_days = tr_days[np.isfinite(Y[tr_days]).any((1, 2))]
    stem = f'{kind}_s{args.seed}' + ('_bench' if args.bench else '')
    last_path, best_path, log_path = (os.path.join(NEURAL, f'{stem}_{k}') for k in ('last.pt', 'best.pt', 'log.csv'))
    state = dict(epoch=0, best=1e9, best_epoch=0, bad=0, hist=[])
    if os.path.exists(last_path) and not args.fresh:
        ck = torch.load(last_path, weights_only=True, map_location='cpu')
        if ck['epochs'] == EPOCHS:
            model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
            np.random.set_state(np_state_in(ck['np_rng'])); torch.set_rng_state(ck['torch_rng'])
            state = ck['state']
            print(f'resumed from the checkpoint after epoch {state["epoch"]}', flush=True)
    print(f'{LABELS[kind]}: seed {args.seed}, device {dev}, {nthreads} threads, {sum(q.numel() for q in model.parameters())} parameters, '
          f'{len(tr_days)} training days, {N} graph nodes, {EPOCHS} epochs at most', flush=True)
    done = state['bad'] >= PATIENCE or state['epoch'] >= EPOCHS
    while not done:
        ep = state['epoch'] + 1
        t0 = time.time(); model.train(); perm = np.random.permutation(tr_days); tl = []
        for i in range(0, len(perm), BATCH):
            x, t, c, y = batch(perm[i:i + BATCH])
            opt.zero_grad()
            pred = model(x, t, c); m = torch.isfinite(y)
            loss = Fn.smooth_l1_loss(pred[m], y[m]); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl.append(loss.item())
        sched.step()
        vm = val_mape()
        sec = time.time() - t0
        state['epoch'] = ep
        if vm < state['best']:
            state.update(best=vm, best_epoch=ep, bad=0); torch.save(model.state_dict(), best_path)
        else:
            state['bad'] += 1
        state['hist'].append(dict(epoch=ep, train_loss=round(float(np.mean(tl)), 5), val_mape=round(vm, 5), seconds=round(sec, 1)))
        pd.DataFrame(state['hist']).to_csv(log_path, index=False)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), np_rng=np_state_out(),
                        torch_rng=torch.get_rng_state(), state=state, epochs=EPOCHS), last_path)
        print(f'epoch {ep} loss {np.mean(tl):.4f} validation error {100 * vm:.2f}% seconds {sec:.1f}', flush=True)
        done = state['bad'] >= PATIENCE or ep >= EPOCHS
    secs = [h['seconds'] for h in state['hist']]
    if args.bench:
        print(f'{LABELS[kind]}: {np.mean(secs):.1f} seconds per epoch (training and validation) on {nthreads} threads over {len(secs)} epochs: {secs}')
        return
    model.load_state_dict(torch.load(best_path, weights_only=True, map_location=dev))
    P = np.full_like(A, np.nan)
    P[:] = predict(np.arange(ND))
    np.save(os.path.join(NEURAL, f'{stem}_predictions.npy'), P)
    Fa = F.copy(); pj = SCORED.index('PJM')
    Fa[:, pj, :] = np.concatenate([[np.nan], F[:, pj, :].ravel()[:-1]]).reshape(F[:, pj, :].shape)   # PJM's forecast moved one hour later, as in step 3
    info = dict(epochs_run=state['epoch'], best_epoch=state['best_epoch'], best_validation_error=round(100 * state['best'], 3), threads=nthreads, device=str(dev),
                seconds_per_epoch=round(float(np.mean(secs)), 1))
    row = score_row(P, A, F, Fa, TEST_MASK, LABELS[kind], args.seed, info)
    json.dump(dict(kind=kind, seed=args.seed, row=row), open(os.path.join(NEURAL, f'{stem}_result.json'), 'w'), indent=1)
    collect()
    print(f'{LABELS[kind]}: seed {args.seed}, stopped after epoch {state["epoch"]}, best validation epoch {state["best_epoch"]} ({100 * state["best"]:.2f}%)')
    print('  test year hourly percent error: ' + ', '.join(f'{o} {row[o]:.2f}' for o in SCORED) + f', pooled {row["pooled"]:.2f}', flush=True)


t_start = time.time()
print('1. interchange files:', flush=True)
download_interchange()
data = prepare()
print(f'2. data prepared: {len(data[3])} operating days, {len(data[4])} graph nodes, {len(data[5])} links ({time.time() - t_start:.0f} s)', flush=True)
print(f'3. seed {args.seed}', flush=True)
for kind in (['lstm', 'gat'] if args.model == 'both' else [args.model]):
    train(kind, data)
if not args.bench:
    s = pd.read_csv(f'{OUT}/neural_scores.csv')
    print(f'\nTest year {TEST_START.date()} to {PERIOD_END.date()}. Hourly MAPE, percent (lower is better; CISO is not comparable):\n')
    print(s.to_string(index=False))
    print('\nMeaning: these are one training run per model at the seed shown. A neural network started from a different seed lands on a slightly different score, '
          'so rerun with --seed 1 and --seed 2 to see the spread before reading any gap between the two models. Limits: no weather input; one test year; '
          'the settings were chosen on validation in a four-setting search, and the graph model\'s links come from interchange reported by EIA, not from a tuned graph.')
print(f'Step 8 total {time.time() - t_start:.0f} s')

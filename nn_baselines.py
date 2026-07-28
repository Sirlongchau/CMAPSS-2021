"""
nn_baselines.py -- same-input neural baselines for the GFT paper.

Compares MLP / 1D-CNN / LSTM against the GFT and the age prior under the
IDENTICAL protocol, so the only thing that differs is the model:

    * same per-unit residualization + causal EWMA (gft.residualize / smooth_inputs)
    * same by-unit splits (datasets.split, seeds 0..K-1) and same LOSO
    * same rul_cap (gft.suggest_rul_cap, from the fit set only)
    * same capped-RUL target, scored with the same RMSE / NASA (gft.*)

Representation choice (see the paper): the NN gets ALL 13 residualized sensors
(its natural advantage, feature learning), NOT the GFT's evidence-selected subset.
    * MLP   : memoryless -- one cycle's features -> RUL (the temporal-free control,
              the fair analogue of the memoryless FIS).
    * CNN   : a length-W window of cycles -> RUL (local temporal context).
    * LSTM  : the same length-W window as a sequence -> RUL.

Windows are built WITHIN a unit and split BY unit -- never across the split
boundary. Random-splitting windows leaks adjacent cycles of the same engine and
inflates the temporal models; this script does not do that.

Requires PyTorch:  pip install torch
    python nn_baselines.py            # 12-split CV + LOSO for all three models
"""
import numpy as np
import pandas as pd

import gft
import datasets as ds
from ncmapss_features import SENSORS, CONDITIONS

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("This script needs PyTorch:  pip install torch")

K_SPLITS = 12
WINDOW = 10                    # cycles of context for CNN / LSTM
EPOCHS, BATCH, LR = 60, 128, 1e-3
REF_CYCLES, SMOOTH = 20, 7     # match the GFT preprocessing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# preprocessing + windowing (shared protocol)
# ---------------------------------------------------------------------------
def prep(frame):
    """Same residual + smooth the GFT applies, on ALL sensors."""
    f = gft.residualize(frame.reset_index(drop=True), SENSORS, CONDITIONS, REF_CYCLES)
    return gft.smooth_inputs(f, SENSORS, SMOOTH)


def windows(frame, cap, w):
    """Per-unit length-w windows ending at each cycle. Returns X (N,w,D),
    y (N,), units (N,). Left-padded with the first cycle when history is short."""
    Xs, ys, us = [], [], []
    for u, g in frame.sort_values(["unit", "cycle"]).groupby("unit"):
        S = g[SENSORS].to_numpy(float)
        y = np.asarray(gft.cap(g["RUL"], cap), float)
        for t in range(len(g)):
            lo = t - w + 1
            win = S[max(lo, 0):t + 1]
            if lo < 0:
                win = np.vstack([np.repeat(S[:1], -lo, axis=0), win])
            Xs.append(win); ys.append(y[t]); us.append(u)
    return np.asarray(Xs), np.asarray(ys), np.asarray(us)


def standardize(Xtr, Xte):
    """Per-feature z-score using TRAIN stats only (no leakage)."""
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    return (Xtr - mu) / sd, (Xte - mu) / sd


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                      # x: (B, W, D) -> use last cycle only
        return self.net(x[:, -1, :]).squeeze(-1)


class CNN1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.fc = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):                      # x: (B, W, D) -> (B, D, W)
        return self.fc(self.conv(x.transpose(1, 2)).squeeze(-1)).squeeze(-1)


class LSTMReg(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(d, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


MODELS = {"MLP": MLP, "1D-CNN": CNN1D, "LSTM": LSTMReg}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_eval(Model, Xtr, ytr, Xte, seed=0):
    torch.manual_seed(seed)
    m = Model(Xtr.shape[-1]).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    n = len(Xtr_t)
    m.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = lossf(m(Xtr_t[idx]), ytr_t[idx])
            loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        yhat = m(torch.tensor(Xte, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    return yhat, n_params(m)


def score(y, yhat):
    return dict(RMSE=gft.rmse(y, yhat), NASA=gft.nasa_score(y, yhat), R2=gft.r2(y, yhat))


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
feats = ds.pooled()


def run_cv():
    rows = []
    for k in range(K_SPLITS):
        train, val, test = ds.split(feats, fracs=(0.8, 0.1, 0.1), seed=k)
        fitset = pd.concat([train, val], ignore_index=True)
        cap = gft.suggest_rul_cap(fitset)
        ftr, fte = prep(fitset), prep(test)
        base = gft.baselines(fitset, test, cap)
        age = base[base["model"].str.startswith("age only")].iloc[0]
        for name, Model in MODELS.items():
            w = 1 if name == "MLP" else WINDOW
            Xtr, ytr, _ = windows(ftr, cap, w)
            Xte, yte, _ = windows(fte, cap, w)
            Xtr, Xte = standardize(Xtr, Xte)
            yhat, npar = train_eval(Model, Xtr, ytr, Xte, seed=k)
            s = score(yte, yhat)
            s.update(model=name, split=k, n_params=npar,
                     beats_age=s["RMSE"] < age["RMSE"],
                     beats_age_nasa=s["NASA"] < age["NASA"])
            rows.append(s)
            print(f"  split {k:2d} {name:7s}: RMSE={s['RMSE']:.2f} NASA={s['NASA']:.2f} "
                  f"(age RMSE={age['RMSE']:.2f})")
    R = pd.DataFrame(rows)
    print("\n== CV over splits (mean +/- sd) ==")
    print(R.groupby("model")[["RMSE", "NASA", "R2"]].agg(["mean", "std"]).round(3).to_string())
    print("\nfraction of splits beating age (RMSE / NASA):")
    print(R.groupby("model")[["beats_age", "beats_age_nasa"]].mean().round(2).to_string())
    R.to_csv("nn_cv.csv", index=False)
    return R


def run_loso():
    rows = []
    for tr, te, name in ds.leave_one_dataset_out(feats):
        cap = gft.suggest_rul_cap(tr)
        ftr, fte = prep(tr), prep(te)
        age = gft.baselines(tr, te, cap)
        age = age[age["model"].str.startswith("age only")].iloc[0]
        for mname, Model in MODELS.items():
            w = 1 if mname == "MLP" else WINDOW
            Xtr, ytr, _ = windows(ftr, cap, w)
            Xte, yte, _ = windows(fte, cap, w)
            Xtr, Xte = standardize(Xtr, Xte)
            yhat, npar = train_eval(Model, Xtr, ytr, Xte, seed=0)
            s = score(yte, yhat)
            s.update(model=mname, held_out=name, age_RMSE=float(age["RMSE"]))
            rows.append(s)
            print(f"  held out {name:12s} {mname:7s}: RMSE={s['RMSE']:.2f} "
                  f"(age {age['RMSE']:.2f})")
    L = pd.DataFrame(rows)
    print("\n== LOSO: mean RMSE per model, and modes beaten vs age ==")
    L["beats_age"] = L["RMSE"] < L["age_RMSE"]
    print(L.groupby("model").agg(RMSE=("RMSE", "mean"),
                                 beats_age=("beats_age", "sum")).round(3).to_string())
    L.to_csv("nn_loso.csv", index=False)
    return L


if __name__ == "__main__":
    print(f"device={DEVICE}  window={WINDOW}  epochs={EPOCHS}")
    run_cv()
    run_loso()
    print("\nwrote nn_cv.csv, nn_loso.csv")
"""
utils.py - shared code for the beat-to-beat dRR prediction research pipeline.

Target: dRR[n+1] = RR[n+1] - RR[n] in ms, from the last WINDOW=30 RR intervals.

Contents
  1. Engine parameters and feature names
  2. Streaming feature engine (numba). ONE code path for training, evaluation and
     closed-loop imputation, written beat-by-beat so it is the reference for the C port.
  3. Data loading (RR .txt series + your JSON manifests, subject-disjoint splits)
  4. Z-score normalisation and model inputs (residual-on-RLS target supported)
  5. Model builders (micro-TCN, TCN-lite, GRU, MLP) and complexity (params / MACs)
  6. Predictors (baselines, trained runs, int8 TFLite, legacy TCN_MHA via utils2)
  7. Metrics, subject-bootstrap CIs, open-loop and closed-loop evaluation

Run `python utils.py` for the self-test (feature definitions, RLS, metric sanity,
TCN window/streaming equivalence).
"""
import json
import math
import warnings
from pathlib import Path

import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view

# =============================================================================
# 1. ENGINE PARAMETERS AND FEATURE NAMES
# =============================================================================
ENGINE = dict(
    window=30,            # RR history length (same as your current pipeline)
    ar_order=8,           # RLS AR order (Boardman 2002 / Barbieri 2005 use 8-16)
    n_drr_lags=8,         # most recent dRR values exposed as features
    rls_lambda=0.99,      # forgetting factor -> ~100-beat memory
    rls_p0=100.0,         # initial covariance (regressors are in units of RLS_SCALE ms)
    rls_p_max_trace=1e4,  # wind-up guard when the signal is not exciting
    rls_scale=100.0,      # ms -> RLS internal units (keeps float32 well conditioned on MCU)
    rls_pred_clamp=300.0, # |RLS forecast| clamp in ms (safety bound for the residual base)
    bp_f0=0.30,           # RSA band-pass centre in cycles/beat (~3.3 beats per breath)
    bp_q=1.0,             # band-pass Q (covers ~0.15-0.45 cycles/beat)
)
# ponytail: fixed RSA band in cycles/beat. Ceiling: patients whose breaths/beat ratio
# drifts far from ~0.3 get a weak phase cue (and a model may pin RSA at 0.30 in closed loop).
# Upgrade, done 2026-10-07: RSA_TRACKER / RSA_FEATS track the patient's RSA (bp kept for old runs).

# Patient-adaptive RSA tracker (appended 2026-10-07, user-approved engine feature change).
# Deliberately NOT in ENGINE: RunPredictor compares cfg["engine"] == ENGINE, so runs trained
# before the tracker stay loadable. Runs that record "rsa_tracker" are checked against it.
RSA_TRACKER = dict(
    f_lo=0.12,     # lowest band centre (cycles/beat)
    f_hi=0.44,     # highest band centre (infant RSA reaches ~0.5 = Nyquist, aliasing above)
    n_bands=8,     # resonator bank size (Mirmohamadsadeghi & Vesin: notch/resonator banks)
    q=3.0,         # band Q (bandwidth f/3, neighbouring bands overlap)
    beta=0.99,     # band-power EMA (~100-beat memory, as the RLS forgetting factor)
    gate_min_imputed=20,  # closed loop: rsa_q = rsa_drr = 0 once >= 20 of the last WINDOW beats were filled
)
# Phase gate (appended 2026-10-08, user-approved RSA engine change). At 80% dropped no estimator recovers the
# RSA phase from the kept beats (Decisions 2026-10-08), so the tracker must stop asserting one. On the val
# closed loop (tcn_lite_ft fills, seed 7, 16273 + nsr022RRcl) corr(rsa_drr, true target) is 0.17-0.37 below
# 50% filled, 0.10-0.23 at 18-19/30, 0.04-0.07 from 20/30 on -> threshold 20. Open loop and training never
# fill, so their features are unchanged. Runs recorded before the gate lack the key: RunPredictor compares
# only the keys a run recorded and warns if it uses rsa_* (their closed-loop rsa_q/rsa_drr are gated now).
# ponytail: a fixed bank + parabolic interpolation instead of an RLS-adapted notch centre.
# On the train split the single adaptive notch (Nehorai/RLS on the centre) tracked only 45%
# of the subjects with an HF peak within 0.03 c/b and its presence measure did not separate
# peaked from flat subjects; the bank reached 78-82% and AUC ~0.8. Ceiling: f resolution
# ~0.01 c/b and nothing outside [f_lo, f_hi]. Upgrade: refine with a notch seeded at rsa_f.

RLS_WARMUP = 30        # first windows of every series are dropped (cold RLS state)
RR_RANGE_MS = (150.0, 3000.0)  # data validation: values outside are reported
DEADBAND_MS = 5.0      # sign accuracy ignores |dRR| <= deadband (sign is noise there)
HF_BAND = (0.15, 0.40)  # RSA band in cycles/beat (same as the legacy Burg features)
EPS = 1e-8

# project layout (utils.py lives in src/): every default path is anchored here, so the code runs from any cwd
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"        # series/, subjects_*.json, ar_model_parameters.json, legacy hrv_*.h5
RUNS_DIR = ROOT / "runs"        # trained models
RESULTS_DIR = ROOT / "results"  # evaluation/, test/, validation/
LEGACY_DIR = ROOT / "legacy"    # utils2.py and the TCN_MHA model
SERIES_DIR = "series"           # relative to DATA_DIR
MANIFESTS = {  # relative to DATA_DIR  # produced by your preprocessing_v2.py -> identical subjects to the current model
    "train": "subjects_train.json",
    "val": "subjects_validation.json",
    "test": "subjects_test_split.json",
}

WINDOW = ENGINE["window"]
WINDOW_FEATS = ["rr_mean", "rr_dev", "rmssd", "rs", "ccm", "guzik", "porta", "nn20", "run_len"]
RECURSIVE_FEATS = ["rls_pred", "rls_err", "bp", "bp_prev"]
DRR_LAGS = [f"drr_lag{k}" for k in range(ENGINE["n_drr_lags"])]  # lag0 = most recent dRR
RLS_W = [f"rls_w{k}" for k in range(ENGINE["ar_order"])]
# rsa_f: tracked RSA frequency (c/b); rsa_amp: amplitude (ms RR, from the band power, frozen on
# filled beats); rsa_cos/rsa_sin: current phase of the RSA band output; rsa_q: peak prominence
# (log2 above the in-band trend, same measure as spectral_fidelity); rsa_drr: the RSA part of the
# next dRR, amp * (cos(phase + 2 pi f) - cos(phase)).
RSA_FEATS = ["rsa_f", "rsa_amp", "rsa_cos", "rsa_sin", "rsa_q", "rsa_drr"]
FEATURE_NAMES = WINDOW_FEATS + RECURSIVE_FEATS + DRR_LAGS + RLS_W + RSA_FEATS  # append only
FI = {n: i for i, n in enumerate(FEATURE_NAMES)}

FEATURE_SETS = {
    # tabular input of sequence models (the RR sequence itself goes through the TCN/GRU)
    "seq": ["rmssd", "rs", "ccm", "guzik", "rls_pred", "rls_err", "bp", "bp_prev", "run_len"],
    # tabular-only models get the recent dynamics explicitly
    "tab": ["rr_dev", "rr_mean"] + DRR_LAGS
           + ["rmssd", "rs", "ccm", "guzik", "rls_pred", "rls_err", "bp", "bp_prev", "run_len"],
    # linear model on these == global (non-adaptive) centred AR(9)
    "ar": ["rr_dev"] + DRR_LAGS,
    # cheap subset of your 12 legacy features (Burg AR and CCM_N5 removed)
    "legacy_cheap": ["rmssd", "rs", "ccm", "guzik", "porta", "nn20"],
    # round 2: 'tab' without the fixed 0.30 c/b band-pass (displacement prior), and with the
    # patient-adaptive RSA tracker in its place
    "tab_nobp": ["rr_dev", "rr_mean"] + DRR_LAGS + ["rmssd", "rs", "ccm", "guzik", "rls_pred", "rls_err", "run_len"],
    "tab_rsa": ["rr_dev", "rr_mean"] + DRR_LAGS + ["rmssd", "rs", "ccm", "guzik", "rls_pred", "rls_err", "run_len"]
               + RSA_FEATS,
    # round 4: the 9 seq features + the gated RSA tracker. rsa_cos/rsa_sin are left out on purpose: raw
    # phase cannot be gated without an out-of-distribution or fake phase; rsa_drr carries amp x phase
    # and is zeroed with rsa_q when >= RSA_TRACKER['gate_min_imputed'] window beats were filled
    "seq_rsa": ["rmssd", "rs", "ccm", "guzik", "rls_pred", "rls_err", "bp", "bp_prev", "run_len",
                "rsa_f", "rsa_amp", "rsa_q", "rsa_drr"],
    "all": FEATURE_NAMES,
}

MODELS = {  # kind: 'seq' -> inputs [sequence(30,2), tabular]; 'tab' -> tabular only
    "micro_tcn": dict(kind="seq", feats="seq", residual=True, framework="keras"),
    "micro_tcn_f8": dict(kind="seq", feats="seq", residual=True, framework="keras"),  # filters 16 -> 8
    "tcn_lite": dict(kind="seq", feats="seq", residual=True, framework="keras"),
    "gru": dict(kind="seq", feats="seq", residual=True, framework="keras"),
    "mlp": dict(kind="tab", feats="tab", residual=True, framework="keras"),
    "gbdt": dict(kind="tab", feats="tab", residual=True, framework="sklearn"),
    "linear": dict(kind="tab", feats="ar", residual=False, framework="numpy"),
}


def resolve_features(spec):
    """'seq' | 'tab' | ... | list of names -> validated list of names."""
    names = FEATURE_SETS[spec] if isinstance(spec, str) else list(spec)
    unknown = [n for n in names if n not in FI]
    if unknown:
        raise KeyError(f"Unknown features {unknown}. Valid: {FEATURE_NAMES}")
    return names


def _rbj_bandpass(f0, q):
    """RBJ band-pass (0 dB peak): b = [a, 0, -a], a = [1+a, -2cos, 1-a] -> (b0, a1, a2) normalised."""
    w0 = 2.0 * np.pi * f0
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return alpha / a0, -2.0 * np.cos(w0) / a0, (1.0 - alpha) / a0


def engine_params(engine=ENGINE, tracker=RSA_TRACKER):
    """Packs the engine settings into the float array the numba kernel reads.
    [0:10] ENGINE (layout unchanged), [10] n_bands, [11] beta, then per band (fc, b0, a1, a2),
    then the phase gate (min imputed beats in the window; > WINDOW = never)."""
    b0, a1, a2 = _rbj_bandpass(engine["bp_f0"], engine["bp_q"])
    K = int(tracker["n_bands"])
    bank = []
    for fc in np.linspace(tracker["f_lo"], tracker["f_hi"], K):
        bank += [fc, *_rbj_bandpass(fc, tracker["q"])]
    return np.array([
        engine["rls_lambda"], engine["rls_p0"], engine["rls_p_max_trace"],
        engine["rls_scale"], engine["rls_pred_clamp"], engine["n_drr_lags"],
        b0, -b0, a1, a2, K, tracker["beta"], *bank, tracker.get("gate_min_imputed", engine["window"] + 1),
    ], dtype=np.float64)


PRM = engine_params()
N_FEATS = len(FEATURE_NAMES)

# =============================================================================
# 2. STREAMING FEATURE ENGINE
# =============================================================================
# State layout (sc vector): number of beats seen, last unclamped RLS forecast, biquad state,
# then per RSA band k: output y[n-1], y[n-2], power EMA at S_BANK + 3k + (0, 1, 2), then the
# imputed-beat window: bitmask of the last W imputed flags (an exact integer in a float64) and its count
S_N, S_PRED, S_X1, S_X2, S_Y1, S_Y2 = 0, 1, 2, 3, 4, 5
S_BANK = 6
P_BANK = 12  # prm index of the first band (fc, b0, a1, a2)


@njit(cache=True)
def _init_state(W, p, nf, prm):
    return (np.zeros(W), np.zeros(S_BANK + 3 * int(prm[10]) + 2), np.zeros(p), np.eye(p) * prm[1],
            np.zeros(p), np.zeros(nf))


@njit(cache=True)
def _push(rr, buf, sc, w, P, phi, out, prm, frz_rls, frz_rsa):
    """Consumes one RR (ms). Returns True when `out` holds the features that
    predict the NEXT dRR. Cost per beat: O(W) window stats + O(p^2) RLS + O(K) RSA bank.
    Beats the pacemaker filled itself (closed loop) may freeze adaptive states:
    frz_rls skips the RLS w/P update, frz_rsa the RSA band-power EMAs. frz_rsa is also the
    imputed flag of the phase gate (rsa_q = rsa_drr = 0 when >= gate beats of the window were
    imputed). Every other feature is still computed. Training and the open loop pass False, False.
    Window features follow the definitions of utils2.py exactly (see self-test)."""
    W = buf.shape[0]
    p = w.shape[0]
    lam, pmax, scale, clamp = prm[0], prm[2], prm[3], prm[4]
    n_lags = int(prm[5])
    b0, b2, a1, a2 = prm[6], prm[7], prm[8], prm[9]
    n = int(sc[S_N])

    # 0) imputed flags of the last W beats (phase gate): shift register + count, O(1)
    K = int(prm[10])
    s_im = S_BANK + 3 * K
    m_im = int(sc[s_im])
    old = (m_im >> (W - 1)) & 1
    new = 1 if frz_rsa else 0
    sc[s_im] = float(((m_im << 1) | new) & ((1 << W) - 1))
    sc[s_im + 1] += new - old

    # 1) RLS update now that the true dRR of this beat is known (skipped when frz_rls)
    e = 0.0
    err = 0.0
    if n >= W:
        e = (rr - buf[W - 1] - sc[S_PRED]) / scale
        err = e * scale
    if n >= W and not frz_rls:
        Pphi = np.zeros(p)
        for i in range(p):
            s = 0.0
            for j in range(p):
                s += P[i, j] * phi[j]
            Pphi[i] = s
        den = lam
        for i in range(p):
            den += phi[i] * Pphi[i]
        tr = 0.0
        if den >= lam:  # phi'P phi < 0 means P lost positive definiteness: skip and reset below
            for i in range(p):
                k = Pphi[i] / den
                w[i] += k * e
                for j in range(p):
                    P[i, j] = (P[i, j] - k * Pphi[j]) / lam
            for i in range(p):  # keep P symmetric (rounding drift is what breaks it)
                for j in range(i + 1, p):
                    s = 0.5 * (P[i, j] + P[j, i])
                    P[i, j] = s
                    P[j, i] = s
                tr += P[i, i]
        if not (tr > 0.0) or not np.isfinite(tr):
            for i in range(p):
                for j in range(p):
                    P[i, j] = 0.0
                P[i, i] = prm[1]
        elif tr > pmax:
            f = pmax / tr
            for i in range(p):
                for j in range(p):
                    P[i, j] *= f

    # 2) RSA-band biquad on raw RR (zero DC gain, so no detrending is needed)
    if n == 0:
        sc[S_X1] = rr
        sc[S_X2] = rr
    bp_prev = sc[S_Y1]
    y = b0 * rr + b2 * sc[S_X2] - a1 * sc[S_Y1] - a2 * sc[S_Y2]

    # 2b) RSA tracker: bank of K RBJ band-passes on raw RR, band-power EMAs frozen on imputed
    # beats (they describe the patient, not the fills). ~10 FLOPs per band.
    beta = prm[11]
    for k in range(K):
        q0 = P_BANK + 4 * k
        s0 = S_BANK + 3 * k
        yk = prm[q0 + 1] * (rr - sc[S_X2]) - prm[q0 + 2] * sc[s0] - prm[q0 + 3] * sc[s0 + 1]
        sc[s0 + 1] = sc[s0]
        sc[s0] = yk
        if not frz_rsa:
            sc[s0 + 2] = beta * sc[s0 + 2] + (1.0 - beta) * yk * yk
    sc[S_X2] = sc[S_X1]
    sc[S_X1] = rr
    sc[S_Y2] = sc[S_Y1]
    sc[S_Y1] = y

    # 3) slide the window (a C port would use a circular index instead)
    for i in range(W - 1):
        buf[i] = buf[i + 1]
    buf[W - 1] = rr
    sc[S_N] = n + 1
    if n + 1 < W:
        return False

    # 4) window features
    mean = 0.0
    for i in range(W):
        mean += buf[i]
    mean /= W
    var = 0.0
    for i in range(W):
        var += (buf[i] - mean) ** 2
    std = math.sqrt(var / W)

    sq = 0.0
    nn20 = 0.0
    n_up = 0.0
    n_down = 0.0
    d_up = 0.0
    d_abs = 0.0
    for i in range(W - 1):
        d = buf[i + 1] - buf[i]
        sq += d * d
        ad = abs(d)
        d_abs += ad
        if ad > 20.0:
            nn20 += 1.0
        if d > 0:
            n_up += 1.0
            d_up += d
        elif d < 0:
            n_down += 1.0
    rmssd = math.sqrt(sq / (W - 1))
    porta = n_down / (n_up + n_down) if (n_up + n_down) > 0 else 0.0
    guzik = d_up / d_abs if d_abs > 0 else 0.0

    cum = 0.0
    cmax = -1e300
    cmin = 1e300
    for i in range(W):
        cum += buf[i] - mean
        if cum > cmax:
            cmax = cum
        if cum < cmin:
            cmin = cum
    rs = (cmax - cmin) / std if std > 0 else 0.0

    # CCM (2-D): Poincare points P_i = (RR_i, RR_i+1); triangles of 3 consecutive points
    m1 = 0.0
    m2 = 0.0
    for i in range(W - 1):
        m1 += (buf[i] - buf[i + 1]) / math.sqrt(2.0)
        m2 += (buf[i] + buf[i + 1]) / math.sqrt(2.0)
    m1 /= W - 1
    m2 /= W - 1
    v1 = 0.0
    v2 = 0.0
    for i in range(W - 1):
        v1 += ((buf[i] - buf[i + 1]) / math.sqrt(2.0) - m1) ** 2
        v2 += ((buf[i] + buf[i + 1]) / math.sqrt(2.0) - m2) ** 2
    sd1 = math.sqrt(v1 / (W - 1))
    sd2 = math.sqrt(v2 / (W - 1))
    area = 0.0
    for i in range(W - 3):
        ux = buf[i + 1] - buf[i]
        uy = buf[i + 2] - buf[i + 1]
        vx = buf[i + 2] - buf[i]
        vy = buf[i + 3] - buf[i + 1]
        area += 0.5 * abs(ux * vy - uy * vx)
    denom = math.pi * sd1 * sd2 * (W - 3)
    ccm = area / denom if denom > 0 else 0.0
    ccm = min(max(ccm, 0.0), 1.0)

    # signed run length of same-sign dRR ending at the newest beat (+ = decelerating)
    run = 0.0
    d_last = buf[W - 1] - buf[W - 2]
    if d_last != 0:
        sgn = 1.0 if d_last > 0 else -1.0
        for i in range(W - 2, -1, -1):
            if (buf[i + 1] - buf[i]) * sgn > 0:
                run += 1.0
            else:
                break
        run *= sgn

    # 5) RLS forecast of the next dRR from centred RR lags
    for k in range(p):
        phi[k] = (buf[W - 1 - k] - mean) / scale
    pred = 0.0
    for k in range(p):
        pred += w[k] * phi[k]
    pred *= scale
    sc[S_PRED] = pred  # unclamped value is what RLS must be corrected against
    pred_c = min(max(pred, -clamp), clamp)

    # 6) write features in FEATURE_NAMES order
    out[0] = mean
    out[1] = buf[W - 1] - mean
    out[2] = rmssd
    out[3] = rs
    out[4] = ccm
    out[5] = guzik
    out[6] = porta
    out[7] = nn20
    out[8] = run
    out[9] = pred_c
    out[10] = err
    out[11] = y
    out[12] = bp_prev
    for k in range(n_lags):
        out[13 + k] = buf[W - 1 - k] - buf[W - 2 - k]
    for k in range(p):
        out[13 + n_lags + k] = w[k]

    # 7) RSA features: peak of the band log-powers above their linear trend (interior bands, as
    # _hf_peak), parabolic interpolation of f, phase of the peak band output by 2-sample quadrature.
    # Per beat: K log2 + ~12K FLOPs + 2 sqrt + cos + sin (K=8: ~100 FLOPs).
    lp = np.empty(K)
    fm = 0.0
    lm = 0.0
    for k in range(K):
        lp[k] = math.log2(sc[S_BANK + 3 * k + 2] + 1e-8)
        fm += prm[P_BANK + 4 * k]
        lm += lp[k]
    fm /= K
    lm /= K
    sxy = 0.0
    sxx = 0.0
    for k in range(K):
        df = prm[P_BANK + 4 * k] - fm
        sxy += df * (lp[k] - lm)
        sxx += df * df
    slope = sxy / sxx
    best = 1
    prom = -1e300
    for k in range(1, K - 1):
        r = lp[k] - lm - slope * (prm[P_BANK + 4 * k] - fm)
        if r > prom:
            prom = r
            best = k
    l_m = lp[best - 1]
    l_0 = lp[best]
    l_p = lp[best + 1]
    curv = l_m - 2.0 * l_0 + l_p
    off = 0.5 * (l_m - l_p) / curv if curv < 0.0 else 0.0
    off = min(max(off, -0.5), 0.5)
    df_band = prm[P_BANK + 4] - prm[P_BANK]
    f_rsa = prm[P_BANK + 4 * best] + off * df_band
    amp = math.sqrt(2.0 * sc[S_BANK + 3 * best + 2])  # sinusoid amplitude from its mean power
    wr = 2.0 * math.pi * f_rsa
    cw = math.cos(wr)
    sw = math.sin(wr)
    yi = sc[S_BANK + 3 * best]                    # y[n]   = A cos(phi)
    yq = (sc[S_BANK + 3 * best + 1] - yi * cw) / sw  # from y[n-1] = A cos(phi - w): A sin(phi)
    mag = math.sqrt(yi * yi + yq * yq)
    c_ph = yi / mag if mag > 1e-8 else 1.0
    s_ph = yq / mag if mag > 1e-8 else 0.0
    o = 13 + n_lags + p
    out[o] = f_rsa
    out[o + 1] = amp
    out[o + 2] = c_ph
    out[o + 3] = s_ph
    out[o + 4] = prom
    out[o + 5] = amp * (c_ph * cw - s_ph * sw - c_ph)
    if sc[s_im + 1] >= prm[P_BANK + 4 * K]:  # phase gate: too many fills to know the RSA phase
        out[o + 4] = 0.0
        out[o + 5] = 0.0
    return True


@njit(cache=True)
def _series_features(rr, W, p, nf, prm):
    """Row j = features after beat j+W-1, i.e. for window rr[j:j+W] -> target rr[j+W]-rr[j+W-1]."""
    nb = rr.shape[0]
    F = np.zeros((nb - W, nf))
    buf, sc, w, P, phi, out = _init_state(W, p, nf, prm)
    for t in range(nb - 1):  # the last beat has no target
        if _push(rr[t], buf, sc, w, P, phi, out, prm, False, False):
            F[t - W + 1, :] = out
    return F


def series_features(rr):
    rr = np.asarray(rr, dtype=np.float64)
    F = _series_features(rr, WINDOW, ENGINE["ar_order"], N_FEATS, PRM)
    y = rr[WINDOW:] - rr[WINDOW - 1:-1]
    return F, y


@njit(cache=True)
def _push_lanes(t, vals, imputed, lens, bufs, scs, ws, Ps, phis, outs, prm, frz_rls_imp, F):
    """Beat t of L independent engines (one per closed-loop lane), the same _push as FeatureEngine.
    Row t-W+1 of F[i] gets the features after beat t (the _series_features layout)."""
    W = bufs.shape[1]
    for i in range(vals.shape[0]):
        if t >= lens[i]:
            continue
        imp = imputed[i]
        ok = _push(vals[i], bufs[i], scs[i], ws[i], Ps[i], phis[i], outs[i], prm, imp and frz_rls_imp, imp)
        j = t - W + 1
        if ok and j < lens[i] - W:
            for k in range(outs.shape[1]):
                F[i, j, k] = outs[i, k]


# Tested 2026-10-07 on val (seed 7, 16 subjects, 5000 beats): freezing the RLS on filled beats made
# the closed loop WORSE at 80% dropped (rls_ar 54.6 -> 69.1 ms, mlp 42.3 -> 44.3, micro_tcn 41.5 -> 43.2;
# invented RSA 11 -> 14, 3 -> 7, 1 -> 6 of 16). A fill is the model's own forecast, so for rls_ar its
# error is 0: the update leaves w unchanged and only shrinks P, a free regulariser. Frozen, P stays
# large and the sparse true beats (whose error spans several filled steps) make w jump.
FREEZE_RLS_ON_IMPUTED = False


class FeatureEngine:
    """Beat-by-beat extractor (same kernel as training). Used by closed-loop evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.buf, self.sc, self.w, self.P, self.phi, self.out = _init_state(
            WINDOW, ENGINE["ar_order"], N_FEATS, PRM)
        self.ready = False

    def push(self, rr, imputed=False, freeze_rls=None):
        """imputed=True: `rr` is the pacemaker's own fill. The RSA band powers then keep describing
        the patient (frozen); the RLS keeps adapting unless freeze_rls (default FREEZE_RLS_ON_IMPUTED)."""
        imputed = bool(imputed)
        frz_rls = imputed and (FREEZE_RLS_ON_IMPUTED if freeze_rls is None else bool(freeze_rls))
        self.ready = _push(float(rr), self.buf, self.sc, self.w, self.P, self.phi, self.out, PRM,
                           frz_rls, imputed)
        return self.ready


# =============================================================================
# 3. DATA LOADING
# =============================================================================
def canonical_code(x):
    s = str(x).strip()
    if s.isdigit():
        return f"{int(s):03d}" if len(s) < 3 else s
    return s


def read_manifest(path):
    """{interval: {"subjects": [...]}} -> [(interval, code), ...]"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for interval, payload in data.items():
        subs = payload["subjects"] if isinstance(payload, dict) else payload
        out += [(str(interval), canonical_code(s)) for s in subs]
    return out


def load_series(path):
    rr = np.loadtxt(path, dtype=np.float64).ravel()
    if not np.all(np.isfinite(rr)):
        raise ValueError(f"{path}: non-finite RR values")
    lo, hi = RR_RANGE_MS
    n_bad = int(np.sum((rr < lo) | (rr > hi)))
    if n_bad:
        warnings.warn(f"{path}: {n_bad} RR values outside {RR_RANGE_MS} ms (kept as-is)")
    return rr


def load_split(split, data_dir=DATA_DIR, series_dir=SERIES_DIR, manifests=MANIFESTS, max_beats=None):
    """Loads one split: list of dicts {code, interval, rr, F, y, offset}.
    Rows of F / y map to windows rr[offset + j : offset + j + WINDOW]."""
    data_dir = Path(data_dir)
    subjects = []
    for interval, code in read_manifest(data_dir / manifests[split]):
        path = data_dir / series_dir / f"{code}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Missing RR file: {path}")
        rr = load_series(path)
        if max_beats:
            rr = rr[:max_beats]
        if len(rr) < WINDOW + RLS_WARMUP + 2:
            warnings.warn(f"{path}: only {len(rr)} beats, skipped")
            continue
        F, y = series_features(rr)
        subjects.append(dict(code=code, interval=interval, rr=rr,
                             F=F[RLS_WARMUP:].astype(np.float32), y=y[RLS_WARMUP:],
                             offset=RLS_WARMUP))
    if not subjects:
        raise ValueError(f"Split '{split}' is empty")
    return subjects


def check_disjoint(**splits):
    """Leakage guard: a subject may appear in only one split."""
    seen = {}
    for name, subs in splits.items():
        for s in subs:
            if s["code"] in seen and seen[s["code"]] != name:
                raise ValueError(f"Subject {s['code']} is in both '{seen[s['code']]}' and '{name}'")
            seen[s["code"]] = name


def subject_windows(s):
    return sliding_window_view(s["rr"], WINDOW)[s["offset"]: s["offset"] + len(s["y"])]


def stack(subjects):
    """-> rr_win (N, W) float32 raw ms, F (N, n_feats) float32 raw, y (N,) ms, sid (N,) subject index"""
    rr_win = np.concatenate([subject_windows(s) for s in subjects]).astype(np.float32)
    F = np.concatenate([s["F"] for s in subjects])
    y = np.concatenate([s["y"] for s in subjects])
    sid = np.concatenate([np.full(len(s["y"]), i, dtype=np.int32) for i, s in enumerate(subjects)])
    return rr_win, F, y, sid


def predict_subjects(predictor, subjects):
    """Predictions subject by subject (never stacks all windows in RAM), concatenated."""
    return np.concatenate([predictor.predict(subject_windows(s).astype(np.float32), s["F"]) for s in subjects])


def file_md5(path):
    import hashlib
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# =============================================================================
# 4. NORMALISATION AND MODEL INPUTS (global z-score, fitted on train only)
# =============================================================================
def _mean_std(x):
    m, s = float(np.mean(x)), float(np.std(x))
    return m, (s if s > EPS else 1.0)


def fit_norm(train_subjects, features, residual):
    rr_all = np.concatenate([s["rr"] for s in train_subjects])
    drr_all = np.concatenate([np.diff(s["rr"]) for s in train_subjects])
    cols = [FI[f] for f in features]
    F = np.concatenate([s["F"][:, cols] for s in train_subjects])  # only the used columns (RAM at 20M windows)
    y = np.concatenate([s["y"] for s in train_subjects])
    if residual:
        y = y - np.concatenate([s["F"][:, FI["rls_pred"]] for s in train_subjects])
    rr_m, rr_s = _mean_std(rr_all)
    d_m, d_s = _mean_std(drr_all)
    y_m, y_s = _mean_std(y)
    f_m = F.mean(axis=0)
    f_s = F.std(axis=0)
    f_s = np.where(f_s > EPS, f_s, 1.0)
    return dict(rr_mean=rr_m, rr_std=rr_s, drr_mean=d_m, drr_std=d_s, y_mean=y_m, y_std=y_s,
                feat_mean=f_m.tolist(), feat_std=f_s.tolist())


def normalize_feats(cfg, F):
    nm = cfg["norm"]
    feats = (F[:, [FI[f] for f in cfg["features"]]] - np.asarray(nm["feat_mean"])) / np.asarray(nm["feat_std"])
    return feats.astype(np.float32)


def make_seq(cfg, rr_win):
    """(N, W) raw RR ms -> (N, W, 2): z-scored RR and z-scored dRR (first step dRR = 0).
    train.py rebuilds this in TensorFlow for on-the-fly batches and asserts it matches."""
    nm = cfg["norm"]
    rr = np.asarray(rr_win, dtype=np.float32)
    drr = np.diff(rr, axis=1, prepend=rr[:, :1])
    return np.stack([(rr - nm["rr_mean"]) / nm["rr_std"],
                     (drr - nm["drr_mean"]) / nm["drr_std"]], axis=-1).astype(np.float32)


def make_inputs(cfg, rr_win, F):
    feats = normalize_feats(cfg, F)
    return feats if cfg["kind"] == "tab" else [make_seq(cfg, rr_win), feats]


def residual_base(cfg, F):
    return F[:, FI["rls_pred"]].astype(np.float64) if cfg["residual"] else 0.0


def to_z(cfg, y_ms, F):
    nm = cfg["norm"]
    return ((y_ms - residual_base(cfg, F) - nm["y_mean"]) / nm["y_std"]).astype(np.float32)


def from_z(cfg, z, F):
    nm = cfg["norm"]
    return np.asarray(z, dtype=np.float64).ravel() * nm["y_std"] + nm["y_mean"] + residual_base(cfg, F)


# =============================================================================
# 5. MODEL BUILDERS AND COMPLEXITY
# =============================================================================
def build_tcn(n_feats, filters=16, dilations=(1, 2, 4, 7), kernel=3, separable=True,
              head=32, seq_len=WINDOW, n_ch=2, name="micro_tcn"):
    """Attention-free causal TCN, MCU friendly: ReLU + foldable BatchNorm, explicit
    left padding + 'valid' convs, last-timestep readout. Receptive field <= seq_len,
    so the window model equals the streaming (ring-buffer) model exactly."""
    from tensorflow import keras
    L = keras.layers
    rf = 1 + (kernel - 1) * sum(dilations)
    if rf > seq_len:
        raise ValueError(f"Receptive field {rf} > window {seq_len}: streaming would differ")
    seq_in = keras.Input((seq_len, n_ch), name="seq")
    feat_in = keras.Input((n_feats,), name="feats")
    x = L.Conv1D(filters, 1, use_bias=False)(seq_in)  # pointwise stem, RF unchanged
    x = L.ReLU()(L.BatchNormalization()(x))
    for d in dilations:
        y = L.ZeroPadding1D(((kernel - 1) * d, 0))(x)
        conv = L.SeparableConv1D if separable else L.Conv1D
        y = conv(filters, kernel, dilation_rate=d, padding="valid", use_bias=False)(y)
        y = L.ReLU()(L.BatchNormalization()(y))
        x = L.Add()([x, y])
    x = L.Flatten()(L.Cropping1D((seq_len - 1, 0))(x))  # newest timestep only
    h = L.Dense(head, activation="relu")(L.Concatenate()([x, feat_in]))
    out = L.Dense(1, name="drr")(h)
    return keras.Model([seq_in, feat_in], out, name=name)


def build_gru(n_feats, units=16, head=32, seq_len=WINDOW, n_ch=2):
    """Small GRU; unroll=True keeps the TFLite graph loop-free. On the MCU it can run
    stateful (1 step per beat) - validate that mode before relying on it."""
    from tensorflow import keras
    L = keras.layers
    seq_in = keras.Input((seq_len, n_ch), name="seq")
    feat_in = keras.Input((n_feats,), name="feats")
    x = L.GRU(units, unroll=True)(seq_in)
    h = L.Dense(head, activation="relu")(L.Concatenate()([x, feat_in]))
    out = L.Dense(1, name="drr")(h)
    return keras.Model([seq_in, feat_in], out, name="gru")


def build_mlp(n_feats, hidden=(32, 16)):
    from tensorflow import keras
    L = keras.layers
    inp = keras.Input((n_feats,), name="feats")
    x = inp
    for h in hidden:
        x = L.Dense(h, activation="relu")(x)
    out = L.Dense(1, name="drr")(x)
    return keras.Model(inp, out, name="mlp")


def build_keras_model(name, n_feats):
    if name == "micro_tcn":
        return build_tcn(n_feats, filters=16, separable=True, name="micro_tcn")
    if name == "micro_tcn_f8":  # round 2: conv vs mlp at equal cost (944 vs 1136 MACs/beat)
        return build_tcn(n_feats, filters=8, separable=True, name="micro_tcn_f8")
    if name == "tcn_lite":
        return build_tcn(n_feats, filters=24, separable=False, name="tcn_lite")
    if name == "gru":
        return build_gru(n_feats)
    if name == "mlp":
        return build_mlp(n_feats)
    raise KeyError(name)


def keras_complexity(model):
    """Params and MACs per prediction. 'window' recomputes the 30-beat window every beat;
    'stream' computes only the newest timestep per layer (ring buffers / stateful GRU).
    BatchNorm is excluded (folded into the conv at export); activations are not counted."""
    win = stream = 0
    streamable = True
    for layer in model.layers:
        cls = layer.__class__.__name__
        if cls == "Dense":
            m = int(layer.input.shape[-1]) * layer.units
            win += m
            stream += m
        elif cls in ("Conv1D", "SeparableConv1D"):
            cin = int(layer.input.shape[-1])
            T = int(layer.output.shape[1])
            k = layer.kernel_size[0]
            if cls == "Conv1D":
                per_t = k * cin * layer.filters
            else:
                dm = layer.depth_multiplier
                per_t = k * cin * dm + cin * dm * layer.filters
            win += T * per_t
            stream += per_t
        elif cls == "GRU":
            T, cin = int(layer.input.shape[1]), int(layer.input.shape[2])
            per_t = 3 * (cin * layer.units + layer.units * layer.units)
            win += T * per_t
            stream += per_t
        elif cls == "MultiHeadAttention":  # Q/K/V projections + scores + weighted sum + output
            q = layer.input[0] if isinstance(layer.input, (list, tuple)) else layer.input
            T, d = int(q.shape[1]), int(q.shape[2])
            hk = layer.num_heads * layer.key_dim
            win += 3 * T * d * hk + 2 * T * T * hk + T * hk * d
            streamable = False
        elif cls.startswith("Global"):  # pooling over time needs the whole window every beat
            streamable = False
    if not streamable:
        stream = win
    return dict(params=int(model.count_params()), macs_window=int(win), macs_stream=int(stream))


# =============================================================================
# 6. PREDICTORS: predict(rr_win (N,W) raw ms, F (N,n_feats) raw) -> dRR ms (N,)
# =============================================================================
class Persistence:
    """dRR = 0. Its RMSE equals the RMSSD of the evaluated beats (skill = 0 by definition)."""
    name = "persistence"

    def predict(self, rr_win, F):
        return np.zeros(len(F))


class EMAMean:
    """Moody & Mark running mean rmean = 0.75 rmean + 0.25 RR; predicts RR_next = rmean."""
    name = "ema_mean"

    def __init__(self):
        k = np.arange(WINDOW)
        w = 0.25 * 0.75 ** k
        self.w = (w / w.sum())[::-1]  # oldest ... newest (0.75^30 truncation is negligible)

    def predict(self, rr_win, F):
        return np.asarray(rr_win, np.float64) @ self.w - np.asarray(rr_win[:, -1], np.float64)


class RLSAR:
    """Adaptive per-patient AR(8) via RLS - the always-on fallback candidate (~250 FLOPs/beat)."""
    name = "rls_ar"

    def predict(self, rr_win, F):
        return F[:, FI["rls_pred"]].astype(np.float64)


class RunPredictor:
    """Loads a trained run directory (config.json + model artifact)."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        with open(self.run_dir / "config.json", "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        if self.cfg.get("engine") != ENGINE:
            raise ValueError(f"{run_dir}: trained with engine {self.cfg.get('engine')}, "
                             f"current ENGINE differs -> retrain or restore settings")
        if "rsa_tracker" in self.cfg:
            rec = self.cfg["rsa_tracker"]
            if any(RSA_TRACKER.get(k) != v for k, v in rec.items()):
                raise ValueError(f"{run_dir}: trained with rsa_tracker {rec}, "
                                 f"current RSA_TRACKER differs -> retrain or restore settings")
            if set(RSA_TRACKER) - set(rec) and set(self.cfg.get("features", [])) & set(RSA_FEATS):
                warnings.warn(f"{run_dir}: trained before the RSA phase gate; open-loop features are "
                              f"unchanged, but its closed-loop rsa_q/rsa_drr are gated now")
        elif set(self.cfg.get("features", [])) & set(RSA_FEATS):
            warnings.warn(f"{run_dir}: uses RSA tracker features but did not record RSA_TRACKER; "
                          f"assuming the current values {RSA_TRACKER}")
        self.name = self.cfg["tag"]
        fw = self.cfg["framework"]
        if fw == "keras":
            import tensorflow as tf
            from tensorflow import keras
            self.model = keras.models.load_model(self.run_dir / "model.keras", compile=False)

            def infer(*xs):
                return self.model(list(xs) if len(xs) > 1 else xs[0], training=False)

            # compiled per-beat call for the closed loop: 0.7 ms vs 15 ms for eager model() (micro_tcn, CPU)
            self._infer = tf.function(infer, autograph=False, reduce_retracing=True,
                                      input_signature=[tf.TensorSpec(i.shape, tf.float32) for i in self.model.inputs])
            probe = [np.zeros((1,) + tuple(i.shape[1:]), np.float32) for i in self.model.inputs]
            assert np.allclose(np.asarray(self._infer(*probe)), np.asarray(infer(*probe)), atol=1e-5), \
                f"{run_dir}: compiled and eager outputs differ"
        elif fw == "sklearn":
            import joblib
            self.model = joblib.load(self.run_dir / "model.joblib")
        elif fw == "numpy":
            self.coef = np.asarray(self.cfg["linear"]["coef"])
            self.intercept = float(self.cfg["linear"]["intercept"])

    def predict_z(self, X):
        fw = self.cfg["framework"]
        if fw == "keras":
            n = len(X[0]) if isinstance(X, list) else len(X)
            if n <= 64:  # compiled direct call: much faster than .predict() for tiny batches
                return np.asarray(self._infer(*X) if isinstance(X, list) else self._infer(X)).ravel()
            return self.model.predict(X, batch_size=8192, verbose=0).ravel()
        if fw == "sklearn":
            return self.model.predict(X)
        return X @ self.coef + self.intercept

    def predict(self, rr_win, F):
        return from_z(self.cfg, self.predict_z(make_inputs(self.cfg, rr_win, F)), F)

    def predict_batch(self, rr_win, F):
        """predict() through the compiled call at any batch size (no Keras .predict() overhead):
        the lockstep closed loop calls it once per beat with one row per lane."""
        if self.cfg["framework"] != "keras":
            return self.predict(rr_win, F)
        X = make_inputs(self.cfg, rr_win, F)
        z = self._infer(*X) if isinstance(X, list) else self._infer(X)
        return from_z(self.cfg, np.asarray(z).ravel(), F)


class TFLitePredictor:
    """Full-integer int8 model produced by train.py (shows quantisation damage per architecture)."""

    def __init__(self, run_dir):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:  # older stacks: the (deprecated) TF interpreter
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        self.base = RunPredictor(run_dir)
        self.cfg = self.base.cfg
        self.name = self.cfg["tag"] + "_int8"
        self.itp = Interpreter(model_path=str(Path(run_dir) / "model_int8.tflite"))
        self.itp.allocate_tensors()
        self.inp = self.itp.get_input_details()
        self.out = self.itp.get_output_details()[0]

    def _q(self, x, d):
        s, z = d["quantization"]
        return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)

    def predict_z(self, X):
        """Normalised inputs (as make_inputs returns them) -> dequantised z outputs."""
        X = X if isinstance(X, list) else [X]
        by_rank = {x.ndim: x for x in X}  # seq is 3-D, tabular 2-D
        res = np.empty(len(X[0]))
        s_o, z_o = self.out["quantization"]
        for i in range(len(res)):
            for d in self.inp:
                self.itp.set_tensor(d["index"], self._q(by_rank[len(d["shape"])][i:i + 1], d))
            self.itp.invoke()
            res[i] = (float(self.itp.get_tensor(self.out["index"]).ravel()[0]) - z_o) * s_o
        return res

    def predict(self, rr_win, F):
        return from_z(self.cfg, self.predict_z(make_inputs(self.cfg, rr_win, F)), F)


LEGACY_STATIC = ["rmssd", "rs", "ccm", "ccm_n5", "guzik", "nn20", "porta",
                 "ar_1", "ar_2", "ar_3", "ar_4", "ar_5"]


class LegacyPredictor:
    """Your current TCN_MHA model. Features are recomputed from the raw window with the
    original utils2.py functions and normalised with the stats stored in hrv_dataset.h5,
    so it can be compared open-loop, closed-loop and used as a distillation teacher."""
    name = "tcn_mha_current"

    def __init__(self, model_path, train_h5):
        import h5py
        from tensorflow import keras
        import sys
        if str(LEGACY_DIR) not in sys.path:
            sys.path.insert(0, str(LEGACY_DIR))
        import utils2  # your original module (legacy/utils2.py)
        self.u2 = utils2
        import tensorflow as tf
        self.model = keras.models.load_model(model_path, compile=False)

        def infer(xs, xf):
            return self.model([xs, xf], training=False)

        # compiled per-beat call for the closed loop (eager model() costs tens of ms per beat)
        self._infer = tf.function(infer, autograph=False, reduce_retracing=True,
                                  input_signature=[tf.TensorSpec(i.shape, tf.float32) for i in self.model.inputs])
        with h5py.File(train_h5, "r") as h5:
            g = h5["normalization"]
            cols = [c.decode() if isinstance(c, bytes) else str(c) for c in g["columns"][()]]
            idx = [cols.index(c) for c in LEGACY_STATIC]
            self.f_mean = g["mean"][()][idx]
            self.f_std = g["std"][()][idx]
            self.rr_mean, self.rr_std = float(g.attrs["rr_mean"]), float(g.attrs["rr_std"])
            self.y_mean, self.y_std = float(g.attrs["target_mean"]), float(g.attrs["target_std"])

    def predict(self, rr_win, F, chunk=4096):
        out = []
        for a in range(0, len(rr_win), chunk):
            win = np.asarray(rr_win[a:a + chunk], dtype=np.float64)
            diffs = np.diff(win, axis=1)
            feats = {**self.u2.compute_asymmetry_features(diffs),
                     **self.u2.compute_statistical_features(win, diffs),
                     **self.u2.compute_phase_space_features(win, tau=1),
                     **self.u2.compute_burg_features(win, ar_order=5, psd_order=8, f_low=0.15, f_high=0.40)}
            xf = ((np.stack([feats[c] for c in LEGACY_STATIC], axis=1) - self.f_mean) / self.f_std).astype(np.float32)
            xs = ((win - self.rr_mean) / self.rr_std).astype(np.float32)[..., None]
            z = self._infer(xs, xf) if len(win) <= 64 else \
                self.model.predict([xs, xf], batch_size=4096, verbose=0)
            out.append(np.asarray(z, dtype=np.float64).ravel() * self.y_std + self.y_mean)
        return np.concatenate(out)


# =============================================================================
# 7. METRICS AND EVALUATION
# =============================================================================
def point_metrics(y, yhat):
    y = np.asarray(y, np.float64)
    yhat = np.asarray(yhat, np.float64)
    e = yhat - y
    rmse = float(np.sqrt(np.mean(e ** 2)))
    persist = float(np.sqrt(np.mean(y ** 2)))
    m = np.abs(y) > DEADBAND_MS
    return dict(
        n=int(len(y)), rmse=rmse, mae=float(np.mean(np.abs(e))),
        p95=float(np.percentile(np.abs(e), 95)),
        skill=1.0 - rmse / persist if persist > 0 else np.nan,
        r=float(np.corrcoef(y, yhat)[0, 1]) if np.std(yhat) > 0 and np.std(y) > 0 else np.nan,
        sign_acc=float(np.mean(np.sign(yhat[m]) == np.sign(y[m]))) if m.any() else np.nan,
        amp_ratio=float(np.std(yhat) / np.std(y)) if np.std(y) > 0 else np.nan,
    )


def bootstrap_ci(sse, sy2, n, reps=1000, seed=0):
    """Subject-level bootstrap of pooled RMSE and skill (subjects, not beats, are independent)."""
    sse, sy2, n = map(np.asarray, (sse, sy2, n))
    idx = np.random.default_rng(seed).integers(0, len(n), size=(reps, len(n)))
    rmse = np.sqrt(sse[idx].sum(1) / n[idx].sum(1))
    skill = 1.0 - np.sqrt(sse[idx].sum(1) / sy2[idx].sum(1))
    lo, hi = 2.5, 97.5
    return dict(rmse_lo=float(np.percentile(rmse, lo)), rmse_hi=float(np.percentile(rmse, hi)),
                skill_lo=float(np.percentile(skill, lo)), skill_hi=float(np.percentile(skill, hi)))


def evaluate_open_loop(predictors, subjects, ref="rls_ar"):
    """One-step-ahead on the true history. Returns (summary, per_subject, per_interval) DataFrames."""
    import pandas as pd
    from scipy.stats import wilcoxon
    rr_win, F, y, sid = stack(subjects)
    summary, per_subj, per_int = [], [], []
    for p in predictors:
        yhat = p.predict(rr_win, F)
        row = dict(model=p.name, **point_metrics(y, yhat))
        sse = np.bincount(sid, (yhat - y) ** 2, minlength=len(subjects))
        sy2 = np.bincount(sid, y ** 2, minlength=len(subjects))
        cnt = np.bincount(sid, minlength=len(subjects))
        row.update(bootstrap_ci(sse, sy2, cnt))
        summary.append(row)
        for i, s in enumerate(subjects):
            m = sid == i
            per_subj.append(dict(model=p.name, subject=s["code"], interval=s["interval"],
                                 **point_metrics(y[m], yhat[m])))
        intervals = np.array([subjects[i]["interval"] for i in sid])
        for iv in sorted(set(intervals)):
            m = intervals == iv
            per_int.append(dict(model=p.name, interval=iv, **point_metrics(y[m], yhat[m])))
    summary, per_subj, per_int = map(pd.DataFrame, (summary, per_subj, per_int))
    # paired test across subjects vs the reference model
    if ref in set(summary["model"]):
        r = per_subj[per_subj.model == ref].set_index("subject")["rmse"]
        pvals = []
        for name in summary["model"]:
            mrm = per_subj[per_subj.model == name].set_index("subject")["rmse"].loc[r.index]
            try:
                pvals.append(float(wilcoxon(mrm, r).pvalue) if name != ref else np.nan)
            except ValueError:
                pvals.append(np.nan)
        summary[f"p_vs_{ref}"] = pvals
    return summary.sort_values("rmse").reset_index(drop=True), per_subj, per_int


def rr_reference(age_years):
    """Age-matched RR mean / std (ms): the start-up anchor of correlation_study_v3 (utils2.get_rr_reference)."""
    mean = 505.0 * age_years ** 0.122
    std = 80.0 * age_years ** 0.26 if age_years <= 12 else 290.0 * age_years ** -0.2
    return float(mean), float(std)


def interval_age_years(interval):
    """Manifest age interval in weeks ('3220-3481', the notebook's range_key) -> mid-range age in years."""
    lo, hi = map(float, str(interval).split("-"))
    return 0.5 * (lo + hi) / 52.14


AR_PARAMS_FILE = DATA_DIR / "ar_model_parameters.json"


class ColdStartAR:
    """Pacemaker start-up fill for beats dropped before the model's 30-beat window exists:
    utils2.ExactARPacingPredictor (Burg AR) in numpy, deterministic (stochastic=False, so the
    evaluation is reproducible; the notebook adds innovation noise)."""

    def __init__(self, phi):
        self.phi = np.asarray(phi, dtype=np.float64)
        p = len(self.phi)
        # exact autocorrelations rho(1..p) of the AR(p) process: rho_k = sum_j phi_j rho_|k-j|, rho_0 = 1
        A = np.eye(p)
        for k in range(1, p + 1):
            for j in range(1, p + 1):
                if k != j:
                    A[k - 1, abs(k - j) - 1] -= self.phi[j - 1]
        self.rho = np.r_[1.0, np.linalg.solve(A, self.phi)]

    @classmethod
    def for_interval(cls, interval, path=AR_PARAMS_FILE):
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        # ponytail: the file holds only the 3220-3481 fit, which the notebook uses for every subject.
        # Ceiling: infants/children get adult AR dynamics in the first 30 fills. Upgrade: per-range fits.
        entry = params.get(str(interval), next(iter(params.values())))
        return cls(entry["phi_burg"])

    def weights(self, n):
        """MMSE weights for a history of n beats (exact Yule-Walker solve when n < p)."""
        k = min(n, len(self.phi))
        if k == len(self.phi):
            return self.phi
        R = self.rho[np.abs(np.subtract.outer(np.arange(k), np.arange(k)))]
        return np.linalg.solve(R, self.rho[1:k + 1])

    def forecast(self, history, ref_mean, ref_std):
        n = len(history)
        if n == 0:
            return float(ref_mean)
        rr = np.asarray(history, dtype=np.float64)
        w_local = min(n / 10.0, 1.0)  # local mean anchor, blended toward the age reference below 10 beats
        mean = w_local * rr.mean() + (1.0 - w_local) * ref_mean
        w = self.weights(n)
        z_recent = (rr[::-1][:len(w)] - mean) / (ref_std + EPS)  # newest first
        return float(w @ z_recent * ref_std + mean)


RSA_PEAK_LOG2 = 1.0   # an RSA peak = >= 2x above the in-band trend (shuffled-beat null on val: max 0.77)
RSA_SHIFT = 0.03      # cycles/beat: a peak further than this from the original's is displaced
RSA_GROW_LOG2 = 0.5   # invented = crosses RSA_PEAK_LOG2 AND grows >= 1.4x: a bump just under the
                      # threshold flickering over it grew 0.2-0.4 on val; genuine cases (rls_ar at 80%) 0.75-5.7


def _hf_peak(x, band, nperseg, edge=2):
    """(log2 height of the HF peak above the in-band log-PSD trend, its frequency, log2 band power).
    The trend removes LF spill-over / 1/f slope; band-edge bins are never peaks."""
    from scipy.signal import welch
    f, p = welch(np.asarray(x, np.float64), nperseg=nperseg, detrend="linear")
    m = (f >= band[0]) & (f <= band[1])
    fb, lp = f[m], np.log2(p[m] + EPS)
    r = lp - np.polyval(np.polyfit(fb, lp, 1), fb)
    i = edge + int(np.argmax(r[edge:-edge]))
    return float(r[i]), float(fb[i]), float(np.log2(p[m].sum() + EPS))


def spectral_fidelity(orig, filled, band=HF_BAND, nperseg=256):
    """RSA fidelity of a filled series vs the original (the TCN_MHA failure: it put an RSA peak into
    series without one and shifted it in others). Welch PSD per beat.
    hf_power_log2: log2(HF power filled / original), 0 = same, -1 = halved (damped).
    hf_peak_log2:  detrended peak height filled minus original (log2); > 1 = 2x sharper.
    hf_peak_shift: peak frequency filled minus original (cycles/beat), meaningful when both have one.
    rsa_invented / rsa_displaced / rsa_lost: 1.0 when the filled series has an RSA peak the original
    lacks (and it grew by >= RSA_GROW_LOG2) / both have one but > RSA_SHIFT apart / the original has
    one and the filled lost it."""
    po, fo, wo = _hf_peak(orig, band, nperseg)
    pf, ff, wf = _hf_peak(filled, band, nperseg)
    has_o, has_f = po >= RSA_PEAK_LOG2, pf >= RSA_PEAK_LOG2
    return dict(hf_power_log2=wf - wo, hf_peak_log2=pf - po, hf_peak_shift=ff - fo,
                rsa_invented=float(has_f and not has_o and pf - po >= RSA_GROW_LOG2),
                rsa_displaced=float(has_o and has_f and abs(ff - fo) > RSA_SHIFT),
                rsa_lost=float(has_o and not has_f))


def closed_loop_subject(predictor, rr, drop_mask, cold_start=None):
    """Streaming imputation as in correlation_study_v3 / utils2._process_single_run: a dropped beat
    is replaced by the prediction, which then feeds the history (and the RLS state) of later beats.
    Beats dropped before WINDOW beats exist get cold_start(history) and are not scored (as there)."""
    eng = FeatureEngine()
    filled = np.array(rr, dtype=np.float64)
    t, p = [], []
    for n in range(len(filled)):
        if drop_mask[n]:
            if eng.ready:
                d = predictor.predict(eng.buf[None, :].astype(np.float32), eng.out[None, :].astype(np.float32))[0]
                filled[n] = eng.buf[-1] + d
                t.append(rr[n])
                p.append(filled[n])
            else:
                filled[n] = cold_start(filled[:n])
        eng.push(filled[n], imputed=drop_mask[n])  # the pacemaker knows which beats it filled
    return np.asarray(t), np.asarray(p), filled


def evaluate_closed_loop(predictors, subjects, percents=(0.1, 0.3, 0.5), seeds=(7,), max_beats=5000):
    """RR-level error (ms) on model-imputed beats, averaged over subjects, per percent of dropped beats.
    Beats are dropped anywhere from beat 0, as random_extraction(start_idx=0) in correlation_study_v3."""
    import pandas as pd
    import time
    from threadpoolctl import threadpool_limits  # ships with scikit-learn
    rows = []
    for p in predictors:
        t0 = time.time()
        # one prediction per beat: a 128-thread OpenMP pool per call (sklearn GBDT) costs more than the work
        with threadpool_limits(limits=1):
            per_model_rows = _closed_loop_model(p, subjects, percents, seeds, max_beats)
        rows += per_model_rows
        print(f"  closed loop {p.name}: {time.time() - t0:.0f}s", flush=True)
    per = pd.DataFrame(rows)
    per["hf_peak_shift_abs"] = per["hf_peak_shift"].abs()
    summ = per.groupby(["model", "percent"])[["rmse", "mae", "r2", "corr_series", "hf_power_log2", "hf_peak_log2",
                                              "hf_peak_shift_abs", "rsa_invented", "rsa_displaced",
                                              "rsa_lost"]].mean().reset_index()  # rsa_*: fraction of subjects
    return summ.sort_values(["percent", "rmse"]).reset_index(drop=True), per


def _closed_loop_model(p, subjects, percents, seeds, max_beats):
    """Per-subject closed-loop rows for one predictor (see evaluate_closed_loop)."""
    rows = []
    for si, s in enumerate(subjects):
        rr = s["rr"][:max_beats]
        ar = ColdStartAR.for_interval(s["interval"])
        ref_mean, ref_std = rr_reference(interval_age_years(s["interval"]))
        cold = lambda h: ar.forecast(h, ref_mean, ref_std)  # noqa: E731
        for seed in seeds:
            for pc in percents:
                n_drop = int(round(len(rr) * pc))
                rng = np.random.default_rng([seed, si, int(round(pc * 1000))])  # same mask for every model
                mask = np.zeros(len(rr), dtype=bool)
                mask[rng.choice(len(rr), size=n_drop, replace=False)] = True
                t, yp, filled = closed_loop_subject(p, rr, mask, cold)
                if len(t) == 0:
                    continue
                e = yp - t
                rows.append(dict(model=p.name, subject=s["code"], interval=s["interval"], seed=seed,
                                 percent=pc, n_imputed=len(t), rmse=float(np.sqrt(np.mean(e ** 2))),
                                 mae=float(np.mean(np.abs(e))),
                                 r2=float(1.0 - np.sum(e ** 2) / (np.sum((t - t.mean()) ** 2) + EPS)),
                                 corr_series=float(np.corrcoef(rr, filled)[0, 1]),
                                 **spectral_fidelity(rr, filled)))
    return rows


# -----------------------------------------------------------------------------
# Closed-loop training data (scheduled sampling / data-as-demonstrator, RESEARCH.md 3)
# -----------------------------------------------------------------------------
DAD_SEG_BEATS = 5000        # = evaluate_closed_loop max_beats: each segment restarts from beat 0 as there
DAD_DROP_RANGE = (0.1, 0.8)  # per-segment drop rate ~ U(lo, hi): the pacemaker's range
DAD_SEED_KEY = 0xDAD         # mask rng = default_rng([DAD_SEED_KEY, seed, subject, segment]): a stream
                             # disjoint from evaluate_closed_loop's default_rng([seed, subject, pc])
# ponytail: one fill per training series (one DAD iteration, the parent as filler) in 5000-beat lanes that
# restart cold, as the evaluator does. Ceiling: the fills come from the parent, not the model being trained,
# and closed-loop states older than 5000 beats are never seen. Upgrade: refill with the fine-tuned model
# (a 2nd DAD iteration, ~5 min) or longer lanes if the pacemaker runs gaps longer than that.


def closed_loop_dataset(filler, subjects, seed=0, drop_range=DAD_DROP_RANGE, seg_beats=DAD_SEG_BEATS,
                        drop_rates=None, max_segments=None, warmup=RLS_WARMUP, progress=False, prm=PRM):
    """Closed-loop-filled copies of `subjects` (load_split dicts; only rr/code/interval are read).
    Each series is cut into segments of seg_beats; each segment is a lane that drops a random
    U(drop_range) share of its beats (drop_rates: one fixed rate per lane instead) and fills them
    exactly as closed_loop_subject does: the filler's forecast for beats after the first WINDOW,
    the Burg-AR cold start before, the engine pushed with imputed=mask. All lanes run in lockstep:
    one batched filler call per beat index (predict_batch if the filler has it) + one numba call.
    Returns load_split-shaped dicts, one per segment:
      rr = filled series (the windows the model sees), F = engine features on it with the closed-loop
      imputed flags, y[j] = rr_true[j+W] - rr_filled[j+W-1] (what the model must output to land on
      the TRUE beat; with residual=True the base is F's rls_pred, the RLS run on the filled series),
      offset = warmup (first rows dropped, as load_split), plus rr_true, mask, drop_rate, start."""
    import time
    W, p, nf = WINDOW, ENGINE["ar_order"], N_FEATS
    lanes = []  # (subject index, start, length)
    for si, s in enumerate(subjects):
        for a in range(0, len(s["rr"]), seg_beats):
            n = min(seg_beats, len(s["rr"]) - a)
            if n >= W + warmup + 2:  # same minimum as load_split
                lanes.append((si, a, n))
    lanes = lanes[:max_segments] if max_segments else lanes
    L = len(lanes)
    T = max(n for _, _, n in lanes)
    lens = np.array([n for _, _, n in lanes], dtype=np.int64)
    true = np.zeros((L, T))
    masks = np.zeros((L, T), dtype=bool)
    rates = np.empty(L)
    seg_idx = {}
    for i, (si, a, n) in enumerate(lanes):
        k = seg_idx[si] = seg_idx.get(si, -1) + 1
        true[i, :n] = subjects[si]["rr"][a:a + n]
        rng = np.random.default_rng([DAD_SEED_KEY, seed, si, k])
        rates[i] = rng.uniform(*drop_range) if drop_rates is None else drop_rates[i]
        masks[i, rng.choice(n, size=int(round(n * rates[i])), replace=False)] = True
    colds = {}
    for si, _, _ in lanes:  # one Burg-AR start-up filler per age interval, as _closed_loop_model
        iv = subjects[si]["interval"]
        if iv not in colds:
            ar = ColdStartAR.for_interval(iv)
            ref_mean, ref_std = rr_reference(interval_age_years(iv))
            colds[iv] = (lambda h, ar=ar, m=ref_mean, sd=ref_std: ar.forecast(h, m, sd))
    bufs, scs, ws, Ps, phis, outs = (np.repeat(a[None], L, axis=0)  # writable copies (L may be 1)
                                     for a in _init_state(W, p, nf, prm))  # == _init_state per lane
    F = np.zeros((L, T - W, nf), dtype=np.float32)
    filled = true.copy()
    predict = getattr(filler, "predict_batch", filler.predict)
    t0 = time.time()
    for t in range(T):
        drop = masks[:, t]
        idx = np.flatnonzero(drop)
        if len(idx) and t >= W:  # every lane has pushed beats 0..t-1 >= W-1: its engine is ready
            d = predict(bufs[idx].astype(np.float32), outs[idx].astype(np.float32))
            filled[idx, t] = bufs[idx, W - 1] + d
        else:
            for i in idx:
                filled[i, t] = colds[subjects[lanes[i][0]]["interval"]](filled[i, :t])
        _push_lanes(t, filled[:, t].copy(), drop, lens, bufs, scs, ws, Ps, phis, outs, prm,
                    FREEZE_RLS_ON_IMPUTED, F)
        if progress and (t + 1) % 1000 == 0:
            print(f"  closed-loop fill: beat {t + 1}/{T}, {L} lanes, {time.time() - t0:.0f}s", flush=True)
    out = []
    for i, (si, a, n) in enumerate(lanes):
        s = subjects[si]
        y = true[i, W:n] - filled[i, W - 1:n - 1]
        out.append(dict(code=s["code"], interval=s["interval"], start=a, drop_rate=float(rates[i]),
                        rr=filled[i, :n], rr_true=true[i, :n], mask=masks[i, :n],
                        F=F[i, warmup:n - W], y=y[warmup:], offset=warmup))
    return out


# =============================================================================
# SELF-TEST  (python utils.py)
# =============================================================================
def _reference_window_features(win):
    """Independent numpy re-implementation of the utils2.py definitions for one window."""
    d = np.diff(win)
    out = dict(rmssd=np.sqrt(np.mean(d ** 2)), nn20=np.sum(np.abs(d) > 20),
               porta=np.sum(d < 0) / max(np.sum(d != 0), 1),
               guzik=np.sum(np.abs(d) * (d > 0)) / np.sum(np.abs(d)))
    cum = np.cumsum(win - win.mean())
    out["rs"] = (cum.max() - cum.min()) / win.std()
    emb = sliding_window_view(win, 2)  # (L, 2)
    H = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2.0)
    sd = np.std(emb @ H.T, axis=0)
    tri = sliding_window_view(emb, 3, axis=0)  # (K, 2, 3)
    tri = np.moveaxis(tri, -1, 1)  # (K, 3, 2)
    dm = tri[:, 1:, :] - tri[:, :1, :]
    vol = 0.5 * np.abs(np.linalg.det(dm))
    out["ccm"] = np.clip(vol.sum() / (np.pi * sd[0] * sd[1] * len(vol)), 0, 1)
    out["rr_dev"] = win[-1] - win.mean()
    return out


def _reference_rsa_features(rr, tracker=RSA_TRACKER, W=WINDOW):
    """Independent scipy re-implementation of the RSA tracker (engine steps 2b + 7) for every row
    of series_features(rr): lfilter band-passes and EMAs, then vectorised peak / phase logic."""
    from scipy.signal import lfilter
    rr = np.asarray(rr, np.float64)[:-1]  # the last beat has no target
    fc = np.linspace(tracker["f_lo"], tracker["f_hi"], int(tracker["n_bands"]))
    u = rr - np.r_[rr[0], rr[0], rr[:-2]]  # x[n] - x[n-2], start-up history = first beat
    Y = np.array([lfilter([b0], [1.0, a1, a2], u) for b0, a1, a2 in
                  (_rbj_bandpass(f, tracker["q"]) for f in fc)]).T  # (beats, K)
    Pw = lfilter([1.0 - tracker["beta"]], [1.0, -tracker["beta"]], Y * Y, axis=0)
    y1, Y, Pw = Y[W - 2:-1], Y[W - 1:], Pw[W - 1:]  # rows = series_features rows
    L = np.log2(Pw + 1e-8)
    df = fc - fc.mean()
    Lc = L - L.mean(1, keepdims=True)
    R = Lc - np.outer(Lc @ df / (df @ df), df)  # residual above the least-squares line
    best = 1 + np.argmax(R[:, 1:-1], axis=1)   # interior bands only
    i = np.arange(len(L))
    lm, l0, lp = L[i, best - 1], L[i, best], L[i, best + 1]
    curv = lm - 2.0 * l0 + lp
    off = np.clip(np.where(curv < 0, 0.5 * (lm - lp) / np.minimum(curv, -1e-300), 0.0), -0.5, 0.5)
    f = fc[best] + off * (fc[1] - fc[0])
    amp = np.sqrt(2.0 * Pw[i, best])
    w = 2.0 * np.pi * f
    yi = Y[i, best]
    yq = (y1[i, best] - yi * np.cos(w)) / np.sin(w)
    ang = np.arctan2(yq, yi)
    return dict(rsa_f=f, rsa_amp=amp, rsa_cos=np.cos(ang), rsa_sin=np.sin(ang), rsa_q=R[i, best],
                rsa_drr=amp * (np.cos(ang + w) - np.cos(ang)))


def _selftest():
    # Real series only (user rule): the validation subjects. The test split is never read here.
    val = load_split("val", DATA_DIR)
    rr = max(val, key=lambda s: len(s["rr"]))["rr"][:6000]  # past the ~3200-beat point where an unguarded RLS diverged

    print("[1/7] feature engine vs utils2 definitions (validation series) ...")
    F, y = series_features(rr)
    import hashlib  # pre-tracker columns are bit-identical to the engine every trained run used
    md5 = hashlib.md5(np.ascontiguousarray(F[:, :29]).tobytes()).hexdigest()
    assert md5 == "c9c3a01750bdca5c6626e651b96f68d8", \
        f"features 0-28 changed ({md5}): trained runs no longer match the engine (on a new CPU/numba, re-derive)"
    md5 = hashlib.md5(np.ascontiguousarray(F).tobytes()).hexdigest()  # all 35, incl. rsa_* (pre-gate engine)
    assert md5 == "80beb054fd9fcb4b3c0fd56d3af5cead", f"open-loop features changed ({md5})"
    for j in (0, 57, 3300, len(y) - 1):
        ref = _reference_window_features(rr[j:j + WINDOW])
        for k, v in ref.items():
            assert np.isclose(F[j, FI[k]], v, rtol=1e-9, atol=1e-9), (k, j, F[j, FI[k]], v)
        assert np.isclose(y[j], rr[j + WINDOW] - rr[j + WINDOW - 1])
        assert np.isclose(F[j, FI["drr_lag0"]], rr[j + WINDOW - 1] - rr[j + WINDOW - 2])
    eng = FeatureEngine()  # incremental engine == batch engine, bit for bit
    rows = [eng.out.copy() for v in rr[:-1] if eng.push(v)]
    assert np.array_equal(np.array(rows), F), "FeatureEngine differs from series_features"

    print("[2/7] RSA tracker vs scipy reference; imputed beats freeze the adaptive states ...")
    for k, v in _reference_rsa_features(rr).items():
        assert np.allclose(F[:, FI[k]], v, rtol=1e-9, atol=1e-9), (k, np.abs(F[:, FI[k]] - v).max())
    eng = FeatureEngine()
    for v in rr[:500]:
        eng.push(v)
    w0, P0, pw0 = eng.w.copy(), eng.P.copy(), eng.sc[S_BANK + 2:S_BANK + 3 * int(PRM[10]):3].copy()
    out0 = eng.out.copy()
    sc0, buf0 = eng.sc.copy(), eng.buf.copy()
    eng.push(rr[500] + 40.0, imputed=True, freeze_rls=True)  # a filled beat, both states frozen
    assert np.array_equal(eng.w, w0) and np.array_equal(eng.P, P0), "RLS adapted on a frozen beat"
    assert np.array_equal(eng.sc[S_BANK + 2:S_BANK + 3 * int(PRM[10]):3], pw0), "RSA band powers adapted on an imputed beat"
    assert not np.array_equal(eng.out, out0) and eng.buf[-1] == rr[500] + 40.0  # features still update
    eng.push(rr[501])
    assert not np.array_equal(eng.w, w0), "RLS did not resume on the next true beat"
    eng.buf[:], eng.sc[:], eng.w[:], eng.P[:] = buf0, sc0, w0, P0
    eng.push(rr[500] + 40.0, imputed=True)  # default protocol: RLS adapts, band powers frozen
    assert not np.array_equal(eng.w, w0) and np.array_equal(eng.sc[S_BANK + 2:S_BANK + 3 * int(PRM[10]):3], pw0)

    print("[3/7] RLS stays stable and beats persistence on every validation subject ...")
    clamp = ENGINE["rls_pred_clamp"] - 1e-6
    for s in val:  # an unguarded RLS put 13-58% of forecasts at the clamp (val RMSE 203 ms)
        frac = float(np.mean(np.abs(s["F"][:, FI["rls_pred"]]) >= clamp))
        assert frac < 0.01, f"RLS diverged on {s['code']}: {frac:.1%} of forecasts at the clamp"
    y = np.concatenate([s["y"] for s in val])
    m_rls = point_metrics(y, np.concatenate([s["F"][:, FI["rls_pred"]] for s in val]))
    m_per = point_metrics(y, np.zeros(len(y)))
    assert m_per["skill"] == 0.0
    assert m_rls["rmse"] < 0.95 * m_per["rmse"], (m_rls, m_per)
    print(f"      {len(val)} subjects, {len(y)} windows | persistence RMSE {m_per['rmse']:.2f} ms | "
          f"RLS RMSE {m_rls['rmse']:.2f} ms")

    print("[4/7] closed loop: 0% dropped reproduces the series; start-up beats get the Burg-AR fill ...")
    t, p, filled = closed_loop_subject(RLSAR(), rr[:300], np.zeros(300, bool))
    assert len(t) == 0 and np.array_equal(filled, rr[:300])
    from scipy.signal import lfilter
    ar = ColdStartAR.for_interval("3220-3481")
    psi = lfilter([1.0], np.r_[1.0, -ar.phi], np.r_[1.0, np.zeros(200000)])  # MA(inf) weights
    gam = np.array([psi[:len(psi) - k] @ psi[k:] for k in range(len(ar.phi) + 1)])
    assert np.allclose(ar.rho, gam / gam[0], atol=1e-6), "Yule-Walker autocorrelations are wrong"
    ref_mean, ref_std = rr_reference(interval_age_years("3220-3481"))
    drop = np.zeros(300, bool)
    drop[[0, 5, 12, 29]] = True  # all before the 30-beat window: filled by the AR, never scored
    t, p, filled = closed_loop_subject(RLSAR(), rr[:300], drop, lambda h: ar.forecast(h, ref_mean, ref_std))
    assert len(t) == 0 and np.array_equal(filled[~drop], rr[:300][~drop])
    assert filled[0] == ref_mean and np.isclose(filled[29], ar.forecast(filled[:29], ref_mean, ref_std))
    sf = spectral_fidelity(rr, rr)  # identical series -> perfect RSA fidelity
    assert all(v == 0.0 for v in sf.values()), sf
    rsa = max(val, key=lambda s: _hf_peak(s["rr"][:5000], HF_BAND, 256)[0])["rr"][:5000]  # clearest RSA peak
    assert _hf_peak(rsa, HF_BAND, 256)[0] >= RSA_PEAK_LOG2, "no validation subject has an RSA peak"
    drop = np.random.default_rng(0).random(len(rsa)) < 0.5  # holding the last beat adds broadband steps
    t, p, filled = closed_loop_subject(Persistence(), rsa, drop, lambda h: ar.forecast(h, ref_mean, ref_std))
    sf = spectral_fidelity(rsa, filled)
    assert sf["hf_peak_log2"] < -0.3 and sf["rsa_invented"] == 0.0, sf  # ... which blur, never sharpen, RSA
    print(f"      persistence at 50% dropped (RSA subject): HF power x{2 ** sf['hf_power_log2']:.2f}, "
          f"peak height x{2 ** sf['hf_peak_log2']:.2f}, shift {sf['hf_peak_shift']:+.3f} c/b, lost={sf['rsa_lost']:.0f}")
    # RSA tracker on real data: it finds the clearest peak and flags it above every peak-free subject
    q_flat, f_err, q_rsa = [], None, None
    for s in val:
        x = s["rr"][:5000]
        Fs = series_features(x)[0][len(x) // 2:]
        pk, fpk, _ = _hf_peak(x, HF_BAND, 256)
        if np.array_equal(x, rsa):
            f_err, q_rsa = abs(np.median(Fs[:, FI["rsa_f"]]) - fpk), np.median(Fs[:, FI["rsa_q"]])
        elif pk < RSA_PEAK_LOG2:
            q_flat.append(np.median(Fs[:, FI["rsa_q"]]))
    assert f_err < RSA_SHIFT, f"rsa_f is {f_err:.3f} c/b from the Welch RSA peak"
    assert q_rsa > max(q_flat), (q_rsa, max(q_flat))
    print(f"      RSA tracker: |rsa_f - Welch peak| {f_err:.3f} c/b; rsa_q {q_rsa:.2f} on the RSA subject "
          f"vs <= {max(q_flat):.2f} on {len(q_flat)} peak-free subjects")

    print("[5/7] causal TCNs: 30-beat window output == output on a longer history ...")
    import tensorflow as tf
    starts = np.array([100, 1000, 2000, 3000])
    seg = np.stack([rr[i:i + 45] for i in starts])  # real 45-beat histories, z-scored per series
    d = np.diff(seg, axis=1, prepend=seg[:, :1])
    x45 = np.stack([(seg - rr.mean()) / rr.std(), d / np.diff(rr).std()], axis=-1).astype(np.float32)
    f = ((F[starts + 15, :5] - F[:, :5].mean(0)) / (F[:, :5].std(0) + EPS)).astype(np.float32)
    for name, filters in (("micro_tcn", 16), ("micro_tcn_f8", 8)):  # every conv model in MODELS
        tf.keras.utils.set_random_seed(0)
        m30 = build_keras_model(name, 5)
        m45 = build_tcn(5, filters=filters, seq_len=45)
        m45.set_weights(m30.get_weights())
        a = np.asarray(m30([x45[:, -WINDOW:], f], training=False))
        b = np.asarray(m45([x45, f], training=False))
        assert np.allclose(a, b, atol=1e-5), f"{name}: receptive field leaks outside the window"
        c = keras_complexity(m30)
        print(f"      {name}: {c['params']} params, {c['macs_window']} MACs/window, "
              f"{c['macs_stream']} MACs/beat streaming")

    print("[6/7] closed-loop training data (DAD): lockstep fill == per-subject closed loop ...")
    W, wu = WINDOW, RLS_WARMUP
    segs = [dict(s, rr=s["rr"][:400]) for s in val[:3]]  # one 400-beat lane per real val subject
    dad = closed_loop_dataset(RLSAR(), segs, drop_rates=[0.0, 0.0, 0.0])
    for g, s in zip(dad, segs):  # nothing dropped: the original series, true dRR, open-loop features
        F0, y0 = series_features(s["rr"])
        assert not g["mask"].any() and np.array_equal(g["rr"], s["rr"]), "0% dropped changed the series"
        assert np.array_equal(g["y"], y0[wu:]) and np.array_equal(g["F"], F0[wu:].astype(np.float32))
    rates = [0.3, 0.5, 0.8]
    fillers = [RLSAR()] + [RunPredictor(RUNS_DIR / r) for r in ("micro_tcn",)
                           if (RUNS_DIR / r / "config.json").exists()]
    for filler in fillers:
        dad = closed_loop_dataset(filler, segs, seed=1, drop_rates=rates)
        dmax = 0.0
        for g, s, pc in zip(dad, segs, rates):
            assert g["mask"].sum() == round(len(s["rr"]) * pc) and np.array_equal(g["rr_true"], s["rr"])
            ar = ColdStartAR.for_interval(s["interval"])
            m, sd = rr_reference(interval_age_years(s["interval"]))
            _, _, ref = closed_loop_subject(filler, s["rr"], g["mask"], lambda h: ar.forecast(h, m, sd))
            dmax = max(dmax, float(np.abs(ref - g["rr"]).max()))
            eng = FeatureEngine()  # features = the engine replayed on the fill with the closed-loop flags
            rows = [eng.out.copy() for v, im in zip(g["rr"][:-1], g["mask"][:-1]) if eng.push(v, imputed=im)]
            assert np.array_equal(np.array(rows, dtype=np.float32)[wu:], g["F"])
            assert np.array_equal(g["y"], (g["rr_true"][W:] - g["rr"][W - 1:-1])[wu:]), "target != true - filled"
            assert np.array_equal(g["rr"][~g["mask"]], s["rr"][~g["mask"]]), "a kept beat was altered"
        # numba-side baseline: bit-exact; Keras: batched vs single-row float32 calls round differently
        assert dmax <= (0.0 if isinstance(filler, RLSAR) else 1e-2), (filler.name, dmax)
        print(f"      {filler.name}: 3 real val lanes at 30/50/80% dropped, max |lockstep - closed_loop_subject| "
              f"{dmax:.2g} ms")
    print("[7/7] RSA phase gate: closed-loop rsa_q/rsa_drr zeroed when >= gate fills in the window, nothing else ...")
    import hashlib
    h = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()  # noqa: E731
    segs = [dict(s, rr=s["rr"][:2000]) for s in val[:3]]
    kw = dict(seed=1, drop_rates=[0.3, 0.5, 0.8])
    dad = closed_loop_dataset(RLSAR(), segs, **kw)
    Fc = np.concatenate([g["F"] for g in dad])  # md5s recorded with the pre-gate engine (2026-10-08)
    assert h(Fc[:, :29]) == "32f3b5cc38ab0ccaed5d09369b96e5d1", "closed-loop non-rsa features changed"
    assert h(np.concatenate([g["rr"] for g in dad])) == "89ee0a391efe77ab76e9d77215815b08", "closed-loop fill changed"
    prm_off = PRM.copy()
    prm_off[-1] = WINDOW + 1  # gate never on == the pre-gate engine, bit for bit
    Fo = np.concatenate([g["F"] for g in closed_loop_dataset(RLSAR(), segs, prm=prm_off, **kw)])
    assert h(Fo) == h(np.c_[Fc[:, :29], Fo[:, 29:]]) and h(Fo[:, 29:]) == "5b254bef018eb7bea0decc6a09e0cab0"
    n_imp = np.concatenate([np.convolve(g["mask"][:-1], np.ones(WINDOW, int), "valid")[RLS_WARMUP:] for g in dad])
    on = n_imp >= RSA_TRACKER["gate_min_imputed"]  # row j sees the flags of beats j .. j+W-1
    gated = [FI["rsa_q"], FI["rsa_drr"]]
    assert on.any() and np.all(Fc[on][:, gated] == 0.0), "gate did not zero rsa_q/rsa_drr"
    assert np.array_equal(Fc[~on], Fo[~on]) and np.array_equal(np.delete(Fc, gated, 1), np.delete(Fo, gated, 1))
    share = {}
    for pc in (0.1, 0.3, 0.8):  # the clearest RSA subject: off at 10-30% dropped, on for most rows at 80%
        g = closed_loop_dataset(RLSAR(), [dict(val[0], rr=rsa)], seed=7, drop_rates=[pc])[0]
        share[pc] = float(np.mean(g["F"][:, FI["rsa_q"]] == 0.0))
    assert share[0.1] == 0.0 and share[0.3] == 0.0 and share[0.8] > 0.9, share
    print(f"      non-rsa features + fills bit-identical (md5); gate on at 30/50/80%: "
          f"{[round(float(np.mean(n_imp[a:a + len(g['y'])] >= RSA_TRACKER['gate_min_imputed'])), 3) for a, g in zip(np.cumsum([0] + [len(g['y']) for g in dad[:-1]]), dad)]}"
          f" of rows; RSA subject rsa_q == 0 at 10/30/80%: {share}")
    print("Self-test passed.")


if __name__ == "__main__":
    _selftest()

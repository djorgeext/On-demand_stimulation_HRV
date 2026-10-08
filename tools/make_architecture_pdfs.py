"""
tools/make_architecture_pdfs.py - one PDF summary per architecture family (written to docs/architectures/) per architecture family, plus an overview
and the shared feature-engineering / preprocessing document.

Every number comes from the project's own files: runs/*/config.json and model files (layers,
params, MACs, training settings), results/evaluation/<round>/ (validation), results/evaluation/final_test/
(the single test evaluation) and results/{test,validation}/ (whole-series closed-loop diagnostics).

Run: HIP_VISIBLE_DEVICES=-1 .venv/bin/python tools/make_architecture_pdfs.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "architectures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # glossary.py
os.chdir(ROOT)

import pandas as pd  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

import glossary as G  # noqa: E402

DATE = "2026-10-08"
RATES = ["0.1", "0.3", "0.5", "0.8"]

# ----------------------------------------------------------------------------- facts
def layer_rows(model):
    rows = []
    for layer in model.layers:
        c = layer.get_config()
        cfg = []
        for k in ("filters", "kernel_size", "dilation_rate", "units", "activation", "padding",
                  "num_heads", "key_dim", "rate", "unroll"):
            if k in c and c[k] not in (None, "linear", "valid", (1,)):
                v = c[k][0] if isinstance(c[k], (list, tuple)) and len(c[k]) == 1 else c[k]
                cfg.append(f"{k}={v}")
        if type(layer).__name__ == "ZeroPadding1D":
            cfg.append(f"left={c['padding'][0]}")
        try:
            shape = str(tuple(layer.output.shape)).replace("None", "B")
        except Exception:
            shape = ""
        rows.append((type(layer).__name__, ", ".join(cfg), shape, int(layer.count_params())))
    return rows


def n_weights(group):
    """Number of scalars in every dataset under an h5py group."""
    sizes = []
    group.visititems(lambda _, o: sizes.append(o.size) if hasattr(o, "size") and hasattr(o, "shape") else None)
    return int(sum(sizes))


def layer_rows_from_file(path):
    """layer_rows without TensorFlow (for hosts whose .venv cannot load it): layer configs from the .keras
    config.json, output shapes from the tensors the next layers consume, params from model.weights.h5,
    whose groups Keras 3 names snake_case(class) + per-class counter in layer order."""
    import io
    import re
    import zipfile
    import h5py
    z = zipfile.ZipFile(path)
    layers = json.loads(z.read("config.json"))["config"]["layers"]
    w = h5py.File(io.BytesIO(z.read("model.weights.h5")), "r")["layers"]
    shapes = {}
    for layer in layers:
        for node in layer["inbound_nodes"]:
            for t in json.dumps(node["args"]).split('"__keras_tensor__"')[1:]:
                m = re.search(r'"shape": (\[[^\]]*\]).*?"keras_history": \["([^"]+)"', t)
                shapes[m.group(2)] = json.loads(m.group(1))
    count, rows, total = {}, [], 0
    for layer in layers:
        cls, c = layer["class_name"], {k: tuple(v) if isinstance(v, list) else v for k, v in layer["config"].items()}
        cfg = []
        for k in ("filters", "kernel_size", "dilation_rate", "units", "activation", "padding",
                  "num_heads", "key_dim", "rate", "unroll"):
            if k in c and c[k] not in (None, "linear", "valid", (1,)):
                v = c[k][0] if isinstance(c[k], tuple) and len(c[k]) == 1 else c[k]
                cfg.append(f"{k}={v}")
        if cls == "ZeroPadding1D":
            cfg.append(f"left={c['padding'][0]}")
        snake = re.sub(r"([a-z])([A-Z])", r"\1_\2", re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", cls)).lower()
        key = snake if snake not in count else f"{snake}_{count[snake]}"
        count[snake] = count.get(snake, 0) + 1
        n = n_weights(w[key]) if key in w else 0
        total += n
        shape = shapes.get(layer["name"]) or c.get("batch_shape")
        if shape is None and cls == "Dense":  # the model output: nothing consumes it
            shape = shapes[layer["inbound_nodes"][0]["args"][0]["config"]["keras_history"][0]][:-1] + [c["units"]]
        rows.append((cls, ", ".join(cfg), str(tuple(shape)).replace("None", "B") if shape else "", n))
    assert total == n_weights(w), f"{path}: per-layer params {total} != weights file {n_weights(w)}"
    return rows


def collect():
    import utils as U
    try:
        from tensorflow import keras
    except ImportError:  # e.g. a ROCm build of TF on a host without ROCm: read the .keras files directly
        keras = None
    f = dict(engine=U.ENGINE, rsa_tracker=U.RSA_TRACKER, feature_sets=U.FEATURE_SETS,
             legacy_static=list(U.LEGACY_STATIC), runs={})
    for run in sorted(p.name for p in (ROOT / "runs").iterdir() if (p / "config.json").exists()):
        c = json.loads((ROOT / "runs" / run / "config.json").read_text())
        h = ROOT / "runs" / run / "history.csv"
        r = dict(cfg=c, epochs=(sum(1 for _ in open(h)) - 1) if h.exists() else None)
        if c.get("framework") == "keras":
            mp = ROOT / "runs" / run / "model.keras"
            r["layers"] = layer_rows(keras.models.load_model(mp, compile=False)) if keras else layer_rows_from_file(mp)
        f["runs"][run] = r
    lp = U.LEGACY_DIR / "tcn_attention_hrv_best.keras"
    if keras:
        leg = keras.models.load_model(lp, compile=False)
        f["tcn_mha"] = dict(layers=layer_rows(leg), params=int(leg.count_params()), cx=U.keras_complexity(leg))
    else:  # complexity as evaluate.py recorded it with TF (round 2)
        rows = layer_rows_from_file(lp)
        s = pd.read_csv(U.RESULTS_DIR / "evaluation" / "round2" / "val_open_loop_summary.csv").set_index("model")
        s = s.loc["tcn_mha_current"]
        assert sum(r[3] for r in rows) == int(s.params), "TCN_MHA params differ from the round-2 record"
        f["tcn_mha"] = dict(layers=rows, params=int(s.params), cx=dict(macs_stream=int(s.macs_stream)))
    res = {}
    ev = U.RESULTS_DIR / "evaluation"
    for tag, pre in [("round4", "val"), ("round3", "val"), ("round2_int8fix", "val"), ("round2", "val"),
                     ("round2_gru", "val"), ("round1", "val"), ("final_test", "test")]:
        o = pd.read_csv(ev / tag / f"{pre}_open_loop_summary.csv")
        c = pd.read_csv(ev / tag / f"{pre}_closed_loop_summary.csv")
        cols = [x for x in ["rmse", "rsa_invented", "hf_power_log2"] if x in c]
        res[tag] = dict(open={r.model: float(r.rmse) for r in o.itertuples()},
                        closed={m: {str(pc): {k: float(g[g.percent == pc][k].iloc[0]) for k in cols}
                                    for pc in sorted(g.percent.unique())} for m, g in c.groupby("model")})
    t = pd.read_csv(U.RESULTS_DIR / "test" / "summary_by_model.csv")
    for split, folder in (("test", "test"), ("val", "validation")):  # closed-loop diagnostics, 80% dropped, seed 7
        xc = pd.read_csv(U.RESULTS_DIR / folder / "crosscorr.csv", dtype={"subject": str})
        res[f"xcorr_{split}"] = {m: dict(rr=g.corrcoef_rr.mean(), drr=g.corrcoef_drr.mean())
                                 for m, g in xc.groupby("model")}
        res[f"xcorr_subj_{split}"] = {(r.subject, r.model): (float(r.corrcoef_rr), float(r.corrcoef_drr))
                                      for r in xc.itertuples()}
        ws = pd.read_csv(U.RESULTS_DIR / folder / "summary_by_model.csv")
        res[f"ws_{split}"] = {r.model: float(r.rmse) for r in ws.itertuples()}
    res["xcorr"] = res["xcorr_test"]
    res["test_ws"] = {r.model: dict(rmse=float(r.rmse), hf=float(r.hf_power_log2), inv=float(r.rsa_invented),
                                    disp=float(r.rsa_displaced)) for r in t.itertuples()}
    f["results"] = res
    return f


F = collect()
R = F["results"]
ORDER = ["round4", "round3", "round2_int8fix", "round2", "round2_gru", "round1"]
LABEL = dict(round4="round 4", round3="round 3", round2_int8fix="round 2 (int8 re-check)", round2="round 2",
             round2_gru="round 2 (gru)", round1="round 1")


def val_open(m):
    for t in ORDER:
        if m in R[t]["open"]:
            return R[t]["open"][m], t
    return None, None


def val_closed(m):
    for t in ORDER:
        if m in R[t]["closed"]:
            return R[t]["closed"][m], t
    return None, None


def fmt(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def cx(run):
    return F["runs"][run]["cfg"].get("complexity", {})


# ----------------------------------------------------------------------------- pdf helpers
SS = getSampleStyleSheet()
ST = dict(
    title=ParagraphStyle("t", parent=SS["Title"], fontSize=17, spaceAfter=4),
    sub=ParagraphStyle("s", parent=SS["Normal"], fontSize=9, textColor=colors.HexColor("#555555"), spaceAfter=10),
    h1=ParagraphStyle("h1", parent=SS["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, keepWithNext=1,
                      textColor=colors.HexColor("#1F3A5F")),
    h2=ParagraphStyle("h2", parent=SS["Heading3"], fontSize=10.5, spaceBefore=6, spaceAfter=2, keepWithNext=1),
    p=ParagraphStyle("p", parent=SS["Normal"], fontSize=9.2, leading=12.2, spaceAfter=4, alignment=TA_LEFT),
    formula=ParagraphStyle("f", parent=SS["Normal"], fontSize=9.2, leading=13, leftIndent=18, spaceBefore=1,
                           spaceAfter=3, textColor=colors.HexColor("#1F3A5F")),
    li=ParagraphStyle("li", parent=SS["Normal"], fontSize=9.2, leading=12.2, leftIndent=12, bulletIndent=2,
                      spaceAfter=1.5),
    cell=ParagraphStyle("c", parent=SS["Normal"], fontSize=7.4, leading=9.0),
    cellb=ParagraphStyle("cb", parent=SS["Normal"], fontSize=7.4, leading=9.0, fontName="Helvetica-Bold"),
    box=ParagraphStyle("box", parent=SS["Normal"], fontSize=9.2, leading=12.2, backColor=colors.HexColor("#EEF3F9"),
                       borderColor=colors.HexColor("#9DB4D3"), borderWidth=0.6, borderPadding=6, spaceBefore=4,
                       spaceAfter=10),
)


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def P(text, style="p"):
    return Paragraph(text, ST[style])


def bullets(items):
    return [Paragraph(t, ST["li"], bulletText="•") for t in items]


def table(header, rows, widths=None, bold_first=False, highlight=None):
    data = [[Paragraph(f"<b>{esc(h)}</b>", ST["cell"]) for h in header]]
    for r in rows:
        data.append([Paragraph(c if isinstance(c, str) and c.startswith("<") else esc(c),
                               ST["cellb"] if (bold_first and i == 0) else ST["cell"]) for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F2")),
             ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#A7B4C4")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for i in (highlight or []):
        style.append(("BACKGROUND", (0, i + 1), (-1, i + 1), colors.HexColor("#FFF4CC")))
    t.setStyle(TableStyle(style))
    return t


def build(fname, title, subtitle, story):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"HRV dRR predictor - {title} - {DATE}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"page {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title=title, author="HRV dRR project",
                            subject=subtitle)
    story = [P(esc(title), "title"), P(subtitle, "sub")] + story
    text = G.plain(" ".join(_texts(story)) + f" HRV dRR predictor - {title}")
    missing = G.undefined_tokens(text)
    if missing and os.environ.get("GLOSSARY_REPORT"):
        print(f"UNDEFINED {fname}: {missing}")
    elif missing:  # runnable check: every acronym in the PDF must be defined in tools/glossary.py
        raise ValueError(f"{fname}: undefined acronyms {missing}; add them to tools/glossary.py")
    feat_terms = [("drr_lag0..7" if k == "drr_lag" else "rls_w0..7" if k == "rls_w" else k, f"feature: {d}")
                  for k, d in FEATURE_DESC.items()
                  if (k in text if k in ("drr_lag", "rls_w") else G._present(k, "word", text))]
    have = {x[0] for x in G.used_terms(text)}
    terms = sorted(G.used_terms(text) + [x for x in feat_terms if x[0] not in have],
                   key=lambda x: x[0].lstrip("_").lower())
    story += [P("Abbreviations and acronyms used in this document", "h1"),
              table(["term", "meaning"], terms, widths=[3.2 * cm, 12.8 * cm], bold_first=True)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / fname, f"({len(terms)} glossary terms)")


def _texts(flowables):
    """Plain text of Paragraphs and Table cells, for the glossary scan."""
    for f in flowables:
        if isinstance(f, Paragraph):
            yield f.text
            if getattr(f, "bulletText", None):
                yield str(f.bulletText)
        elif isinstance(f, Table):
            for row in f._cellvalues:
                yield from _texts([c if isinstance(c, Paragraph) else Paragraph(esc(c), ST["cell"]) for c in row])
        elif isinstance(f, (list, tuple)):
            yield from _texts(f)


# ----------------------------------------------------------------------------- shared tables
def layer_table(rows):
    return table(["layer", "configuration", "output shape", "params"],
                 [(a, b, c, str(d)) for a, b, c, d in rows], widths=[3.4 * cm, 7.6 * cm, 3.4 * cm, 1.6 * cm])


def variants_table(runs, notes, highlight=None):
    rows = []
    for run in runs:
        c = F["runs"][run]["cfg"]
        x = cx(run)
        feats = c.get("features") or []
        fs = next((k for k, v in F["feature_sets"].items() if v == feats), f"{len(feats)} features")
        a = c.get("args", {})
        init = c.get("init_from")
        init = init.get("run") if isinstance(init, dict) else init
        how = f"fine-tune of {Path(init).name}, lr {a.get('lr')}" if init else f"from scratch, lr {a.get('lr')}"
        if (c.get("dad") or {}).get("filler"):
            how += f", closed-loop data share {c['dad'].get('share')}"
        rows.append((run, notes.get(run, ""), f"{fs} ({len(feats)})", str(x.get("params", "-")),
                     str(x.get("macs_stream", "-")), how, str(F["runs"][run].get("epochs") or "-")))
    return table(["run", "change vs parent", "feature set (n)", "params", "MACs/beat", "training", "epochs"],
                 rows, widths=[2.4 * cm, 4.3 * cm, 2.2 * cm, 1.2 * cm, 1.7 * cm, 3.4 * cm, 1.2 * cm],
                 bold_first=True, highlight=highlight)


def results_table(models, highlight=None):
    rows = []
    for m in models:
        vo, to = val_open(m)
        vc, tc = val_closed(m + "_int8")
        kind = "int8"
        if vc is None:
            vc, tc = val_closed(m)
            kind = "float"
        cl = [fmt(vc[r]["rmse"]) if vc and r in vc else "-" for r in RATES]
        inv = "-"
        if vc and "0.8" in vc and "rsa_invented" in vc["0.8"]:
            inv = f"{100 * vc['0.8']['rsa_invented']:.0f}%"
        ts = R["test_ws"].get(m + "_int8") or R["test_ws"].get(m)
        rows.append([m, fmt(vo), kind] + cl + [inv, fmt(ts["rmse"]) if ts else "-",
                                               LABEL.get(tc or to, "-")])
    return table(["model", "val open loop", "closed-loop model", "10%", "30%", "50%", "80%",
                  "RSA invented at 80%", "test whole-series 80%", "val source"], rows,
                 widths=[2.6 * cm, 1.5 * cm, 1.5 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm, 1.6 * cm, 1.8 * cm,
                         2.6 * cm], bold_first=True, highlight=highlight)


RES_NOTE = ("RMSE in ms. <b>val open loop</b>: one-step-ahead on the true history, 16 validation subjects. "
            "<b>10-80%</b>: closed loop (pacemaker use) on validation: that share of beats dropped at random from "
            "beat 0, each filled by the model and fed back; RMSE on model-filled beats, first 5000 beats per "
            "subject, mean of the seeds of that round (3 seeds from round 2 on). int8 is used where it exists. "
            "<b>RSA invented</b>: share of subject-runs whose filled series gains an RSA peak the original lacks. "
            "<b>test whole-series 80%</b>: diagnostic on the 10 test subjects (whole series, seed 7, results/test/ folder); "
            "it was not used for any selection.")

FEATURE_DESC = {
    "rr_mean": "mean RR of the 30-beat window", "rr_dev": "newest RR minus the window mean",
    "rmssd": "root mean square of the 29 successive differences",
    "rs": "rescaled range: (max - min of the cumulative sum of RR - mean) / std",
    "ccm": "complex correlation measure: Poincare-triangle area / (pi SD1 SD2 (W-3)), clipped to [0, 1]",
    "guzik": "Guzik asymmetry: sum of positive dRR / sum of |dRR|",
    "porta": "Porta asymmetry: count of negative dRR / count of non-zero dRR",
    "nn20": "count of |dRR| > 20 ms in the window",
    "run_len": "signed length of the current run of same-sign dRR (+ = decelerating)",
    "rls_pred": "RLS AR(8) forecast of the next dRR (ms), clamped to +-300",
    "rls_err": "last RLS innovation: true dRR minus the previous forecast (ms)",
    "bp": "RSA band-pass output: RBJ biquad on raw RR, centre 0.30 cycles/beat, Q 1",
    "bp_prev": "previous band-pass output (gives the phase with bp)",
    "drr_lag": "drr_lag0..7: the 8 most recent dRR (lag0 = newest)",
    "rls_w": "rls_w0..7: current RLS AR(8) weights",
    "rsa_f": "tracked RSA frequency (cycles/beat): peak of an 8-band resonator bank, parabolic interpolation",
    "rsa_amp": "RSA amplitude (ms) from the peak band's mean power",
    "rsa_cos": "cos of the current RSA phase (2-sample quadrature of the peak band)", "rsa_sin": "sin of that phase",
    "rsa_q": "RSA presence: peak height above the in-band log-power trend (log2); 0 when gated",
    "rsa_drr": "predicted RSA part of the next dRR: amp (cos(phase + 2 pi f) - cos(phase)); 0 when gated",
}


XC_NOTE = ("Computed on the 16 validation and the 10 test subjects (results/validation/ and results/test/ folders), on the ENTIRE "
           "series, after dropping 80% of the beats (seed 7, from beat 0) and filling them in closed loop with the "
           "model (Burg-AR fill for the first 30 beats): <b>corr RR</b> = np.corrcoef(original_serie, "
           "modified_serie)[0, 1]; <b>corr dRR</b> = the same function on the beat-to-beat changes, "
           "np.corrcoef(np.diff(original), np.diff(filled))[0, 1], which isolates the short-term dynamics. Mean "
           "over subjects; per-subject values in &lt;folder&gt;/crosscorr.csv. <b>RMSE filled</b>: RMSE on the "
           "model-filled beats of the same runs. Diagnostic: the selection used the round-4 protocol, not these runs.")


def xcorr_table(models, highlight=None):
    rows = []
    for m in models:
        xt, xv = R["xcorr_test"].get(m), R["xcorr_val"].get(m)
        if xt is None and xv is None:
            continue
        g = lambda x, k, d: f"{x[k]:.{d}f}" if x else "-"  # noqa: E731
        rows.append([m, g(xv, "rr", 4), g(xt, "rr", 4), g(xv, "drr", 3), g(xt, "drr", 3),
                     fmt(R["ws_val"].get(m)), fmt(R["ws_test"].get(m))])
    return table(["model", "corr RR validation", "corr RR test", "corr dRR validation", "corr dRR test",
                  "RMSE filled, validation (ms)", "RMSE filled, test (ms)"], rows,
                 widths=[3.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm], bold_first=True,
                 highlight=highlight)


XC_ORDER = sorted(R["xcorr_val"], key=lambda m: -R["xcorr_val"][m]["rr"])  # validation: the selection split


def _rank(m, key, split):
    d = R[f"xcorr_{split}"]
    return 1 + sorted(d, key=lambda x: -d[x][key]).index(m)


def _xc(m, key, split, d=4):
    return f"{R[f'xcorr_{split}'][m][key]:.{d}f}"


def _ext(key, split, fn):
    d = R[f"xcorr_{split}"]
    return fn(d, key=lambda m: d[m][key])


XC_BULLETS = [
    "RR level: every model keeps a high correlation with the original (validation "
    f"{min(v['rr'] for v in R['xcorr_val'].values()):.3f}-{max(v['rr'] for v in R['xcorr_val'].values()):.3f}, test "
    f"{min(v['rr'] for v in R['xcorr_test'].values()):.3f}-{max(v['rr'] for v in R['xcorr_test'].values()):.3f}): the "
    "slow heart-rate changes dominate and the 20% kept beats anchor them. The deployed tcn_lite_ft_int8: "
    f"validation {_xc('tcn_lite_ft_int8', 'rr', 'val')} (rank {_rank('tcn_lite_ft_int8', 'rr', 'val')} of "
    f"{len(R['xcorr_val'])}), test {_xc('tcn_lite_ft_int8', 'rr', 'test')} (rank "
    f"{_rank('tcn_lite_ft_int8', 'rr', 'test')}); rls_ar {_xc('rls_ar', 'rr', 'val')} / {_xc('rls_ar', 'rr', 'test')}; "
    f"TCN_MHA {_xc('tcn_mha_current', 'rr', 'val')} / {_xc('tcn_mha_current', 'rr', 'test')} (validation / test).",
    "dRR level (beat-to-beat dynamics) separates the models more: tcn_lite_ft_int8 "
    f"{_xc('tcn_lite_ft_int8', 'drr', 'val', 3)} / {_xc('tcn_lite_ft_int8', 'drr', 'test', 3)}, rls_ar "
    f"{_xc('rls_ar', 'drr', 'val', 3)} / {_xc('rls_ar', 'drr', 'test', 3)}, TCN_MHA "
    f"{_xc('tcn_mha_current', 'drr', 'val', 3)} / {_xc('tcn_mha_current', 'drr', 'test', 3)} (validation / test); "
    f"lowest: {_ext('drr', 'val', min)} on validation, {_ext('drr', 'test', min)} on test. Values around 0.2-0.3 are "
    "expected at 80% dropped: 4 of every 5 beat-to-beat changes are the model's own.",
    f"Highest RR correlation: {_ext('rr', 'val', max)} on validation, {_ext('rr', 'test', max)} on test. These "
    "whole-series runs are diagnostics; the selection was made with the round-4 validation protocol and does not "
    "change.",
]


def xcorr_per_subject(models, split="test"):
    d = R[f"xcorr_subj_{split}"]
    subjects = sorted({s for s, _ in d})
    rows = []
    for s in subjects:
        rows.append([s] + [f"{d[(s, m)][0]:.4f}" for m in models] + [f"{d[(s, m)][1]:.3f}" for m in models])
    head = ["subject"] + [f"RR {m}" for m in models] + [f"dRR {m}" for m in models]
    name = "test" if split == "test" else "validation"
    return [P(f"Per {name} subject ({len(subjects)}): deployed model vs TCN_MHA and rls_ar", "h2"),
            table(head, rows, widths=[2.2 * cm] + [2.3 * cm] * len(models) + [2.3 * cm] * len(models), bold_first=True)]


def features_list(feats):
    out, seen = [], set()
    for f in feats:
        key = "drr_lag" if f.startswith("drr_lag") else "rls_w" if f.startswith("rls_w") else f
        if key in seen:
            continue
        seen.add(key)
        name = "drr_lag0..7" if key == "drr_lag" else "rls_w0..7" if key == "rls_w" else f
        out.append((name, FEATURE_DESC.get(key, "")))
    return table(["feature", "definition"], out, widths=[2.6 * cm, 13.4 * cm])


INPUT_PREP = [
    "<b>Sequence input</b> (TCN/GRU): the last 30 RR intervals as two channels, z-RR and z-dRR (first step dRR = 0), "
    "z-scored with mean/std fitted on the training subjects only (zero variance -&gt; std 1).",
    "<b>Tabular input</b>: the engineered features of the newest beat (see 01_feature_engineering.pdf), z-scored per "
    "feature with training statistics.",
    "<b>Target</b>: dRR[n+1] = RR[n+1] - RR[n]. Residual models predict the z-scored correction over the RLS "
    "forecast (dRR - rls_pred); the output is de-normalised and added back to rls_pred.",
    "The first 30 windows of every series are dropped from training and scoring (cold RLS state).",
]
TRAIN_KERAS = ("Keras, AdamW (weight decay 1e-4), Huber loss (delta 1, z units), batch 512, steps_per_execution 64, "
               "epochs of 2M random training windows (about 10% of the 20.8M), EarlyStopping on val_loss "
               "(restore best weights) and ReduceLROnPlateau (factor 0.5, min 1e-5), seed 211. int8: full-integer "
               "post-training quantisation (TFLite, int8 in/out), deterministic best-of-K calibration (8 draws of "
               "2k/5k training windows; the draw with the smallest gap on held-out training subjects is kept).")


# ----------------------------------------------------------------------------- documents
def doc_overview():
    fams = [("tcn_lite", "tcn_lite_ft", "08_tcn_lite.pdf"), ("micro_tcn", "micro_tcn", "07_micro_tcn.pdf"),
            ("gru", "gru", "06_gru.pdf"), ("mlp", "mlp", "04_mlp.pdf"), ("gbdt", "gbdt", "05_gbdt.pdf"),
            ("linear", "linear", "03_linear.pdf")]
    rows = []
    for fam, best, pdf in fams:
        x = cx(best)
        vo, _ = val_open(best)
        vc, _ = val_closed(best + "_int8")
        vc = vc or val_closed(best)[0]
        ts = R["test_ws"].get(best + "_int8") or R["test_ws"].get(best)
        size = ("238 KB (30,500 nodes)" if fam == "gbdt" else f"{x.get('params', 0)} params"
                if x.get("params", 0) < 100 else f"{x.get('params', 0) / 1024:.1f} KB")
        rows.append([fam, best, str(x.get("macs_stream", "3000 compares")), size, fmt(vo),
                     fmt(vc["0.1"]["rmse"]) if vc else "-", fmt(vc["0.8"]["rmse"]) if vc else "-",
                     fmt(ts["rmse"]) if ts else "-", pdf])
    vo, _ = val_open("tcn_mha_current")
    vc, _ = val_closed("tcn_mha_current")
    rows.append(["TCN_MHA (legacy)", "tcn_mha_current", f"{F['tcn_mha']['cx']['macs_stream']:,}",
                 f"{F['tcn_mha']['params'] / 1024:.1f} KB", fmt(vo), fmt(vc["0.1"]["rmse"]), fmt(vc["0.8"]["rmse"]),
                 fmt(R["test_ws"]["tcn_mha_current"]["rmse"]), "09_tcn_mha.pdf"])
    vo, _ = val_open("rls_ar")
    vc = R["round4"]["closed"]["rls_ar"]
    rows.append(["baseline", "rls_ar", "152 FLOPs", "-", fmt(vo), fmt(vc["0.1"]["rmse"]), fmt(vc["0.8"]["rmse"]),
                 fmt(R["test_ws"]["rls_ar"]["rmse"]), "02_baselines.pdf"])
    ft = R["final_test"]
    ftc = ft["closed"]
    final_rows = []
    for m, lab in [("tcn_lite_ft_int8", "tcn_lite_ft_int8 (deployed)"), ("tcn_lite_ft", "tcn_lite_ft (float)"),
                   ("tcn_mha_current", "TCN_MHA (legacy)"), ("rls_ar", "rls_ar"), ("persistence", "persistence")]:
        final_rows.append([lab, fmt(ft["open"][m])] + [fmt(ftc[m][r]["rmse"]) for r in RATES])
    story = [
        P("<b>Final selection: tcn_lite_ft_int8</b> - a causal TCN (24 filters, dilations 1/2/4/7, receptive field "
          "29 beats) on the last 30 RR intervals plus 9 engineered features, predicting a correction on top of an "
          "adaptive RLS AR(8) forecast. 8,048 MACs per beat and 8.4 KB of int8 weights. It loses to none of the "
          "other candidates in the int8 closed loop on validation and passed every rule on the held-out test "
          "subjects.", "box"),
        P("Goal and constraints", "h1"),
        *bullets([
            "Predict dRR[n+1] = RR[n+1] - RR[n] beat by beat from the last 30 RR intervals, to fill missing beats in "
            "real time on a pacemaker (notebooks/correlation_study_v3.ipynb).",
            "First hardware: STM32H750 (Cortex-M7); the implant will be weaker. Hard budget: at most 10k MACs per "
            "beat (3k preferred), int8 weights at most 32 KB, int8 RMSE within 2% of float, MCU-native ops only "
            "(Conv1D, SeparableConv1D, Dense, BatchNorm, ReLU, Add, causal padding, GRU), receptive field at most 30.",
            "Data: RR series in data/series/*.txt, split by subject: 142 training, 16 validation, 10 test subjects "
            "(data/subjects_*.json, stratified by age interval). Selection on validation only; the test split was opened "
            "once, for the final evaluation.",
            "Selection rule (user decisions): inside the budget, precision first, decided by the CLOSED loop of the "
            "int8 model with paired per-subject Wilcoxon tests (pooled over drop rates or per rate with Bonferroni, "
            "never significantly worse at any rate); true ties go to the smaller model. Every model is also checked "
            "for RSA fidelity (no invented or displaced respiratory peak)."]),
        P("Architectures compared (best variant of each family)", "h1"),
        table(["family", "best variant", "MACs/beat", "int8 size", "val open loop", "val closed 10%",
               "val closed 80%", "test whole-series 80%", "document"], rows,
              widths=[2.3 * cm, 2.4 * cm, 1.6 * cm, 2.2 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm, 2.1 * cm],
              bold_first=True, highlight=[0]),
        P(RES_NOTE, "sub"),
        P("Final test (held-out test split, run once, 10 subjects, 942,526 windows)", "h1"),
        table(["model", "open loop", "closed 10%", "closed 30%", "closed 50%", "closed 80%"], final_rows,
              widths=[4.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], bold_first=True, highlight=[0]),
        *bullets([
            "tcn_lite_ft_int8 is better than rls_ar in 10/10 test subjects open loop (-8.2%) and in the pooled closed "
            "loop; at 80% dropped it is 9 ms (23.6%) better. The 10% gain is no longer significant on test.",
            "On the test subjects it invents and displaces no RSA peak at any drop rate; at 80% dropped it damps RSA power, which a "
            "feasibility analysis showed is an information limit (the RSA phase cannot be recovered from 20% of "
            "the beats), not a model flaw.",
            "The legacy TCN_MHA is worse in 10/10 subjects open loop, invents RSA peaks, and cannot run on the MCU."]),
        P("Correlation between the original and the filled series (validation and test subjects, all models)", "h1"),
        xcorr_table(XC_ORDER, [XC_ORDER.index("tcn_lite_ft_int8")]),
        P(XC_NOTE, "sub"),
        *bullets(XC_BULLETS),
        P("Documents in this folder", "h1"),
        table(["file", "content"], [
            ("00_overview.pdf", "this summary"),
            ("01_feature_engineering.pdf", "the streaming feature engine, preprocessing, targets, closed-loop and int8 protocols"),
            ("02_baselines.pdf", "persistence, EMA mean, adaptive RLS AR(8)"),
            ("03_linear.pdf", "ridge regression on lagged dRR (global AR)"),
            ("04_mlp.pdf", "MLP on engineered features: mlp, mlp_nobp, mlp_rsa"),
            ("05_gbdt.pdf", "gradient-boosted trees: gbdt, gbdt_all (accuracy and feature-importance reference)"),
            ("06_gru.pdf", "GRU on the RR sequence + features"),
            ("07_micro_tcn.pdf", "separable causal TCN, 16 filters, and its variants"),
            ("08_tcn_lite.pdf", "full causal TCN, 24 filters, and its variants incl. the selected tcn_lite_ft"),
            ("09_tcn_mha.pdf", "the legacy TCN with multi-head attention (current production model)"),
        ], widths=[4.2 * cm, 11.8 * cm]),
    ]
    build("00_overview.pdf", "Overview of all architectures",
          "Summary of every architecture trained for beat-by-beat dRR prediction, with the final selection and test result.",
          story)


def doc_features():
    e, t = F["engine"], F["rsa_tracker"]
    fs_rows = []
    users = {"seq": "micro_tcn, micro_tcn_f8, micro_tcn_dad, tcn_lite, tcn_lite_ft, tcn_lite_dad, gru",
             "tab": "mlp, gbdt", "ar": "linear", "legacy_cheap": "(not used by a trained run)", "tab_nobp": "mlp_nobp",
             "tab_rsa": "mlp_rsa", "seq_rsa": "tcn_lite_rsa, tcn_lite_rsa_ws, micro_tcn_rsa, micro_tcn_rsa_ws",
             "all": "gbdt_all (29 features: trained before the 6 RSA features were appended)"}
    for k, v in F["feature_sets"].items():
        fs_rows.append((k, str(len(v)), ", ".join(v) if len(v) < 25 else "all features (FEATURE_NAMES)", users.get(k, "")))
    story = [
        P("All models get their inputs from one streaming engine (src/utils.py, numba kernel _push): it consumes one RR "
          "interval per beat and keeps O(1) state, so training, evaluation and the pacemaker compute identical "
          "features. It is the reference for the C port. The legacy TCN_MHA is the only exception (see the end).", "box"),
        P("1. Per-beat streaming engine", "h1"),
        *bullets([
            f"Window: the last W = {e['window']} RR intervals (ms), kept in a ring buffer. Cost per beat: O(W) window "
            "statistics + O(p<super>2</super>) RLS update (p = 8, about 152 FLOPs) + the RSA bank (about 180 FLOPs) "
            "+ a biquad (5 multiplies).",
            f"RLS AR({e['ar_order']}): predicts the next dRR from the centred RR lags (RR[n-k] - window mean) / "
            f"{e['rls_scale']:g}. Forgetting factor {e['rls_lambda']} (about 100-beat memory), initial covariance "
            f"{e['rls_p0']:g}, covariance-trace cap {e['rls_p_max_trace']:g}, forecast clamp +-{e['rls_pred_clamp']:g} ms. "
            "Stability guard: the update is skipped when the covariance loses positive definiteness, the covariance "
            "is kept symmetric and reset if its trace becomes non-finite (without it the RLS diverged after about "
            "3,200 beats on real data).",
            f"Fixed RSA band-pass: RBJ biquad on the raw RR, centre {e['bp_f0']} cycles/beat, Q {e['bp_q']} (about "
            "0.15-0.45 cycles/beat), zero DC gain.",
            f"RSA tracker (round 2): a bank of {t['n_bands']} RBJ band-passes from {t['f_lo']} to {t['f_hi']} "
            f"cycles/beat (Q {t['q']}), band-power EMAs with factor {t['beta']}. The peak band above the in-band "
            "log-power trend gives frequency, amplitude, phase and presence. In the closed loop its band powers are "
            f"frozen on filled beats and a gate sets rsa_q = rsa_drr = 0 once {t['gate_min_imputed']} or more of the "
            "last 30 beats were filled (the RSA phase cannot be recovered there)."]),
        P("2. Engineered features (35, in FEATURE_NAMES order)", "h1"),
        features_list(__import__("utils").FEATURE_NAMES),
        *__import__("feature_details").story("en", P, bullets, table, cm),
        P("4. Feature sets", "h1"),
        table(["set", "n", "features", "used by"], fs_rows, widths=[1.8 * cm, 0.7 * cm, 8.7 * cm, 4.8 * cm], bold_first=True),
        P("5. Model inputs, normalisation and target", "h1"),
        *bullets(INPUT_PREP),
        P("6. Closed-loop preprocessing (the pacemaker use, notebooks/correlation_study_v3.ipynb)", "h1"),
        *bullets([
            "Dropped beats: a random share (10/30/50/80%) of beats is removed, uniformly from beat 0 (as "
            "utils2.random_extraction with start_idx = 0); the same mask is used for every model.",
            "First 30 beats (no full window yet): filled by the notebook's Burg-AR predictor (AR(15) coefficients "
            "from data/ar_model_parameters.json, exact finite-history MMSE weights, anchored to the age-matched RR "
            "reference 505 x age<super>0.122</super> ms); these fills are not scored.",
            "From beat 30: each dropped beat is filled by the model; the fill enters the window, the features and "
            "the RLS state of later beats. Only model-filled beats are scored.",
            "RSA fidelity: Welch PSD of the filled vs the original series in 0.15-0.40 cycles/beat; a peak counts as "
            "RSA when it is at least 2x above the in-band trend; invented = new peak (and grown >= 1.4x), displaced "
            "= moved > 0.03 cycles/beat, lost = disappeared."]),
        P("7. Closed-loop training data (DAD, round 3)", "h1"),
        P("For the _dad runs, every training series was cut into 5000-beat segments with a random 10-80% of beats "
          "dropped and filled in closed loop by the parent model itself; the engine was run on the filled series "
          "and the target was the TRUE next beat (rr_true[n+1] - rr_filled[n]). Batches mixed 50% of these windows "
          "with 50% ordinary ones. It did not improve the deployed int8 model significantly."),
        P("8. int8 export", "h1"),
        P("Full-integer post-training quantisation with TFLite (int8 weights, activations, input and output, the "
          "CMSIS-NN / ST Edge AI format). Calibration is deterministic best-of-K: 8 random draws of 2k or 5k "
          "training windows are tried and the export with the smallest int8-vs-float gap on 20 held-out training "
          "subjects (ordinary and closed-loop windows) is kept. Validation is never used for calibration."),
        P("9. Legacy TCN_MHA features (not from the engine)", "h1"),
        P("The legacy model recomputes its inputs per window with legacy/utils2.py: 12 static features "
          f"({', '.join(F['legacy_static'])}) z-scored with the statistics stored in data/hrv_dataset.h5, plus the 30 RR "
          "values z-scored with global statistics (one channel). The Burg AR(5) coefficients need an O(order x W) "
          "fit per beat."),
    ]
    build("01_feature_engineering.pdf", "Feature engineering and preprocessing",
          "The streaming feature engine shared by every model, the feature sets, inputs, targets, closed-loop protocol and int8 export.",
          story)


def doc_family(fname, title, subtitle, intro, arch_text, layer_run, runs, notes, verdict, highlight=None,
               extra=None, extra_after=None):
    c = F["runs"][layer_run]["cfg"]
    feats = c.get("features") or []
    story = [P(intro, "box"), P("Architecture", "h1"), *bullets(arch_text)]
    if F["runs"][layer_run].get("layers"):
        story += [P(f"Layer table of <b>{layer_run}</b> (B = batch)", "h2"), layer_table(F["runs"][layer_run]["layers"])]
    story += [P(f"Inputs and feature engineering ({layer_run})", "h1"), *bullets(INPUT_PREP[:3] if c.get("kind") == "seq"
              else [INPUT_PREP[1], INPUT_PREP[2]] if c.get("residual") else [INPUT_PREP[1],
              "<b>Target</b>: dRR[n+1] directly (no residual)."]),
              P(f"Tabular features ({len(feats)}):", "h2"), features_list(feats)]
    if c.get("framework") == "keras":
        story += [P("Training", "h1"), P(TRAIN_KERAS)]
    if extra:
        story += extra
    story += [P("Variants trained", "h1"), variants_table(runs, notes, highlight),
              P("Results", "h1"), results_table(runs, highlight), P(RES_NOTE, "sub"),
              P("Correlation between the original and the filled series (validation and test subjects)", "h1"),
              xcorr_table(runs + [m + "_int8" for m in runs if m + "_int8" in R["xcorr_val"]],
                          [i for i, m in enumerate(runs) if highlight and i in highlight]),
              P(XC_NOTE, "sub")]
    if extra_after:
        story += extra_after
    story += [P("Verdict", "h1"), *bullets(verdict)]
    build(fname, title, subtitle, story)


def doc_baselines():
    rows = []
    for m, desc, cost in [("persistence", "next dRR = 0 (repeat the last RR)", "0"),
                          ("ema_mean", "next RR = exponential moving average of the window (factor 0.75 per beat), "
                                       "dRR = EMA - last RR", "30"),
                          ("rls_ar", "adaptive per-patient AR(8) by recursive least squares on centred RR lags "
                                     "(forgetting 0.99); the forecast is the rls_pred feature", "152 FLOPs")]:
        vo, _ = val_open(m)
        vc = R["round4"]["closed"][m]
        rows.append([m, desc, cost, fmt(vo)] + [fmt(vc[r]["rmse"]) for r in RATES] + [fmt(R["test_ws"][m]["rmse"])])
    story = [
        P("Baselines need no training. Every candidate is reported against rls_ar and persistence; a model that does "
          "not beat rls_ar is not progress.", "box"),
        table(["baseline", "definition", "cost/beat", "val open", "10%", "30%", "50%", "80%", "test whole-series 80%"], rows,
              widths=[2.0 * cm, 6.0 * cm, 1.4 * cm, 1.2 * cm, 1.0 * cm, 1.0 * cm, 1.0 * cm, 1.0 * cm, 1.4 * cm],
              bold_first=True),
        P(RES_NOTE, "sub"),
        P("Correlation between the original and the filled series (validation and test subjects)", "h1"),
        xcorr_table(["persistence", "ema_mean", "rls_ar"]), P(XC_NOTE, "sub"),
        P("Observations", "h1"),
        *bullets(["rls_ar is the strongest baseline one step ahead (37.1 ms on validation) and is the residual base "
                  "of every good model.",
                  "In the closed loop rls_ar is unstable at high drop rates: at 80% dropped it is worse than "
                  "persistence and invents an RSA peak in about 75% of subject-runs, because an AR model fed its own "
                  "predictions resonates. Freezing its update on filled beats made it worse (tested in round 4).",
                  "persistence never invents RSA peaks but damps RSA power and has the worst one-step error."]),
    ]
    build("02_baselines.pdf", "Baselines", "Persistence, EMA mean and the adaptive RLS AR(8) reference.", story)


def doc_tcn_mha():
    ft = R["final_test"]
    vo, _ = val_open("tcn_mha_current")
    vc, _ = val_closed("tcn_mha_current")
    ws = R["test_ws"]["tcn_mha_current"]
    story = [
        P("The current production model, evaluated as one more candidate (not as a reference). It is not "
          "deployable on the microcontroller and is beaten by tcn_lite in almost every subject.", "box"),
        P("Architecture", "h1"),
        *bullets([
            "Multi-scale causal convolution stem (kernels 3, 5 and 9, 16 filters each, GELU), concatenated to 48 "
            "channels.",
            "Four residual blocks of two causal Conv1D(48, kernel 3) with LayerNormalization, GELU and "
            "SpatialDropout(0.1), dilations 1, 2, 4 and 8.",
            "MultiHeadAttention over the 30 time steps (4 heads, key dim 12) with a residual connection and "
            "LayerNorm, then GlobalAveragePooling1D.",
            "Static-feature branch: Dense(32, GELU) + LayerNorm, used to modulate the pooled sequence (multiply and "
            "add gates) before a Dense(64, GELU) + LayerNorm, Dense(32, GELU), Dense(1) head.",
            f"{F['tcn_mha']['params']:,} parameters (about {F['tcn_mha']['params'] / 1024:.1f} KB int8) and about "
            f"{F['tcn_mha']['cx']['macs_stream']:,} MACs per prediction; not streamable.",
            "Trained in the legacy pipeline with RSAPhaseAwareLoss: Huber + a sign penalty + a variance-matching "
            "penalty. The variance term rewards oscillation, a plausible cause of its invented RSA peaks."]),
        P("Layer table (B = batch)", "h2"), layer_table(F["tcn_mha"]["layers"]),
        P("Inputs and preprocessing", "h1"),
        *bullets([f"Sequence: the last 30 RR, one channel, z-scored with global statistics from data/hrv_dataset.h5.",
                  f"12 static features recomputed per window with legacy/utils2.py: {', '.join(F['legacy_static'])}, "
                  "z-scored with data/hrv_dataset.h5 statistics.",
                  "Target: dRR[n+1], de-normalised with the stored target mean/std (no residual over RLS)."]),
        P("Results", "h1"),
        table(["split", "open loop", "closed 10%", "closed 30%", "closed 50%", "closed 80%", "notes"], [
            ["validation (round 2)", fmt(vo)] + [fmt(vc[r]["rmse"]) for r in RATES] +
            [f"RSA invented at 80%: {100 * vc['0.8']['rsa_invented']:.0f}% of subject-runs"],
            ["test (final)", fmt(ft["open"]["tcn_mha_current"])] + [fmt(ft["closed"]["tcn_mha_current"][r]["rmse"]) for r in RATES] +
            ["float only"],
            ["test whole series, 80%", "-", "-", "-", "-", fmt(ws["rmse"]),
             f"RSA power x{2 ** ws['hf']:.2f}, invented {ws['inv']:.1f}, displaced {ws['disp']:.1f} per subject"]],
            widths=[3.0 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 6.0 * cm], bold_first=True),
        P("Correlation between the original and the filled series (validation and test subjects)", "h1"),
        xcorr_table(["tcn_mha_current", "tcn_lite_ft_int8", "rls_ar"], [1]), P(XC_NOTE, "sub"),
        P("Verdict", "h1"),
        *bullets(["Invalid for the MCU (attention, LayerNorm, GELU, global pooling, 75.5 KB, about 2M MACs).",
                  "Beaten by tcn_lite in 16/16 validation subjects open loop and 14/16 closed loop, and by "
                  "tcn_lite_ft_int8 in 10/10 test subjects open loop.",
                  "Invents or displaces RSA peaks in the closed loop (visible in results/test/&lt;subject&gt;/tcn_mha_current/ "
                  "spectra), the failure that started this work. The TCNs did so in at most 4% of validation subject-runs at 80% "
                  "dropped and never on the test subjects."]),
    ]
    build("09_tcn_mha.pdf", "TCN_MHA (legacy production model)",
          "The previous production model: causal TCN with multi-head attention, judged as one more candidate.", story)


def main():
    OUT.mkdir(exist_ok=True)
    doc_overview()
    doc_features()
    doc_baselines()
    lin = F["runs"]["linear"]["cfg"].get("linear") or {}
    doc_family("03_linear.pdf", "Linear model (ridge, global AR)", "Ridge regression on the lagged dRR: a global, non-adaptive AR(9).",
               "The simplest learned model: one weight per input. It is a global AR model shared by all patients, "
               "unlike rls_ar, which adapts to each patient.",
               ["Ridge regression (penalty 1e-3) solved by float64 normal equations accumulated in chunks.",
                f"{len(lin.get('coef', []))} weights + intercept = {cx('linear').get('params')} parameters, "
                f"{cx('linear').get('macs_stream')} MACs per beat.",
                "Not residual: it predicts dRR directly."],
               "linear", ["linear"], {"linear": "baseline learned model"},
               ["Worse than rls_ar one step ahead (40.5 vs 37.1 ms) and up to 24.5% worse in one age interval: "
                "rejected. A global AR cannot follow per-patient dynamics the way the adaptive RLS does."])
    doc_family("04_mlp.pdf", "MLP on engineered features", "Dense network on the tabular features only (no raw sequence).",
               "A two-hidden-layer MLP that sees only the engineered features, including the recent dRR lags and "
               "the RLS forecast. Selected in round 1 under the old rule; beaten by the TCNs once ties were judged "
               "by paired tests.",
               ["Dense(32, ReLU) - Dense(16, ReLU) - Dense(1).",
                f"{cx('mlp').get('params')} parameters, {cx('mlp').get('macs_stream')} MACs per beat, about 1.2 KB int8."],
               "mlp", ["mlp", "mlp_nobp", "mlp_rsa"],
               {"mlp": "round 1 base", "mlp_nobp": "without the fixed 0.30 band-pass (bp, bp_prev)",
                "mlp_rsa": "mlp_nobp + the 6 adaptive RSA-tracker features (ungated)"},
               ["mlp: better than rls_ar in every age interval, but worse than the TCNs in the closed loop in "
                "13-15 of 16 subjects.",
                "mlp_nobp: removing the fixed band-pass made the closed loop worse (p = 0.018): bp is useful.",
                "mlp_rsa: better one step ahead (15/16 subjects vs mlp) but worse at 80% dropped (+1.64 ms): the "
                "tracker's filters run on the smooth fills and lose the RSA phase. Led to the gated design of "
                "round 4."])
    gb = F["runs"]["gbdt"]
    doc_family("05_gbdt.pdf", "Gradient-boosted trees", "HistGradientBoosting on the tabular features: accuracy and feature-importance reference.",
               "Gradient-boosted trees are a strong tabular reference but far too large for the microcontroller "
               "(about 8 bytes per node).",
               ["scikit-learn HistGradientBoostingRegressor: squared error, learning rate 0.05, up to 500 trees of "
                "depth 6 (31 leaves), min 100 samples per leaf, L2 1.0, early stopping on the validation split.",
                "500 trees, 30,500 nodes: about 238 KB at 8 B per node, far over the 32 KB budget, and about 3,000 "
                "comparisons per prediction.",
                "Residual over the RLS forecast like the neural models."],
               "gbdt", ["gbdt", "gbdt_all"],
               {"gbdt": "19 tabular features", "gbdt_all": "every engine feature (29 at the time)"},
               ["Not deployable (size). Accuracy reference only: 34.2 ms on validation, between the MLP and the TCNs.",
                "gbdt_all (34.15 ms) shows that the extra features add nothing; its permutation importance ranks "
                "rls_pred first by far, then rr_dev, drr_lag0, guzik and bp; the 8 RLS weights contribute nothing."])
    doc_family("06_gru.pdf", "GRU", "A small recurrent network over the RR sequence plus the engineered features.",
               "A 16-unit GRU over the 30-beat sequence. Accurate in float but rejected because int8 quantisation "
               "costs it 1.6-3.6 ms.",
               ["GRU(16, unrolled for a loop-free TFLite graph) over the (30, 2) sequence; its last state is "
                "concatenated with the 9 features, then Dense(32, ReLU) - Dense(1).",
                f"{cx('gru').get('params')} parameters, {cx('gru').get('macs_stream')} MACs per beat (stateful "
                "streaming), about 1.8 KB int8."],
               "gru", ["gru"], {"gru": "same inputs as the TCNs"},
               ["Float: ties micro_tcn at every drop rate with 19% fewer MACs and beats mlp in 15/16 subjects.",
                "int8: gap +2.71% on validation (over the 2% limit) and 1.6-3.6 ms lost in the closed loop; it also "
                "invents an RSA peak at 50% on one subject: rejected. Quantising recurrent state is known to be hard "
                "on MCUs."])
    doc_family("07_micro_tcn.pdf", "micro_tcn (separable causal TCN)", "Depthwise-separable causal TCN, 16 filters: the cheap convolutional family.",
               "The economical TCN: 2,080 MACs per beat and 2.4 KB. It ties the larger tcn_lite in the pooled closed "
               "loop but loses to tcn_lite_ft at 10-30% dropped in int8.",
               ["Pointwise Conv1D stem (16 filters) + BatchNorm + ReLU, then 4 residual blocks of causal "
                "SeparableConv1D (kernel 3, dilations 1, 2, 4, 7) with explicit left padding, BatchNorm (folded at "
                "export) and ReLU.",
                "Receptive field 1 + 2 x (1+2+4+7) = 29 beats, so the 30-beat window model equals the streaming model "
                "(asserted in the self-test). Only the newest time step is read out, concatenated with the 9 "
                "features, then Dense(32, ReLU) - Dense(1).",
                f"{cx('micro_tcn').get('params')} parameters, {cx('micro_tcn').get('macs_stream')} MACs per beat in "
                "streaming mode, 2.4 KB int8."],
               "micro_tcn", ["micro_tcn", "micro_tcn_f8", "micro_tcn_dad", "micro_tcn_rsa", "micro_tcn_rsa_ws"],
               {"micro_tcn": "round 1 base", "micro_tcn_f8": "filters 16 -> 8",
                "micro_tcn_dad": "fine-tuned on closed-loop-filled data (DAD)",
                "micro_tcn_rsa": "+ gated RSA features (seq_rsa), from scratch",
                "micro_tcn_rsa_ws": "+ gated RSA features, warm start (new inputs zero-initialised)"},
               ["micro_tcn: runner-up; in int8 it loses to tcn_lite_ft at 10% and 30% dropped (p &lt;= 0.003).",
                "micro_tcn_f8: halving the filters costs about 1.8 ms: rejected.",
                "micro_tcn_dad: better than micro_tcn at 80% in int8 (p = 0.002) but worse at 10%: not selected.",
                "micro_tcn_rsa (from scratch, about 3.6 passes over the data vs about 48 for its parent: confounded) "
                "and micro_tcn_rsa_ws (clean test): no gain from the gated RSA features."])
    ft = R["final_test"]
    doc_family("08_tcn_lite.pdf", "tcn_lite (full causal TCN) - selected family", "Full-convolution causal TCN, 24 filters: the selected model tcn_lite_ft.",
               "<b>Selected: tcn_lite_ft_int8.</b> Same graph as tcn_lite, fine-tuned for a few more epochs. It loses "
               "to no other candidate in the int8 closed loop on validation. Final test (10 subjects): open loop "
               f"{fmt(ft['open']['tcn_lite_ft_int8'])} ms vs rls_ar {fmt(ft['open']['rls_ar'])}; closed loop "
               f"{' / '.join(fmt(ft['closed']['tcn_lite_ft_int8'][r]['rmse']) for r in RATES)} ms at 10/30/50/80% "
               "dropped; no invented or displaced RSA peak.",
               ["Pointwise Conv1D stem (24 filters) + BatchNorm + ReLU, then 4 residual blocks of causal full "
                "Conv1D (kernel 3, dilations 1, 2, 4, 7) with explicit left padding, BatchNorm (folded at export) "
                "and ReLU.",
                "Receptive field 29 beats (window model = streaming model). The newest time step is concatenated "
                "with the 9 features, then Dense(32, ReLU) - Dense(1).",
                f"{cx('tcn_lite').get('params')} parameters, {cx('tcn_lite').get('macs_stream')} MACs per beat in "
                "streaming mode (under the 10k limit), 8.4 KB int8; int8 costs +0.5% on test."],
               "tcn_lite", ["tcn_lite", "tcn_lite_ft", "tcn_lite_dad", "tcn_lite_rsa", "tcn_lite_rsa_ws"],
               {"tcn_lite": "round 1 base (about 48 full passes)",
                "tcn_lite_ft": "same model, fine-tuned (lr 3e-4); control for the DAD run. SELECTED",
                "tcn_lite_dad": "fine-tuned on closed-loop-filled data (DAD)",
                "tcn_lite_rsa": "+ gated RSA features (seq_rsa), from scratch",
                "tcn_lite_rsa_ws": "+ gated RSA features, warm start (new inputs zero-initialised)"},
               ["tcn_lite_ft_int8 beats micro_tcn at 10% and 30% dropped (p &lt;= 0.003, float agrees) and tcn_lite "
                "pooled; it ties only tcn_lite_dad (same size; DAD's 0.28 ms gain at 80% is not significant).",
                "The gated RSA features add nothing at 10-50% and cost 0.2 ms at 80%; tcn_lite_rsa from scratch is "
                "confounded by a 13x smaller training budget.",
                "At 80% dropped every model damps RSA power (about x0.5): an information limit, not a defect; the "
                "selected model invented no RSA peak on validation or test; its displaced-peak flags on validation (only "
                "the 2 RSA subjects) were no more frequent than rls_ar's, and there were none on test."],
               highlight=[1], extra_after=xcorr_per_subject(["tcn_lite_ft_int8", "tcn_mha_current", "rls_ar"], "val")
               + xcorr_per_subject(["tcn_lite_ft_int8", "tcn_mha_current", "rls_ar"], "test"))
    doc_tcn_mha()


if __name__ == "__main__":
    main()

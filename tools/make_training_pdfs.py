"""
tools/make_training_pdfs.py - how every model was trained: one PDF per model family (plus the shared
pipeline and the baselines), in English and Spanish, written to docs/training/<nn>_<name>_{en,es}.pdf.

Facts come from runs/*/config.json and history.csv (settings, epochs, timing, learning-rate schedule,
int8 calibration), src/train.py (the procedure), legacy/TCN_MHA.py (the legacy model) and
docs/EXPERIMENTS.md (incidents and outcomes, quoted in the per-run notes). Layout helpers, results and
glossaries are shared with make_architecture_pdfs.py / make_architecture_pdfs_es.py.

Run: HIP_VISIBLE_DEVICES=-1 .venv/bin/python tools/make_training_pdfs.py
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_architecture_pdfs as M  # noqa: E402  (loads every run and result once)
import make_architecture_pdfs_es as MES  # noqa: E402
import glossary_es as GE  # noqa: E402
from reportlab.graphics.charts.lineplots import LinePlot  # noqa: E402
from reportlab.graphics.shapes import Drawing, Line, String  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer  # noqa: E402

OUT = M.ROOT / "docs" / "training"
F, R, DATE = M.F, M.R, M.DATE
P, bullets, table, esc, fmt = M.P, M.bullets, M.table, M.esc, M.fmt
RUNS = M.ROOT / "runs"
DEFAULTS = dict(epochs=100, epoch_samples=2_000_000, batch_size=512, lr=1e-3, warmup_steps=0, patience=10,
                steps_per_execution=64, seed=211)
ALLOW_TRAIN = {"ModelCheckpoint", "CSVLogger",  # Keras class names, not acronyms
               "TRAINING", "WITH", "TRUE", "ONLINE", "NOT", "ENTRENAMIENTO", "CON", "VERDADERO", "EN", "LINEA", "NO",
               "EPOCHS"}  # emphasis; EPOCHS: the legacy script constant
LANG = "en"


def t(en, es):
    return es if LANG == "es" else en


def num(n):  # thousands separator: comma in English, space in Spanish
    s = f"{n:,}"
    return s.replace(",", " ") if LANG == "es" else s


def dec(x, d=2):
    return f"{x:.{d}f}"


# ----------------------------------------------------------------------------- run facts
def history(run):
    h = RUNS / run / "history.csv"
    if not h.exists():
        return []
    return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(h))]


def facts(run):
    c = F["runs"][run]["cfg"]
    a = c["args"]
    h = history(run)
    n_tr = c["data"]["n_train_windows"]
    round1 = a.get("epoch_samples") is None
    ep_size = n_tr if round1 else min(a["epoch_samples"], n_tr)
    ep_size = (ep_size // a["batch_size"]) * a["batch_size"]
    x = dict(cfg=c, args=a, hist=h, round1=round1, ep_size=ep_size, n_tr=n_tr)
    if h:
        best = min(range(len(h)), key=lambda i: h[i]["val_loss"])
        lrs = [r["learning_rate"] for r in h]
        x.update(epochs=len(h), best=best + 1, best_val=h[best]["val_loss"], first_val=h[0]["val_loss"],
                 lr0=lrs[0], lr1=lrs[-1], halvings=sum(1 for p, q in zip(lrs, lrs[1:]) if q < p * 0.75),
                 seen=len(h) * ep_size, passes=len(h) * ep_size / n_tr)
    return x


def init_text(c):
    i = c.get("init_from")
    if not i:
        return t("from scratch (random initialisation, seed 211)", "desde cero (inicialización aleatoria, semilla 211)")
    parent = Path(i["run"]).name
    if i.get("mode") == "expand":
        return t(f"warm start from {parent}: every weight copied, the {len(i['new_features'])} new inputs "
                 f"({', '.join(i['new_features'])}) start at zero weight, so at step 0 it computes exactly the "
                 f"parent's function (checked: max difference {i['init_max_abs_diff_z']:.1e} z on real validation windows)",
                 f"arranque en caliente desde {parent}: se copian todos los pesos, las {len(i['new_features'])} "
                 f"entradas nuevas ({', '.join(i['new_features'])}) empiezan con peso cero, así que en el paso 0 "
                 f"calcula exactamente la función del padre (comprobado: diferencia máxima {i['init_max_abs_diff_z']:.1e} "
                 "z en ventanas reales de validación)")
    return t(f"fine-tune of {parent}: its weights, features, residual flag and normalisation; a fresh AdamW optimiser",
             f"ajuste fino de {parent}: sus pesos, características, indicador residual y normalización; un optimizador "
             "AdamW nuevo")


def command(run):
    c, a = F["runs"][run]["cfg"], F["runs"][run]["cfg"]["args"]
    parts = ["python src/train.py", f"--models {c['model']}"]
    if a.get("features"):
        parts.append("--features " + " ".join(a["features"]))
    if a.get("init_from"):
        parts.append(f"--init_from {a['init_from']}")
    if a.get("dad_filler"):
        parts += [f"--dad_filler {a['dad_filler']}", f"--dad_share {a['dad_share']}"]
    for k in ("lr", "epochs", "patience", "epoch_samples", "batch_size"):
        v = a.get(k)
        if v is not None and v != DEFAULTS[k]:
            parts.append(f"--{k} {v:g}" if isinstance(v, float) else f"--{k} {v}")
    parts.append(f"--tag {run}")
    return " ".join(parts)


def int8_text(run):
    c = F["runs"][run]["cfg"]
    vo, _ = M.val_open(run)
    vi, _ = M.val_open(run + "_int8")
    full = t(f"; full validation: int8 {fmt(vi)} vs float {fmt(vo)} ms ({100 * (vi / vo - 1):+.2f}%)",
             f"; validación completa: int8 {fmt(vi)} frente a float {fmt(vo)} ms ({100 * (vi / vo - 1):+.2f}%)") \
        if vi and vo else ""
    if "int8_calib" in c:
        ch, ref = c["int8_calib"]["chosen"], c["int8_calib"]["ref_20k_minmax"]
        when = c.get("int8_reexport", {}).get("date", "")[:10]
        how = t(f"re-exported {when} with " if when else "", f"reexportado el {when} con " if when else "")
        return t(f"{how}best-of-K calibration: chose draw {ch['name']} ({ch['n']} windows), int8-vs-float gap "
                 f"{ch['gap']:+.2f}% on the held-out training windows (plain 20k min/max: {ref['gap']:+.2f}%)",
                 f"{how}calibración del mejor de K: se eligió la extracción {ch['name']} ({ch['n']} ventanas), "
                 f"diferencia int8-frente-a-float {ch['gap']:+.2f}% en las ventanas de entrenamiento reservadas "
                 f"(min/max simple con 20k: {ref['gap']:+.2f}%)") + full
    if "val_int8_check" in c:
        ck = c["val_int8_check"]
        when = c.get("int8_reexport", {}).get("date", "")[:10]
        how = t(f"re-exported {when} after the 500-window calibration bug, " if when else "",
                f"reexportado el {when} tras el error de calibración con 500 ventanas, " if when else "")
        return t(f"{how}min/max calibration on {num(ck['calib_windows'])} random training windows; first 20k "
                 f"validation windows: {ck['rmse_ms_int8']:.2f} vs {ck['rmse_ms_float']:.2f} ms float",
                 f"{how}calibración min/max con {num(ck['calib_windows'])} ventanas de entrenamiento aleatorias; "
                 f"primeras 20k ventanas de validación: {ck['rmse_ms_int8']:.2f} frente a {ck['rmse_ms_float']:.2f} "
                 "ms en float") + full
    return t("exported by the round-1 train.py (its calibration settings were not recorded in config.json)",
             "exportado por el train.py de la ronda 1 (sus ajustes de calibración no quedaron en config.json)") + full


def curve(run, x):
    """Training and validation loss per epoch (z units), each on its own axis, best epoch dashed."""
    h = x["hist"]
    w, hgt, f = 16.0 * cm, 4.8 * cm, "Helvetica"
    d = Drawing(w, hgt)
    panels = [("loss", t("training loss (Huber, z)", "pérdida de entrenamiento (Huber, z)"), "#2B6CB0"),
              ("val_loss", t("validation loss (Huber, z)", "pérdida de validación (Huber, z)"), "#C05621")]
    pw = (w - 0.4 * cm) / 2
    for j, (key, label, col) in enumerate(panels):
        pts = [(i + 1, r[key]) for i, r in enumerate(h)]
        lp = LinePlot()
        lp.x, lp.y, lp.width, lp.height = j * pw + 1.5 * cm, 0.9 * cm, pw - 1.9 * cm, hgt - 1.6 * cm
        lp.data = [pts]
        lp.lines[0].strokeColor, lp.lines[0].strokeWidth = colors.HexColor(col), 1.2
        lo, hi = min(v for _, v in pts), max(v for _, v in pts)
        pad = 0.08 * (hi - lo or 1e-3)
        lp.yValueAxis.valueMin, lp.yValueAxis.valueMax = lo - pad, hi + pad
        lp.yValueAxis.labelTextFormat = "%.4f"
        lp.xValueAxis.valueMin, lp.xValueAxis.valueMax = 1, max(2, len(h))
        lp.xValueAxis.valueStep = max(1, len(h) // 8)
        for ax in (lp.xValueAxis, lp.yValueAxis):
            ax.labels.fontName, ax.labels.fontSize = f, 6.5
        d.add(lp)
        bx = lp.x + lp.width * (x["best"] - 1) / max(1, len(h) - 1)
        d.add(Line(bx, lp.y, bx, lp.y + lp.height, strokeColor=colors.HexColor("#888888"), strokeDashArray=[2, 2]))
        d.add(String(lp.x, hgt - 0.4 * cm, label, fontName=f, fontSize=7, fillColor=colors.HexColor(col)))
        d.add(String(lp.x + lp.width / 2 - 0.4 * cm, 0.1 * cm, t("epoch", "época"), fontName=f, fontSize=7))
    d.add(String(w, 0.1 * cm, t(f"dashed: best epoch ({x['best']})", f"discontinua: mejor época ({x['best']})"),
                 fontName=f, fontSize=7, fillColor=colors.HexColor("#666666"), textAnchor="end"))
    return d


def run_section(run, role, notes):
    """One run: settings table, learning curve, outcome, notes."""
    x = facts(run)
    c, a = x["cfg"], x["args"]
    feats = c["features"]
    fs = next((k for k, v in F["feature_sets"].items() if v == feats), None)
    rows = [
        (t("command", "comando"), f"<font face='Courier' size='6.8'>{esc(command(run))}</font>"),
        (t("start", "inicio"), init_text(c)),
        (t("inputs", "entradas"),
         t(f"{'30-beat RR sequence (z-RR, z-dRR) + ' if c['kind'] == 'seq' else ''}{len(feats)} tabular features"
           f"{' (set ' + fs + ')' if fs else ''}",
           f"{'secuencia RR de 30 latidos (z-RR, z-dRR) + ' if c['kind'] == 'seq' else ''}{len(feats)} características "
           f"tabulares{' (conjunto ' + fs + ')' if fs else ''}")),
        (t("target", "objetivo"),
         t("z-scored correction over the RLS forecast (dRR - rls_pred)", "corrección normalizada z sobre el pronóstico RLS "
           "(dRR - rls_pred)") if c["residual"] else t("z-scored dRR (no residual)", "dRR normalizado z (sin residual)")),
    ]
    if x["hist"]:
        ep = t(f"{num(x['ep_size'])} windows = one full pass over the {num(x['n_tr'])} training windows (round-1 train.py)",
               f"{num(x['ep_size'])} ventanas = una pasada completa por las {num(x['n_tr'])} ventanas de entrenamiento "
               "(train.py de la ronda 1)") if x["round1"] else \
            t(f"{num(x['ep_size'])} windows drawn at random with replacement ({a['batch_size']} x "
              f"{x['ep_size'] // a['batch_size']} steps)",
              f"{num(x['ep_size'])} ventanas extraídas al azar con reemplazo ({a['batch_size']} x "
              f"{x['ep_size'] // a['batch_size']} pasos)")
        ti = c.get("train_info") or {}
        ms = ti.get("ms_per_step_median")
        rows += [
            (t("epoch", "época"), ep),
            (t("budget", "presupuesto"),
             t(f"{x['epochs']} of at most {a['epochs']} epochs (early stopping, patience {a['patience']}); "
               f"{num(x['seen'])} windows drawn = {x['passes']:.1f} passes over the training set",
               f"{x['epochs']} de como máximo {a['epochs']} épocas (parada temprana, paciencia {a['patience']}); "
               f"{num(x['seen'])} ventanas extraídas = {x['passes']:.1f} pasadas por el conjunto de entrenamiento")),
            (t("learning rate", "tasa de aprendizaje"),
             t(f"{x['lr0']:g} at the start, halved {x['halvings']} times on validation plateaus, {x['lr1']:.2g} at the end",
               f"{x['lr0']:g} al inicio, reducida a la mitad {x['halvings']} veces en mesetas de validación, "
               f"{x['lr1']:.2g} al final")),
            (t("best epoch", "mejor época"),
             t(f"{x['best']} (validation loss {x['first_val']:.4f} after epoch 1, {x['best_val']:.4f} at best); "
               "its weights are the saved model", f"{x['best']} (pérdida de validación {x['first_val']:.4f} tras la "
               f"época 1, {x['best_val']:.4f} en la mejor); sus pesos son el modelo guardado")),
            (t("time", "tiempo"),
             t(f"{c['train_seconds'] / 3600:.2f} h on one GPU" + (f", {ms:.1f} ms per step" if ms else ""),
               f"{c['train_seconds'] / 3600:.2f} h en una GPU" + (f", {ms:.1f} ms por paso" if ms else ""))),
        ]
    if c.get("dad"):
        d = c["dad"]
        rows.append((t("closed-loop data", "datos de lazo cerrado"),
                     t(f"{Path(d['filler']).name} filled {num(d['n_train_segments'])} training segments of {d['seg_beats']} "
                       f"beats ({num(d['n_train_windows'])} windows, {d['drop_range'][0]:.0%}-{d['drop_range'][1]:.0%} "
                       f"dropped per segment, seed {d['seed']}) in {d['fill_seconds']:.0f} s on CPU; each training window "
                       f"is closed-loop with probability {d['share']} (realised {d['share_realised_100_batches']:.3f}); "
                       "early stopping on a 50/50 mix of ordinary and closed-loop validation windows",
                       f"{Path(d['filler']).name} rellenó {num(d['n_train_segments'])} segmentos de entrenamiento de "
                       f"{d['seg_beats']} latidos ({num(d['n_train_windows'])} ventanas, {d['drop_range'][0]:.0%}-"
                       f"{d['drop_range'][1]:.0%} eliminado por segmento, semilla {d['seed']}) en {d['fill_seconds']:.0f} s "
                       f"en CPU; cada ventana de entrenamiento es de lazo cerrado con probabilidad {d['share']} (real "
                       f"{d['share_realised_100_batches']:.3f}); parada temprana sobre una mezcla 50/50 de ventanas de "
                       "validación ordinarias y de lazo cerrado")))
    if c["framework"] == "keras":
        rows.append(("int8", int8_text(run)))
    vm, vr = c["val_metrics"]["rmse"], c["val_metrics_rls_ar"]["rmse"]
    res = t(f"{vm:.2f} ms on all {num(c['data']['n_val_windows'])} validation windows ({100 * (vm / vr - 1):+.1f}% vs "
            f"rls_ar {vr:.2f})", f"{vm:.2f} ms en las {num(c['data']['n_val_windows'])} ventanas de validación "
            f"({100 * (vm / vr - 1):+.1f}% frente a rls_ar {vr:.2f})")
    if "val_metrics_parent" in c:
        res += t(f"; parent {c['val_metrics_parent']['rmse']:.2f} ms", f"; padre {c['val_metrics_parent']['rmse']:.2f} ms")
    if "val_metrics_dad" in c:
        res += t(f"; closed-loop validation windows {c['val_metrics_dad']['rmse']:.2f} ms (parent "
                 f"{c['val_metrics_dad_parent']['rmse']:.2f}, rls_ar {c['val_metrics_dad_rls_ar']['rmse']:.2f})",
                 f"; ventanas de validación de lazo cerrado {c['val_metrics_dad']['rmse']:.2f} ms (padre "
                 f"{c['val_metrics_dad_parent']['rmse']:.2f}, rls_ar {c['val_metrics_dad_rls_ar']['rmse']:.2f})")
    rows.append((t("validation RMSE (float)", "RMSE de validación (float)"), res))
    out = [KeepTogether([P(f"{esc(run)} - {role}", "h2"),
                         table([t("item", "concepto"), t("value", "valor")], [(k, v) for k, v in rows],
                               widths=[3.0 * cm, 13.0 * cm], bold_first=True)])]
    if x["hist"]:
        out += [Spacer(1, 4), KeepTogether([curve(run, x)])]
    if notes:
        out += bullets(notes)
    return out


# ----------------------------------------------------------------------------- shared text
def pipeline_bullets():
    return [
        t("Data: RR series in data/series/*.txt split by subject (142 training, 16 validation; one-recording subjects go "
          "to validation). Training uses the training subjects only; validation monitors early stopping and is scored at "
          "the end. The 10 test subjects were never opened during training.",
          "Datos: series RR en data/series/*.txt divididas por sujeto (142 de entrenamiento, 16 de validación; los "
          "sujetos con un solo registro van a validación). El entrenamiento usa solo los sujetos de entrenamiento; la "
          "validación vigila la parada temprana y se evalúa al final. Los 10 sujetos de test nunca se abrieron durante "
          "el entrenamiento."),
        t("Windows: the streaming engine runs over every series; each beat with a full 30-beat window gives one "
          "example (features of beat n, target dRR[n+1]). The first 30 windows of every series are dropped (cold RLS). "
          "20,755,925 training and 1,421,225 validation windows.",
          "Ventanas: el motor en streaming recorre cada serie; cada latido con una ventana completa de 30 latidos da un "
          "ejemplo (características del latido n, objetivo dRR[n+1]). Se descartan las primeras 30 ventanas de cada "
          "serie (RLS en frío). 20 755 925 ventanas de entrenamiento y 1 421 225 de validación."),
    ]


def doc_pipeline():
    tc = F["runs"]["tcn_lite_ft"]["cfg"]
    story = [
        P(t("Every learned model except the legacy TCN_MHA was trained by src/train.py with the procedure below. The "
            "per-family documents give each run's settings, learning curve and outcome.",
            "Todos los modelos aprendidos excepto el TCN_MHA heredado se entrenaron con src/train.py siguiendo el "
            "procedimiento de abajo. Los documentos por familia dan los ajustes, la curva de aprendizaje y el resultado "
            "de cada ejecución."), "box"),
        P(t("1. Data and examples", "1. Datos y ejemplos"), "h1"),
        *bullets(pipeline_bullets() + [
            t("Normalisation: z-score statistics (RR, dRR, each feature, the target) are fitted on the training subjects "
              "only and stored in the run's config.json; a zero variance gives std 1.",
              "Normalización: las estadísticas z (RR, dRR, cada característica, el objetivo) se ajustan solo con los "
              "sujetos de entrenamiento y se guardan en el config.json de la ejecución; una varianza cero da desviación 1."),
            t("Target: dRR[n+1] = RR[n+1] - RR[n]. Residual models (all but linear) learn the z-scored correction "
              "dRR - rls_pred; the prediction adds rls_pred back.",
              "Objetivo: dRR[n+1] = RR[n+1] - RR[n]. Los modelos residuales (todos menos linear) aprenden la corrección "
              "normalizada z dRR - rls_pred; la predicción vuelve a sumar rls_pred."),
            t("RAM: the (N, 30, 2) sequence tensor is never built. Each window is stored as its features, its target and "
              "the start index into the concatenated RR series; batches are gathered on the fly by tf.data, and an "
              "assert checks that this path equals the one used at inference (utils.make_inputs).",
              "RAM: el tensor de secuencias (N, 30, 2) nunca se construye. Cada ventana se guarda como sus "
              "características, su objetivo y el índice de inicio en la serie RR concatenada; los lotes se forman al "
              "vuelo con tf.data, y una aserción comprueba que este camino es igual al usado en inferencia "
              "(utils.make_inputs)."),
        ]),
        P(t("2. Optimisation (Keras models: mlp, gru, micro_tcn, tcn_lite)", "2. Optimización (modelos Keras: mlp, gru, "
            "micro_tcn, tcn_lite)"), "h1"),
        table([t("setting", "ajuste"), t("value", "valor")], [
            (t("optimiser", "optimizador"), t("AdamW, weight decay 1e-4", "AdamW, decaimiento de pesos 1e-4")),
            (t("loss", "pérdida"), t("Huber, delta 1 (in z units: quadratic up to about one target standard deviation, "
                                     "linear beyond, so outlier beats weigh less than with MSE)",
                                     "Huber, delta 1 (en unidades z: cuadrática hasta aproximadamente una desviación "
                                     "estándar del objetivo y lineal más allá, así que los latidos atípicos pesan menos "
                                     "que con MSE)")),
            (t("batch", "lote"), t("512 windows; 64 steps per Keras call (steps_per_execution) to remove the per-step "
                                   "overhead that dominates small models", "512 ventanas; 64 pasos por llamada de Keras "
                                   "(steps_per_execution) para eliminar la sobrecarga por paso que domina en modelos "
                                   "pequeños")),
            (t("epoch", "época"), t("from round 2 on: 2,000,000 windows drawn uniformly at random WITH replacement "
                                    "(3,906 steps, about 10% of the data, a fresh subset every epoch). Round-1 runs "
                                    "(linear, mlp, gbdt, micro_tcn, tcn_lite): one full pass over all training windows",
                                    "desde la ronda 2: 2 000 000 de ventanas extraídas uniformemente al azar CON "
                                    "reemplazo (3906 pasos, alrededor del 10% de los datos, un subconjunto nuevo en cada "
                                    "época). Ejecuciones de la ronda 1 (linear, mlp, gbdt, micro_tcn, tcn_lite): una "
                                    "pasada completa por todas las ventanas de entrenamiento")),
            (t("learning rate", "tasa de aprendizaje"),
             t("1e-3 from scratch, 3e-4 for fine-tunes and warm starts; ReduceLROnPlateau halves it when the validation "
               "loss stops improving (patience max(2, patience / 3) epochs), floor 1e-5",
               "1e-3 desde cero, 3e-4 para ajustes finos y arranques en caliente; ReduceLROnPlateau la divide por dos "
               "cuando la pérdida de validación deja de mejorar (paciencia max(2, paciencia / 3) épocas), mínimo 1e-5")),
            (t("stopping", "parada"), t("EarlyStopping on the validation loss, patience 10 epochs (6 for fine-tunes), "
                                        "restoring the best epoch's weights; at most 100 epochs (30 for fine-tunes, 90 "
                                        "for gru)", "EarlyStopping sobre la pérdida de validación, paciencia de 10 épocas "
                                        "(6 en los ajustes finos), restaurando los pesos de la mejor época; como máximo "
                                        "100 épocas (30 en los ajustes finos, 90 en gru)")),
            (t("validation monitor", "monitor de validación"),
             t("a fixed random subset of 200,000 validation windows each epoch (round 2 on); the final metrics use all "
               "validation windows", "un subconjunto aleatorio fijo de 200 000 ventanas de validación en cada época "
               "(desde la ronda 2); las métricas finales usan todas las ventanas de validación")),
            (t("reproducibility", "reproducibilidad"),
             t("seed 211 for Python, NumPy, Keras and the window draws; deterministic TensorFlow ops; the data manifests' "
               "md5 and the library versions are stored in config.json", "semilla 211 para Python, NumPy, Keras y las "
               "extracciones de ventanas; operaciones de TensorFlow deterministas; el md5 de los manifiestos de datos y "
               "las versiones de las bibliotecas se guardan en config.json")),
            (t("hardware", "hardware"),
             t(f"one GPU (TensorFlow {tc['versions']['tensorflow']}), one job at a time through the jobs/ queue; "
               "per-step time 3 ms (mlp) to 25 ms (gru)", f"una GPU (TensorFlow {tc['versions']['tensorflow']}), un "
               "trabajo cada vez mediante la cola jobs/; tiempo por paso de 3 ms (mlp) a 25 ms (gru)")),
        ], widths=[3.4 * cm, 12.6 * cm], bold_first=True),
        P(t("3. Non-neural models", "3. Modelos no neuronales"), "h1"),
        *bullets([
            t("linear: ridge regression (penalty 1e-3, intercept not penalised) solved in closed form from float64 "
              "normal equations accumulated in chunks of 1M windows; no epochs, 0.6 s.",
              "linear: regresión ridge (penalización 1e-3, término independiente sin penalizar) resuelta de forma "
              "cerrada con ecuaciones normales en float64 acumuladas en bloques de 1M ventanas; sin épocas, 0.6 s."),
            t("gbdt: scikit-learn HistGradientBoostingRegressor on CPU, early stopping on the validation windows "
              "(see 04_gbdt).", "gbdt: HistGradientBoostingRegressor de scikit-learn en CPU, parada temprana sobre las "
              "ventanas de validación (ver 04_gbdt)."),
        ]),
        P(t("4. After training (every run)", "4. Después del entrenamiento (cada ejecución)"), "h1"),
        *bullets([
            t("The saved model is reloaded from disk and scored in ms on all validation windows, next to rls_ar (and the "
              "parent for fine-tunes); everything goes to runs/&lt;tag&gt;/config.json, the per-epoch curve to history.csv.",
              "El modelo guardado se recarga desde disco y se evalúa en ms sobre todas las ventanas de validación, junto "
              "a rls_ar (y al padre en los ajustes finos); todo va a runs/&lt;tag&gt;/config.json y la curva por época a "
              "history.csv."),
            t("int8 export (Keras): full-integer post-training quantisation with TFLite (int8 weights, activations, input "
              "and output). Since 2026-10-08 the calibration is best-of-K: 8 fixed draws (2,000 and 5,000 training "
              "windows x 4 seeds) plus a plain 20,000-window reference are exported, and the draw with the smallest "
              "int8-vs-float gap on held-out TRAINING windows is kept: 20 training subjects excluded from every draw, "
              "20,000 ordinary + 20,000 closed-loop windows filled by the run's own float model. An assert requires "
              "best-of-K to be no worse than the reference. Validation never takes part in the choice.",
              "Exportación int8 (Keras): cuantización post-entrenamiento totalmente entera con TFLite (pesos, "
              "activaciones, entrada y salida int8). Desde el 2026-10-08 la calibración es del mejor de K: se exportan 8 "
              "extracciones fijas (2000 y 5000 ventanas de entrenamiento x 4 semillas) más una referencia simple de "
              "20 000 ventanas, y se conserva la extracción con la menor diferencia int8-frente-a-float en ventanas de "
              "ENTRENAMIENTO reservadas: 20 sujetos de entrenamiento excluidos de todas las extracciones, 20 000 ventanas "
              "ordinarias + 20 000 de lazo cerrado rellenadas por el propio modelo float. Una aserción exige que el mejor "
              "de K no sea peor que la referencia. La validación nunca participa en la elección."),
            t("Calibration history: round-1 runs were exported by the first train.py; from 2026-10-07 18:14 a bug "
              "calibrated on only 500 windows (int8 output saturated, gaps 3.0-3.7%), fixed the same evening with 20,000 "
              "windows and train.py --reexport_int8; best-of-K replaced it on 2026-10-08 for the round-3/4 finalists. "
              "Each run's document says which export it has.",
              "Historia de la calibración: las ejecuciones de la ronda 1 se exportaron con el primer train.py; desde el "
              "2026-10-07 a las 18:14 un error calibraba con solo 500 ventanas (salida int8 saturada, diferencias del "
              "3.0-3.7%), corregido esa misma tarde con 20 000 ventanas y train.py --reexport_int8; el mejor de K lo "
              "sustituyó el 2026-10-08 para los finalistas de las rondas 3/4. El documento de cada ejecución indica qué "
              "exportación tiene."),
        ]),
        P(t("5. Training variants used in later rounds", "5. Variantes de entrenamiento usadas en rondas posteriores"), "h1"),
        *bullets([
            t("<b>Fine-tune</b> (--init_from): continue from a trained run's weights with lr 3e-4, at most 30 epochs, "
              "patience 6 (tcn_lite_ft).", "<b>Ajuste fino</b> (--init_from): continuar desde los pesos de una ejecución "
              "entrenada con lr 3e-4, como máximo 30 épocas, paciencia 6 (tcn_lite_ft)."),
            t("<b>Warm start with extra inputs</b> (--init_from + --features): the new inputs get zero weights and their "
              "own training-fitted normalisation, so training starts from the parent's exact function (_rsa_ws runs).",
              "<b>Arranque en caliente con entradas extra</b> (--init_from + --features): las entradas nuevas reciben "
              "pesos cero y su propia normalización ajustada en entrenamiento, así que el entrenamiento parte de la "
              "función exacta del padre (ejecuciones _rsa_ws)."),
            t("<b>Closed-loop data, DAD</b> (--dad_filler, --dad_share 0.5): the training series are cut into 5000-beat "
              "segments, 10-80% of the beats are dropped and filled by the parent model as on the pacemaker, the engine "
              "recomputes the features on the filled series and the target stays the TRUE next beat (rr_true[n+1] - "
              "rr_filled[n]); half of the batches' windows come from these segments (_dad runs). A first attempt only "
              "reached the first 429k windows of each half (a tf.data index-range bug) and was deleted; a check now stops "
              "training if the draws cannot reach every window.",
              "<b>Datos de lazo cerrado, DAD</b> (--dad_filler, --dad_share 0.5): las series de entrenamiento se cortan "
              "en segmentos de 5000 latidos, se elimina el 10-80% de los latidos y los rellena el modelo padre como en el "
              "marcapasos, el motor recalcula las características sobre la serie rellenada y el objetivo sigue siendo el "
              "latido siguiente VERDADERO (rr_true[n+1] - rr_filled[n]); la mitad de las ventanas de los lotes viene de "
              "estos segmentos (ejecuciones _dad). Un primer intento solo alcanzaba las primeras 429k ventanas de cada "
              "mitad (un error de rango de índices en tf.data) y se borró; ahora una comprobación detiene el "
              "entrenamiento si las extracciones no pueden alcanzar todas las ventanas."),
        ]),
        P(t("Documents in this folder (each in English _en and Spanish _es)",
            "Documentos de esta carpeta (cada uno en inglés _en y en español _es)"), "h1"),
        table([t("file", "archivo"), t("content", "contenido")], [
            ("00_training_pipeline", t("this document: the shared procedure", "este documento: el procedimiento común")),
            ("01_baselines", t("persistence, ema_mean, rls_ar: not trained; how rls_ar adapts online",
                               "persistence, ema_mean, rls_ar: sin entrenamiento; cómo se adapta rls_ar en línea")),
            ("02_linear", "linear"), ("03_mlp", "mlp, mlp_nobp, mlp_rsa"), ("04_gbdt", "gbdt, gbdt_all"),
            ("05_gru", "gru"),
            ("06_micro_tcn", "micro_tcn, micro_tcn_f8, micro_tcn_dad, micro_tcn_rsa, micro_tcn_rsa_ws"),
            ("07_tcn_lite", t("tcn_lite, tcn_lite_ft (selected), tcn_lite_dad, tcn_lite_rsa, tcn_lite_rsa_ws",
                              "tcn_lite, tcn_lite_ft (seleccionado), tcn_lite_dad, tcn_lite_rsa, tcn_lite_rsa_ws")),
            ("08_tcn_mha", t("the legacy TCN_MHA (trained outside this project)",
                             "el TCN_MHA heredado (entrenado fuera de este proyecto)")),
        ], widths=[4.2 * cm, 11.8 * cm], bold_first=True),
    ]
    build("00_training_pipeline", t("Training pipeline", "Proceso de entrenamiento"),
          t("How every model in this project was trained: data, optimisation, int8 export and the training variants.",
            "Cómo se entrenó cada modelo del proyecto: datos, optimización, exportación int8 y variantes de "
            "entrenamiento."), story)


def doc_baselines():
    e = F["engine"]
    story = [
        P(t("The baselines are not trained: they have no parameters fitted on the training split. They are what every "
            "learned model is compared against.", "Las referencias no se entrenan: no tienen parámetros ajustados con el "
            "conjunto de entrenamiento. Son aquello contra lo que se compara cada modelo aprendido."), "box"),
        *bullets([
            t("<b>persistence</b>: predicts dRR[n+1] = 0 (the next RR repeats the last one). Nothing to fit.",
              "<b>persistence</b>: predice dRR[n+1] = 0 (el siguiente RR repite el último). Nada que ajustar."),
            t("<b>ema_mean</b>: the next RR is an exponential moving average of the window (factor 0.75 per beat); the "
              "factor is fixed, not fitted.", "<b>ema_mean</b>: el siguiente RR es una media móvil exponencial de la "
              "ventana (factor 0.75 por latido); el factor es fijo, no ajustado."),
            t(f"<b>rls_ar</b>: an AR({e['ar_order']}) model whose weights are learned ONLINE, per patient, beat by beat, by "
              f"recursive least squares, never on the training split. It starts at zero weights (it predicts 0, like "
              f"persistence) with covariance {e['rls_p0']:g} I. When each new beat arrives, the error of the previous "
              f"forecast updates the 8 weights through the RLS gain; the forgetting factor {e['rls_lambda']} weights the "
              "past exponentially (about 100 beats of memory). The regressors are the last 8 RR minus the 30-beat window "
              f"mean, divided by {e['rls_scale']:g}; the forecast is clamped to +-{e['rls_pred_clamp']:g} ms.",
              f"<b>rls_ar</b>: un modelo AR({e['ar_order']}) cuyos pesos se aprenden EN LÍNEA, por paciente, latido a "
              "latido, por mínimos cuadrados recursivos, nunca con el conjunto de entrenamiento. Empieza con pesos cero "
              f"(predice 0, como persistence) y covarianza {e['rls_p0']:g} I. Al llegar cada latido nuevo, el error del "
              "pronóstico anterior actualiza los 8 pesos mediante la ganancia del RLS; el factor de olvido "
              f"{e['rls_lambda']} pondera el pasado exponencialmente (unos 100 latidos de memoria). Los regresores son "
              f"los últimos 8 RR menos la media de la ventana de 30 latidos, divididos por {e['rls_scale']:g}; el "
              f"pronóstico se recorta a +-{e['rls_pred_clamp']:g} ms."),
            t("Its only design choices (order 8, forgetting 0.99, initial covariance, trace cap 1e4, clamp) are fixed in "
              "the ENGINE settings of src/utils.py; changing them would change the rls_pred feature and invalidate every "
              "trained run. The stability guards (skip the update when the covariance loses positive definiteness, keep "
              "it symmetric, reset it if its trace becomes non-finite) were added on 2026-10-07, after an unguarded RLS "
              "diverged after about 3,200 beats on real data.",
              "Sus únicas decisiones de diseño (orden 8, olvido 0.99, covarianza inicial, límite de traza 1e4, recorte) "
              "están fijadas en los ajustes ENGINE de src/utils.py; cambiarlas cambiaría la característica rls_pred e "
              "invalidaría todas las ejecuciones entrenadas. Las protecciones de estabilidad (omitir la actualización "
              "cuando la covarianza pierde la definición positiva, mantenerla simétrica, reiniciarla si su traza deja de "
              "ser finita) se añadieron el 2026-10-07, después de que un RLS sin protección divergiera tras unos 3200 "
              "latidos en datos reales."),
            t("Its forecast is also the rls_pred feature and the residual base: every residual model is trained to correct "
              "it, so rls_ar's online learning runs inside every model as well.",
              "Su pronóstico es también la característica rls_pred y la base residual: cada modelo residual se entrena "
              "para corregirlo, así que el aprendizaje en línea de rls_ar se ejecuta también dentro de cada modelo."),
        ]),
        P(t("Validation reference", "Referencia de validación"), "h1"),
        table([t("baseline", "referencia"), t("validation RMSE, open loop (ms)", "RMSE de validación, lazo abierto (ms)")],
              [(m, fmt(M.val_open(m)[0])) for m in ("persistence", "ema_mean", "rls_ar")],
              widths=[4 * cm, 6 * cm], bold_first=True),
    ]
    build("01_baselines", t("Baselines (not trained)", "Referencias (sin entrenamiento)"),
          t("persistence, ema_mean and rls_ar: no offline training; rls_ar learns online per patient.",
            "persistence, ema_mean y rls_ar: sin entrenamiento previo; rls_ar aprende en línea por paciente."), story)


def doc_family(fname, title, subtitle, intro, how, runs, verdict):
    story = [P(intro, "box"), P(t("How this family is trained", "Cómo se entrena esta familia"), "h1"), *bullets(how),
             P(t("Runs", "Ejecuciones"), "h1")]
    for run, role, notes in runs:
        story += run_section(run, role, notes)
    story += [P(t("What the training showed", "Qué mostró el entrenamiento"), "h1"), *bullets(verdict)]
    build(fname, title, subtitle, story)


def doc_linear():
    c = F["runs"]["linear"]["cfg"]
    lin = c["linear"]
    story = [
        P(t("A global ridge regression, solved in closed form in 0.6 s. No epochs and no int8 export (10 parameters).",
            "Una regresión ridge global, resuelta de forma cerrada en 0.6 s. Sin épocas ni exportación int8 (10 "
            "parámetros)."), "box"),
        P(t("How it is trained", "Cómo se entrena"), "h1"),
        *bullets([
            t(f"Inputs: the 'ar' feature set, {', '.join(c['features'])}, z-scored with training statistics.",
              f"Entradas: el conjunto de características 'ar', {', '.join(c['features'])}, normalizadas z con "
              "estadísticas de entrenamiento."),
            t("Target: dRR[n+1] z-scored, NOT residual (the model is a global AR, the counterpart of the per-patient "
              "rls_ar).", "Objetivo: dRR[n+1] normalizado z, NO residual (el modelo es un AR global, la contrapartida "
              "del rls_ar por paciente)."),
            t("Solver: A = [features, 1] for all 20,755,925 training windows; A'A and A'z are accumulated in float64 in "
              "chunks of 1M windows (no 20M x 10 float64 copy), then w = solve(A'A + 1e-3 I, A'z) with the intercept left "
              "unpenalised. Exact least squares in one pass, so no learning rate, epochs or early stopping.",
              "Resolución: A = [características, 1] para las 20 755 925 ventanas de entrenamiento; A'A y A'z se acumulan "
              "en float64 por bloques de 1M ventanas (sin copia float64 de 20M x 10), y después w = solve(A'A + 1e-3 I, "
              "A'z) dejando sin penalizar el término independiente. Mínimos cuadrados exactos en una pasada, así que no "
              "hay tasa de aprendizaje, épocas ni parada temprana."),
            t(f"Command: python src/train.py --models linear. Training time {c['train_seconds']} s on CPU.",
              f"Comando: python src/train.py --models linear. Tiempo de entrenamiento {c['train_seconds']} s en CPU."),
        ]),
        P(t("Fitted weights (z units)", "Pesos ajustados (unidades z)"), "h1"),
        table([t("input", "entrada"), t("weight", "peso")],
              [(f, f"{w:+.4f}") for f, w in zip(c["features"], lin["coef"])] + [(t("intercept", "término independiente"),
                                                                                  f"{lin['intercept']:+.4f}")],
              widths=[4 * cm, 4 * cm], bold_first=True),
        P(t("Outcome", "Resultado"), "h1"),
        *bullets([t(f"Validation RMSE {c['val_metrics']['rmse']:.2f} ms vs rls_ar {c['val_metrics_rls_ar']['rmse']:.2f}: "
                    "worse than the adaptive baseline overall and in every age interval (up to +24.5%), so rejected. A "
                    "single set of weights for all patients cannot follow per-patient dynamics.",
                    f"RMSE de validación {c['val_metrics']['rmse']:.2f} ms frente a rls_ar "
                    f"{c['val_metrics_rls_ar']['rmse']:.2f}: peor que la referencia adaptativa en total y en todos los "
                    "intervalos de edad (hasta +24.5%), así que se rechazó. Un único conjunto de pesos para todos los "
                    "pacientes no puede seguir la dinámica de cada uno.")]),
    ]
    build("02_linear", t("Training: linear (ridge)", "Entrenamiento: linear (ridge)"),
          t("Closed-form ridge regression on the lagged dRR.", "Regresión ridge de forma cerrada sobre los dRR retardados."),
          story)


def doc_gbdt():
    rows = []
    for run in ("gbdt", "gbdt_all"):
        c = F["runs"][run]["cfg"]
        rows.append((run, str(len(c["features"])), str(c["complexity"]["trees"]), f"{c['train_seconds']} s",
                     f"{c['val_metrics']['rmse']:.2f}", f"{c['val_metrics_rls_ar']['rmse']:.2f}"))
    story = [
        P(t("Gradient-boosted trees, fitted on CPU in about 2 minutes. They are an accuracy and feature-importance "
            "reference only: 500 trees are far too large for the microcontroller.",
            "Árboles potenciados por gradiente, ajustados en CPU en unos 2 minutos. Son solo una referencia de precisión "
            "y de importancia de características: 500 árboles son demasiado grandes para el microcontrolador."), "box"),
        P(t("How it is trained", "Cómo se entrena"), "h1"),
        *bullets([
            t("scikit-learn HistGradientBoostingRegressor: squared error, learning rate 0.05, at most 500 trees of depth "
              "6 and 31 leaves, at least 100 windows per leaf, L2 1.0, random_state 211.",
              "HistGradientBoostingRegressor de scikit-learn: error cuadrático, tasa de aprendizaje 0.05, como máximo 500 "
              "árboles de profundidad 6 y 31 hojas, al menos 100 ventanas por hoja, L2 1.0, random_state 211."),
            t("Data: all 20,755,925 training windows (features binned into histograms by the library), residual target "
              "dRR - rls_pred in z units. Early stopping on the subject-disjoint validation windows (passed as X_val), "
              "after 20 trees without improvement; it never triggered: both runs used all 500 trees.",
              "Datos: las 20 755 925 ventanas de entrenamiento (la biblioteca agrupa las características en "
              "histogramas), objetivo residual dRR - rls_pred en unidades z. Parada temprana sobre las ventanas de "
              "validación de sujetos disjuntos (pasadas como X_val), tras 20 árboles sin mejora; nunca se activó: ambas "
              "ejecuciones usaron los 500 árboles."),
            t("After fitting: permutation importance on up to 50,000 validation windows (5 repeats) and the Spearman "
              "correlation of all engine features on training windows, saved as feature_importance.csv and "
              "feature_correlation.csv in the run folder.",
              "Tras el ajuste: importancia por permutación sobre hasta 50 000 ventanas de validación (5 repeticiones) y la "
              "correlación de Spearman de todas las características del motor sobre ventanas de entrenamiento, guardadas "
              "como feature_importance.csv y feature_correlation.csv en la carpeta de la ejecución."),
            t("Commands: python src/train.py --models gbdt; python src/train.py --models gbdt --features all --tag gbdt_all "
              "(run on CPU).", "Comandos: python src/train.py --models gbdt; python src/train.py --models gbdt --features "
              "all --tag gbdt_all (ejecutados en CPU)."),
        ]),
        P(t("Runs", "Ejecuciones"), "h1"),
        table([t("run", "ejecución"), t("features", "características"), t("trees", "árboles"), t("fit time", "ajuste"),
               t("val RMSE (ms)", "RMSE val (ms)"), "rls_ar (ms)"], rows,
              widths=[2.6 * cm, 2.4 * cm, 2 * cm, 2.2 * cm, 2.6 * cm, 2.4 * cm], bold_first=True),
        P(t("What the training showed", "Qué mostró el entrenamiento"), "h1"),
        *bullets([
            t("Both runs reached the 500-tree cap: more trees might still help, but the model is already about 238 KB at "
              "8 B per node (budget 32 KB), so it was not pursued.",
              "Ambas ejecuciones llegaron al tope de 500 árboles: más árboles quizá ayudarían, pero el modelo ya ocupa "
              "unos 238 KB a 8 B por nodo (presupuesto 32 KB), así que no se continuó."),
            t("gbdt_all (every engine feature) is only 0.04 ms better: its permutation importance ranks rls_pred first by "
              "far (0.118 z), then rr_dev, drr_lag0, guzik and bp; the 8 RLS weights add nothing.",
              "gbdt_all (todas las características del motor) es solo 0.04 ms mejor: su importancia por permutación sitúa "
              "rls_pred primera con mucha diferencia (0.118 z), después rr_dev, drr_lag0, guzik y bp; los 8 pesos del RLS "
              "no aportan nada."),
        ]),
    ]
    build("04_gbdt", t("Training: gbdt (gradient-boosted trees)", "Entrenamiento: gbdt (árboles potenciados por gradiente)"),
          t("HistGradientBoosting fitted on the tabular features, with early stopping on validation.",
            "HistGradientBoosting ajustado sobre las características tabulares, con parada temprana en validación."), story)


def doc_tcn_mha():
    vo, _ = M.val_open("tcn_mha_current")
    story = [
        P(t("TCN_MHA was not trained by this project. It is the previous production model, trained by the script "
            "legacy/TCN_MHA.py (also legacy/TCN_MHA.ipynb) on the old pipeline's h5 datasets, and loaded here only for "
            "comparison. Its training history was not saved, so only the script's settings can be documented.",
            "TCN_MHA no se entrenó en este proyecto. Es el modelo de producción anterior, entrenado con el script "
            "legacy/TCN_MHA.py (también legacy/TCN_MHA.ipynb) sobre los conjuntos h5 del pipeline antiguo, y aquí solo "
            "se carga para comparar. Su historial de entrenamiento no se guardó, así que solo se pueden documentar los "
            "ajustes del script."), "box"),
        P(t("Training settings in legacy/TCN_MHA.py", "Ajustes de entrenamiento en legacy/TCN_MHA.py"), "h1"),
        table([t("setting", "ajuste"), t("value", "valor")], [
            (t("data", "datos"), t("data/hrv_dataset.h5 (training) and data/hrv_validation.h5 (validation), loaded "
                                   "entirely into RAM and copied to the GPU; built by the old preprocessing, whose subject "
                                   "split was not re-checked against this project's",
                                   "data/hrv_dataset.h5 (entrenamiento) y data/hrv_validation.h5 (validación), cargados "
                                   "enteros en RAM y copiados a la GPU; construidos por el preprocesado antiguo, cuya "
                                   "división por sujetos no se volvió a comprobar frente a la de este proyecto")),
            (t("inputs", "entradas"), t("the 30 RR of the window (one channel) + 12 static features: rmssd, rs, ccm, "
                                        "ccm_n5, guzik, nn20, porta, ar_1..ar_5 (Burg AR(5) coefficients), as stored in "
                                        "the h5 files", "los 30 RR de la ventana (un canal) + 12 características "
                                        "estáticas: rmssd, rs, ccm, ccm_n5, guzik, nn20, porta, ar_1..ar_5 (coeficientes "
                                        "AR(5) de Burg), tal como están en los archivos h5")),
            (t("target", "objetivo"), t("dRR[n+1] as stored in the h5 files (no RLS residual)",
                                        "dRR[n+1] tal como está en los archivos h5 (sin residual RLS)")),
            (t("loss", "pérdida"), t("RSAPhaseAwareLoss = Huber (delta 1) + 0.2 x mean(relu(-y_true y_pred)) (a sign "
                                     "penalty) + 0.1 x |var(y_true) - var(y_pred)| over the batch (a variance-matching "
                                     "penalty)", "RSAPhaseAwareLoss = Huber (delta 1) + 0.2 x mean(relu(-y_true y_pred)) "
                                     "(penalización de signo) + 0.1 x |var(y_true) - var(y_pred)| sobre el lote "
                                     "(penalización de ajuste de varianza)")),
            (t("optimiser", "optimizador"), t("AdamW, learning rate 5e-4, weight decay 1e-4",
                                              "AdamW, tasa de aprendizaje 5e-4, decaimiento de pesos 1e-4")),
            (t("batch, epochs", "lote, épocas"), t("batch 128 (chosen 'to preserve local gradient variance in the custom "
                                                   "loss'); EPOCHS = 3 in the script as it is in the repository",
                                                   "lote 128 (elegido 'para preservar la varianza local del gradiente en "
                                                   "la pérdida personalizada'); EPOCHS = 3 en el script tal como está en "
                                                   "el repositorio")),
            (t("callbacks", "callbacks"), t("EarlyStopping (patience 12, restore best), ReduceLROnPlateau (factor 0.5, "
                                            "patience 5, min 1e-6), ModelCheckpoint of the best val_loss to "
                                            "tcn_attention_hrv_best.keras", "EarlyStopping (paciencia 12, restaurar el "
                                            "mejor), ReduceLROnPlateau (factor 0.5, paciencia 5, mínimo 1e-6), "
                                            "ModelCheckpoint del mejor val_loss en tcn_attention_hrv_best.keras")),
            (t("regularisation", "regularización"), t("SpatialDropout1D(0.1) in every residual block, weight decay",
                                                      "SpatialDropout1D(0.1) en cada bloque residual, decaimiento de pesos")),
            (t("reproducibility", "reproducibilidad"), t("seed 211, deterministic TensorFlow ops; the saved weights are "
                                                         "dated 2026-10-06 22:01", "semilla 211, operaciones de TensorFlow "
                                                         "deterministas; los pesos guardados tienen fecha 2026-10-06 22:01")),
        ], widths=[3.2 * cm, 12.8 * cm], bold_first=True),
        P(t("Differences from this project's training", "Diferencias con el entrenamiento de este proyecto"), "h1"),
        *bullets([
            t("Its features are recomputed per window by legacy/utils2.py (including a Burg AR fit), not by the streaming "
              "engine, so they cannot run beat by beat on the microcontroller.",
              "Sus características se recalculan en cada ventana con legacy/utils2.py (incluido un ajuste AR de Burg), no "
              "con el motor en streaming, así que no pueden ejecutarse latido a latido en el microcontrolador."),
            t("No RLS residual: it learns the whole dRR, not a correction of an adaptive forecast.",
              "Sin residual RLS: aprende el dRR completo, no una corrección de un pronóstico adaptativo."),
            t("The variance term of its loss rewards predictions that oscillate as much as the true dRR; with shuffled "
              "batches it mostly measures the spread between subjects. This project dropped it for plain Huber, and it "
              "is a plausible cause of the RSA peaks TCN_MHA invents in the closed loop.",
              "El término de varianza de su pérdida premia predicciones que oscilan tanto como el dRR verdadero; con "
              "lotes barajados mide sobre todo la dispersión entre sujetos. Este proyecto lo abandonó por Huber simple, y "
              "es una causa plausible de los picos de RSA que TCN_MHA inventa en el lazo cerrado."),
            t(f"Evaluated here as a float model only (no int8 export): validation open loop {fmt(vo)} ms.",
              f"Evaluado aquí solo como modelo float (sin exportación int8): validación en lazo abierto {fmt(vo)} ms."),
        ]),
    ]
    build("08_tcn_mha", t("Training: TCN_MHA (legacy)", "Entrenamiento: TCN_MHA (heredado)"),
          t("How the previous production model was trained, from the legacy script.",
            "Cómo se entrenó el modelo de producción anterior, según el script heredado."), story)


# ----------------------------------------------------------------------------- build
def build(stem, title, subtitle, story):
    fname = f"{stem}_{LANG}.pdf"
    foot = t("HRV dRR predictor", "Predictor de dRR para HRV")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"{foot} - {title} - {DATE}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, t(f"page {doc.page}", f"página {doc.page}"))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title=title,
                            author=t("HRV dRR project", "Proyecto HRV dRR"), subject=subtitle, lang=LANG)
    story = [P(esc(title), "title"), P(subtitle, "sub")] + story
    text = GE._ascii(M.G.plain(" ".join(M._texts(story)) + f" {foot} - {title}"))
    missing = [x for x in (GE.undefined_tokens(text) if LANG == "es" else M.G.undefined_tokens(text))
               if x not in ALLOW_TRAIN and not re.fullmatch(r"n\d+", x)]  # n2000: calibration draw names
    if missing:  # runnable check: every acronym in the PDF must be defined in tools/glossary.py
        raise ValueError(f"{fname}: undefined acronyms {missing}; add them to tools/glossary.py and glossary_es.py")
    used = GE.used_terms(text) if LANG == "es" else M.G.used_terms(text)
    desc = MES.FEATURE_DESC if LANG == "es" else M.FEATURE_DESC
    feat_terms = [("drr_lag0..7" if k == "drr_lag" else "rls_w0..7" if k == "rls_w" else k,
                   t(f"feature: {d}", f"característica: {d}"))
                  for k, d in desc.items() if (k in text if k in ("drr_lag", "rls_w") else M.G._present(k, "word", text))]
    have = {x[0] for x in used}
    terms = sorted(used + [x for x in feat_terms if x[0] not in have], key=lambda x: x[0].lstrip("_").lower())
    story += [P(t("Abbreviations and acronyms used in this document", "Abreviaturas y siglas usadas en este documento"),
                "h1"),
              table([t("term", "término"), t("meaning", "significado")], terms, widths=[3.2 * cm, 12.8 * cm],
                    bold_first=True)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / fname)


def families():
    doc_family(
        "03_mlp", t("Training: mlp (dense network)", "Entrenamiento: mlp (red densa)"),
        t("Three runs of the two-hidden-layer MLP on the tabular features.",
          "Tres ejecuciones del MLP de dos capas ocultas sobre las características tabulares."),
        t("The MLP sees only the engineered features of the newest beat (no RR sequence). It is the cheapest network "
          "(about 1,100 MACs per beat) and trains in minutes.",
          "El MLP solo ve las características diseñadas del latido más reciente (sin secuencia RR). Es la red más barata "
          "(unos 1100 MACs por latido) y se entrena en minutos."),
        [t("Dense(32, ReLU) - Dense(16, ReLU) - Dense(1), Keras default initialisation, residual target, the shared "
           "Keras recipe (00_training_pipeline).", "Dense(32, ReLU) - Dense(16, ReLU) - Dense(1), inicialización por "
           "defecto de Keras, objetivo residual, la receta Keras común (00_training_pipeline)."),
         t("Each run changes only the feature set; architecture, seed and optimiser settings are the same.",
           "Cada ejecución cambia solo el conjunto de características; arquitectura, semilla y ajustes del optimizador "
           "son los mismos.")],
        [("mlp", t("round 1, 19 features", "ronda 1, 19 características"),
          [t("Round-1 epochs were full passes (40,538 steps of 512), so 18 epochs are about 18 passes; the median step "
             "took about 4 ms.", "Las épocas de la ronda 1 eran pasadas completas (40 538 pasos de 512), así que 18 épocas "
             "son unas 18 pasadas; el paso mediano tardó unos 4 ms.")]),
         ("mlp_nobp", t("round 2, without bp and bp_prev", "ronda 2, sin bp ni bp_prev"),
          [t("Trained with the 500-window int8 calibration bug; re-exported with 20,000 windows the same evening.",
             "Entrenado con el error de calibración int8 de 500 ventanas; reexportado con 20 000 ventanas esa misma "
             "tarde.")]),
         ("mlp_rsa", t("round 2, + 6 RSA-tracker features", "ronda 2, + 6 características del seguidor de RSA"),
          [t("Same re-export as mlp_nobp. The tracker features were ungated at the time.",
             "La misma reexportación que mlp_nobp. Las características del seguidor no tenían bloqueo entonces.")])],
        [t("All three converge to within 0.4 ms of each other; the learning curves are flat after a few epochs, so the "
           "MLP is limited by its inputs, not by the training budget.",
           "Las tres convergen a menos de 0.4 ms entre sí; las curvas de aprendizaje son planas tras unas pocas épocas, "
           "así que el MLP está limitado por sus entradas, no por el presupuesto de entrenamiento."),
         t("The closed-loop evaluation (not used in training) rejected all three in favour of the TCNs.",
           "La evaluación en lazo cerrado (no usada en el entrenamiento) rechazó las tres en favor de las TCNs.")])
    doc_family(
        "05_gru", t("Training: gru", "Entrenamiento: gru"),
        t("A 16-unit GRU over the RR sequence plus the 9 features.", "Una GRU de 16 unidades sobre la secuencia RR más "
          "las 9 características."),
        t("The GRU is the slowest network to train per step: its 30 time steps are unrolled, so each step runs 30 "
          "recurrent updates forward and backward.",
          "La GRU es la red más lenta de entrenar por paso: sus 30 pasos temporales están desenrollados, así que cada "
          "paso ejecuta 30 actualizaciones recurrentes hacia delante y hacia atrás."),
        [t("GRU(16, unroll=True) over the (30, 2) sequence, its last state concatenated with the 9 'seq' features, then "
           "Dense(32, ReLU) - Dense(1); residual target; the shared Keras recipe with at most 90 epochs.",
           "GRU(16, unroll=True) sobre la secuencia (30, 2), su último estado concatenado con las 9 características "
           "'seq', después Dense(32, ReLU) - Dense(1); objetivo residual; la receta Keras común con como máximo 90 "
           "épocas."),
         t("Before the run, a throughput probe compared 64, 128 and 256 steps per Keras call: 18-24, 24 and 23 ms per "
           "step, so 64 was kept (the cost is on the device, not in the dispatch).",
           "Antes de la ejecución, una prueba de rendimiento comparó 64, 128 y 256 pasos por llamada de Keras: 18-24, 24 "
           "y 23 ms por paso, así que se mantuvo 64 (el coste está en el dispositivo, no en el despacho).")],
        [("gru", t("round 3", "ronda 3"),
          [t("A first round-1 attempt was interrupted; its folder runs/gru_interrupted was kept but holds no model or "
             "config. This run was queued on 2026-10-07 at 21:46 and is the only gru model.",
             "Un primer intento en la ronda 1 se interrumpió; su carpeta runs/gru_interrupted se conservó pero no "
             "contiene modelo ni configuración. Esta ejecución se encoló el 2026-10-07 a las 21:46 y es el único modelo "
             "gru."),
           t("Exported with the 20,000-window calibration (trained after the calibration fix, before best-of-K).",
             "Exportado con la calibración de 20 000 ventanas (entrenado después de la corrección de calibración y antes "
             "del mejor de K).")])],
        [t("In float the GRU is competitive (it ties micro_tcn), but int8 quantisation of the recurrent state costs it "
           "+2.71% on validation, over the 2% limit, and 1.6-3.6 ms in the closed loop: rejected. Better training would "
           "not fix the quantisation.",
           "En float la GRU es competitiva (empata con micro_tcn), pero la cuantización int8 del estado recurrente le "
           "cuesta +2.71% en validación, por encima del límite del 2%, y 1.6-3.6 ms en lazo cerrado: rechazada. Un mejor "
           "entrenamiento no arreglaría la cuantización.")])
    doc_family(
        "06_micro_tcn", t("Training: micro_tcn", "Entrenamiento: micro_tcn"),
        t("The separable causal TCN, 16 filters, and its four variants.",
          "La TCN causal separable, 16 filtros, y sus cuatro variantes."),
        t("micro_tcn was trained once from scratch in round 1 (about 11 hours, 45 full passes). Its variants either "
          "change one setting from scratch (fewer filters, more inputs) or continue from its weights.",
          "micro_tcn se entrenó una vez desde cero en la ronda 1 (unas 11 horas, 45 pasadas completas). Sus variantes o "
          "cambian un ajuste desde cero (menos filtros, más entradas) o continúan desde sus pesos."),
        [t("Inputs: the (30, 2) z-scored RR/dRR sequence + the 9 'seq' features; residual target; the shared Keras "
           "recipe. BatchNorm layers train normally and are folded into the convolutions at int8 export.",
           "Entradas: la secuencia RR/dRR (30, 2) normalizada z + las 9 características 'seq'; objetivo residual; la "
           "receta Keras común. Las capas BatchNorm se entrenan normalmente y se fusionan con las convoluciones al "
           "exportar a int8."),
         t("Variants compare against their parent with the same seed (211), one change each.",
           "Las variantes se comparan con su padre con la misma semilla (211), un cambio cada una.")],
        [("micro_tcn", t("round 1 base", "base de la ronda 1"),
          [t("The int8 file was re-exported twice: with 20,000 windows (old file kept as model_int8_prev_calib.tflite) "
             "and then best-of-K (20k file kept as model_int8_20k.tflite).",
             "El archivo int8 se reexportó dos veces: con 20 000 ventanas (archivo antiguo conservado como "
             "model_int8_prev_calib.tflite) y después con el mejor de K (archivo de 20k conservado como "
             "model_int8_20k.tflite).")]),
         ("micro_tcn_f8", t("round 2, 8 filters, from scratch", "ronda 2, 8 filtros, desde cero"),
          [t("Re-exported with 20,000 windows after the calibration bug.",
             "Reexportado con 20 000 ventanas tras el error de calibración.")]),
         ("micro_tcn_dad", t("round 3, fine-tune on closed-loop data", "ronda 3, ajuste fino con datos de lazo cerrado"),
          [t("Its first attempt was stopped by the coordinator because of the tf.data index bug and deleted; this is the "
             "re-queued run.", "Su primer intento lo detuvo el coordinador por el error de índices de tf.data y se "
             "borró; esta es la ejecución reencolada."),
           t("The validation loss is higher than in the other runs because it is measured on a 50/50 mix with "
             "closed-loop windows, which are harder.", "La pérdida de validación es mayor que en las otras ejecuciones "
             "porque se mide sobre una mezcla 50/50 con ventanas de lazo cerrado, que son más difíciles.")]),
         ("micro_tcn_rsa", t("round 4, + gated RSA features, from scratch",
                             "ronda 4, + características de RSA con bloqueo, desde cero"),
          [t("A new input layer cannot load the parent's weights, so it started from scratch with the 2M-window epochs: "
             "about 3.9 passes vs about 45 for micro_tcn. The comparison is confounded by this budget.",
             "Una capa de entrada nueva no puede cargar los pesos del padre, así que empezó desde cero con épocas de 2M "
             "ventanas: unas 3.9 pasadas frente a unas 45 de micro_tcn. La comparación queda confundida por este "
             "presupuesto.")]),
         ("micro_tcn_rsa_ws", t("round 4, + gated RSA features, warm start",
                                "ronda 4, + características de RSA con bloqueo, arranque en caliente"),
          [t("The clean test of the RSA inputs: same function as micro_tcn at step 0, then the fine-tune schedule.",
             "La prueba limpia de las entradas de RSA: la misma función que micro_tcn en el paso 0, después el calendario "
             "de ajuste fino.")])],
        [t("micro_tcn's round-1 run reached the 1e-5 learning-rate floor with a flat validation loss: it is trained to "
           "convergence.", "La ejecución de la ronda 1 de micro_tcn llegó al mínimo de tasa de aprendizaje 1e-5 con una "
           "pérdida de validación plana: está entrenada hasta la convergencia."),
         t("Halving the filters (f8) costs about 1.8 ms; DAD and the warm-started RSA inputs move the validation RMSE by "
           "less than 0.6 ms. The selection in the closed loop went to the larger tcn_lite_ft.",
           "Reducir los filtros a la mitad (f8) cuesta unos 1.8 ms; DAD y las entradas de RSA con arranque en caliente "
           "mueven el RMSE de validación menos de 0.6 ms. La selección en lazo cerrado fue para la tcn_lite_ft, más "
           "grande.")])
    doc_family(
        "07_tcn_lite", t("Training: tcn_lite (selected family)", "Entrenamiento: tcn_lite (familia seleccionada)"),
        t("The full causal TCN, 24 filters, and the training of the selected model tcn_lite_ft.",
          "La TCN causal completa, 24 filtros, y el entrenamiento del modelo seleccionado tcn_lite_ft."),
        t("<b>The deployed model tcn_lite_ft_int8 was trained in two stages:</b> tcn_lite from scratch in round 1 (48 "
          "full passes over the 20.8M training windows, 9.5 hours, ending at the 1e-5 learning-rate floor), then a "
          "fine-tune of those weights at lr 3e-4 (7 epochs of 2M windows; the best was the first, so the weights barely "
          "moved). Finally a best-of-K int8 export.",
          "<b>El modelo desplegado tcn_lite_ft_int8 se entrenó en dos etapas:</b> tcn_lite desde cero en la ronda 1 (48 "
          "pasadas completas por las 20.8M ventanas de entrenamiento, 9.5 horas, terminando en el mínimo de tasa de "
          "aprendizaje 1e-5) y después un ajuste fino de esos pesos con lr 3e-4 (7 épocas de 2M ventanas; la mejor fue "
          "la primera, así que los pesos apenas se movieron). Por último, una exportación int8 del mejor de K."),
        [t("Inputs: the (30, 2) z-scored RR/dRR sequence + the 9 'seq' features (rmssd, rs, ccm, guzik, rls_pred, "
           "rls_err, bp, bp_prev, run_len); residual target; the shared Keras recipe; BatchNorm folded at export.",
           "Entradas: la secuencia RR/dRR (30, 2) normalizada z + las 9 características 'seq' (rmssd, rs, ccm, guzik, "
           "rls_pred, rls_err, bp, bp_prev, run_len); objetivo residual; la receta Keras común; BatchNorm fusionada al "
           "exportar."),
         t("tcn_lite_ft was planned as the control of tcn_lite_dad: the same fine-tune schedule without closed-loop "
           "data, to separate the effect of DAD from that of restarting the learning rate.",
           "tcn_lite_ft se planificó como control de tcn_lite_dad: el mismo calendario de ajuste fino sin datos de lazo "
           "cerrado, para separar el efecto de DAD del de reiniciar la tasa de aprendizaje.")],
        [("tcn_lite", t("round 1 base", "base de la ronda 1"),
          [t("Re-exported with 20,000 windows and then best-of-K on 2026-10-08.",
             "Reexportado con 20 000 ventanas y después con el mejor de K el 2026-10-08.")]),
         ("tcn_lite_ft", t("SELECTED: fine-tune, control for DAD", "SELECCIONADO: ajuste fino, control de DAD"),
          [t("The first epoch at lr 3e-4 already gave the best validation loss (0.2407 vs 0.2417 for the parent); the "
             "next 6 did not improve it and early stopping restored epoch 1. The gain over tcn_lite is tiny (0.02 ms open "
             "loop) but consistent in the int8 closed loop (pooled, 14/16 subjects).",
             "La primera época con lr 3e-4 ya dio la mejor pérdida de validación (0.2407 frente a 0.2417 del padre); las "
             "6 siguientes no la mejoraron y la parada temprana restauró la época 1. La ganancia sobre tcn_lite es "
             "minúscula (0.02 ms en lazo abierto) pero consistente en el lazo cerrado int8 (agrupado, 14/16 sujetos)."),
           t("int8: draw n2000_s2 (2,000 windows), md5 6219d6ae...; this file is the one evaluated on the test split.",
             "int8: extracción n2000_s2 (2000 ventanas), md5 6219d6ae...; este archivo es el que se evaluó en el "
             "conjunto de test.")]),
         ("tcn_lite_dad", t("round 3, fine-tune on closed-loop data", "ronda 3, ajuste fino con datos de lazo cerrado"),
          [t("A first attempt (2026-10-08 01:07) drew only the first 429k windows of each half because of the tf.data "
             "index bug: training loss 0.083 and validation 56.76 ms, an overfit to the first subjects. It was deleted "
             "and this run re-queued after the fix.",
             "Un primer intento (2026-10-08 01:07) solo extraía las primeras 429k ventanas de cada mitad por el error de "
             "índices de tf.data: pérdida de entrenamiento 0.083 y validación 56.76 ms, un sobreajuste a los primeros "
             "sujetos. Se borró y esta ejecución se reencoló tras la corrección.")]),
         ("tcn_lite_rsa", t("round 4, + gated RSA features, from scratch",
                            "ronda 4, + características de RSA con bloqueo, desde cero"),
          [t("About 3.6 passes vs 48 for its parent: the +0.42 ms is confounded by the smaller training budget.",
             "Unas 3.6 pasadas frente a 48 de su padre: los +0.42 ms quedan confundidos por el menor presupuesto de "
             "entrenamiento.")]),
         ("tcn_lite_rsa_ws", t("round 4, + gated RSA features, warm start",
                               "ronda 4, + características de RSA con bloqueo, arranque en caliente"),
          [t("Same schedule as tcn_lite_ft, so the two differ only by the 4 new inputs: no gain at 10-50% dropped and a "
             "small loss at 80%.", "El mismo calendario que tcn_lite_ft, así que las dos solo difieren en las 4 entradas "
             "nuevas: sin ganancia con el 10-50% eliminado y una pequeña pérdida al 80%.")])],
        [t("tcn_lite reached the learning-rate floor with a flat validation loss: more epochs would not help, and the "
           "fine-tune confirms it (best epoch = first).",
           "tcn_lite llegó al mínimo de tasa de aprendizaje con una pérdida de validación plana: más épocas no ayudarían, "
           "y el ajuste fino lo confirma (mejor época = la primera)."),
         t("Neither closed-loop data (DAD) nor RSA inputs gave a significant int8 closed-loop gain over tcn_lite_ft, which "
           "was selected and passed the single test evaluation.",
           "Ni los datos de lazo cerrado (DAD) ni las entradas de RSA dieron una ganancia significativa en lazo cerrado "
           "int8 sobre tcn_lite_ft, que fue la seleccionada y superó la evaluación única de test.")])


def main():
    global LANG
    OUT.mkdir(parents=True, exist_ok=True)
    for LANG in ("en", "es"):
        doc_pipeline()
        doc_baselines()
        doc_linear()
        families()
        doc_gbdt()
        doc_tcn_mha()


if __name__ == "__main__":
    main()

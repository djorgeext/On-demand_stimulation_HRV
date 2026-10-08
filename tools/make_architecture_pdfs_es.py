"""
tools/make_architecture_pdfs_es.py - Spanish version of the architecture PDFs, written to
docs/architectures/architectures-spanish/ with the same file names as the English ones.

Facts (runs, layers, results) and the table/paragraph helpers come from make_architecture_pdfs.py, so the
numbers are identical to the English PDFs; only the text is translated here. Acronym meanings come from
tools/glossary_es.py. Numbers keep the decimal point, as in the tables and the code.

Run: HIP_VISIBLE_DEVICES=-1 .venv/bin/python tools/make_architecture_pdfs_es.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_architecture_pdfs as M  # noqa: E402  (loads every run and result once)
import glossary_es as GE  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import SimpleDocTemplate  # noqa: E402

OUT = M.OUT / "architectures-spanish"
F, R, RATES, DATE = M.F, M.R, M.RATES, M.DATE
P, bullets, table, esc, fmt, cx = M.P, M.bullets, M.table, M.esc, M.fmt, M.cx
val_open, val_closed, _xc, _rank, _ext = M.val_open, M.val_closed, M._xc, M._rank, M._ext
LABEL = dict(round4="ronda 4", round3="ronda 3", round2_int8fix="ronda 2 (recomprobación int8)", round2="ronda 2",
             round2_gru="ronda 2 (gru)", round1="ronda 1")


def miles(n):
    return f"{n:,}".replace(",", " ")  # thousands separated by a space (the decimal point stays a point)


# ----------------------------------------------------------------------------- pdf helpers
def build(fname, title, subtitle, story):
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#777777"))
        canvas.drawString(1.8 * cm, 1.1 * cm, f"Predictor de dRR para HRV - {title} - {DATE}")
        canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, f"página {doc.page}")
        canvas.restoreState()
    doc = SimpleDocTemplate(str(OUT / fname), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.6 * cm, bottomMargin=1.8 * cm, title=title, author="Proyecto HRV dRR",
                            subject=subtitle, lang="es")
    story = [P(esc(title), "title"), P(subtitle, "sub")] + story
    text = GE._ascii(M.G.plain(" ".join(M._texts(story)) + f" Predictor de dRR para HRV - {title}"))
    missing = GE.undefined_tokens(text)
    if missing:  # runnable check: every acronym in the PDF must be defined in tools/glossary.py (+ glossary_es.py)
        raise ValueError(f"{fname}: undefined acronyms {missing}; add them to tools/glossary.py and glossary_es.py")
    feat_terms = [("drr_lag0..7" if k == "drr_lag" else "rls_w0..7" if k == "rls_w" else k, f"característica: {d}")
                  for k, d in FEATURE_DESC.items()
                  if (k in text if k in ("drr_lag", "rls_w") else M.G._present(k, "word", text))]
    have = {x[0] for x in GE.used_terms(text)}
    terms = sorted(GE.used_terms(text) + [x for x in feat_terms if x[0] not in have],
                   key=lambda x: x[0].lstrip("_").lower())
    story += [P("Abreviaturas y siglas usadas en este documento", "h1"),
              table(["término", "significado"], terms, widths=[3.2 * cm, 12.8 * cm], bold_first=True)]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT / fname, f"({len(terms)} glossary terms)")


# ----------------------------------------------------------------------------- shared tables
def layer_table(rows):
    return table(["capa", "configuración", "forma de salida", "parám."],
                 [(a, b, c, str(d)) for a, b, c, d in rows], widths=[3.4 * cm, 7.6 * cm, 3.4 * cm, 1.6 * cm])


def variants_table(runs, notes, highlight=None):
    rows = []
    for run in runs:
        c = F["runs"][run]["cfg"]
        x = cx(run)
        feats = c.get("features") or []
        fs = next((k for k, v in F["feature_sets"].items() if v == feats), f"{len(feats)} características")
        a = c.get("args", {})
        init = c.get("init_from")
        init = init.get("run") if isinstance(init, dict) else init
        how = f"ajuste fino de {Path(init).name}, lr {a.get('lr')}" if init else f"desde cero, lr {a.get('lr')}"
        if (c.get("dad") or {}).get("filler"):
            how += f", proporción de datos de lazo cerrado {c['dad'].get('share')}"
        rows.append((run, notes.get(run, ""), f"{fs} ({len(feats)})", str(x.get("params", "-")),
                     str(x.get("macs_stream", "-")), how, str(F["runs"][run].get("epochs") or "-")))
    return table(["ejecución", "cambio respecto al padre", "conjunto de características (n)", "parám.",
                  "MACs/latido", "entrenamiento", "épocas"],
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
        rows.append([m, fmt(vo), kind] + cl + [inv, fmt(ts["rmse"]) if ts else "-", LABEL.get(tc or to, "-")])
    return table(["modelo", "val lazo abierto", "modelo en lazo cerrado", "10%", "30%", "50%", "80%",
                  "RSA inventada al 80%", "test serie completa 80%", "fuente val"], rows,
                 widths=[2.6 * cm, 1.5 * cm, 1.5 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm, 1.1 * cm, 1.6 * cm, 1.8 * cm,
                         2.6 * cm], bold_first=True, highlight=highlight)


RES_NOTE = ("RMSE en ms. <b>val lazo abierto</b>: predicción a un paso sobre la historia verdadera, 16 sujetos de "
            "validación. <b>10-80%</b>: lazo cerrado (uso en el marcapasos) en validación: se elimina al azar esa "
            "proporción de latidos desde el latido 0, cada uno se rellena con el modelo y se realimenta; RMSE sobre "
            "los latidos rellenados por el modelo, primeros 5000 latidos por sujeto, media de las semillas de esa "
            "ronda (3 semillas desde la ronda 2). Se usa int8 donde existe. <b>RSA inventada</b>: proporción de "
            "pares sujeto-ejecución cuya serie rellenada gana un pico de RSA que la original no tiene. <b>test serie "
            "completa 80%</b>: diagnóstico sobre los 10 sujetos de test (serie completa, semilla 7, carpeta "
            "results/test/); no se usó para ninguna selección.")

FEATURE_DESC = {
    "rr_mean": "RR medio de la ventana de 30 latidos", "rr_dev": "RR más reciente menos la media de la ventana",
    "rmssd": "raíz cuadrada media de las 29 diferencias sucesivas",
    "rs": "rango reescalado: (máx - mín de la suma acumulada de RR - media) / desviación estándar",
    "ccm": "medida de correlación compleja: área de los triángulos de Poincaré / (pi SD1 SD2 (W-3)), recortada a [0, 1]",
    "guzik": "asimetría de Guzik: suma de dRR positivos / suma de |dRR|",
    "porta": "asimetría de Porta: número de dRR negativos / número de dRR no nulos",
    "nn20": "número de |dRR| > 20 ms en la ventana",
    "run_len": "longitud con signo de la racha actual de dRR del mismo signo (+ = desaceleración)",
    "rls_pred": "pronóstico RLS AR(8) del siguiente dRR (ms), recortado a +-300",
    "rls_err": "última innovación del RLS: dRR verdadero menos el pronóstico anterior (ms)",
    "bp": "salida del pasa banda RSA: biquad RBJ sobre el RR crudo, centro 0.30 ciclos/latido, Q 1",
    "bp_prev": "salida anterior del pasa banda (junto con bp da la fase)",
    "drr_lag": "drr_lag0..7: los 8 dRR más recientes (lag0 = el más nuevo)",
    "rls_w": "rls_w0..7: pesos actuales del AR(8) del RLS",
    "rsa_f": "frecuencia de RSA seguida (ciclos/latido): pico de un banco de 8 resonadores, interpolación parabólica",
    "rsa_amp": "amplitud de la RSA (ms) a partir de la potencia media de la banda del pico",
    "rsa_cos": "coseno de la fase actual de la RSA (cuadratura de 2 muestras de la banda del pico)",
    "rsa_sin": "seno de esa fase",
    "rsa_q": "presencia de RSA: altura del pico sobre la tendencia de log-potencia de la banda (log2); 0 si está bloqueada",
    "rsa_drr": "parte de RSA predicha del siguiente dRR: amp (cos(fase + 2 pi f) - cos(fase)); 0 si está bloqueada",
}

XC_NOTE = ("Calculado sobre los 16 sujetos de validación y los 10 de test (carpetas results/validation/ y "
           "results/test/), sobre la serie COMPLETA, tras eliminar el 80% de los latidos (semilla 7, desde el latido 0) "
           "y rellenarlos en lazo cerrado con el modelo (relleno Burg-AR en los primeros 30 latidos): <b>corr RR</b> = "
           "np.corrcoef(original_serie, modified_serie)[0, 1]; <b>corr dRR</b> = la misma función sobre los cambios "
           "latido a latido, np.corrcoef(np.diff(original), np.diff(filled))[0, 1], que aísla la dinámica a corto "
           "plazo. Media sobre sujetos; valores por sujeto en &lt;carpeta&gt;/crosscorr.csv. <b>RMSE rellenados</b>: "
           "RMSE sobre los latidos rellenados por el modelo en esas mismas ejecuciones. Diagnóstico: la selección usó "
           "el protocolo de la ronda 4, no estas ejecuciones.")


def xcorr_table(models, highlight=None):
    rows = []
    for m in models:
        xt, xv = R["xcorr_test"].get(m), R["xcorr_val"].get(m)
        if xt is None and xv is None:
            continue
        g = lambda x, k, d: f"{x[k]:.{d}f}" if x else "-"  # noqa: E731
        rows.append([m, g(xv, "rr", 4), g(xt, "rr", 4), g(xv, "drr", 3), g(xt, "drr", 3),
                     fmt(R["ws_val"].get(m)), fmt(R["ws_test"].get(m))])
    return table(["modelo", "corr RR validación", "corr RR test", "corr dRR validación", "corr dRR test",
                  "RMSE rellenados, validación (ms)", "RMSE rellenados, test (ms)"], rows,
                 widths=[3.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.2 * cm, 2.0 * cm], bold_first=True,
                 highlight=highlight)


XC_BULLETS = [
    "Nivel RR: todos los modelos mantienen una correlación alta con la original (validación "
    f"{min(v['rr'] for v in R['xcorr_val'].values()):.3f}-{max(v['rr'] for v in R['xcorr_val'].values()):.3f}, test "
    f"{min(v['rr'] for v in R['xcorr_test'].values()):.3f}-{max(v['rr'] for v in R['xcorr_test'].values()):.3f}): "
    "dominan los cambios lentos de frecuencia cardiaca y el 20% de latidos conservados los ancla. El modelo desplegado "
    f"tcn_lite_ft_int8: validación {_xc('tcn_lite_ft_int8', 'rr', 'val')} (puesto "
    f"{_rank('tcn_lite_ft_int8', 'rr', 'val')} de {len(R['xcorr_val'])}), test {_xc('tcn_lite_ft_int8', 'rr', 'test')} "
    f"(puesto {_rank('tcn_lite_ft_int8', 'rr', 'test')}); rls_ar {_xc('rls_ar', 'rr', 'val')} / "
    f"{_xc('rls_ar', 'rr', 'test')}; TCN_MHA {_xc('tcn_mha_current', 'rr', 'val')} / "
    f"{_xc('tcn_mha_current', 'rr', 'test')} (validación / test).",
    "El nivel dRR (dinámica latido a latido) separa más a los modelos: tcn_lite_ft_int8 "
    f"{_xc('tcn_lite_ft_int8', 'drr', 'val', 3)} / {_xc('tcn_lite_ft_int8', 'drr', 'test', 3)}, rls_ar "
    f"{_xc('rls_ar', 'drr', 'val', 3)} / {_xc('rls_ar', 'drr', 'test', 3)}, TCN_MHA "
    f"{_xc('tcn_mha_current', 'drr', 'val', 3)} / {_xc('tcn_mha_current', 'drr', 'test', 3)} (validación / test); "
    f"el más bajo: {_ext('drr', 'val', min)} en validación, {_ext('drr', 'test', min)} en test. Valores en torno a "
    "0.2-0.3 son esperables con el 80% eliminado: 4 de cada 5 cambios latido a latido son del propio modelo.",
    f"Mayor correlación RR: {_ext('rr', 'val', max)} en validación, {_ext('rr', 'test', max)} en test. Estas "
    "ejecuciones sobre la serie completa son diagnósticos; la selección se hizo con el protocolo de validación de la "
    "ronda 4 y no cambia.",
]


def xcorr_per_subject(models, split="test"):
    d = R[f"xcorr_subj_{split}"]
    subjects = sorted({s for s, _ in d})
    rows = [[s] + [f"{d[(s, m)][0]:.4f}" for m in models] + [f"{d[(s, m)][1]:.3f}" for m in models] for s in subjects]
    head = ["sujeto"] + [f"RR {m}" for m in models] + [f"dRR {m}" for m in models]
    name = "test" if split == "test" else "validación"
    return [P(f"Por sujeto de {name} ({len(subjects)}): modelo desplegado frente a TCN_MHA y rls_ar", "h2"),
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
    return table(["característica", "definición"], out, widths=[2.6 * cm, 13.4 * cm])


INPUT_PREP = [
    "<b>Entrada de secuencia</b> (TCN/GRU): los últimos 30 intervalos RR como dos canales, z-RR y z-dRR (dRR del "
    "primer paso = 0), normalizados z con media/desviación ajustadas solo sobre los sujetos de entrenamiento "
    "(varianza cero -&gt; desviación 1).",
    "<b>Entrada tabular</b>: las características diseñadas del latido más reciente (ver 01_feature_engineering.pdf), "
    "normalizadas z por característica con estadísticas de entrenamiento.",
    "<b>Objetivo</b>: dRR[n+1] = RR[n+1] - RR[n]. Los modelos residuales predicen la corrección normalizada z sobre "
    "el pronóstico RLS (dRR - rls_pred); la salida se desnormaliza y se suma de nuevo a rls_pred.",
    "Las primeras 30 ventanas de cada serie se descartan del entrenamiento y de la evaluación (estado RLS en frío).",
]
TRAIN_KERAS = ("Keras, AdamW (decaimiento de pesos 1e-4), pérdida de Huber (delta 1, en unidades z), lote 512, "
               "steps_per_execution 64, épocas de 2M ventanas de entrenamiento aleatorias (alrededor del 10% de las "
               "20.8M), EarlyStopping sobre val_loss (restaurando los mejores pesos) y ReduceLROnPlateau (factor 0.5, "
               "mínimo 1e-5), semilla 211. int8: cuantización post-entrenamiento totalmente entera (TFLite, entrada y "
               "salida int8), calibración determinista del mejor de K (8 extracciones de 2k/5k ventanas de "
               "entrenamiento; se conserva la que tiene la menor diferencia sobre sujetos de entrenamiento reservados).")


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
        size = ("238 KB (30 500 nodos)" if fam == "gbdt" else f"{x.get('params', 0)} parám."
                if x.get("params", 0) < 100 else f"{x.get('params', 0) / 1024:.1f} KB")
        rows.append([fam, best, str(x.get("macs_stream", "3000 comparaciones")), size, fmt(vo),
                     fmt(vc["0.1"]["rmse"]) if vc else "-", fmt(vc["0.8"]["rmse"]) if vc else "-",
                     fmt(ts["rmse"]) if ts else "-", pdf])
    vo, _ = val_open("tcn_mha_current")
    vc, _ = val_closed("tcn_mha_current")
    rows.append(["TCN_MHA (heredado)", "tcn_mha_current", miles(F["tcn_mha"]["cx"]["macs_stream"]),
                 f"{F['tcn_mha']['params'] / 1024:.1f} KB", fmt(vo), fmt(vc["0.1"]["rmse"]), fmt(vc["0.8"]["rmse"]),
                 fmt(R["test_ws"]["tcn_mha_current"]["rmse"]), "09_tcn_mha.pdf"])
    vo, _ = val_open("rls_ar")
    vc = R["round4"]["closed"]["rls_ar"]
    rows.append(["referencia", "rls_ar", "152 FLOPs", "-", fmt(vo), fmt(vc["0.1"]["rmse"]), fmt(vc["0.8"]["rmse"]),
                 fmt(R["test_ws"]["rls_ar"]["rmse"]), "02_baselines.pdf"])
    ft = R["final_test"]
    ftc = ft["closed"]
    final_rows = []
    for m, lab in [("tcn_lite_ft_int8", "tcn_lite_ft_int8 (desplegado)"), ("tcn_lite_ft", "tcn_lite_ft (float)"),
                   ("tcn_mha_current", "TCN_MHA (heredado)"), ("rls_ar", "rls_ar"), ("persistence", "persistence")]:
        final_rows.append([lab, fmt(ft["open"][m])] + [fmt(ftc[m][r]["rmse"]) for r in RATES])
    story = [
        P("<b>Selección final: tcn_lite_ft_int8</b> - una TCN causal (24 filtros, dilataciones 1/2/4/7, campo "
          "receptivo de 29 latidos) sobre los últimos 30 intervalos RR más 9 características diseñadas, que predice "
          "una corrección sobre un pronóstico RLS AR(8) adaptativo. 8 048 MACs por latido y 8.4 KB de pesos int8. No "
          "pierde frente a ningún otro candidato en el lazo cerrado int8 en validación y cumplió todas las reglas "
          "sobre los sujetos de test reservados.", "box"),
        P("Objetivo y restricciones", "h1"),
        *bullets([
            "Predecir dRR[n+1] = RR[n+1] - RR[n] latido a latido a partir de los últimos 30 intervalos RR, para "
            "rellenar latidos perdidos en tiempo real en un marcapasos (notebooks/correlation_study_v3.ipynb).",
            "Primer hardware: STM32H750 (Cortex-M7); el implante será menos potente. Presupuesto estricto: como máximo "
            "10k MACs por latido (preferible 3k), pesos int8 de como máximo 32 KB, RMSE int8 a menos del 2% del float, "
            "solo operaciones nativas del MCU (Conv1D, SeparableConv1D, Dense, BatchNorm, ReLU, Add, relleno causal, "
            "GRU), campo receptivo de como máximo 30.",
            "Datos: series RR en data/series/*.txt, divididas por sujeto: 142 sujetos de entrenamiento, 16 de "
            "validación y 10 de test (data/subjects_*.json, estratificados por intervalo de edad). La selección se "
            "hizo solo en validación; el conjunto de test se abrió una única vez, para la evaluación final.",
            "Regla de selección (decisiones del usuario): dentro del presupuesto, primero la precisión, decidida por "
            "el lazo CERRADO del modelo int8 con pruebas de Wilcoxon pareadas por sujeto (agrupando las tasas de "
            "eliminación o por tasa con Bonferroni, nunca significativamente peor en ninguna tasa); los empates "
            "reales los gana el modelo más pequeño. Además se comprueba en cada modelo la fidelidad de la RSA (ningún "
            "pico respiratorio inventado ni desplazado)."]),
        P("Arquitecturas comparadas (mejor variante de cada familia)", "h1"),
        table(["familia", "mejor variante", "MACs/latido", "tamaño int8", "val lazo abierto", "val cerrado 10%",
               "val cerrado 80%", "test serie completa 80%", "documento"], rows,
              widths=[2.3 * cm, 2.4 * cm, 1.6 * cm, 2.2 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.6 * cm, 2.1 * cm],
              bold_first=True, highlight=[0]),
        P(RES_NOTE, "sub"),
        P("Test final (conjunto de test reservado, ejecutado una vez, 10 sujetos, 942 526 ventanas)", "h1"),
        table(["modelo", "lazo abierto", "cerrado 10%", "cerrado 30%", "cerrado 50%", "cerrado 80%"], final_rows,
              widths=[4.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm], bold_first=True, highlight=[0]),
        *bullets([
            "tcn_lite_ft_int8 es mejor que rls_ar en 10/10 sujetos de test en lazo abierto (-8.2%) y en el lazo "
            "cerrado agrupado; con el 80% eliminado es 9 ms (23.6%) mejor. La ganancia al 10% ya no es "
            "significativa en test.",
            "En los sujetos de test no inventa ni desplaza ningún pico de RSA en ninguna tasa de eliminación; con el "
            "80% eliminado amortigua la potencia de la RSA, lo que un análisis de viabilidad mostró que es un límite "
            "de información (la fase de la RSA no se puede recuperar con el 20% de los latidos), no un defecto del "
            "modelo.",
            "El TCN_MHA heredado es peor en 10/10 sujetos en lazo abierto, inventa picos de RSA y no puede "
            "ejecutarse en el MCU."]),
        P("Correlación entre la serie original y la rellenada (sujetos de validación y test, todos los modelos)", "h1"),
        xcorr_table(M.XC_ORDER, [M.XC_ORDER.index("tcn_lite_ft_int8")]),
        P(XC_NOTE, "sub"),
        *bullets(XC_BULLETS),
        P("Documentos de esta carpeta", "h1"),
        table(["archivo", "contenido"], [
            ("00_overview.pdf", "este resumen"),
            ("01_feature_engineering.pdf", "el motor de características en streaming, preprocesado, objetivos, "
                                           "protocolos de lazo cerrado e int8"),
            ("02_baselines.pdf", "persistencia, media EMA, RLS AR(8) adaptativo"),
            ("03_linear.pdf", "regresión ridge sobre dRR retardados (AR global)"),
            ("04_mlp.pdf", "MLP sobre características diseñadas: mlp, mlp_nobp, mlp_rsa"),
            ("05_gbdt.pdf", "árboles potenciados por gradiente: gbdt, gbdt_all (referencia de precisión y de "
                            "importancia de características)"),
            ("06_gru.pdf", "GRU sobre la secuencia RR + características"),
            ("07_micro_tcn.pdf", "TCN causal separable, 16 filtros, y sus variantes"),
            ("08_tcn_lite.pdf", "TCN causal completa, 24 filtros, y sus variantes, incluida la seleccionada tcn_lite_ft"),
            ("09_tcn_mha.pdf", "la TCN heredada con atención multicabeza (modelo de producción actual)"),
        ], widths=[4.2 * cm, 11.8 * cm]),
    ]
    build("00_overview.pdf", "Resumen de todas las arquitecturas",
          "Resumen de todas las arquitecturas entrenadas para la predicción latido a latido de dRR, con la selección "
          "final y el resultado en test.", story)


def doc_features():
    e, t = F["engine"], F["rsa_tracker"]
    users = {"seq": "micro_tcn, micro_tcn_f8, micro_tcn_dad, tcn_lite, tcn_lite_ft, tcn_lite_dad, gru",
             "tab": "mlp, gbdt", "ar": "linear", "legacy_cheap": "(no lo usa ninguna ejecución entrenada)",
             "tab_nobp": "mlp_nobp", "tab_rsa": "mlp_rsa",
             "seq_rsa": "tcn_lite_rsa, tcn_lite_rsa_ws, micro_tcn_rsa, micro_tcn_rsa_ws",
             "all": "gbdt_all (29 características: entrenado antes de añadir las 6 características de RSA)"}
    fs_rows = [(k, str(len(v)), ", ".join(v) if len(v) < 25 else "todas las características (FEATURE_NAMES)",
                users.get(k, "")) for k, v in F["feature_sets"].items()]
    story = [
        P("Todos los modelos obtienen sus entradas de un único motor en streaming (src/utils.py, núcleo numba _push): "
          "consume un intervalo RR por latido y mantiene un estado O(1), de modo que el entrenamiento, la evaluación "
          "y el marcapasos calculan características idénticas. Es la referencia para el port a C. El TCN_MHA heredado "
          "es la única excepción (ver al final).", "box"),
        P("1. Motor en streaming latido a latido", "h1"),
        *bullets([
            f"Ventana: los últimos W = {e['window']} intervalos RR (ms), guardados en un búfer circular. Coste por "
            "latido: estadísticas de ventana O(W) + actualización RLS O(p<super>2</super>) (p = 8, unos 152 FLOPs) + "
            "el banco de RSA (unos 180 FLOPs) + un biquad (5 multiplicaciones).",
            f"RLS AR({e['ar_order']}): predice el siguiente dRR a partir de los retardos de RR centrados (RR[n-k] - "
            f"media de la ventana) / {e['rls_scale']:g}. Factor de olvido {e['rls_lambda']} (memoria de unos 100 "
            f"latidos), covarianza inicial {e['rls_p0']:g}, límite de la traza de la covarianza "
            f"{e['rls_p_max_trace']:g}, recorte del pronóstico +-{e['rls_pred_clamp']:g} ms. Protección de "
            "estabilidad: la actualización se omite cuando la covarianza pierde la definición positiva, la covarianza "
            "se mantiene simétrica y se reinicia si su traza deja de ser finita (sin ella el RLS divergía tras unos "
            "3200 latidos en datos reales).",
            f"Pasa banda RSA fijo: biquad RBJ sobre el RR crudo, centro {e['bp_f0']} ciclos/latido, Q {e['bp_q']} "
            "(aprox. 0.15-0.45 ciclos/latido), ganancia DC nula.",
            f"Seguidor de RSA (ronda 2): un banco de {t['n_bands']} pasa banda RBJ de {t['f_lo']} a {t['f_hi']} "
            f"ciclos/latido (Q {t['q']}), EMAs de potencia por banda con factor {t['beta']}. La banda del pico sobre la "
            "tendencia de log-potencia de la banda da frecuencia, amplitud, fase y presencia. En el lazo cerrado sus "
            "potencias de banda se congelan en los latidos rellenados y un bloqueo pone rsa_q = rsa_drr = 0 cuando "
            f"{t['gate_min_imputed']} o más de los últimos 30 latidos fueron rellenados (ahí la fase de la RSA no se "
            "puede recuperar)."]),
        P("2. Características diseñadas (35, en el orden de FEATURE_NAMES)", "h1"),
        features_list(__import__("utils").FEATURE_NAMES),
        *__import__("feature_details").story("es", P, bullets, table, cm),
        P("4. Conjuntos de características", "h1"),
        table(["conjunto", "n", "características", "usado por"], fs_rows,
              widths=[1.8 * cm, 0.7 * cm, 8.7 * cm, 4.8 * cm], bold_first=True),
        P("5. Entradas del modelo, normalización y objetivo", "h1"),
        *bullets(INPUT_PREP),
        P("6. Preprocesado del lazo cerrado (el uso en el marcapasos, notebooks/correlation_study_v3.ipynb)", "h1"),
        *bullets([
            "Latidos eliminados: se quita una proporción aleatoria (10/30/50/80%) de latidos, de forma uniforme desde "
            "el latido 0 (como utils2.random_extraction con start_idx = 0); se usa la misma máscara para todos los "
            "modelos.",
            "Primeros 30 latidos (aún sin ventana completa): se rellenan con el predictor Burg-AR del notebook "
            "(coeficientes AR(15) de data/ar_model_parameters.json, pesos MMSE exactos de historia finita, anclados "
            "a la referencia de RR según la edad 505 x edad<super>0.122</super> ms); estos rellenos no se evalúan.",
            "Desde el latido 30: cada latido eliminado lo rellena el modelo; el relleno entra en la ventana, en las "
            "características y en el estado RLS de los latidos siguientes. Solo se evalúan los latidos rellenados "
            "por el modelo.",
            "Fidelidad de la RSA: PSD de Welch de la serie rellenada frente a la original en 0.15-0.40 ciclos/latido; "
            "un pico cuenta como RSA cuando está al menos 2x por encima de la tendencia de la banda; inventado = pico "
            "nuevo (y crecido >= 1.4x), desplazado = movido > 0.03 ciclos/latido, perdido = desaparecido."]),
        P("7. Datos de entrenamiento de lazo cerrado (DAD, ronda 3)", "h1"),
        P("Para las ejecuciones _dad, cada serie de entrenamiento se cortó en segmentos de 5000 latidos con un 10-80% "
          "aleatorio de latidos eliminados y rellenados en lazo cerrado por el propio modelo padre; el motor se "
          "ejecutó sobre la serie rellenada y el objetivo fue el latido siguiente VERDADERO (rr_true[n+1] - "
          "rr_filled[n]). Los lotes mezclaban un 50% de estas ventanas con un 50% de ventanas ordinarias. No mejoró "
          "significativamente el modelo int8 desplegado."),
        P("8. Exportación int8", "h1"),
        P("Cuantización post-entrenamiento totalmente entera con TFLite (pesos, activaciones, entrada y salida int8, "
          "el formato de CMSIS-NN / ST Edge AI). La calibración es determinista, el mejor de K: se prueban 8 "
          "extracciones aleatorias de 2k o 5k ventanas de entrenamiento y se conserva la exportación con la menor "
          "diferencia int8-frente-a-float sobre 20 sujetos de entrenamiento reservados (ventanas ordinarias y de lazo "
          "cerrado). La validación nunca se usa para calibrar."),
        P("9. Características del TCN_MHA heredado (no salen del motor)", "h1"),
        P("El modelo heredado recalcula sus entradas en cada ventana con legacy/utils2.py: 12 características "
          f"estáticas ({', '.join(F['legacy_static'])}) normalizadas z con las estadísticas guardadas en "
          "data/hrv_dataset.h5, más los 30 valores RR normalizados z con estadísticas globales (un canal). Los "
          "coeficientes AR(5) de Burg necesitan un ajuste O(orden x W) en cada latido."),
    ]
    build("01_feature_engineering.pdf", "Ingeniería de características y preprocesado",
          "El motor de características en streaming compartido por todos los modelos, los conjuntos de "
          "características, entradas, objetivos, protocolo de lazo cerrado y exportación int8.", story)


def doc_family(fname, title, subtitle, intro, arch_text, layer_run, runs, notes, verdict, highlight=None,
               extra=None, extra_after=None):
    c = F["runs"][layer_run]["cfg"]
    feats = c.get("features") or []
    story = [P(intro, "box"), P("Arquitectura", "h1"), *bullets(arch_text)]
    if F["runs"][layer_run].get("layers"):
        story += [P(f"Tabla de capas de <b>{layer_run}</b> (B = lote)", "h2"),
                  layer_table(F["runs"][layer_run]["layers"])]
    story += [P(f"Entradas e ingeniería de características ({layer_run})", "h1"),
              *bullets(INPUT_PREP[:3] if c.get("kind") == "seq"
                       else [INPUT_PREP[1], INPUT_PREP[2]] if c.get("residual") else
                       [INPUT_PREP[1], "<b>Objetivo</b>: dRR[n+1] directamente (sin residual)."]),
              P(f"Características tabulares ({len(feats)}):", "h2"), features_list(feats)]
    if c.get("framework") == "keras":
        story += [P("Entrenamiento", "h1"), P(TRAIN_KERAS)]
    if extra:
        story += extra
    story += [P("Variantes entrenadas", "h1"), variants_table(runs, notes, highlight),
              P("Resultados", "h1"), results_table(runs, highlight), P(RES_NOTE, "sub"),
              P("Correlación entre la serie original y la rellenada (sujetos de validación y test)", "h1"),
              xcorr_table(runs + [m + "_int8" for m in runs if m + "_int8" in R["xcorr_val"]],
                          [i for i, m in enumerate(runs) if highlight and i in highlight]),
              P(XC_NOTE, "sub")]
    if extra_after:
        story += extra_after
    story += [P("Veredicto", "h1"), *bullets(verdict)]
    build(fname, title, subtitle, story)


def doc_baselines():
    rows = []
    for m, desc, cost in [("persistence", "siguiente dRR = 0 (repetir el último RR)", "0"),
                          ("ema_mean", "siguiente RR = media móvil exponencial de la ventana (factor 0.75 por latido), "
                                       "dRR = EMA - último RR", "30"),
                          ("rls_ar", "AR(8) adaptativo por paciente mediante mínimos cuadrados recursivos sobre los "
                                     "retardos de RR centrados (olvido 0.99); el pronóstico es la característica "
                                     "rls_pred", "152 FLOPs")]:
        vo, _ = val_open(m)
        vc = R["round4"]["closed"][m]
        rows.append([m, desc, cost, fmt(vo)] + [fmt(vc[r]["rmse"]) for r in RATES] + [fmt(R["test_ws"][m]["rmse"])])
    story = [
        P("Las referencias no necesitan entrenamiento. Cada candidato se presenta frente a rls_ar y persistence; un "
          "modelo que no supera a rls_ar no es un avance.", "box"),
        table(["referencia", "definición", "coste/latido", "val abierto", "10%", "30%", "50%", "80%",
               "test serie completa 80%"], rows,
              widths=[2.0 * cm, 6.0 * cm, 1.4 * cm, 1.2 * cm, 1.0 * cm, 1.0 * cm, 1.0 * cm, 1.0 * cm, 1.4 * cm],
              bold_first=True),
        P(RES_NOTE, "sub"),
        P("Correlación entre la serie original y la rellenada (sujetos de validación y test)", "h1"),
        xcorr_table(["persistence", "ema_mean", "rls_ar"]), P(XC_NOTE, "sub"),
        P("Observaciones", "h1"),
        *bullets(["rls_ar es la referencia más fuerte a un paso (37.1 ms en validación) y es la base residual de "
                  "todos los buenos modelos.",
                  "En el lazo cerrado rls_ar es inestable con tasas de eliminación altas: con el 80% eliminado es peor "
                  "que persistence e inventa un pico de RSA en alrededor del 75% de los pares sujeto-ejecución, porque "
                  "un modelo AR alimentado con sus propias predicciones entra en resonancia. Congelar su actualización "
                  "en los latidos rellenados lo empeoró (probado en la ronda 4).",
                  "persistence nunca inventa picos de RSA, pero amortigua la potencia de la RSA y tiene el peor error "
                  "a un paso."]),
    ]
    build("02_baselines.pdf", "Referencias (baselines)",
          "Persistencia, media EMA y la referencia RLS AR(8) adaptativa.", story)


def doc_tcn_mha():
    ft = R["final_test"]
    vo, _ = val_open("tcn_mha_current")
    vc, _ = val_closed("tcn_mha_current")
    ws = R["test_ws"]["tcn_mha_current"]
    story = [
        P("El modelo de producción actual, evaluado como un candidato más (no como referencia). No se puede "
          "desplegar en el microcontrolador y tcn_lite lo supera en casi todos los sujetos.", "box"),
        P("Arquitectura", "h1"),
        *bullets([
            "Tronco de convoluciones causales multiescala (núcleos 3, 5 y 9, 16 filtros cada uno, GELU), concatenado "
            "a 48 canales.",
            "Cuatro bloques residuales de dos Conv1D causales (48, núcleo 3) con LayerNormalization, GELU y "
            "SpatialDropout(0.1), dilataciones 1, 2, 4 y 8.",
            "MultiHeadAttention sobre los 30 pasos temporales (4 cabezas, dimensión de clave 12) con conexión "
            "residual y LayerNorm, y después GlobalAveragePooling1D.",
            "Rama de características estáticas: Dense(32, GELU) + LayerNorm, usada para modular la secuencia agregada "
            "(compuertas de multiplicación y suma) antes de una cabeza Dense(64, GELU) + LayerNorm, Dense(32, GELU), "
            "Dense(1).",
            f"{miles(F['tcn_mha']['params'])} parámetros (unos {F['tcn_mha']['params'] / 1024:.1f} KB en int8) y unos "
            f"{miles(F['tcn_mha']['cx']['macs_stream'])} MACs por predicción; no se puede ejecutar en streaming.",
            "Entrenado en el pipeline heredado con RSAPhaseAwareLoss: Huber + una penalización de signo + una "
            "penalización de ajuste de varianza. El término de varianza premia la oscilación, una causa plausible "
            "de sus picos de RSA inventados."]),
        P("Tabla de capas (B = lote)", "h2"), layer_table(F["tcn_mha"]["layers"]),
        P("Entradas y preprocesado", "h1"),
        *bullets(["Secuencia: los últimos 30 RR, un canal, normalizados z con estadísticas globales de "
                  "data/hrv_dataset.h5.",
                  f"12 características estáticas recalculadas en cada ventana con legacy/utils2.py: "
                  f"{', '.join(F['legacy_static'])}, normalizadas z con las estadísticas de data/hrv_dataset.h5.",
                  "Objetivo: dRR[n+1], desnormalizado con la media/desviación del objetivo guardadas (sin residual "
                  "sobre el RLS)."]),
        P("Resultados", "h1"),
        table(["conjunto", "lazo abierto", "cerrado 10%", "cerrado 30%", "cerrado 50%", "cerrado 80%", "notas"], [
            ["validación (ronda 2)", fmt(vo)] + [fmt(vc[r]["rmse"]) for r in RATES] +
            [f"RSA inventada al 80%: {100 * vc['0.8']['rsa_invented']:.0f}% de los pares sujeto-ejecución"],
            ["test (final)", fmt(ft["open"]["tcn_mha_current"])] +
            [fmt(ft["closed"]["tcn_mha_current"][r]["rmse"]) for r in RATES] + ["solo float"],
            ["test serie completa, 80%", "-", "-", "-", "-", fmt(ws["rmse"]),
             f"potencia de RSA x{2 ** ws['hf']:.2f}, inventados {ws['inv']:.1f}, desplazados {ws['disp']:.1f} "
             "por sujeto"]],
            widths=[3.0 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 1.4 * cm, 6.0 * cm], bold_first=True),
        P("Correlación entre la serie original y la rellenada (sujetos de validación y test)", "h1"),
        xcorr_table(["tcn_mha_current", "tcn_lite_ft_int8", "rls_ar"], [1]), P(XC_NOTE, "sub"),
        P("Veredicto", "h1"),
        *bullets(["No válido para el MCU (atención, LayerNorm, GELU, agregación global, 75.5 KB, unos 2M MACs).",
                  "tcn_lite lo supera en 16/16 sujetos de validación en lazo abierto y en 14/16 en lazo cerrado, y "
                  "tcn_lite_ft_int8 en 10/10 sujetos de test en lazo abierto.",
                  "Inventa o desplaza picos de RSA en el lazo cerrado (visible en los espectros de "
                  "results/test/&lt;sujeto&gt;/tcn_mha_current/), el fallo que dio origen a este trabajo. Las TCNs "
                  "lo hicieron como mucho en el 4% de los pares sujeto-ejecución de validación con el 80% eliminado y "
                  "nunca en los sujetos de test."]),
    ]
    build("09_tcn_mha.pdf", "TCN_MHA (modelo de producción heredado)",
          "El modelo de producción anterior: TCN causal con atención multicabeza, juzgado como un candidato más.", story)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc_overview()
    doc_features()
    doc_baselines()
    lin = F["runs"]["linear"]["cfg"].get("linear") or {}
    doc_family("03_linear.pdf", "Modelo lineal (ridge, AR global)",
               "Regresión ridge sobre los dRR retardados: un AR(9) global, no adaptativo.",
               "El modelo aprendido más simple: un peso por entrada. Es un modelo AR global compartido por todos los "
               "pacientes, a diferencia de rls_ar, que se adapta a cada paciente.",
               ["Regresión ridge (penalización 1e-3) resuelta con ecuaciones normales en float64 acumuladas por "
                "bloques.",
                f"{len(lin.get('coef', []))} pesos + término independiente = {cx('linear').get('params')} parámetros, "
                f"{cx('linear').get('macs_stream')} MACs por latido.",
                "No residual: predice dRR directamente."],
               "linear", ["linear"], {"linear": "modelo aprendido de referencia"},
               ["Peor que rls_ar a un paso (40.5 frente a 37.1 ms) y hasta un 24.5% peor en un intervalo de edad: "
                "rechazado. Un AR global no puede seguir la dinámica de cada paciente como lo hace el RLS adaptativo."])
    doc_family("04_mlp.pdf", "MLP sobre características diseñadas",
               "Red densa solo sobre las características tabulares (sin la secuencia cruda).",
               "Un MLP de dos capas ocultas que solo ve las características diseñadas, incluidos los dRR recientes y "
               "el pronóstico RLS. Seleccionado en la ronda 1 con la regla antigua; superado por las TCNs cuando los "
               "empates pasaron a juzgarse con pruebas pareadas.",
               ["Dense(32, ReLU) - Dense(16, ReLU) - Dense(1).",
                f"{cx('mlp').get('params')} parámetros, {cx('mlp').get('macs_stream')} MACs por latido, unos 1.2 KB "
                "en int8."],
               "mlp", ["mlp", "mlp_nobp", "mlp_rsa"],
               {"mlp": "base de la ronda 1", "mlp_nobp": "sin el pasa banda fijo de 0.30 (bp, bp_prev)",
                "mlp_rsa": "mlp_nobp + las 6 características del seguidor de RSA adaptativo (sin bloqueo)"},
               ["mlp: mejor que rls_ar en todos los intervalos de edad, pero peor que las TCNs en lazo cerrado en "
                "13-15 de 16 sujetos.",
                "mlp_nobp: quitar el pasa banda fijo empeoró el lazo cerrado (p = 0.018): bp es útil.",
                "mlp_rsa: mejor a un paso (15/16 sujetos frente a mlp) pero peor con el 80% eliminado (+1.64 ms): los "
                "filtros del seguidor trabajan sobre los rellenos suaves y pierden la fase de la RSA. Llevó al "
                "diseño con bloqueo de la ronda 4."])
    doc_family("05_gbdt.pdf", "Árboles potenciados por gradiente",
               "HistGradientBoosting sobre las características tabulares: referencia de precisión y de importancia "
               "de características.",
               "Los árboles potenciados por gradiente son una referencia tabular fuerte, pero demasiado grandes para "
               "el microcontrolador (unos 8 bytes por nodo).",
               ["HistGradientBoostingRegressor de scikit-learn: error cuadrático, tasa de aprendizaje 0.05, hasta 500 "
                "árboles de profundidad 6 (31 hojas), mínimo 100 muestras por hoja, L2 1.0, parada temprana sobre el "
                "conjunto de validación.",
                "500 árboles, 30 500 nodos: unos 238 KB a 8 B por nodo, muy por encima del presupuesto de 32 KB, y "
                "unas 3000 comparaciones por predicción.",
                "Residual sobre el pronóstico RLS, como los modelos neuronales."],
               "gbdt", ["gbdt", "gbdt_all"],
               {"gbdt": "19 características tabulares", "gbdt_all": "todas las características del motor (29 entonces)"},
               ["No desplegable (tamaño). Solo referencia de precisión: 34.2 ms en validación, entre el MLP y las "
                "TCNs.",
                "gbdt_all (34.15 ms) muestra que las características extra no aportan nada; su importancia por "
                "permutación sitúa rls_pred primera con mucha diferencia, después rr_dev, drr_lag0, guzik y bp; los 8 "
                "pesos del RLS no aportan nada."])
    doc_family("06_gru.pdf", "GRU", "Una pequeña red recurrente sobre la secuencia RR más las características diseñadas.",
               "Una GRU de 16 unidades sobre la secuencia de 30 latidos. Precisa en float, pero rechazada porque la "
               "cuantización int8 le cuesta 1.6-3.6 ms.",
               ["GRU(16, desenrollada para obtener un grafo TFLite sin bucles) sobre la secuencia (30, 2); su último "
                "estado se concatena con las 9 características y después Dense(32, ReLU) - Dense(1).",
                f"{cx('gru').get('params')} parámetros, {cx('gru').get('macs_stream')} MACs por latido (streaming con "
                "estado), unos 1.8 KB en int8."],
               "gru", ["gru"], {"gru": "mismas entradas que las TCNs"},
               ["Float: empata con micro_tcn en todas las tasas de eliminación con un 19% menos de MACs y supera a "
                "mlp en 15/16 sujetos.",
                "int8: diferencia +2.71% en validación (por encima del límite del 2%) y pérdida de 1.6-3.6 ms en lazo "
                "cerrado; además inventa un pico de RSA al 50% en un sujeto: rechazada. Cuantizar el estado "
                "recurrente es notoriamente difícil en los MCUs."])
    doc_family("07_micro_tcn.pdf", "micro_tcn (TCN causal separable)",
               "TCN causal separable en profundidad, 16 filtros: la familia convolucional barata.",
               "La TCN económica: 2080 MACs por latido y 2.4 KB. Empata con la tcn_lite, más grande, en el lazo "
               "cerrado agrupado, pero pierde frente a tcn_lite_ft con el 10-30% eliminado en int8.",
               ["Tronco Conv1D puntual (16 filtros) + BatchNorm + ReLU, después 4 bloques residuales de "
                "SeparableConv1D causal (núcleo 3, dilataciones 1, 2, 4, 7) con relleno explícito a la izquierda, "
                "BatchNorm (fusionada al exportar) y ReLU.",
                "Campo receptivo 1 + 2 x (1+2+4+7) = 29 latidos, así que el modelo de ventana de 30 latidos equivale "
                "al modelo en streaming (comprobado en el autotest). Solo se lee el paso temporal más reciente, que se "
                "concatena con las 9 características, y después Dense(32, ReLU) - Dense(1).",
                f"{cx('micro_tcn').get('params')} parámetros, {cx('micro_tcn').get('macs_stream')} MACs por latido en "
                "modo streaming, 2.4 KB en int8."],
               "micro_tcn", ["micro_tcn", "micro_tcn_f8", "micro_tcn_dad", "micro_tcn_rsa", "micro_tcn_rsa_ws"],
               {"micro_tcn": "base de la ronda 1", "micro_tcn_f8": "filtros 16 -> 8",
                "micro_tcn_dad": "ajuste fino con datos rellenados en lazo cerrado (DAD)",
                "micro_tcn_rsa": "+ características de RSA con bloqueo (seq_rsa), desde cero",
                "micro_tcn_rsa_ws": "+ características de RSA con bloqueo, arranque en caliente (entradas nuevas a cero)"},
               ["micro_tcn: segundo clasificado; en int8 pierde frente a tcn_lite_ft con el 10% y el 30% eliminado "
                "(p &lt;= 0.003).",
                "micro_tcn_f8: reducir los filtros a la mitad cuesta unos 1.8 ms: rechazado.",
                "micro_tcn_dad: mejor que micro_tcn al 80% en int8 (p = 0.002) pero peor al 10%: no seleccionado.",
                "micro_tcn_rsa (desde cero, unas 3.6 pasadas sobre los datos frente a unas 48 de su padre: resultado "
                "confundido) y micro_tcn_rsa_ws (prueba limpia): sin ganancia con las características de RSA con "
                "bloqueo."])
    ft = R["final_test"]
    doc_family("08_tcn_lite.pdf", "tcn_lite (TCN causal completa) - familia seleccionada",
               "TCN causal de convolución completa, 24 filtros: el modelo seleccionado tcn_lite_ft.",
               "<b>Seleccionado: tcn_lite_ft_int8.</b> El mismo grafo que tcn_lite, con unas pocas épocas más de ajuste "
               "fino. No pierde frente a ningún otro candidato en el lazo cerrado int8 en validación. Test final (10 "
               f"sujetos): lazo abierto {fmt(ft['open']['tcn_lite_ft_int8'])} ms frente a rls_ar "
               f"{fmt(ft['open']['rls_ar'])}; lazo cerrado "
               f"{' / '.join(fmt(ft['closed']['tcn_lite_ft_int8'][r]['rmse']) for r in RATES)} ms con el 10/30/50/80% "
               "eliminado; ningún pico de RSA inventado ni desplazado.",
               ["Tronco Conv1D puntual (24 filtros) + BatchNorm + ReLU, después 4 bloques residuales de Conv1D causal "
                "completa (núcleo 3, dilataciones 1, 2, 4, 7) con relleno explícito a la izquierda, BatchNorm "
                "(fusionada al exportar) y ReLU.",
                "Campo receptivo de 29 latidos (modelo de ventana = modelo en streaming). El paso temporal más "
                "reciente se concatena con las 9 características y después Dense(32, ReLU) - Dense(1).",
                f"{cx('tcn_lite').get('params')} parámetros, {cx('tcn_lite').get('macs_stream')} MACs por latido en "
                "modo streaming (por debajo del límite de 10k), 8.4 KB en int8; int8 cuesta +0.5% en test."],
               "tcn_lite", ["tcn_lite", "tcn_lite_ft", "tcn_lite_dad", "tcn_lite_rsa", "tcn_lite_rsa_ws"],
               {"tcn_lite": "base de la ronda 1 (unas 48 pasadas completas)",
                "tcn_lite_ft": "mismo modelo, ajuste fino (lr 3e-4); control de la ejecución DAD. SELECCIONADO",
                "tcn_lite_dad": "ajuste fino con datos rellenados en lazo cerrado (DAD)",
                "tcn_lite_rsa": "+ características de RSA con bloqueo (seq_rsa), desde cero",
                "tcn_lite_rsa_ws": "+ características de RSA con bloqueo, arranque en caliente (entradas nuevas a cero)"},
               ["tcn_lite_ft_int8 supera a micro_tcn con el 10% y el 30% eliminado (p &lt;= 0.003, el float coincide) "
                "y a tcn_lite en el agrupado; solo empata con tcn_lite_dad (mismo tamaño; la ganancia de 0.28 ms de "
                "DAD al 80% no es significativa).",
                "Las características de RSA con bloqueo no aportan nada al 10-50% y cuestan 0.2 ms al 80%; "
                "tcn_lite_rsa desde cero queda confundido por un presupuesto de entrenamiento 13 veces menor.",
                "Con el 80% eliminado todos los modelos amortiguan la potencia de la RSA (aprox. x0.5): un límite de "
                "información, no un defecto; el modelo seleccionado no inventó ningún pico de RSA en validación ni en "
                "test; sus marcas de pico desplazado en validación (solo en los 2 sujetos con RSA) no fueron más "
                "frecuentes que las de rls_ar, y en test no hubo ninguna."],
               highlight=[1], extra_after=xcorr_per_subject(["tcn_lite_ft_int8", "tcn_mha_current", "rls_ar"], "val")
               + xcorr_per_subject(["tcn_lite_ft_int8", "tcn_mha_current", "rls_ar"], "test"))
    doc_tcn_mha()


if __name__ == "__main__":
    main()

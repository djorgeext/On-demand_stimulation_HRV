"""
tools/feature_details.py - the detailed "how each feature is computed" section of
docs/architectures/01_feature_engineering.pdf and its Spanish copy (one text, two languages).

It also carries the section's runnable check: reference_features() re-implements the whole streaming engine
in plain NumPy, following the formulas printed in the PDF, and check_reference() asserts that it reproduces
utils.series_features on a real validation series. If the engine and the documented formulas ever diverge,
the PDF build fails.
"""
import json
import math

import numpy as np

import utils as U

EXAMPLE_SUBJECT = "16273"  # validation subject with a clear RSA peak
EXAMPLE_BEAT = 2000        # 0-based index of the newest beat of the example window
CHECK_BEATS = 4000


# ----------------------------------------------------------------------------- reference implementation
def reference_features(rr, engine=None, tracker=None):
    """The engine, written as the PDF describes it (open loop: no beat is imputed). Same rows as series_features."""
    e, tk = engine or U.ENGINE, tracker or U.RSA_TRACKER
    W, p, n_lags = e["window"], e["ar_order"], e["n_drr_lags"]
    lam, p0, pmax, scale, clamp = e["rls_lambda"], e["rls_p0"], e["rls_p_max_trace"], e["rls_scale"], e["rls_pred_clamp"]
    b0, a1, a2 = U._rbj_bandpass(e["bp_f0"], e["bp_q"])
    K = tk["n_bands"]
    fc = np.linspace(tk["f_lo"], tk["f_hi"], K)
    bank = np.array([U._rbj_bandpass(f, tk["q"]) for f in fc])  # (K, 3): b0, a1, a2
    beta = tk["beta"]
    w, P, phi, pred_prev = np.zeros(p), np.eye(p) * p0, np.zeros(p), 0.0
    x1 = x2 = y1 = y2 = 0.0
    yb1, yb2, pw = np.zeros(K), np.zeros(K), np.zeros(K)
    rows = []
    for n, x in enumerate(rr[:-1]):  # the last beat has no target
        err = 0.0
        if n >= W:  # RLS update with the error of the previous forecast
            err = (x - rr[n - 1]) - pred_prev
            Pphi = P @ phi
            den = lam + phi @ Pphi
            tr = 0.0
            if den >= lam:
                k = Pphi / den
                w = w + k * err / scale
                P = (P - np.outer(k, Pphi)) / lam
                P = 0.5 * (P + P.T)
                tr = np.trace(P)
            if not (tr > 0 and np.isfinite(tr)):
                P = np.eye(p) * p0
            elif tr > pmax:
                P = P * (pmax / tr)
        if n == 0:
            x1 = x2 = x
        bp_prev = y1
        y = b0 * (x - x2) - a1 * y1 - a2 * y2
        yb = bank[:, 0] * (x - x2) - bank[:, 1] * yb1 - bank[:, 2] * yb2
        yb2, yb1 = yb1, yb
        pw = beta * pw + (1 - beta) * yb ** 2
        x2, x1, y2, y1 = x1, x, y1, y
        if n + 1 < W:
            continue
        win = rr[n + 1 - W:n + 1]
        d = np.diff(win)
        mean = win.mean()
        std = win.std()
        cum = np.cumsum(win - mean)
        u, v = (win[:-1] - win[1:]) / math.sqrt(2), (win[:-1] + win[1:]) / math.sqrt(2)
        sd1, sd2 = u.std(), v.std()
        area = 0.5 * np.abs(d[:-2] * (win[3:] - win[1:-2]) - d[1:-1] * (win[2:-1] - win[:-3])).sum()
        den_c = math.pi * sd1 * sd2 * (W - 3)
        run = 0.0
        if d[-1] != 0:
            s = np.sign(d[-1])
            same = (d[::-1] * s) > 0
            run = s * (np.argmin(same) if not same.all() else len(same))
        phi = (win[::-1][:p] - mean) / scale
        pred = scale * (w @ phi)
        pred_prev = pred
        nz = (d != 0).sum()
        f = [mean, win[-1] - mean, math.sqrt(np.mean(d ** 2)),
             (cum.max() - cum.min()) / std if std > 0 else 0.0,
             min(max(area / den_c, 0.0), 1.0) if den_c > 0 else 0.0,
             d[d > 0].sum() / np.abs(d).sum() if nz else 0.0,
             (d < 0).sum() / nz if nz else 0.0,
             float((np.abs(d) > 20).sum()), run,
             min(max(pred, -clamp), clamp), err, y, bp_prev]
        f += list(d[::-1][:n_lags]) + list(w)
        # RSA tracker
        lp = np.log2(pw + 1e-8)
        slope = np.sum((fc - fc.mean()) * (lp - lp.mean())) / np.sum((fc - fc.mean()) ** 2)
        resid = lp - lp.mean() - slope * (fc - fc.mean())
        best = 1 + int(np.argmax(resid[1:-1]))
        curv = lp[best - 1] - 2 * lp[best] + lp[best + 1]
        off = float(np.clip(0.5 * (lp[best - 1] - lp[best + 1]) / curv, -0.5, 0.5)) if curv < 0 else 0.0
        f_rsa = fc[best] + off * (fc[1] - fc[0])
        amp = math.sqrt(2 * pw[best])
        om = 2 * math.pi * f_rsa
        yi, yq = yb1[best], (yb2[best] - yb1[best] * math.cos(om)) / math.sin(om)
        mag = math.hypot(yi, yq)
        c, s_ = (yi / mag, yq / mag) if mag > 1e-8 else (1.0, 0.0)
        f += [f_rsa, amp, c, s_, resid[best], amp * (math.cos(om) * c - math.sin(om) * s_ - c)]
        rows.append(f)
    return np.array(rows)


def example_series():
    codes = [c for _, c in U.read_manifest(U.DATA_DIR / U.MANIFESTS["val"])]
    assert EXAMPLE_SUBJECT in codes, f"{EXAMPLE_SUBJECT} is not a validation subject"
    return U.load_series(U.DATA_DIR / U.SERIES_DIR / f"{EXAMPLE_SUBJECT}.txt")


def check_reference(rr):
    """The one check of this section: the documented formulas reproduce the engine (all 35 features)."""
    F, _ = U.series_features(rr[:CHECK_BEATS])
    G = reference_features(rr[:CHECK_BEATS])
    assert F.shape == G.shape, (F.shape, G.shape)
    err = np.abs(F - G) / (1.0 + np.abs(F))
    worst = int(np.argmax(err.max(axis=0)))
    assert err.max() < 1e-6, f"{U.FEATURE_NAMES[worst]} differs from the engine by {err.max():.2e}"
    return F


def train_stats():
    """Training-split mean and std of every feature, as fitted by fit_norm (stored in run configs)."""
    out = {}
    for run in ("gbdt_all", "mlp_rsa"):
        c = json.loads((U.RUNS_DIR / run / "config.json").read_text())
        for f, m, s in zip(c["features"], c["norm"]["feat_mean"], c["norm"]["feat_std"]):
            out.setdefault(f, (m, s))
    return out


# ----------------------------------------------------------------------------- the PDF section
def story(lang, P, bullets, table, cm):
    def t(en, es):
        return es if lang == "es" else en

    def g(ch):  # Greek / math glyph from the Symbol font
        return f'<font face="Symbol">{ch}</font>'

    def eq(s):
        return P(s, "formula")

    def h(title):
        return P(title, "h2")

    al, be, ph, om, la, pi, Si, De, ne, sq = (g(c) for c in "αβφωλπΣΔ≠√")
    e, tk = U.ENGINE, U.RSA_TRACKER
    W = e["window"]
    b0, a1, a2 = U._rbj_bandpass(e["bp_f0"], e["bp_q"])
    fc = np.linspace(tk["f_lo"], tk["f_hi"], tk["n_bands"])
    rr = example_series()
    F = check_reference(rr)
    stats = train_stats()
    j = EXAMPLE_BEAT - (W - 1)  # feature row of the example window
    win = rr[EXAMPLE_BEAT + 1 - W:EXAMPLE_BEAT + 1]
    val = dict(zip(U.FEATURE_NAMES, F[j]))
    d = np.diff(win)
    half = (len(U.FEATURE_NAMES) + 1) // 2
    out = [
        P(t("3. How each feature is computed", "3. Cómo se calcula cada característica"), "h1"),
        P(t(f"Notation: when beat n arrives, the window holds its last W = {W} RR intervals, RR<sub>0</sub> (oldest) to "
            f"RR<sub>{W - 1}</sub> = RR[n] (newest), in ms. d<sub>i</sub> = RR<sub>i+1</sub> - RR<sub>i</sub> are its "
            f"{W - 1} successive differences (dRR). Every feature of beat n uses only beats up to n (causal) and is "
            "the input for predicting dRR[n+1]. Window statistics are recomputed from the 30 values at each beat; the "
            "RLS, the band-pass and the RSA tracker are recursive: they keep a small state that is updated once per "
            "beat. Every formula below is checked when this document is generated: a NumPy re-implementation written from "
            f"these formulas must reproduce the engine on {CHECK_BEATS} beats of validation subject {EXAMPLE_SUBJECT} "
            "(relative difference below 1e-6 for all 35 features).",
            f"Notación: cuando llega el latido n, la ventana contiene sus últimos W = {W} intervalos RR, de "
            f"RR<sub>0</sub> (el más antiguo) a RR<sub>{W - 1}</sub> = RR[n] (el más reciente), en ms. d<sub>i</sub> = "
            f"RR<sub>i+1</sub> - RR<sub>i</sub> son sus {W - 1} diferencias sucesivas (dRR). Cada característica del "
            "latido n usa solo latidos hasta n (causal) y es la entrada para predecir dRR[n+1]. Las estadísticas de "
            "ventana se recalculan con los 30 valores en cada latido; el RLS, el pasa banda y el seguidor de RSA son "
            "recursivos: guardan un pequeño estado que se actualiza una vez por latido. Cada fórmula de abajo se "
            "comprueba al generar este documento: una reimplementación en NumPy escrita a partir de estas fórmulas debe "
            f"reproducir el motor en {CHECK_BEATS} latidos del sujeto de validación {EXAMPLE_SUBJECT} (diferencia "
            "relativa menor que 1e-6 en las 35 características)."), "box"),

        h(t("3.1 Level and variability of the window", "3.1 Nivel y variabilidad de la ventana")),
        P("<b>rr_mean</b> (ms)"),
        eq(f"rr_mean = (1/{W}) {Si}<sub>i=0..{W - 1}</sub> RR<sub>i</sub>"),
        P(t("The local heart period. It is also the centre the RLS regressors are measured from.",
            "El periodo cardiaco local. Es también el centro desde el que se miden los regresores del RLS.")),
        P("<b>rr_dev</b> (ms)"),
        eq(f"rr_dev = RR<sub>{W - 1}</sub> - rr_mean"),
        P(t("How far the newest beat sits from the local level. A positive value (a long beat) tends to be followed by "
            "a shortening, so it carries mean reversion.",
            "Cuánto se aleja el latido más reciente del nivel local. Un valor positivo (un latido largo) suele ir "
            "seguido de un acortamiento, así que aporta la reversión a la media.")),
        P("<b>rmssd</b> (ms)"),
        eq(f"rmssd = {sq}( (1/{W - 1}) {Si}<sub>i=0..{W - 2}</sub> d<sub>i</sub><super>2</super> )"),
        P(t("Root mean square of the 29 successive differences: the short-term (beat-to-beat) variability, the classic "
            "vagal index. It sets the expected size of the next dRR.",
            "Raíz cuadrática media de las 29 diferencias sucesivas: la variabilidad a corto plazo (latido a latido), el "
            "índice vagal clásico. Fija el tamaño esperado del siguiente dRR.")),
        P("<b>nn20</b> (count, 0-29)"),
        eq(f"nn20 = #{{ i : |d<sub>i</sub>| &gt; 20 ms }}"),
        P(t("How many of the 29 differences exceed 20 ms (strictly). A count, not a percentage.",
            "Cuántas de las 29 diferencias superan 20 ms (estrictamente). Un recuento, no un porcentaje.")),
        P("<b>rs</b> " + t("(rescaled range, no unit)", "(rango reescalado, sin unidad)")),
        eq(f"C<sub>k</sub> = {Si}<sub>i=0..k</sub> (RR<sub>i</sub> - rr_mean),  k = 0..{W - 1};   "
           f"rs = (max<sub>k</sub> C<sub>k</sub> - min<sub>k</sub> C<sub>k</sub>) / s"),
        P(t(f"s is the population standard deviation of the window ({sq}((1/30) {Si}(RR<sub>i</sub> - rr_mean)"
            "<super>2</super>)). The cumulative deviation C<sub>k</sub> wanders far when the RR stay on one side of "
            "the mean for long stretches (a trend or a slow wave) and stays small when they alternate, so rs measures "
            "persistence (Hurst-type analysis). rs = 0 if all 30 RR are equal.",
            f"s es la desviación estándar poblacional de la ventana ({sq}((1/30) {Si}(RR<sub>i</sub> - rr_mean)"
            "<super>2</super>)). La desviación acumulada C<sub>k</sub> se aleja mucho cuando los RR se quedan a un "
            "lado de la media durante tramos largos (una tendencia o una onda lenta) y se mantiene pequeña cuando "
            "alternan, así que rs mide la persistencia (análisis de tipo Hurst). rs = 0 si los 30 RR son iguales.")),

        h(t("3.2 Shape of the beat-to-beat changes", "3.2 Forma de los cambios latido a latido")),
        P("<b>guzik</b> " + t("(0-1)", "(0-1)")),
        eq(f"guzik = {Si}<sub>d<sub>i</sub> &gt; 0</sub> d<sub>i</sub> / {Si}<sub>i</sub> |d<sub>i</sub>|"),
        P(t("Share of the total beat-to-beat change that is lengthening (deceleration). 0.5 means symmetric "
            "accelerations and decelerations; 0 if all differences are zero.",
            "Proporción del cambio total latido a latido que es alargamiento (deceleración). 0.5 significa "
            "aceleraciones y deceleraciones simétricas; 0 si todas las diferencias son cero.")),
        P("<b>porta</b> " + t("(0-1)", "(0-1)")),
        eq(f"porta = #{{d<sub>i</sub> &lt; 0}} / #{{d<sub>i</sub> {ne} 0}}"),
        P(t("Share of the non-zero differences that are shortenings (accelerations). Zero differences are ignored; 0 "
            "if all are zero.", "Proporción de las diferencias no nulas que son acortamientos (aceleraciones). Las "
            "diferencias nulas se ignoran; 0 si todas son cero.")),
        P("<b>run_len</b> " + t("(signed count, -29 to +29)", "(recuento con signo, -29 a +29)")),
        eq(t(f"run_len = sign(d<sub>{W - 2}</sub>) · (number of consecutive differences, counted back from "
             f"d<sub>{W - 2}</sub>, with that same sign)",
             f"run_len = signo(d<sub>{W - 2}</sub>) · (número de diferencias consecutivas, contadas hacia atrás desde "
             f"d<sub>{W - 2}</sub>, con ese mismo signo)")),
        P(t("+3 means the last 3 beats each got longer (decelerating for 3 beats); -2 means the last 2 got shorter. "
            "0 if the newest difference is exactly 0. It tells the model where it is inside a slow wave.",
            "+3 significa que cada uno de los 3 últimos latidos se alargó (3 latidos decelerando); -2 que los 2 "
            "últimos se acortaron. 0 si la diferencia más reciente es exactamente 0. Indica al modelo en qué punto de "
            "una onda lenta se encuentra.")),
        P("<b>drr_lag0..7</b> (ms)"),
        eq(f"drr_lag<sub>k</sub> = RR<sub>{W - 1}-k</sub> - RR<sub>{W - 2}-k</sub> = d<sub>{W - 2}-k</sub>,  k = 0..7"),
        P(t("The 8 most recent beat-to-beat changes, newest first (drr_lag0 is the dRR that just happened).",
            "Los 8 cambios latido a latido más recientes, el más nuevo primero (drr_lag0 es el dRR que acaba de "
            "ocurrir).")),
        P("<b>ccm</b> " + t("(complex correlation measure, 0-1)", "(medida de correlación compleja, 0-1)")),
        eq(f"Poincaré: P<sub>i</sub> = (RR<sub>i</sub>, RR<sub>i+1</sub>), i = 0..{W - 2};   "
           f"SD1 = std<sub>i</sub>((RR<sub>i</sub> - RR<sub>i+1</sub>)/{sq}2),   "
           f"SD2 = std<sub>i</sub>((RR<sub>i</sub> + RR<sub>i+1</sub>)/{sq}2)"),
        eq(t(f"A<sub>i</sub> = area of the triangle P<sub>i</sub>, P<sub>i+1</sub>, P<sub>i+2</sub> = ½ |u<sub>x</sub> "
             f"v<sub>y</sub> - u<sub>y</sub> v<sub>x</sub>|, u = P<sub>i+1</sub> - P<sub>i</sub>, v = P<sub>i+2</sub> "
             f"- P<sub>i</sub>, i = 0..{W - 4}",
             f"A<sub>i</sub> = área del triángulo P<sub>i</sub>, P<sub>i+1</sub>, P<sub>i+2</sub> = ½ |u<sub>x</sub> "
             f"v<sub>y</sub> - u<sub>y</sub> v<sub>x</sub>|, u = P<sub>i+1</sub> - P<sub>i</sub>, v = P<sub>i+2</sub> "
             f"- P<sub>i</sub>, i = 0..{W - 4}")),
        eq(f"ccm = clip( {Si}<sub>i</sub> A<sub>i</sub> / ({pi} · SD1 · SD2 · {W - 3}), 0, 1 )"),
        P(t("The 29 Poincaré points form 27 triangles of consecutive points. If the RR follow a smooth, predictable "
            "pattern the points lie on a line and the triangles are flat; irregular, non-linear dynamics open them up. "
            "The sum of their areas is normalised by the area of the SD1 x SD2 Poincaré ellipse (std = population "
            "standard deviation over the 29 points). 0 if SD1 or SD2 is 0.",
            "Los 29 puntos de Poincaré forman 27 triángulos de puntos consecutivos. Si los RR siguen un patrón suave y "
            "predecible, los puntos quedan sobre una línea y los triángulos son planos; una dinámica irregular y no "
            "lineal los abre. La suma de sus áreas se normaliza por el área de la elipse de Poincaré SD1 x SD2 (std = "
            "desviación estándar poblacional sobre los 29 puntos). 0 si SD1 o SD2 es 0.")),

        h(t("3.3 Adaptive RLS AR(8) forecast", "3.3 Pronóstico adaptativo RLS AR(8)")),
        P(t("A per-patient autoregressive model whose 8 weights w are re-estimated at every beat by recursive least "
            "squares. It runs in three steps per beat:",
            "Un modelo autorregresivo por paciente cuyos 8 pesos w se reestiman en cada latido por mínimos cuadrados "
            "recursivos. Funciona en tres pasos por latido:")),
        *bullets([
            t(f"<b>1. Error of the previous forecast</b> (when beat n arrives): e = (RR[n] - RR[n-1]) - "
              f"pred<sub>n-1</sub>, using the unclamped forecast. This is <b>rls_err</b> (ms); 0 for the first window.",
              f"<b>1. Error del pronóstico anterior</b> (al llegar el latido n): e = (RR[n] - RR[n-1]) - "
              f"pred<sub>n-1</sub>, con el pronóstico sin recortar. Es <b>rls_err</b> (ms); 0 en la primera ventana."),
            t(f"<b>2. Update</b> with the regressor vector {ph} of the previous forecast: g = P{ph} / ({la} + "
              f"{ph}<super>T</super>P{ph}), w &lt;- w + g · e/{e['rls_scale']:g}, P &lt;- (P - g{ph}<super>T</super>P) / "
              f"{la}, with {la} = {e['rls_lambda']} (about 100 beats of memory), P<sub>0</sub> = {e['rls_p0']:g} I and "
              f"w<sub>0</sub> = 0. Guards: the update is skipped and P reset to P<sub>0</sub> if {ph}<super>T</super>P{ph} "
              f"&lt; 0 or its trace is not finite; P is symmetrised each beat and scaled down if its trace exceeds "
              f"{e['rls_p_max_trace']:g}.",
              f"<b>2. Actualización</b> con el vector de regresores {ph} del pronóstico anterior: g = P{ph} / ({la} + "
              f"{ph}<super>T</super>P{ph}), w &lt;- w + g · e/{e['rls_scale']:g}, P &lt;- (P - g{ph}<super>T</super>P) / "
              f"{la}, con {la} = {e['rls_lambda']} (unos 100 latidos de memoria), P<sub>0</sub> = {e['rls_p0']:g} I y "
              f"w<sub>0</sub> = 0. Protecciones: la actualización se omite y P se reinicia a P<sub>0</sub> si "
              f"{ph}<super>T</super>P{ph} &lt; 0 o su traza no es finita; P se simetriza en cada latido y se reduce si su "
              f"traza supera {e['rls_p_max_trace']:g}."),
            t(f"<b>3. New forecast</b> on the window that now ends at beat n: {ph}<sub>k</sub> = (RR<sub>{W - 1}-k</sub> - "
              f"rr_mean) / {e['rls_scale']:g}, k = 0..7 (the last 8 RR, centred on the window mean); pred = "
              f"{e['rls_scale']:g} · {Si}<sub>k</sub> w<sub>k</sub> {ph}<sub>k</sub>. <b>rls_pred</b> = clip(pred, "
              f"-{e['rls_pred_clamp']:g}, +{e['rls_pred_clamp']:g}) ms, the forecast of dRR[n+1]. <b>rls_w0..7</b> are "
              "the weights w just used (dimensionless): they describe the patient's current dynamics.",
              f"<b>3. Nuevo pronóstico</b> sobre la ventana que ahora termina en el latido n: {ph}<sub>k</sub> = "
              f"(RR<sub>{W - 1}-k</sub> - rr_mean) / {e['rls_scale']:g}, k = 0..7 (los últimos 8 RR, centrados en la "
              f"media de la ventana); pred = {e['rls_scale']:g} · {Si}<sub>k</sub> w<sub>k</sub> {ph}<sub>k</sub>. "
              f"<b>rls_pred</b> = clip(pred, -{e['rls_pred_clamp']:g}, +{e['rls_pred_clamp']:g}) ms, el pronóstico de "
              "dRR[n+1]. <b>rls_w0..7</b> son los pesos w recién usados (sin unidad): describen la dinámica actual del "
              "paciente."),
        ]),

        h(t("3.4 Fixed RSA band-pass", "3.4 Pasa banda RSA fijo")),
        eq(f"y[n] = b<sub>0</sub> (RR[n] - RR[n-2]) - a<sub>1</sub> y[n-1] - a<sub>2</sub> y[n-2]"),
        P(t(f"A second-order RBJ band-pass run on the raw RR series: centre f<sub>0</sub> = {e['bp_f0']} cycles/beat "
            f"({1 / e['bp_f0']:.1f} beats per breath), Q = {e['bp_q']}, so {om}<sub>0</sub> = 2{pi}f<sub>0</sub>, "
            f"{al} = sin {om}<sub>0</sub> / 2Q, b<sub>0</sub> = {al}/(1+{al}) = {b0:.5f}, a<sub>1</sub> = "
            f"-2cos {om}<sub>0</sub>/(1+{al}) = {a1:.5f}, a<sub>2</sub> = (1-{al})/(1+{al}) = {a2:.5f}. Gain 1 at "
            "f<sub>0</sub> and 0 at both ends (constant level and 0.5 cycles/beat), so no detrending is needed; the "
            "-3 dB band is about 0.15-0.45 cycles/beat. The input history starts at the first RR, so there is no "
            "start-up step. <b>bp</b> = y[n] and <b>bp_prev</b> = y[n-1] (ms): together they give the amplitude and "
            "phase of the respiratory oscillation.",
            f"Un pasa banda RBJ de segundo orden aplicado a la serie RR cruda: centro f<sub>0</sub> = {e['bp_f0']} "
            f"ciclos/latido ({1 / e['bp_f0']:.1f} latidos por respiración), Q = {e['bp_q']}, de modo que "
            f"{om}<sub>0</sub> = 2{pi}f<sub>0</sub>, {al} = sen {om}<sub>0</sub> / 2Q, b<sub>0</sub> = {al}/(1+{al}) = "
            f"{b0:.5f}, a<sub>1</sub> = -2cos {om}<sub>0</sub>/(1+{al}) = {a1:.5f}, a<sub>2</sub> = (1-{al})/(1+{al}) = "
            f"{a2:.5f}. Ganancia 1 en f<sub>0</sub> y 0 en ambos extremos (nivel constante y 0.5 ciclos/latido), así "
            "que no hace falta quitar la tendencia; la banda a -3 dB es aprox. 0.15-0.45 ciclos/latido. La historia de "
            "entrada empieza en el primer RR, así que no hay escalón inicial. <b>bp</b> = y[n] y <b>bp_prev</b> = "
            "y[n-1] (ms): juntas dan la amplitud y la fase de la oscilación respiratoria.")),

        h(t("3.5 Patient-adaptive RSA tracker", "3.5 Seguidor de RSA adaptativo por paciente")),
        *bullets([
            t(f"<b>Bank</b>: {tk['n_bands']} band-passes of the same form with Q = {tk['q']} and centres "
              f"f<sub>k</sub> = {', '.join(f'{f:.3f}' for f in fc)} cycles/beat ({De}f = {fc[1] - fc[0]:.4f}). Each "
              f"band's power is an exponential average p<sub>k</sub> &lt;- {be} p<sub>k</sub> + (1-{be}) "
              f"y<sub>k</sub>[n]<super>2</super>, {be} = {tk['beta']} (about 100 beats); in the closed loop it is "
              "frozen on filled beats.",
              f"<b>Banco</b>: {tk['n_bands']} pasa banda de la misma forma con Q = {tk['q']} y centros "
              f"f<sub>k</sub> = {', '.join(f'{f:.3f}' for f in fc)} ciclos/latido ({De}f = {fc[1] - fc[0]:.4f}). La "
              f"potencia de cada banda es una media exponencial p<sub>k</sub> &lt;- {be} p<sub>k</sub> + (1-{be}) "
              f"y<sub>k</sub>[n]<super>2</super>, {be} = {tk['beta']} (unos 100 latidos); en el lazo cerrado se "
              "congela en los latidos rellenados."),
            t(f"<b>rsa_q</b> (log2 units): L<sub>k</sub> = log2(p<sub>k</sub> + 1e-8); a least-squares line "
              "L ~ a + b·f is fitted over the 8 bands (the background spectrum); the residual r<sub>k</sub> = "
              "L<sub>k</sub> - line(f<sub>k</sub>) is searched over the 6 interior bands, and rsa_q = max r<sub>k</sub>: "
              "how far the strongest band rises above the trend (1 = twice the power). The peak band is k*.",
              f"<b>rsa_q</b> (unidades log2): L<sub>k</sub> = log2(p<sub>k</sub> + 1e-8); se ajusta por mínimos "
              "cuadrados una recta L ~ a + b·f sobre las 8 bandas (el espectro de fondo); el residuo r<sub>k</sub> = "
              "L<sub>k</sub> - recta(f<sub>k</sub>) se busca en las 6 bandas interiores, y rsa_q = max r<sub>k</sub>: "
              "cuánto sobresale la banda más fuerte sobre la tendencia (1 = el doble de potencia). La banda del pico es "
              "k*."),
            t(f"<b>rsa_f</b> (cycles/beat): parabolic interpolation of L around k*: offset = ½ (L<sub>k*-1</sub> - "
              f"L<sub>k*+1</sub>) / (L<sub>k*-1</sub> - 2L<sub>k*</sub> + L<sub>k*+1</sub>) when the denominator is "
              f"negative (a true peak), else 0, clipped to {chr(177)}0.5; rsa_f = f<sub>k*</sub> + offset · {De}f.",
              f"<b>rsa_f</b> (ciclos/latido): interpolación parabólica de L alrededor de k*: desplazamiento = ½ "
              f"(L<sub>k*-1</sub> - L<sub>k*+1</sub>) / (L<sub>k*-1</sub> - 2L<sub>k*</sub> + L<sub>k*+1</sub>) cuando "
              f"el denominador es negativo (un pico real), si no 0, recortado a {chr(177)}0.5; rsa_f = f<sub>k*</sub> + "
              f"desplazamiento · {De}f."),
            t(f"<b>rsa_amp</b> (ms): {sq}(2 p<sub>k*</sub>), the amplitude of a sinusoid whose mean power is "
              "p<sub>k*</sub>.", f"<b>rsa_amp</b> (ms): {sq}(2 p<sub>k*</sub>), la amplitud de una sinusoide cuya "
              "potencia media es p<sub>k*</sub>."),
            t(f"<b>rsa_cos, rsa_sin</b>: the phase {ph} of the peak band's oscillation from its last two outputs, "
              f"treating them as samples of A cos({ph}) and A cos({ph} - {om}) with {om} = 2{pi}·rsa_f: A cos {ph} = "
              f"y[n], A sin {ph} = (y[n-1] - y[n] cos {om}) / sin {om}; then normalised to cos {ph} and sin {ph} "
              "(1 and 0 if the band is silent).",
              f"<b>rsa_cos, rsa_sin</b>: la fase {ph} de la oscilación de la banda del pico a partir de sus dos últimas "
              f"salidas, tratadas como muestras de A cos({ph}) y A cos({ph} - {om}) con {om} = 2{pi}·rsa_f: A cos {ph} = "
              f"y[n], A sen {ph} = (y[n-1] - y[n] cos {om}) / sen {om}; después se normalizan a cos {ph} y sen {ph} (1 y "
              "0 si la banda está en silencio)."),
            t(f"<b>rsa_drr</b> (ms): the RSA part of the next change, rsa_amp · (cos({ph} + {om}) - cos {ph}): how much "
              "the respiratory sinusoid alone would move the next RR.",
              f"<b>rsa_drr</b> (ms): la parte de RSA del siguiente cambio, rsa_amp · (cos({ph} + {om}) - cos {ph}): "
              "cuánto movería el siguiente RR la sinusoide respiratoria por sí sola."),
            t(f"<b>Gate</b> (closed loop only): when {tk['gate_min_imputed']} or more of the last {W} beats were filled "
              "by the model, rsa_q = rsa_drr = 0, because the phase can no longer be recovered. Training and the open "
              "loop never fill, so the gate never acts there.",
              f"<b>Bloqueo</b> (solo en lazo cerrado): cuando {tk['gate_min_imputed']} o más de los últimos {W} latidos "
              "fueron rellenados por el modelo, rsa_q = rsa_drr = 0, porque la fase ya no se puede recuperar. El "
              "entrenamiento y el lazo abierto nunca rellenan, así que el bloqueo nunca actúa ahí."),
        ]),

        h(t("3.6 Training-split statistics", "3.6 Estadísticas en el conjunto de entrenamiento")),
        P(t("Mean and standard deviation of each feature over the 20.8M training windows, as fitted by the "
            "normalisation (fit_norm) and stored in runs/gbdt_all and runs/mlp_rsa. Every model input is (value - "
            "mean) / std.",
            "Media y desviación estándar de cada característica sobre las 20.8M ventanas de entrenamiento, tal como "
            "las ajusta la normalización (fit_norm) y quedan guardadas en runs/gbdt_all y runs/mlp_rsa. Cada entrada "
            "del modelo es (valor - media) / desviación.")),
        table([t("feature", "característica"), t("unit", "unidad"), t("mean", "media"), t("std", "desviación")],
              [(f, unit(f, lang), f"{stats[f][0]:.4g}", f"{stats[f][1]:.4g}") for f in U.FEATURE_NAMES],
              widths=[3.0 * cm, 3.4 * cm, 2.6 * cm, 2.6 * cm], bold_first=True),

        h(t(f"3.7 Worked example: validation subject {EXAMPLE_SUBJECT}, beat {EXAMPLE_BEAT}",
            f"3.7 Ejemplo resuelto: sujeto de validación {EXAMPLE_SUBJECT}, latido {EXAMPLE_BEAT}")),
        P(t(f"The window (RR<sub>0</sub> ... RR<sub>{W - 1}</sub>, ms) and the 29 differences d<sub>i</sub>:",
            f"La ventana (RR<sub>0</sub> ... RR<sub>{W - 1}</sub>, ms) y las 29 diferencias d<sub>i</sub>:")),
        table(["i"] + [str(i) for i in range(10)],
              [[f"RR {r * 10}-{r * 10 + 9}"] + [f"{v:.0f}" for v in win[r * 10:r * 10 + 10]] for r in range(3)]
              + [[f"d {r * 10}-{min(r * 10 + 9, W - 2)}"] + [f"{v:+.0f}" for v in d[r * 10:r * 10 + 10]] for r in range(3)],
              widths=[2.2 * cm] + [1.2 * cm] * 10, bold_first=True),
        P(t("Hand check of the simplest ones: ", "Comprobación a mano de las más sencillas: ")
          + f"rr_mean = {win.sum():.0f} / {W} = {val['rr_mean']:.2f}; rr_dev = {win[-1]:.0f} - {val['rr_mean']:.2f} = "
          f"{val['rr_dev']:.2f}; rmssd = {sq}({(d ** 2).sum():.0f} / {W - 1}) = {val['rmssd']:.2f}; nn20 = "
          f"{int(val['nn20'])}; drr_lag0 = {win[-1]:.0f} - {win[-2]:.0f} = {val['drr_lag0']:+.0f}; run_len = "
          f"{val['run_len']:+.0f}."),
        table([t("feature", "característica"), t("value", "valor"), t("feature", "característica"), t("value", "valor")],
              [(a, f"{val[a]:.4g}", b, f"{val[b]:.4g}" if b else "")
               for a, b in zip(U.FEATURE_NAMES[:half], U.FEATURE_NAMES[half:] + [""])],
              widths=[3.0 * cm, 2.6 * cm, 3.0 * cm, 2.6 * cm], bold_first=False),
        P(t(f"Here the RLS forecasts dRR[n+1] = {val['rls_pred']:+.1f} ms; the true next change was "
            f"{rr[EXAMPLE_BEAT + 1] - rr[EXAMPLE_BEAT]:+.0f} ms. The tracker sees an RSA peak at {val['rsa_f']:.3f} "
            f"cycles/beat ({1 / val['rsa_f']:.1f} beats per breath), {val['rsa_q']:.2f} log2 above the background, "
            f"amplitude {val['rsa_amp']:.1f} ms, and expects the oscillation alone to move the next RR by "
            f"{val['rsa_drr']:+.1f} ms.",
            f"Aquí el RLS pronostica dRR[n+1] = {val['rls_pred']:+.1f} ms; el cambio siguiente real fue "
            f"{rr[EXAMPLE_BEAT + 1] - rr[EXAMPLE_BEAT]:+.0f} ms. El seguidor ve un pico de RSA en {val['rsa_f']:.3f} "
            f"ciclos/latido ({1 / val['rsa_f']:.1f} latidos por respiración), {val['rsa_q']:.2f} log2 por encima del "
            f"fondo, amplitud {val['rsa_amp']:.1f} ms, y espera que la oscilación por sí sola mueva el siguiente RR "
            f"{val['rsa_drr']:+.1f} ms.")),
    ]
    return out


def unit(f, lang):
    es = lang == "es"
    if f in ("rr_mean", "rr_dev", "rmssd", "rls_pred", "rls_err", "bp", "bp_prev", "rsa_amp", "rsa_drr") \
            or f.startswith("drr_lag"):
        return "ms"
    if f in ("nn20",):
        return "recuento" if es else "count"
    if f == "run_len":
        return "recuento con signo" if es else "signed count"
    if f == "rsa_f":
        return "ciclos/latido" if es else "cycles/beat"
    if f == "rsa_q":
        return "log2"
    return "sin unidad" if es else "none"

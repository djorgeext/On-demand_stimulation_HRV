"""
tools/glossary.py - every acronym / abbreviation the architecture PDFs may use, with its meaning.

make_architecture_pdfs.py appends to each PDF the entries that occur in that PDF, and refuses to build
a PDF that contains an acronym-like token that is neither defined here nor in ALLOW (non-acronyms:
proper names, Keras class names, code identifiers). So a new acronym cannot slip in undefined.
"""
import html
import re

# (term, meaning, how to match): "word" = whole token, case-sensitive; "sub" = substring (run-name parts)
GLOSSARY = [
    ("1D", "one-dimensional (a layer that slides along the time axis only, e.g. Conv1D)", "word"),
    ("AdamW", "Adam optimiser with decoupled weight decay (the training algorithm of the neural models)", "word"),
    ("AR", "autoregressive: a value predicted as a weighted sum of past values; AR(p) uses p past values", "word"),
    ("Burg-AR", "AR model whose coefficients were fitted with Burg's method (used for the first 30 beats)", "word"),
    ("ccm", "complex correlation measure: non-linearity index of the Poincare plot of consecutive RR intervals", "word"),
    ("CCM", "complex correlation measure (see ccm)", "word"),
    ("CMSIS-NN", "Cortex Microcontroller Software Interface Standard - Neural Network kernels: Arm's optimised "
                 "neural-network library for Cortex-M microcontrollers", "word"),
    ("Conv1D", "one-dimensional convolution layer", "word"),
    ("Cortex-M7", "Arm Cortex-M7, the processor core of the STM32H750 microcontroller", "word"),
    ("CPU", "central processing unit", "word"),
    ("CSV", "comma-separated values (plain-text table file)", "word"),
    ("DAD", "data as demonstrator: training on inputs that contain the model's own closed-loop fills, with the "
            "true target (a remedy for exposure bias)", "word"),
    ("_dad", "run-name suffix: fine-tuned with DAD (closed-loop-filled) training data", "sub"),
    ("DC", "direct current, i.e. the constant (zero-frequency) component of a signal", "word"),
    ("dRR", "beat-to-beat change of the RR interval: dRR[n+1] = RR[n+1] - RR[n] (ms); the quantity predicted", "word"),
    ("ECG", "electrocardiogram", "word"),
    ("EMA", "exponential moving average", "word"),
    ("ema_mean", "baseline predicting the next RR as an exponential moving average of the window", "word"),
    ("_f8", "run-name suffix: 8 convolution filters instead of 16", "sub"),
    ("FLOPs", "floating-point operations (arithmetic cost)", "word"),
    ("_ft", "run-name suffix: fine-tuned (training continued from the parent's weights)", "sub"),
    ("GBDT", "gradient-boosted decision trees", "word"),
    ("gbdt", "gradient-boosted decision trees (run name)", "word"),
    ("GELU", "Gaussian error linear unit (an activation function)", "word"),
    ("GPU", "graphics processing unit", "word"),
    ("GRU", "gated recurrent unit (a recurrent neural-network cell)", "word"),
    ("gru", "gated recurrent unit (run name)", "word"),
    ("HRV", "heart rate variability", "word"),
    ("int8", "8-bit integer: weights and activations quantised to 8-bit integers, as run on the microcontroller; "
             "_int8 = the int8 version of a run", "word"),
    ("KB", "kilobyte (1,024 bytes)", "word"),
    ("log2", "base-2 logarithm", "word"),
    ("lr", "learning rate (step size of the optimiser)", "word"),
    ("MAC", "multiply-accumulate operation (one multiplication plus one addition): the unit of compute cost on "
            "the microcontroller; MACs/beat = how many the model needs per heartbeat", "word"),
    ("MACs", "multiply-accumulate operations (see MAC)", "word"),
    ("MCU", "microcontroller unit", "word"),
    ("MHA", "multi-head attention (a transformer layer, not supported by the MCU toolchain)", "word"),
    ("MLP", "multilayer perceptron (a fully connected neural network)", "word"),
    ("mlp", "multilayer perceptron (run name)", "word"),
    ("MMSE", "minimum mean square error", "word"),
    ("ms", "milliseconds", "word"),
    ("MSE", "mean squared error", "word"),
    ("nn20", "number of successive RR differences larger than 20 ms in the window", "word"),
    ("nobp", "run-name part: no band-pass (the fixed RSA band-pass features bp and bp_prev removed)", "sub"),
    ("np.corrcoef", "NumPy's Pearson correlation-coefficient function", "word"),
    ("np.diff", "NumPy's successive-difference function", "word"),
    ("O(1)", "constant cost per beat (big-O notation); O(W) = proportional to the window length W", "sub"),
    ("p", "p-value of the paired per-subject Wilcoxon test: probability of a difference at least this large if "
          "the two models were equally accurate (p < 0.05 = significant); in AR(p), the number of past values", "word"),
    ("B", "batch dimension (number of windows processed together) in the layer output shapes", "word"),
    ("PSD", "power spectral density (signal power per frequency)", "word"),
    ("Q", "quality factor of a band-pass filter: centre frequency / bandwidth", "word"),
    ("ReLU", "rectified linear unit, max(0, x) (an activation function)", "word"),
    ("RBJ", "Robert Bristow-Johnson's standard biquad (second-order) filter formulas", "word"),
    ("RLS", "recursive least squares: an adaptive filter that updates its coefficients at every beat", "word"),
    ("rls_ar", "baseline: adaptive per-patient AR(8) forecast by RLS", "word"),
    ("RMSE", "root mean square error (ms): the main accuracy measure, lower is better", "word"),
    ("rmssd", "root mean square of successive differences of the RR intervals in the window", "word"),
    ("RR", "RR interval: time between two consecutive heartbeats (R waves of the ECG), in ms", "word"),
    ("rs", "rescaled range: (max - min of the cumulative deviations from the mean) / standard deviation", "word"),
    ("RSA", "respiratory sinus arrhythmia: the oscillation of the heart rate driven by breathing", "word"),
    ("_rsa", "run-name suffix: uses the RSA-tracker features", "sub"),
    ("SD1", "Poincare-plot standard deviation across the identity line (short-term variability)", "word"),
    ("SD2", "Poincare-plot standard deviation along the identity line (long-term variability)", "word"),
    ("seq", "feature-set / model-kind name: sequence model (takes the raw RR window plus tabular features)", "word"),
    ("SeparableConv1D", "depthwise-separable 1-D convolution: a per-channel convolution followed by a "
                        "pointwise one (much cheaper than Conv1D)", "word"),
    ("STM32H750", "STMicroelectronics 32-bit microcontroller (Arm Cortex-M7, 128 KB flash, 1 MB RAM)", "word"),
    ("RAM", "random-access memory", "word"),
    ("tab", "feature-set / model-kind name: tabular model (engineered features only)", "word"),
    ("TCN", "temporal convolutional network (causal, dilated 1-D convolutions)", "word"),
    ("tcn", "temporal convolutional network (run-name part)", "sub"),
    ("TFLite", "TensorFlow Lite: the runtime and file format for models on embedded devices", "word"),
    ("W", "window length: the 30 most recent RR intervals", "word"),
    ("_ws", "run-name suffix: warm start (initialised from the parent's weights, new inputs set to zero)", "sub"),
    ("z", "z-score: (value - mean) / standard deviation, with statistics from the training subjects", "word"),
    ("ST", "STMicroelectronics (ST Edge AI: its toolchain for running neural networks on STM32 microcontrollers)", "word"),
    ("AI", "artificial intelligence", "word"),
    ("val", "validation: the 16 validation subjects, used for model selection", "word"),
    ("ar", "feature-set name: autoregressive inputs (rr_dev and the 8 most recent dRR) of the linear model", "word"),
    ("L2", "L2 regularisation: a penalty on the sum of squared model weights", "word"),
    ("float64", "64-bit (double-precision) floating-point numbers", "word"),
    ("nsrNNNRRcl", "subject code of an RR series from the normal sinus rhythm (NSR) recordings "
                      "(file data/series/<code>.txt); purely numeric codes such as 16273 or 006 are other subjects", "re"),
]

# capitalised or code tokens that are not acronyms (proper names, Keras classes, identifiers)
ALLOW = {
    "BatchNorm", "BatchNormalization", "LayerNorm", "LayerNormalization", "MultiHeadAttention", "InputLayer",
    "ZeroPadding1D", "Cropping1D", "GlobalAveragePooling1D", "SpatialDropout1D", "ReduceLROnPlateau",
    "EarlyStopping", "HistGradientBoostingRegressor", "RSAPhaseAwareLoss", "TCN_MHA", "FEATURE_NAMES",
    "RSA_TRACKER", "ENGINE", "LEGACY_STATIC", "INT8_CALIB", "TensorFlow", "NumPy", "TCNs", "CLOSED",
    "ENTIRE", "SELECTED", "RR_mean", "SeparableConv1Ds", "TRUE", "HistGradientBoosting", "SpatialDropout",
}


def plain(text):
    return html.unescape(re.sub(r"<[^>]+>", " ", text))


REGEX = {"nsrNNNRRcl": r"nsr\d{3}RRcl"}  # display term -> pattern for "re" entries


def _present(term, mode, text):
    if mode == "sub":
        return term in text
    pat = REGEX[term] if mode == "re" else re.escape(term) + "s?"  # plurals (MACs, MCUs) count as the term
    return re.search(r"(?<![A-Za-z0-9])" + pat + r"(?![A-Za-z0-9])", text) is not None


def used_terms(text):
    return [(t, m) for t, m, mode in GLOSSARY if _present(t, mode, text)]


def undefined_tokens(text):
    """Acronym-like tokens (>= 2 capitals/digits, or mixed case like dRR) that are neither defined nor allowed."""
    defined = {t for t, _, mode in GLOSSARY if mode != "re"}
    patterns = [REGEX[t] for t, _, mode in GLOSSARY if mode == "re"]
    ok = lambda s: s in defined or s in ALLOW or (s.endswith("s") and s[:-1] in defined) \
        or any(re.fullmatch(p, s) for p in patterns)  # noqa: E731
    bad = set()
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]*[A-Za-z0-9]|[A-Z]", text):
        if ok(tok):
            continue
        parts = re.split(r"[_.\-]", tok) if re.search(r"[_.\-]", tok) else [tok]  # compound: check its parts
        for part in parts:
            caps = sum(ch.isupper() or ch.isdigit() for ch in part)
            if part and caps >= 2 and not ok(part) and not re.fullmatch(r"\d+[A-Za-z]*", part):
                bad.add(part)
    return sorted(bad)

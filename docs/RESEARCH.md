# Research notes for the architect (2026-10-07)

What the literature says about our problem, and what it suggests we try. "Hypothesis" marks my
own inference, not a published result. Sources are at the end.

## 1. Why a population-trained net invents or displaces the RSA peak (hypothesis)
At high drop rates (the pacemaker sees up to 80%) the filled series is mostly the predictor's own
output fed back into its window: in closed loop the model behaves like a recursive filter, and the
spectrum of the filled series follows that filter's resonances, not the patient's. A net trained on
the whole population learns an average respiratory oscillation (roughly 0.25-0.35 cycles/beat in
adults). Free-running, it rings at that frequency: a peak appears where the patient had none, or
sits at the population frequency instead of the patient's (= displaced). This matches the TCN_MHA
symptom. Its loss made it worse: the variance and sign penalties in `RSAPhaseAwareLoss` pay the
model to oscillate.
Consequences:
- Plain MSE/Huber is "contractive": in closed loop it damps towards the mean (loses some HF power,
  as interpolation does in the literature) but does not invent peaks. Keep it.
- The patient-specific oscillation must come from patient-adaptive inputs, not from learned weights.

## 2. Patient-adaptive RSA inputs (the engine change the user approved)
- Our `bp`/`bp_prev` use a FIXED band-pass at 0.30 cycles/beat. A model that leans on it is told
  "RSA is at 0.30" for every patient. That is the displacement mechanism above, built into a feature.
- Literature: RSA/respiratory frequency is tracked in real time with a minimal-order adaptive notch
  filter whose centre frequency is updated by RLS (Caiani et al., CinC 1999), or with notch-filter
  banks / oscillator trackers (Mirmohamadsadeghi & Vesin, CinC 2014/2015). These are O(1) per beat,
  a few dozen FLOPs: MCU-friendly.
- Candidate features (append after the last block of `FEATURE_NAMES`, keep the old ones):
  - `rsa_f`: tracked RSA frequency (cycles/beat) from an adaptive notch/biquad on RR whose centre
    adapts by RLS/LMS (replaces the fixed 0.30).
  - `rsa_amp`, and the phase as `rsa_cos`/`rsa_sin`: the next beat's RSA contribution is
    amp * cos(phase + 2*pi*rsa_f).
  - `rsa_q`: RSA presence = power in the tracked band / total power over the recent window.
    It lets the model output no oscillation when there is no RSA (fixes the invented peak).
- Cheaper alternative with no new filter: the RLS AR(8) weights `rls_w*` already describe the
  patient's spectrum. The frequency of the dominant pole pair, from the AR polynomial, is a
  patient-adaptive RSA frequency. But root-finding per beat is costly on an MCU, so prefer the notch.
- Beat-domain caveat: f [cycles/beat] = breaths/min / HR [bpm]. Adults: about 0.15-0.35. Infants
  (in our train set): breathing 30-60/min at HR 100-160 gives 0.2-0.5, close to the 0.5 Nyquist
  limit, so it aliases. The evaluator's HF band (0.15-0.40) fits the adult validation subjects only.

## 3. Training for the closed loop (exposure bias)
Teacher-forced training (always the true history) leaves a model unprepared for its own errors in
the window. The standard remedies are scheduled sampling (Bengio et al. 2015) and "data as
demonstrator" (DAD): train partly on inputs that contain the model's own predictions.
- Cheap version for us: run the engine on training series in which a random 10-80% of beats were
  closed-loop filled (by `rls_ar`, or by the previous model), and keep the TRUE dRR as the target.
  The numba engine processes the whole training set in about a minute. This is a `train.py`
  option (trainer) plus a data hook in `src/utils.py` (architect).
- Hypothesis: this mainly fixes RSA displacement at high drop rates, because the model learns that
  a window full of its own fills is a reason to trust the patient-adaptive inputs, not its prior.

## 4. MCU evidence: what to build
- 1D-CNN vs LSTM on MCUs (2026 preprint, 5 datasets): CNN about 35% less RAM, about 25% less flash,
  27.6 ms vs 2038 ms latency, and negligible int8 loss, while the LSTM lost accuracy under int8.
- GRU int8 is harder (quantising its internal state is delicate); one study swapped a GRU for a
  vanilla RNN because CMSIS-NN GRU support needed extra work.
  -> Prefer causal conv (micro_tcn family). Keep `gru` as one data point only.
- Our numbers already agree: micro_tcn (2.4 KB int8, 2.1k MACs/beat) has an int8 gap of +1.4%.

## 5. Evaluation lessons
- RNN imputation can get the HRV summary metrics right and still be wrong beat by beat (Svane et al.
  2023). Keep the per-beat RMSE AND the spectral check (we do both).
- Linear/spline fills under-estimate HF power. Deletion distorts the spectrum. HF/LF metrics are
  unstable from about 10% missing data. So some HF loss under heavy dropping is expected even for a
  good model; what we must not accept is a NEW or MOVED peak (evaluator rule 5).

## Sources
- Barbieri et al. 2005 / Barbieri & Brown 2006, point-process adaptive RR models: https://pmc.ncbi.nlm.nih.gov/articles/PMC2561955 , https://dspace.mit.edu/handle/1721.1/70072
- Chen, Brown, Barbieri 2009, RSA gain with a point-process adaptive filter: https://pmc.ncbi.nlm.nih.gov/articles/PMC2804879
- Caiani et al. / adaptive notch tracking of respiratory frequency: https://www.measurement.sk/ICE2014/proceedings/057.pdf
- Mirmohamadsadeghi & Vesin, real-time RSA/RPA respiratory rate (notch banks, oscillators): https://infoscience.epfl.ch/record/210216/files/CINC15_LM.pdf , https://infoscience.epfl.ch/record/201743/files/CINC14_LM.pdf
- Tarvainen et al., time-varying AR with a Kalman smoother for HRV: https://www.doi.org/10.1088/0967-3334/27/3/002
- Svane et al. 2023, RNN imputation of RR intervals: https://uis.brage.unit.no/uis-xmlui/bitstream/11250/3105846/1/Svane%252Bet%252Bal%252B2023.pdf
- Interpolation/deletion effects on the HRV spectrum: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6679245/ , https://metabase-lcp.mit.edu/pdf/CliffordTBE05.pdf , https://www.mdpi.com/1424-8220/19/14/3163
- Scheduled sampling / exposure bias for time series: https://arxiv.org/pdf/2210.08959 , https://www.research-collection.ethz.ch/entities/publication/4435165a-67e0-46dd-8759-efa8f1426c72
- LSTM vs 1D-CNN on MCUs: https://arxiv.org/pdf/2603.04860 ; GRU quantisation: https://arxiv.org/pdf/2001.10876 , https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/rnn-models-ethos-u
- TCN on Cortex-M7: https://arxiv.org/pdf/2011.05260

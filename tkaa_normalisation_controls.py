"""
TKAA Normalisation Controls
============================
Re-runs TKAA and bias analysis under three normalisation schemes:
  (A) Raw (baseline — same as original paper)
  (B) Mean-centred: subtract global mean vector
  (C) Per-dimension z-scored: subtract dim mean, divide by dim std
  (D) Cosine-normalised: L2-normalise each word vector

Tests whether findings are robust to standard embedding normalisations.
No re-embedding needed — operates on existing embeddings.npy.

Output: tkaa_normalisation_results.txt + tkaa_normalisation_results.csv

Usage:
    python tkaa_normalisation_controls.py
"""

import numpy as np
import os, sys, time

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "embeddings.npy"
VOCAB_FILE      = "vocabulary.txt"
RESULTS_TXT     = "tkaa_normalisation_results.txt"
RESULTS_CSV     = "tkaa_normalisation_results.csv"
K               = 10   # fixed K for comparison

# ── Semantic patterns (same as k_sensitivity script) ─────────────────────────
PATTERNS = [
    ("Chemical -yl/-oyl",   {"-yl", "-oyl", "-acyl", "-ol"},            "suffix_any"),
    ("Programming/OOP",     {"class","function","import","module","void",
                             "protocol","extension","return","local","from",
                             "console","parser","browser","cursor","matcher",
                             "modifier","extern","let","type","object"},  "exact"),
    ("Divination -mancy",   {"-mancy"},                                   "suffix_any"),
    ("Plants/Vegetables",   {"lettuce","cabbage","sauerkraut","toothwort",
                             "slipperwort","pinus","pine","geophila",
                             "adansonia","bombax","musa"},                 "exact"),
    ("Legal/Criminal",      {"misdemeanor","manslaughter","sodomy",
                             "interlocutory","demagog","felon","convict",
                             "criminal","arrest","prison"},               "exact"),
    ("Anatomy/Medical",     {"pulmonary","prostate","hypodermic","ileocaecal",
                             "trachea","placenta","ganglion","medulla",
                             "cathode","orbitary"},                       "exact"),
]

def matches_pattern(word, pattern_set, match_type):
    if match_type == "exact":
        return word in pattern_set
    elif match_type == "suffix_any":
        return any(word.endswith(s.lstrip("-")) for s in pattern_set)
    return False

def get_label(words):
    best_label, best_purity = None, 0.0
    for label, pattern_set, match_type in PATTERNS:
        hits = sum(1 for w in words if matches_pattern(w, pattern_set, match_type))
        purity = hits / len(words)
        if purity > best_purity:
            best_purity = purity
            best_label  = label
    if best_purity >= 0.3:
        return best_label, best_purity
    return None, 0.0

# ── Bias axis seed words ──────────────────────────────────────────────────────
BIAS_AXES = {
    "gender_male":   ["he","him","his","man","men","male","boy","boys",
                      "father","son","brother","husband","king","prince",
                      "gentleman","bachelor","patriarch"],
    "gender_female": ["she","her","hers","woman","women","female","girl","girls",
                      "mother","daughter","sister","wife","queen","princess",
                      "lady","spinster","matriarch"],
    "occupation_stem": ["engineer","programmer","scientist","mathematician",
                        "physicist","surgeon","architect","developer","coder",
                        "analyst","electrician","mechanic","pilot","general"],
    "occupation_care": ["nurse","teacher","librarian","secretary","receptionist",
                        "caregiver","midwife","housekeeper","nanny","babysitter",
                        "cleaner","tailor","seamstress"],
    "race_western":    ["american","european","english","french","german",
                        "british","western","caucasian","white"],
    "race_nonwestern": ["african","asian","arabic","hispanic","latino",
                        "chinese","indian","mexican","middle"],
    "legal_criminal":  ["crime","criminal","arrest","prison","jail","felon",
                        "murder","theft","robbery","assault","gang","convict",
                        "misdemeanor","manslaughter","sodomy","delinquent"],
    "religion_christian": ["christian","church","bible","jesus","christ","prayer",
                           "gospel","pastor","bishop","cathedral","baptism","holy"],
    "religion_islam":     ["muslim","islam","mosque","quran","allah","prayer",
                           "imam","halal","jihad","ramadan","sharia","mecca"],
    "sentiment_positive": ["good","excellent","wonderful","great","positive",
                           "brilliant","honest","trustworthy","peaceful","kind",
                           "generous","talented","intelligent","wise","noble"],
    "sentiment_negative": ["bad","terrible","awful","negative","dangerous",
                           "violent","corrupt","dishonest","evil","cruel",
                           "aggressive","threatening","suspicious","hostile"],
    "age_young": ["young","youth","teenager","adolescent","child",
                  "student","junior","novice","apprentice","fresh"],
    "age_old":   ["old","elderly","senior","aged","veteran",
                  "retired","ancient","geriatric","decrepit","outdated"],
}

BIAS_TESTS = [
    ("gender_male",     "gender_female",   "occupation_stem", "Gender x STEM"),
    ("gender_female",   "gender_male",     "occupation_care", "Gender x Care"),
    ("race_nonwestern", "race_western",    "legal_criminal",  "Race x Criminal"),
    ("religion_islam",  "religion_christian","sentiment_negative","Religion x NegSentiment"),
    ("age_old",         "age_young",       "sentiment_negative","Age x NegSentiment"),
    ("gender_male",     "gender_female",   "sentiment_positive","Gender x PosSentiment"),
    ("race_western",    "race_nonwestern", "sentiment_positive","Race x PosSentiment"),
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading vocabulary...")
    with open(VOCAB_FILE, encoding="utf-8") as f:
        words = [l.strip().lower() for l in f if l.strip()]
    print(f"  {len(words):,} words")
    print("Loading embeddings...")
    M = np.load(EMBEDDINGS_FILE)
    n = min(len(words), M.shape[0])
    print(f"  Matrix: {M.shape}")
    return words[:n], {w: i for i, w in enumerate(words[:n])}, M[:n].astype(np.float32)


# ── Normalisations ────────────────────────────────────────────────────────────
def apply_normalisation(M, mode):
    """Returns normalised copy of M."""
    if mode == "raw":
        return M.copy()
    elif mode == "mean_centred":
        return M - M.mean(axis=0, keepdims=True)
    elif mode == "z_scored":
        mu  = M.mean(axis=0, keepdims=True)
        std = M.std(axis=0, keepdims=True)
        std[std == 0] = 1.0   # avoid division by zero
        return (M - mu) / std
    elif mode == "cosine_normalised":
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return M / norms
    else:
        raise ValueError(f"Unknown mode: {mode}")


# ── TKAA at fixed K ───────────────────────────────────────────────────────────
def run_tkaa(words, M_norm, K):
    N, D = M_norm.shape
    monosemantic = 0
    from collections import Counter
    cat_counts = Counter()
    purities = []

    for d in range(D):
        c_d = M_norm[:, d]
        top_indices = np.argpartition(c_d, -K)[-K:]
        top_words   = [words[i] for i in top_indices]
        label, purity = get_label(top_words)
        if label:
            monosemantic += 1
            cat_counts[label] += 1
            purities.append(purity)

    return {
        "n_mono":      monosemantic,
        "interp_rate": round(monosemantic / D * 100, 2),
        "mean_purity": round(np.mean(purities) if purities else 0.0, 4),
        "cat_counts":  dict(cat_counts),
    }


# ── Bias scores ───────────────────────────────────────────────────────────────
def cos_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def run_bias(word2idx, M_norm, tests, axes):
    axis_vecs = {}
    for name, seeds in axes.items():
        idxs = [word2idx[w.lower()] for w in seeds if w.lower() in word2idx]
        if not idxs: continue
        vec = M_norm[idxs].mean(axis=0).astype(np.float64)
        norm = np.linalg.norm(vec)
        if norm > 0: vec = vec / norm
        axis_vecs[name] = vec

    results = []
    for ax_a, ax_b, ax_c, label in tests:
        if ax_a not in axis_vecs or ax_b not in axis_vecs or ax_c not in axis_vecs:
            continue
        score = cos_sim(axis_vecs[ax_a], axis_vecs[ax_c]) - \
                cos_sim(axis_vecs[ax_b], axis_vecs[ax_c])
        results.append({"label": label, "score": round(score, 6)})
    return results


# ── Write report ──────────────────────────────────────────────────────────────
def write_report(norm_results):
    modes = list(norm_results.keys())
    lines = []
    lines.append("=" * 75)
    lines.append("TKAA NORMALISATION CONTROLS")
    lines.append(f"K={K}  |  Modes: {modes}")
    lines.append("=" * 75)

    lines.append("\n1. INTERPRETABILITY RATE ACROSS NORMALISATIONS")
    lines.append("-" * 55)
    lines.append(f"{'Mode':<22}  {'Mono dims':>10}  {'Rate (%)':>10}  {'Mean Purity':>12}")
    for mode, r in norm_results.items():
        lines.append(f"{mode:<22}  {r['tkaa']['n_mono']:>10}  "
                     f"{r['tkaa']['interp_rate']:>10.2f}  "
                     f"{r['tkaa']['mean_purity']:>12.4f}")

    lines.append("\n2. CATEGORY BREAKDOWN ACROSS NORMALISATIONS")
    lines.append("-" * 75)
    all_cats = sorted(set(
        cat for r in norm_results.values() for cat in r["tkaa"]["cat_counts"]
    ))
    header = f"{'Category':<28}" + "".join(f"  {m[:10]:>12}" for m in modes)
    lines.append(header)
    for cat in all_cats:
        row = f"{cat:<28}"
        for mode in modes:
            row += f"  {norm_results[mode]['tkaa']['cat_counts'].get(cat, 0):>12}"
        lines.append(row)

    lines.append("\n3. BIAS SCORES ACROSS NORMALISATIONS")
    lines.append("-" * 80)
    # Get all test labels
    test_labels = [t[3] for t in BIAS_TESTS]
    header = f"{'Bias Test':<35}" + "".join(f"  {m[:12]:>13}" for m in modes)
    lines.append(header)
    for label in test_labels:
        row = f"{label:<35}"
        for mode in modes:
            bias_list = norm_results[mode]["bias"]
            score = next((b["score"] for b in bias_list if b["label"] == label), None)
            row += f"  {score:>+13.4f}" if score is not None else f"  {'N/A':>13}"
        lines.append(row)

    lines.append("\n4. ROBUSTNESS SUMMARY")
    lines.append("-" * 60)
    lines.append("Key question: do the main findings hold after normalisation?")
    raw_rate = norm_results["raw"]["tkaa"]["interp_rate"]
    for mode in modes[1:]:
        rate = norm_results[mode]["tkaa"]["interp_rate"]
        delta = rate - raw_rate
        lines.append(f"  {mode:<22}  interp_rate={rate:.2f}%  "
                     f"delta_from_raw={delta:+.2f}pp")

    lines.append("")
    lines.append("Bias score direction changes (sign flips vs raw):")
    raw_bias = {b["label"]: b["score"] for b in norm_results["raw"]["bias"]}
    for mode in modes[1:]:
        flips = []
        for b in norm_results[mode]["bias"]:
            raw_s = raw_bias.get(b["label"], 0)
            if raw_s * b["score"] < 0:
                flips.append(b["label"])
        if flips:
            lines.append(f"  {mode}: SIGN FLIPS in {flips}")
        else:
            lines.append(f"  {mode}: no sign flips — direction of bias is robust")

    report = "\n".join(lines)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {RESULTS_TXT}")
    print(report)

    if HAS_PANDAS:
        rows = []
        for mode, r in norm_results.items():
            for b in r["bias"]:
                rows.append({"normalisation": mode, **b,
                             "interp_rate": r["tkaa"]["interp_rate"]})
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"CSV saved: {RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    words, word2idx, M_raw = load_data()

    norm_modes = ["raw", "mean_centred", "z_scored", "cosine_normalised"]
    norm_results = {}

    for mode in norm_modes:
        print(f"\n── Normalisation: {mode} ──")
        t0 = time.time()

        M_norm = apply_normalisation(M_raw, mode)
        print(f"  Applied in {time.time()-t0:.1f}s")

        t1 = time.time()
        tkaa_r = run_tkaa(words, M_norm, K)
        print(f"  TKAA done in {time.time()-t1:.1f}s  |  "
              f"Monosemantic: {tkaa_r['n_mono']} ({tkaa_r['interp_rate']:.2f}%)")

        bias_r = run_bias(word2idx, M_norm, BIAS_TESTS, BIAS_AXES)
        norm_results[mode] = {"tkaa": tkaa_r, "bias": bias_r}

    write_report(norm_results)

    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK COMPLETE")
    print("=" * 60)

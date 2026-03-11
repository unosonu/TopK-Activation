"""
TKAA Permutation Tests for Bias Significance
=============================================
WEAT-style permutation testing on existing embeddings.npy.
No re-embedding needed.

Output: tkaa_permutation_results.txt + tkaa_permutation_results.csv

Usage:
    pip install numpy pandas scipy
    python tkaa_permutation_tests.py
"""

import numpy as np
import os, sys, time
from itertools import combinations

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_FILE  = "embeddings.npy"
VOCAB_FILE       = "vocabulary.txt"
RESULTS_TXT      = "tkaa_permutation_results.txt"
RESULTS_CSV      = "tkaa_permutation_results.csv"
N_PERMUTATIONS   = 10000   # standard for WEAT; increase to 50000 for camera-ready
RANDOM_SEED      = 42
ALPHA            = 0.05    # significance threshold

# ── Bias axis seed words (must match tkaa_bias_analysis.py exactly) ───────────
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
    "legal_civil":     ["law","court","judge","attorney","lawyer","legal",
                        "statute","contract","plaintiff","defendant","verdict",
                        "counsel","barrister","solicitor","litigation"],
    "religion_christian": ["christian","church","bible","jesus","christ","prayer",
                           "gospel","pastor","bishop","cathedral","baptism","holy"],
    "religion_islam":     ["muslim","islam","mosque","quran","allah","prayer",
                           "imam","halal","jihad","ramadan","sharia","mecca"],
    "religion_other":     ["hindu","buddhist","jewish","synagogue","temple",
                           "torah","buddha","karma","dharma","rabbi","monk"],
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

# Tests: (group_A, group_B, concept, label)
BIAS_TESTS = [
    ("gender_male",     "gender_female",   "occupation_stem", "Gender x STEM Occupations"),
    ("gender_female",   "gender_male",     "occupation_care", "Gender x Care Occupations"),
    ("race_nonwestern", "race_western",    "legal_criminal",  "Race x Criminal Associations"),
    ("religion_islam",  "religion_christian","sentiment_negative","Religion x Negative Sentiment"),
    ("age_old",         "age_young",       "sentiment_negative","Age x Negative Sentiment"),
    ("gender_male",     "gender_female",   "sentiment_positive","Gender x Positive Sentiment"),
    ("race_western",    "race_nonwestern", "sentiment_positive","Race x Positive Sentiment"),
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    print("Loading vocabulary...")
    with open(VOCAB_FILE, encoding="utf-8") as f:
        words = [l.strip().lower() for l in f if l.strip()]
    word2idx = {w: i for i, w in enumerate(words)}
    print(f"  {len(words):,} words")

    print("Loading embeddings...")
    M = np.load(EMBEDDINGS_FILE)
    n = min(len(words), M.shape[0])
    print(f"  Matrix: {M.shape}")
    return words[:n], {w:i for i,w in enumerate(words[:n])}, M[:n]


# ── Axis vectors ──────────────────────────────────────────────────────────────
def build_axis_vectors(word2idx, M):
    axis_vecs  = {}   # axis -> (mean_vec, found_words)
    axis_idxs  = {}   # axis -> list of row indices (for permutation)
    for name, seeds in BIAS_AXES.items():
        idxs = [word2idx[w.lower()] for w in seeds if w.lower() in word2idx]
        if not idxs:
            continue
        vecs = M[idxs]
        mean = vecs.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        axis_vecs[name]  = mean
        axis_idxs[name]  = idxs
    return axis_vecs, axis_idxs


# ── Cosine similarity ─────────────────────────────────────────────────────────
def cos_sim(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ── WEAT-style permutation test ───────────────────────────────────────────────
def permutation_test(axis_a, axis_b, axis_c, axis_vecs, axis_idxs, M, n_perm):
    """
    Observed effect: bias = cos_sim(v_A, v_C) - cos_sim(v_B, v_C)

    Null distribution: repeatedly shuffle the combined word pool of A and B,
    split into two groups of the same sizes as A and B, recompute effect.
    p-value = fraction of permutations with |effect| >= |observed|  (two-tailed)
    """
    if axis_a not in axis_vecs or axis_b not in axis_vecs or axis_c not in axis_vecs:
        return None

    v_c = axis_vecs[axis_c]

    # Observed effect
    observed = cos_sim(axis_vecs[axis_a], v_c) - cos_sim(axis_vecs[axis_b], v_c)

    # Combined word pool (row indices)
    idxs_a = np.array(axis_idxs[axis_a])
    idxs_b = np.array(axis_idxs[axis_b])
    na, nb = len(idxs_a), len(idxs_b)
    combined = np.concatenate([idxs_a, idxs_b])

    rng = np.random.default_rng(RANDOM_SEED)
    null_dist = np.empty(n_perm)

    for i in range(n_perm):
        shuffled = rng.permutation(combined)
        perm_a = shuffled[:na]
        perm_b = shuffled[na:]

        mean_a = M[perm_a].mean(axis=0)
        mean_b = M[perm_b].mean(axis=0)

        norm_a = np.linalg.norm(mean_a)
        norm_b = np.linalg.norm(mean_b)
        if norm_a > 0: mean_a = mean_a / norm_a
        if norm_b > 0: mean_b = mean_b / norm_b

        null_dist[i] = cos_sim(mean_a, v_c) - cos_sim(mean_b, v_c)

    # Two-tailed p-value
    p_value = float(np.mean(np.abs(null_dist) >= np.abs(observed)))

    # Effect size (Cohen's d style: observed / std of null)
    effect_size = float(observed / (null_dist.std() + 1e-10))

    # 95% CI of null distribution
    ci_low, ci_high = float(np.percentile(null_dist, 2.5)), float(np.percentile(null_dist, 97.5))

    return {
        "observed":    round(observed,    6),
        "p_value":     round(p_value,     4),
        "effect_size": round(effect_size, 4),
        "null_mean":   round(float(null_dist.mean()), 6),
        "null_std":    round(float(null_dist.std()),  6),
        "ci_95_low":   round(ci_low,  6),
        "ci_95_high":  round(ci_high, 6),
        "significant": p_value < ALPHA,
        "n_perm":      n_perm,
        "na":          na,
        "nb":          nb,
    }


# ── Bonferroni correction ─────────────────────────────────────────────────────
def apply_bonferroni(results):
    n_tests = len(results)
    for r in results:
        r["p_bonferroni"] = round(min(r["p_value"] * n_tests, 1.0), 4)
        r["sig_bonferroni"] = r["p_bonferroni"] < ALPHA
    return results


# ── Write report ──────────────────────────────────────────────────────────────
def write_report(results):
    lines = []
    lines.append("=" * 75)
    lines.append("TKAA WEAT-STYLE PERMUTATION TEST RESULTS")
    lines.append(f"N permutations: {N_PERMUTATIONS:,}  |  Alpha: {ALPHA}  |  Seed: {RANDOM_SEED}")
    lines.append("=" * 75)
    lines.append("")
    lines.append(f"{'Test':<45} {'Obs':>8} {'p':>7} {'p_bonf':>8} {'ES':>7}  {'Sig?':<6}  {'95% CI null'}")
    lines.append("-" * 110)

    for r in results:
        sig_raw  = "*" if r["significant"]    else " "
        sig_bonf = "*" if r["sig_bonferroni"] else " "
        ci = f"[{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}]"
        lines.append(
            f"{r['label']:<45} {r['observed']:>+8.4f} "
            f"{r['p_value']:>7.4f} {r['p_bonferroni']:>8.4f} "
            f"{r['effect_size']:>7.3f}  "
            f"raw={sig_raw} bonf={sig_bonf}  {ci}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("  Observed = cos_sim(Group_A, Concept) - cos_sim(Group_B, Concept)")
    lines.append("  p-value  = two-tailed permutation p (fraction of |null| >= |observed|)")
    lines.append("  p_bonf   = Bonferroni-corrected p (multiply by number of tests)")
    lines.append("  ES       = effect size (observed / std of null distribution)")
    lines.append("  95% CI null = 95th percentile range of null distribution")
    lines.append("  * = significant at alpha=0.05")

    report = "\n".join(lines)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {RESULTS_TXT}")
    print(report)

    if HAS_PANDAS:
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_CSV, index=False)
        print(f"CSV saved:    {RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)

    words, word2idx, M = load_data()
    axis_vecs, axis_idxs = build_axis_vectors(word2idx, M)

    results = []
    print(f"\nRunning {N_PERMUTATIONS:,} permutations per test...")
    print("-" * 60)

    for axis_a, axis_b, axis_c, label in BIAS_TESTS:
        t0 = time.time()
        r = permutation_test(axis_a, axis_b, axis_c,
                             axis_vecs, axis_idxs, M, N_PERMUTATIONS)
        if r is None:
            print(f"  SKIPPED: {label}")
            continue
        r["label"] = label
        r["axis_a"] = axis_a
        r["axis_b"] = axis_b
        r["axis_c"] = axis_c
        results.append(r)
        elapsed = time.time() - t0
        sig = "SIGNIFICANT" if r["significant"] else "not sig"
        print(f"  {label:<45}  p={r['p_value']:.4f}  ES={r['effect_size']:.3f}  "
              f"({elapsed:.1f}s)  [{sig}]")

    results = apply_bonferroni(results)
    write_report(results)

    print("\n" + "=" * 60)
    sig_raw  = [r for r in results if r["significant"]]
    sig_bonf = [r for r in results if r["sig_bonferroni"]]
    print(f"Significant (raw p<{ALPHA}):        {len(sig_raw)}/{len(results)}")
    print(f"Significant (Bonferroni p<{ALPHA}): {len(sig_bonf)}/{len(results)}")
    print("=" * 60)

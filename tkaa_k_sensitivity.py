"""
TKAA K-Sensitivity Analysis
============================
Runs TKAA at K = {5, 10, 20, 50} and reports how the interpretability
rate and purity change. No re-embedding needed.

Output: tkaa_k_sensitivity.txt + tkaa_k_sensitivity.csv

Usage:
    python tkaa_k_sensitivity.py
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
RESULTS_TXT     = "tkaa_k_sensitivity.txt"
RESULTS_CSV     = "tkaa_k_sensitivity.csv"

K_VALUES = [5, 10, 20, 50]

# ── Semantic label patterns ───────────────────────────────────────────────────
# Each pattern: (label, set_of_anchor_words_or_suffixes, match_type)
# match_type: "suffix" = word.endswith(suffix)
#             "exact"  = word in set

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
    """Return (label, purity) for a word list, or (None, 0) if no pattern matches."""
    best_label, best_purity = None, 0.0
    for label, pattern_set, match_type in PATTERNS:
        hits = sum(1 for w in words if matches_pattern(w, pattern_set, match_type))
        purity = hits / len(words)
        if purity > best_purity:
            best_purity = purity
            best_label  = label
    if best_purity >= 0.3:   # minimum threshold to count as monosemantic
        return best_label, best_purity
    return None, 0.0


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
    return words[:n], M[:n]


# ── Run TKAA at one K value ───────────────────────────────────────────────────
def run_tkaa_at_k(words, M, K):
    N, D = M.shape
    results_per_dim = []

    for d in range(D):
        c_d = M[:, d]
        top_indices = np.argpartition(c_d, -K)[-K:]
        top_indices = top_indices[np.argsort(c_d[top_indices])[::-1]]
        top_words = [words[i] for i in top_indices]
        label, purity = get_label(top_words)
        results_per_dim.append({
            "dim": d,
            "label": label,
            "purity": purity,
            "top_words": top_words[:5],   # store first 5 for inspection
        })

    monosemantic = [r for r in results_per_dim if r["label"] is not None]
    interp_rate  = len(monosemantic) / D

    # Per-category counts
    from collections import Counter
    cat_counts = Counter(r["label"] for r in monosemantic)

    # Mean purity of monosemantic dims
    mean_purity = np.mean([r["purity"] for r in monosemantic]) if monosemantic else 0.0

    return {
        "K":            K,
        "D":            D,
        "n_monosemantic": len(monosemantic),
        "interp_rate":  round(interp_rate * 100, 2),
        "mean_purity":  round(mean_purity, 4),
        "cat_counts":   dict(cat_counts),
        "per_dim":      results_per_dim,
    }


# ── Stability: Jaccard of monosemantic dim sets across K values ───────────────
def jaccard_stability(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


# ── Write report ──────────────────────────────────────────────────────────────
def write_report(all_results):
    lines = []
    lines.append("=" * 70)
    lines.append("TKAA K-SENSITIVITY ANALYSIS")
    lines.append(f"Vocabulary: {all_results[0]['D']:,} dims  |  "
                 f"K values tested: {K_VALUES}")
    lines.append("=" * 70)

    lines.append("\n1. INTERPRETABILITY RATE BY K")
    lines.append("-" * 50)
    lines.append(f"{'K':>5}  {'Monosemantic dims':>18}  {'Rate (%)':>10}  {'Mean Purity':>12}")
    for r in all_results:
        lines.append(f"{r['K']:>5}  {r['n_monosemantic']:>18}  "
                     f"{r['interp_rate']:>10.2f}  {r['mean_purity']:>12.4f}")

    lines.append("\n2. CATEGORY BREAKDOWN BY K")
    lines.append("-" * 70)
    all_cats = sorted(set(
        cat for r in all_results for cat in r["cat_counts"]
    ))
    header = f"{'Category':<30}" + "".join(f"  K={r['K']:>3}" for r in all_results)
    lines.append(header)
    for cat in all_cats:
        row = f"{cat:<30}"
        for r in all_results:
            row += f"  {r['cat_counts'].get(cat, 0):>5}"
        lines.append(row)

    lines.append("\n3. STABILITY: JACCARD OVERLAP OF MONOSEMANTIC DIM SETS")
    lines.append("-" * 60)
    lines.append("(How stable is the set of monosemantic dims across K values?)")
    mono_sets = {r["K"]: set(d["dim"] for d in r["per_dim"] if d["label"]) for r in all_results}
    for i in range(len(K_VALUES)):
        for j in range(i+1, len(K_VALUES)):
            ka, kb = K_VALUES[i], K_VALUES[j]
            j_score = jaccard_stability(mono_sets[ka], mono_sets[kb])
            lines.append(f"  K={ka} vs K={kb}:  Jaccard={j_score:.3f}  "
                         f"(shared={len(mono_sets[ka] & mono_sets[kb])} dims)")

    lines.append("\n4. SAMPLE: TOP MONOSEMANTIC DIMS AT EACH K (first 5)")
    lines.append("-" * 70)
    for r in all_results:
        lines.append(f"\n  K={r['K']}:")
        mono = [d for d in r["per_dim"] if d["label"]][:5]
        for d in mono:
            lines.append(f"    Dim {d['dim']:04d}  [{d['label']}]  "
                         f"purity={d['purity']:.2f}  "
                         f"words={', '.join(d['top_words'][:4])}")

    report = "\n".join(lines)
    with open(RESULTS_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved: {RESULTS_TXT}")
    print(report)

    if HAS_PANDAS:
        rows = []
        for r in all_results:
            for cat, cnt in r["cat_counts"].items():
                rows.append({"K": r["K"], "category": cat, "count": cnt,
                             "interp_rate_pct": r["interp_rate"],
                             "mean_purity": r["mean_purity"]})
        pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        print(f"CSV saved:    {RESULTS_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    words, M = load_data()
    N, D = M.shape
    print(f"\nRunning TKAA at K = {K_VALUES} across {D:,} dimensions...")

    all_results = []
    for K in K_VALUES:
        t0 = time.time()
        print(f"\n  K={K}...")
        r = run_tkaa_at_k(words, M, K)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s  |  "
              f"Monosemantic: {r['n_monosemantic']}/{D} ({r['interp_rate']:.2f}%)")
        all_results.append(r)

    write_report(all_results)

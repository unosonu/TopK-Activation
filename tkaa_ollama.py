import os
import time
import json
import numpy as np
import urllib.request

# =====================================================================
# Top-K Activation Analysis (TKAA) — Ollama Local Edition
# Based on: "Top-K Activation Analysis of Dense Word Embeddings"
# Model: llama3.2 via Ollama (3072-dim embeddings, fully local)
# =====================================================================

# --- Configuration ---
DICT_URL = "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json"
DICT_FILE = "dictionary.json"
VOCAB_FILE = "vocabulary.txt"
EMBEDDINGS_FILE = "embeddings.npy"
RESULTS_FILE = "tkaa_results.txt"

OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "llama3.2"

BATCH_SIZE = 500       # Words per Ollama call (local = no rate limit, bigger = faster)
TOP_K = 10             # Top-K words per dimension
SAMPLE_DIMS = 50       # How many dimensions to print to console


# =====================================================================
# 1. DICTIONARY — Download & Extract Pure Words
# =====================================================================

def get_dictionary_words():
    """
    Downloads a real English dictionary (Webster's Unabridged) and
    extracts all clean, alphabetical words. Saves them to vocabulary.txt.
    Returns a sorted list of lowercase words.
    """
    # Use cached vocabulary if it exists
    if os.path.exists(VOCAB_FILE):
        print(f"Loading vocabulary from {VOCAB_FILE}...")
        with open(VOCAB_FILE, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(words)} words.")
        return words

    # Download the dictionary JSON
    if not os.path.exists(DICT_FILE):
        print("Downloading Webster's Unabridged Dictionary (~9MB)...")
        urllib.request.urlretrieve(DICT_URL, DICT_FILE)
        print("Download complete.")

    print("Extracting words from dictionary...")
    with open(DICT_FILE, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)

    # Extract ONLY the words (keys), not definitions
    # Filter: alphabetical only, at least 3 chars, lowercase
    words = sorted(set(
        w.lower() for w in dictionary.keys()
        if w.isalpha() and len(w) >= 3
    ))

    # Save to file
    with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words))

    print(f"Extracted and saved {len(words)} dictionary words to {VOCAB_FILE}")
    return words


# =====================================================================
# 2. EMBEDDINGS — Fetch from Ollama (fully local, no rate limits)
# =====================================================================

def get_ollama_embeddings(words_batch):
    """Sends a batch of words to Ollama's /api/embed endpoint and returns embeddings."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "input": words_batch
    }).encode('utf-8')

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    response = urllib.request.urlopen(req)
    data = json.loads(response.read())
    return data["embeddings"]


def fetch_all_embeddings(words):
    """
    Fetches embeddings for ALL words via Ollama, with resume support.
    Saves progress to embeddings.npy after each batch.
    """
    # Resume from existing progress
    if os.path.exists(EMBEDDINGS_FILE):
        existing = np.load(EMBEDDINGS_FILE)
        start_idx = len(existing)
        embeddings = existing.tolist()
        print(f"Resuming from word {start_idx}/{len(words)}...")
    else:
        start_idx = 0
        embeddings = []

    if start_idx >= len(words):
        print("All embeddings already fetched!")
        return np.array(embeddings)

    total = len(words)
    t_start = time.time()

    for i in range(start_idx, total, BATCH_SIZE):
        batch = words[i:i + BATCH_SIZE]

        try:
            batch_embeddings = get_ollama_embeddings(batch)
            embeddings.extend(batch_embeddings)

            # Save progress
            np.save(EMBEDDINGS_FILE, np.array(embeddings))

            # Progress reporting
            done = len(embeddings)
            elapsed = time.time() - t_start
            rate = (done - start_idx) / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"Progress: {done}/{total} "
                  f"({done*100/total:.1f}%) "
                  f"| {rate:.0f} words/sec "
                  f"| ETA: {remaining/60:.1f} min")

        except Exception as e:
            print(f"Error at batch starting index {i}: {e}")
            print("Saving progress and stopping. Re-run to resume.")
            np.save(EMBEDDINGS_FILE, np.array(embeddings))
            break

    return np.array(embeddings)


# =====================================================================
# 3. TKAA — Top-K Activation Analysis
# =====================================================================

def perform_tkaa(words, M, top_k=TOP_K):
    """
    For each dimension d in the embedding matrix M:
      1. Extract column c_d = M[:, d]
      2. Find the top-K word indices by value
      3. Report the semantic cluster
    Saves full results to tkaa_results.txt.
    """
    N, D = M.shape
    print(f"\n{'='*60}")
    print(f"  TKAA: {N} words × {D} dimensions | Top-{top_k}")
    print(f"{'='*60}\n")

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        f.write(f"TKAA ANALYSIS RESULTS\n")
        f.write(f"Model: {OLLAMA_MODEL} | Words: {N} | Dims: {D} | Top-K: {top_k}\n")
        f.write(f"{'='*60}\n\n")

        for d in range(D):
            # Step 1: Extract dimension column
            c_d = M[:, d]

            # Step 2: argtopk — indices of highest activations
            top_indices = np.argsort(c_d)[-top_k:][::-1]

            # Step 3: Decode to words
            top_words = [words[idx] for idx in top_indices]
            top_vals = [c_d[idx] for idx in top_indices]

            # Format output
            word_str = ", ".join(f"{w} ({v:.3f})" for w, v in zip(top_words, top_vals))
            line = f"Dim {d:04d} | {word_str}"

            # Write all to file
            f.write(line + "\n")

            # Print first N to console
            if d < SAMPLE_DIMS:
                print(line)

        if D > SAMPLE_DIMS:
            print(f"\n... ({D - SAMPLE_DIMS} more dimensions in {RESULTS_FILE})")

    print(f"\nFull results saved to: {RESULTS_FILE}")
    print(f"Embedding matrix saved to: {EMBEDDINGS_FILE}")
    print(f"Vocabulary saved to: {VOCAB_FILE}")


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Top-K Activation Analysis (TKAA) — Ollama Local Edition")
    print("=" * 60)

    # 1. Get all dictionary words (just the words, no definitions)
    words = get_dictionary_words()

    # 2. Fetch embeddings locally via Ollama
    matrix = fetch_all_embeddings(words)

    # 3. Perform TKAA if we have data
    if matrix.shape[0] == len(words):
        perform_tkaa(words, matrix)
    elif matrix.shape[0] > 0:
        # Partial results — analyze what we have
        partial_words = words[:matrix.shape[0]]
        print(f"\nNote: Analyzing partial results ({matrix.shape[0]}/{len(words)} words)")
        perform_tkaa(partial_words, matrix)
    else:
        print("\nNo embeddings fetched. Is Ollama running?")

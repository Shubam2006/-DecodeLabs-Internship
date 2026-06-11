"""
Tech Stack Recommender — AI Recommendation Logic
DecodeLabs | Industrial Training Kit | Batch 2026 | Project 3

Algorithm: Content-Based Filtering with TF-IDF + Cosine Similarity
Pipeline: Ingestion → Scoring → Sorting → Filtering (Top-N)
"""

import csv
import math
import os


# ─────────────────────────────────────────────
# STEP 0: DATA INGESTION — Load the dataset
# ─────────────────────────────────────────────

def load_dataset(filepath: str) -> dict[str, list[str]]:
    """
    Reads raw_skills.csv and returns a dict:
        { job_role: [skill1, skill2, ...] }
    """
    dataset = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row["job_role"].strip()
            skills = [s.strip().lower() for s in row["skills"].split()]
            dataset[role] = skills
    return dataset


# ─────────────────────────────────────────────
# STEP 1: VECTOR MAPPING — Build shared vocabulary
# ─────────────────────────────────────────────

def build_vocabulary(dataset: dict[str, list[str]]) -> list[str]:
    """
    Collects every unique skill across all job roles.
    Returns a sorted list — this is the shared vocabulary space.
    """
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(vocab)


# ─────────────────────────────────────────────
# STEP 2: FEATURE EXTRACTION — TF-IDF Weighting
# ─────────────────────────────────────────────

def compute_tf(doc_skills: list[str]) -> dict[str, float]:
    """
    Term Frequency: how often each skill appears relative to doc size.
    TF(t, d) = count(t in d) / total_terms(d)
    """
    tf = {}
    total = len(doc_skills)
    for skill in doc_skills:
        tf[skill] = tf.get(skill, 0) + 1
    for skill in tf:
        tf[skill] /= total
    return tf


def compute_idf(dataset: dict[str, list[str]]) -> dict[str, float]:
    """
    Inverse Document Frequency: penalises skills common across all roles.
    IDF(t) = log( N / df(t) )
    where N = total docs, df(t) = docs containing term t
    """
    N = len(dataset)
    df = {}
    for skills in dataset.values():
        for skill in set(skills):
            df[skill] = df.get(skill, 0) + 1
    idf = {}
    for skill, freq in df.items():
        idf[skill] = math.log(N / freq)
    return idf


def vectorize(skills: list[str], vocabulary: list[str], idf: dict[str, float]) -> list[float]:
    """
    Converts a skill list into a TF-IDF weighted numeric vector
    aligned to the shared vocabulary.
    """
    tf = compute_tf(skills)
    vector = []
    for term in vocabulary:
        tfidf = tf.get(term, 0) * idf.get(term, 0)
        vector.append(tfidf)
    return vector


# ─────────────────────────────────────────────
# STEP 3: SIMILARITY ENGINE — Cosine Similarity
# ─────────────────────────────────────────────

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    cos(θ) = (A · B) / (||A|| × ||B||)
    Score 1 → perfectly aligned
    Score 0 → no overlap
    Score -1 → opposite (won't occur with TF-IDF ≥ 0)
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    # Cold-start guard: zero vector means no preferences provided
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ─────────────────────────────────────────────
# STEP 4: THE 4-STEP RANKING PIPELINE
# ─────────────────────────────────────────────

def recommend(user_skills: list[str],
              dataset: dict[str, list[str]],
              top_n: int = 3) -> list[tuple[str, float, list[str]]]:
    """
    Full IPO pipeline:
      INPUT   → user_skills (≥3 required for data density)
      PROCESS → TF-IDF vectorization + Cosine Similarity scoring
      OUTPUT  → Top-N ranked job roles with scores and matched skills

    Returns list of (job_role, similarity_score, matched_skills)
    """
    # --- Validate minimum inputs ---
    if len(user_skills) < 3:
        raise ValueError("Minimum 3 skills required for accurate matching.")

    # Normalise user input
    user_skills_clean = [s.strip().lower() for s in user_skills]

    # Build shared vocabulary from dataset
    vocabulary = build_vocabulary(dataset)

    # Pre-compute IDF across the corpus
    idf = compute_idf(dataset)

    # Vectorize the user profile
    user_vector = vectorize(user_skills_clean, vocabulary, idf)

    # --- Step 1: Ingestion + Step 2: Scoring ---
    scored = []
    for role, skills in dataset.items():
        item_vector = vectorize(skills, vocabulary, idf)
        score = cosine_similarity(user_vector, item_vector)

        # Find which user skills matched this role
        matched = [s for s in user_skills_clean if s in skills]
        scored.append((role, score, matched))

    # --- Step 3: Sorting (descending by score) ---
    scored.sort(key=lambda x: x[1], reverse=True)

    # --- Step 4: Filtering (Top-N) ---
    return scored[:top_n]


# ─────────────────────────────────────────────
# STEP 5: PRESENTATION LAYER — CLI Interface
# ─────────────────────────────────────────────

def display_banner():
    print("=" * 60)
    print("  🤖  TECH STACK RECOMMENDER  |  DecodeLabs Project 3")
    print("       AI Recommendation Logic — Batch 2026")
    print("=" * 60)
    print()


def display_results(results: list[tuple[str, float, list[str]]]):
    print()
    print("━" * 60)
    print("  🏆  YOUR TOP CAREER RECOMMENDATIONS")
    print("━" * 60)
    medals = ["🥇", "🥈", "🥉"]
    for i, (role, score, matched) in enumerate(results):
        medal = medals[i] if i < 3 else f"#{i+1}"
        print(f"\n  {medal}  {role}")
        print(f"      Match Score : {score:.4f}  ({score*100:.1f}%)")
        if matched:
            print(f"      Matched Skills: {', '.join(matched)}")
        else:
            print(f"      Matched Skills: (none directly — profile similarity used)")
    print()
    print("━" * 60)
    print()


def run_cli():
    """Interactive command-line interface."""
    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "raw_skills.csv")
    dataset = load_dataset(csv_path)

    display_banner()
    print("  This engine uses TF-IDF + Cosine Similarity to match")
    print("  your skills to the best-fit tech career roles.\n")

    print("  Popular skills you can enter:")
    print("  Python, SQL, JavaScript, Java, React, Docker, AWS,")
    print("  Machine_Learning, TensorFlow, Git, Linux, Node.js ...\n")

    # Collect user skills (minimum 3)
    while True:
        raw = input("  Enter your skills (comma-separated, min 3):\n  > ").strip()
        user_skills = [s.strip() for s in raw.split(",") if s.strip()]
        if len(user_skills) >= 3:
            break
        print(f"  ⚠  You entered {len(user_skills)} skill(s). Please enter at least 3.\n")

    # Top-N preference
    while True:
        n_input = input("\n  How many recommendations? (default: 3): ").strip()
        if n_input == "":
            top_n = 3
            break
        if n_input.isdigit() and 1 <= int(n_input) <= 10:
            top_n = int(n_input)
            break
        print("  ⚠  Please enter a number between 1 and 10.")

    print(f"\n  ⏳ Analysing profile for: {', '.join(user_skills)} ...")

    # Run the recommendation pipeline
    results = recommend(user_skills, dataset, top_n=top_n)
    display_results(results)

    # Show similarity matrix option
    show_all = input("  View full similarity scores for all roles? (y/n): ").strip().lower()
    if show_all == "y":
        all_results = recommend(user_skills, dataset, top_n=len(dataset))
        print("\n  Full Ranking:")
        print(f"  {'Rank':<5} {'Job Role':<35} {'Score':>8}")
        print("  " + "-" * 50)
        for idx, (role, score, _) in enumerate(all_results, 1):
            bar = "█" * int(score * 20)
            print(f"  {idx:<5} {role:<35} {score:.4f}  {bar}")
        print()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_cli()
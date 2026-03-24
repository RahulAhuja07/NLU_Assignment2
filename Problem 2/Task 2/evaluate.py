"""
Task-2: Quantitative Evaluation - Novelty Rate & Diversity
Computes metrics on generated names from each model.
"""
import os
import json


def load_names(filepath):
    """Load names from file (one per line), returns list to preserve duplicates for novelty."""
    names = []
    if not os.path.exists(filepath):
        return names
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if n:
                names.append(n)
    return names


def load_training_names(filepath="TrainingNames.txt"):
    """Load training set (lowercase for comparison)."""
    path = filepath
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), filepath)
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip().lower()
            if n and n.isalpha():
                names.add(n)
    return names


def novelty_rate(generated, training):
    """Percentage of generated names NOT in training set."""
    gen_lower = [g.lower() for g in generated]
    novel = sum(1 for g in gen_lower if g not in training)
    return 100.0 * novel / len(generated) if generated else 0.0


def diversity(generated):
    """Unique names / total generated (as percentage)."""
    unique = len(set(g.lower() for g in generated))
    total = len(generated)
    return 100.0 * unique / total if total else 0.0


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    training = load_training_names()
    print("=" * 70)
    print("Task-2: Quantitative Evaluation")
    print("=" * 70)
    print(f"Training set size: {len(training)} names\n")

    models = [
        ("Vanilla_RNN", os.path.join(base, "vanilla_rnn_names.txt")),
        ("BLSTM", os.path.join(base, "blstm_names.txt")),
        ("RNN_Attention", os.path.join(base, "rnn_attention_names.txt")),
    ]

    results = []
    for name, path in models:
        gen = list(load_names(path))
        if not gen:
            print(f"{name}: No generated names found. Run sequence_models.py first.")
            continue

        nr = novelty_rate(gen, training)
        div = diversity(gen)
        results.append({
            "model": name,
            "novelty_rate": nr,
            "diversity": div,
            "total": len(gen),
            "unique": len(set(g.lower() for g in gen)),
        })
        print(f"{name}:")
        print(f"  Novelty Rate: {nr:.2f}%")
        print(f"  Diversity:    {div:.2f}%")
        print(f"  Total/Unique: {len(gen)} / {len(set(g.lower() for g in gen))}")
        print()

    # Save results
    out = os.path.join(base, "evaluation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()

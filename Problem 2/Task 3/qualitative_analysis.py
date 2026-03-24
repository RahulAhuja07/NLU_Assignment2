"""
Task-3: Qualitative Analysis - Realism, Failure Modes, Sample Outputs
Analyzes generated names from each model.
"""
import os


def load_names(filepath):
    names = []
    if not os.path.exists(filepath):
        return names
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()
            if n:
                names.append(n)
    return names


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    models = [
        ("Vanilla RNN", "vanilla_rnn_names.txt"),
        ("BLSTM", "blstm_names.txt"),
        ("RNN with Attention", "rnn_attention_names.txt"),
    ]

    lines = [
        "=" * 70,
        "Task-3: Qualitative Analysis",
        "=" * 70,
        "",
        "1. REALISM: Do generated names look like real Indian names?",
        "   - Good models produce names with typical syllable patterns (consonant-vowel)",
        "   - Realistic names: Aarav, Priya, Rajesh, Kavya, etc.",
        "",
        "2. COMMON FAILURE MODES:",
        "   - Repetition: same substring repeated (e.g., rarara...)",
        "   - Too short or too long names",
        "   - Invalid character combinations or unpronounceable strings",
        "   - Memorization: exact copies from training set",
        "",
        "3. SAMPLE OUTPUTS BY MODEL:",
        "",
    ]

    for model_name, fname in models:
        path = os.path.join(base, fname)
        names = load_names(path)
        lines.append("-" * 70)
        lines.append(f"{model_name} (first 30 unique):")
        lines.append("-" * 70)
        seen = set()
        for n in names:
            if n.lower() not in seen and len(seen) < 30:
                seen.add(n.lower())
                lines.append(f"  {n}")
        lines.append("")

    out = os.path.join(base, "qualitative_analysis.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()

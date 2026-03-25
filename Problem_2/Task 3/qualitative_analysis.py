#b23ch1037
#Rahul Ahuja
import os

#load the names from the file
def load_names(filepath):
    names = []#list to store the names
    if not os.path.exists(filepath):
        return names
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip()#strip the line
            if n:
                names.append(n)#append the name to the list
    return names


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task 1")
    models = [
        ("Vanilla RNN", "vanilla_rnn_names.txt"),
        ("BLSTM", "blstm_names.txt"),
        ("RNN with Attention", "rnn_attention_names.txt"),
    ]

    lines = [
        "Qualitative Analysis",
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
            if n.lower() not in seen and len(seen) < 30:#check if the name is not in the set and the length of the set is less than 30
                seen.add(n.lower())
                lines.append(f"  {n}")
        lines.append("")

    out = os.path.join(base, "qualitative_analysis.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved!")


if __name__ == "__main__":
    main()

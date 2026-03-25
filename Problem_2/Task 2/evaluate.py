#b23ch1037
#Rahul Ahuja
import os
import json


def load_names(filepath):
    #load names from file (one per line), returns list to preserve duplicates for novelty.
    names = []
    if not os.path.exists(filepath):
        return names
    with open(filepath, "r", encoding="utf-8") as f:#open the file in read mode
        for line in f:
            n = line.strip()#strip the line
            if n:
                names.append(n)#append the name to the list
    return names


def load_training_names(filepath=os.path.join("..", "Task 0", "TrainingNames.txt")):
    #load training set (lowercase for comparison)
    path = filepath#path to the training names file
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), filepath)
    names = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            n = line.strip().lower()#strip the line and convert to lowercase
            if n and n.isalpha():
                names.add(n)#add the name to the set
    return names


def novelty_rate(generated, training):
    #percentage of generated names NOT in training set.
    gen_lower = [g.lower() for g in generated]
    novel = sum(1 for g in gen_lower if g not in training)
    return 100.0 * novel / len(generated) if generated else 0#calculate the novelty rate


def diversity(generated):
    #unique names / total generated (as percentage).
    unique = len(set(g.lower() for g in generated))
    total = len(generated)
    return 100.0 * unique / total if total else 0#calculate the diversity


def main():
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task 1")#path to the generated output directory
    training = load_training_names()
    print("Task-2: Quantitative Evaluation")
    print(f"Training size: {len(training)} names\n")
    #models to evaluate
    models = [
        ("Vanilla RNN", os.path.join(base, "vanilla_rnn_names.txt")),
        ("BLSTM", os.path.join(base, "blstm_names.txt")),
        ("RNN Attention", os.path.join(base, "rnn_attention_names.txt")),
    ]

    results = []#list to store the results
    for name, path in models:
        gen = list(load_names(path))#load the names from the file
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
        print(f" Novelty Rate: {nr:.2f}%")
        print(f" Diversity:    {div:.2f}%")
        print(f" Total/Unique: {len(gen)} / {len(set(g.lower() for g in gen))}")#print the total and unique names
        print()

    # save results
    out = os.path.join(base, "evaluation_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved !")


if __name__ == "__main__":
    main()

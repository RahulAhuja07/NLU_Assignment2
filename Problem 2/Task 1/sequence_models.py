"""
Problem 2: Character-Level Name Generation using RNN Variants
Implements from scratch: Vanilla RNN, BLSTM, RNN with Attention.
Uses PyTorch. Trained on Indian names from TrainingNames.txt.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DATA_PATH = "TrainingNames.txt"
BATCH_SIZE = 64
HIDDEN_SIZE = 128
EMBED_SIZE = 32
NUM_LAYERS = 2
LEARNING_RATE = 0.003
EPOCHS = int(os.environ.get("Q2_EPOCHS", 40))
MAX_NAME_LEN = 20
GEN_NAMES_PER_MODEL = 200
TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5]

# Special tokens (PAD must be index 0 for padding in batches)
PAD = "\0"
EOS = "."


# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
class NameDataset(Dataset):
    """
    Character-level dataset. Each name is a sequence of character indices.
    Target: predict next character. We append EOS to mark end of name.
    """
    def __init__(self, filepath):
        self.names = self._load(filepath)
        chars = sorted(set("".join(self.names) + EOS))
        # PAD at 0 for batch padding; Embedding uses padding_idx=0
        self.char2idx = {PAD: 0}
        for i, c in enumerate(chars):
            self.char2idx[c] = i + 1
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.pad_idx = 0

    def _load(self, filepath):
        names = []
        candidates = [
            filepath,
            os.path.join(os.path.dirname(__file__), filepath),
            os.path.join(os.path.dirname(__file__), "..", filepath),
        ]
        path = None
        for p in candidates:
            if os.path.exists(p):
                path = p
                break
        if path is None:
            raise FileNotFoundError(
                "TrainingNames.txt not found. Place it in Ques_2/ with one Indian name per line."
            )
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip().lower()
                if name and name.isalpha():
                    names.append(name)
        return names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i] + EOS
        ids = [self.char2idx[c] for c in name]
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate_pad(batch):
    """Pad sequences to same length. PAD index 0 for x; -100 for y (ignored in loss)."""
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
    return xs, ys


# ---------------------------------------------------------------------------
# 1. VANILLA RNN
# ---------------------------------------------------------------------------
class VanillaRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network.
    Architecture: Embedding -> Multi-layer RNN -> Linear -> Vocab logits.
    """
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, h = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, h


# ---------------------------------------------------------------------------
# 2. BIDIRECTIONAL LSTM (BLSTM)
# ---------------------------------------------------------------------------
class BLSTM(nn.Module):
    """
    Bidirectional LSTM. Processes sequence both forward and backward.
    For generation: we pass the prefix each step; backward sees reversed prefix.
    """
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, h = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, h


# ---------------------------------------------------------------------------
# 3. RNN WITH BASIC ATTENTION
# ---------------------------------------------------------------------------
class RNNAttention(nn.Module):
    """
    RNN with causal (autoregressive) attention over previous hidden states.
    At each step, attends only to previous positions (masked).
    """
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # Attention: query from current hidden, keys/values from all previous hiddens
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        rnn_out, h = self.rnn(emb, hidden)  # (B, T, H)
        # Simple additive self-attention with causal mask
        B, T, H = rnn_out.shape
        scores = torch.bmm(rnn_out, rnn_out.transpose(1, 2))  # (B, T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, rnn_out)
        combined = torch.cat([rnn_out, context], dim=-1)
        logits = self.fc(combined)
        return logits, h


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, pad_idx):
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        # Flatten for cross entropy; ignore padded positions
        logits_flat = logits.view(-1, logits.size(-1))
        y_flat = y.view(-1)
        mask = y_flat != -100
        loss = criterion(logits_flat[mask], y_flat[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n += mask.sum().item()
    return total_loss / max(n, 1)


def is_valid_name(name, min_len=4, max_len=16):
    """Filter out garbage: too short, bad consonant clusters, no vowels."""
    if len(name) < min_len or len(name) > max_len:
        return False
    name_lower = name.lower()
    vowels = set("aeiou")
    consonant_streak = 0
    vowel_count = 0
    for c in name_lower:
        if c in vowels:
            vowel_count += 1
            consonant_streak = 0
        elif c.isalpha():
            consonant_streak += 1
            if consonant_streak >= 4:  # 4+ consecutive consonants is rare in names
                return False
    if vowel_count == 0 or vowel_count / len(name_lower) < 0.15:  # need some vowels
        return False
    return True


def generate_names(model, dataset, num_names, temperature=0.8, max_len=20, device="cpu", top_k=50, min_valid_len=4):
    """Generate names with top-k sampling and quality filtering."""
    model.eval()
    gen = []
    pad_idx = dataset.pad_idx
    vowels = set("aeiou")
    start_chars = [c for c in dataset.char2idx if c.isalpha()]
    if not start_chars:
        start_chars = [dataset.idx2char[i] for i in range(1, min(27, dataset.vocab_size))]

    max_attempts = num_names * 3  # allow retries for valid names
    attempts = 0

    with torch.no_grad():
        while len(gen) < num_names and attempts < max_attempts:
            attempts += 1
            start = np.random.choice(start_chars)
            idx = dataset.char2idx[start]
            seq = [idx]
            for _ in range(max_len - 1):
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits, _ = model(x)
                logits = logits[0, -1, :].clone() / temperature
                logits[pad_idx] = float("-inf")

                # Top-k sampling: only sample from top-k likely tokens
                if top_k and top_k < logits.size(0):
                    k = min(top_k, (logits > float("-inf")).sum().item())
                    if k > 1:
                        top_vals, top_idx = torch.topk(logits, k)
                        logits = torch.full_like(logits, float("-inf"))
                        logits[top_idx] = top_vals

                probs = torch.softmax(logits, dim=0).cpu().numpy()
                next_idx = np.random.choice(len(probs), p=probs)
                next_char = dataset.idx2char[next_idx]
                if next_char == EOS:
                    break
                seq.append(next_idx)

            name = "".join(dataset.idx2char[i] for i in seq if i != pad_idx).capitalize()
            if is_valid_name(name, min_len=min_valid_len):
                gen.append(name)

    # If we still don't have enough, relax filters slightly
    if len(gen) < num_names:
        with torch.no_grad():
            for _ in range(num_names - len(gen)):
                start = np.random.choice(start_chars)
                idx = dataset.char2idx[start]
                seq = [idx]
                for _ in range(max_len - 1):
                    x = torch.tensor([seq], dtype=torch.long, device=device)
                    logits, _ = model(x)
                    logits = logits[0, -1, :].clone() / max(temperature, 0.5)
                    logits[pad_idx] = float("-inf")
                    if top_k:
                        k = min(top_k, (logits > float("-inf")).sum().item())
                        if k > 1:
                            top_vals, top_idx = torch.topk(logits, k)
                            logits = torch.full_like(logits, float("-inf"))
                            logits[top_idx] = top_vals
                    probs = torch.softmax(logits, dim=0).cpu().numpy()
                    next_idx = np.random.choice(len(probs), p=probs)
                    next_char = dataset.idx2char[next_idx]
                    if next_char == EOS:
                        break
                    seq.append(next_idx)
                name = "".join(dataset.idx2char[i] for i in seq if i != pad_idx).capitalize()
                if len(name) >= 3:
                    gen.append(name)

    return gen


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_training_names(dataset):
    """Get set of training names (lowercase) for novelty calculation."""
    return set(dataset.names)


def novelty_rate(generated, training):
    gen_lower = [g.lower() for g in generated]
    novel = sum(1 for g in gen_lower if g not in training)
    return 100.0 * novel / len(generated) if generated else 0.0


def diversity(generated):
    unique = len(set(g.lower() for g in generated))
    total = len(generated)
    return 100.0 * unique / total if total else 0.0


def plot_loss_curves(loss_history, output_path):
    """Plot loss curves for all three models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, losses in loss_history.items():
        ax.plot(losses, label=model_name, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves - All Models")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve saved to {output_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 70)
    print("Problem 2: Character-Level Name Generation")
    print("=" * 70)
    print("Note: If TrainingNames.txt shows content in your editor but script says empty, SAVE the file first (Cmd+S).")

    # Load data (no auto-overwrite: if file is empty, fail with clear message)
    dataset = NameDataset(DATA_PATH)
    if len(dataset) == 0:
        data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)
        raise ValueError(
            "TrainingNames.txt is empty or has no valid names. "
            "If you see names in your editor, SAVE the file (Cmd+S / Ctrl+S) and try again."
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad, num_workers=0)
    print(f"\nLoaded {len(dataset)} names, vocab size {dataset.vocab_size}")

    # Criterion: ignore padding index in target
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    models_config = [
        ("Vanilla_RNN", VanillaRNN(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS
        )),
        ("BLSTM", BLSTM(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS
        )),
        ("RNN_Attention", RNNAttention(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE
        )),
    ]

    # Architecture summary (Task-1)
    print("\n" + "-" * 70)
    print("TASK-1: Model Architectures & Hyperparameters")
    print("-" * 70)
    for name, model in models_config:
        n = count_params(model)
        print(f"  {name}: {n:,} trainable parameters")
    print(f"  Hyperparameters: hidden_size={HIDDEN_SIZE}, embed_size={EMBED_SIZE},")
    print(f"  layers={NUM_LAYERS}, lr={LEARNING_RATE}, epochs={EPOCHS}")

    training_names = load_training_names(dataset)
    results = {}
    loss_history = {}
    metrics = {
        "training": {},
        "generation": {},
        "per_temperature": {},
    }
    output_dir = os.path.dirname(os.path.abspath(__file__))

    for name, model in models_config:
        print(f"\n--- Training {name} ---")
        model = model.to(device)
        lr = LEARNING_RATE * 0.5 if "BLSTM" in name else LEARNING_RATE
        epochs = int(EPOCHS * 1.2) if "BLSTM" in name else EPOCHS
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []

        for ep in range(epochs):
            loss = train_epoch(model, loader, criterion, optimizer, device, dataset.pad_idx)
            losses.append(round(loss, 4))
            if (ep + 1) % 25 == 0 or ep == 0:
                print(f"  Epoch {ep+1}/{epochs}  loss={loss:.4f}")

        loss_history[name] = losses
        metrics["training"][name] = {"losses": losses, "final_loss": losses[-1], "epochs": epochs}

        top_k = 35 if "BLSTM" in name else 50
        gen_by_temp = {}
        default_gen = []

        for temp in TEMPERATURES:
            gen = generate_names(model, dataset, GEN_NAMES_PER_MODEL, temp, MAX_NAME_LEN, device, top_k=top_k)
            nr = novelty_rate(gen, training_names)
            div = diversity(gen)
            gen_by_temp[str(temp)] = {
                "novelty_rate": round(nr, 2),
                "diversity": round(div, 2),
                "count": len(gen),
                "samples": gen[:15],
            }
            if temp == 0.8:
                default_gen = gen

        metrics["per_temperature"][name] = gen_by_temp
        nr_default = novelty_rate(default_gen, training_names)
        div_default = diversity(default_gen)
        metrics["generation"][name] = {
            "novelty_rate": round(nr_default, 2),
            "diversity": round(div_default, 2),
            "total_generated": len(default_gen),
            "temperatures_tested": TEMPERATURES,
        }

        out_file = os.path.join(output_dir, f"{name.lower()}_names.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            for n in default_gen:
                f.write(n + "\n")
        results[name] = {"generated": default_gen, "count": len(default_gen), "file": out_file}
        print(f"  Saved {len(default_gen)} names (temp=0.8) to {out_file}")
        print(f"  Novelty: {nr_default:.1f}%  Diversity: {div_default:.1f}%")

    plot_loss_curves(loss_history, os.path.join(output_dir, "loss_curves.png"))

    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  metrics.json saved")

    print("\n" + "=" * 70)
    print("Done. See metrics.json, loss_curves.png")
    print("=" * 70)


if __name__ == "__main__":
    main()

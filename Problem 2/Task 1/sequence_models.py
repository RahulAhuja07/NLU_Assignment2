#B23CH1037
#Rahul Ahuja
#importing all the necessary libraries
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


DATA_PATH = "TrainingNames.txt" #path to the training names file
BATCH_SIZE = 64 #batch size
HIDDEN_SIZE = 128 #hidden size
EMBED_SIZE = 32 #embedding size
NUM_LAYERS = 2 #number of layers
LEARNING_RATE = 0.003 #learning rate
EPOCHS = int(os.environ.get("Q2_EPOCHS", 40)) #number of epochs
MAX_NAME_LEN = 20 #maximum name length
GEN_NAMES_PER_MODEL = 200 #number of names to generate per model
TEMPERATURES = [0.5, 0.8, 1.0, 1.2, 1.5] #temperatures to test


class NameDataset(Dataset):
    #class to load the training names file
    def __init__(self, filepath):
        self.names = self._load(filepath)
        chars = sorted(set("".join(self.names) + EOS))
        # PAD at 0 for batch padding;
        self.char2idx = {PAD: 0}
        for i, c in enumerate(chars):
            self.char2idx[c] = i + 1
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)
        self.pad_idx = 0

    def _load(self, filepath):#function to load the training names file from the given path
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
                "TrainingNames.txt not found."#check for if the file is present
            )
        #open the file and read the names
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip().lower()
                if name and name.isalpha():
                    names.append(name)#append the names to the list
        return names

    def __len__(self):#function to return the length of the dataset
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i] + EOS
        ids = [self.char2idx[c] for c in name]
        x = torch.tensor(ids[:-1], dtype=torch.long)#convert the ids to a tensor
        y = torch.tensor(ids[1:], dtype=torch.long)#convert the ids to a tensor
        return x, y


def collate_pad(batch):
    #function to pad the sequences to the same length
    xs, ys = zip(*batch)
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, pad_value=0)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, pad_value=-100)
    return xs, ys


#BLSTM model
class BLSTM(nn.Module):
    #class to initialize the BLSTM model with parameters as vocab_size, embed_size, hidden_size, num_layers and dropout
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)#embedding layer to convert the input ids to embeddings
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout
        )#LSTM layer to process the input sequence
        self.fc = nn.Linear(hidden_size * 2, vocab_size)#linear layer to convert the hidden state to logits

    def forward(self, x, hidden=None):#function to forward the input through the model
        emb = self.embed(x)
        out, h = self.lstm(emb, hidden)#process the input sequence through the LSTM layer
        logits = self.fc(out)
        return logits, h#convert the hidden state to logits


#Vanilla RNN model
class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, num_layers=2, dropout=0.2):#function to initialize the Vanilla RNN model with parameters as vocab_size, embed_size, hidden_size, num_layers and dropout
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)#embedding layer to convert the input ids to embeddings
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)#linear layer to convert the hidden state to logits

    def forward(self, x, hidden=None):#function to forward the input through the model
        emb = self.embed(x)
        out, h = self.rnn(emb, hidden)
        logits = self.fc(out)#convert the hidden state to logits
        return logits, h


#RNN with attention model
class RNNAttention(nn.Module):
    #class to initialize the RNN with attention model with parameters as vocab_size, embed_size, hidden_size and dropout
    def __init__(self, vocab_size, embed_size=32, hidden_size=128, dropout=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)#embedding layer to convert the input ids to embeddings
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        # Attention: query from current hidden, keys/values from all previous hiddens
        self.attn_w = nn.Linear(hidden_size * 2, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x, hidden=None):#function to forward the input through the model
        emb = self.embed(x)
        rnn_out, h = self.rnn(emb, hidden) #B, T, H
        # Simple additive self-attention with causal mask
        B, T, H = rnn_out.shape
        scores = torch.bmm(rnn_out, rnn_out.transpose(1, 2)) #B, T, T
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn, rnn_out)#B, T, H
        combined = torch.cat([rnn_out, context], dim=-1)#B, T, 2H
        logits = self.fc(combined)#B, T, V
        return logits, h#convert the hidden state to logits


#train
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
#used to filter the names such as too short, bad consonant clusters, no vowels. eg abcd
    if len(name) < min_len or len(name) > max_len:
        return False
    name_lower = name.lower()#convert the name to lowercase
    vowels = set("aeiou")
    consonant_streak = 0#used to count the number of consecutive consonants
    vowel_count = 0#used to count the number of vowels
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

#generating names 
def generate_names(model, dataset, num_names, temperature=0.8, max_len=20, device="cpu", top_k=50, min_valid_len=4):
    #top k sampling
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
        while len(gen) < num_names and attempts < max_attempts:#generate names until the number of names is reached or the maximum number of attempts is reached
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

            name = "".join(dataset.idx2char[i] for i in seq if i != pad_idx).capitalize()#convert the sequence to a name
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


#counting the number of parameters in the model
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#loading the training names
def load_training_names(dataset):
    return set(dataset.names)

#calc for novelty rate.
def novelty_rate(generated, training):
    gen_lower = [g.lower() for g in generated]
    novel = sum(1 for g in gen_lower if g not in training)
    return 100.0 * novel / len(generated) if generated else 0.0

#calc for diversity.
def diversity(generated):
    unique = len(set(g.lower() for g in generated))
    total = len(generated)
    return 100.0 * unique / total if total else 0.0

#plot the loss curves
def plot_loss_curves(loss_history, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_name, losses in loss_history.items():
        ax.plot(losses, label=model_name, alpha=0.9)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Loss curve saved! ")


def main():
    print("Character-Level Name Generation")

    # Load data from the training names file
    dataset = NameDataset(DATA_PATH)
    if len(dataset) == 0:
        data_path = os.path.join(os.path.dirname(__file__), DATA_PATH)
        raise ValueError(
            "TrainingNames.txt is empty or has no valid names. "
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad, num_workers=0)
    print(f"\nLoaded {len(dataset)} names, vocab size {dataset.vocab_size}")

    # Criterion: ignore padding index in target
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    models_config = [
        ("Vanilla RNN", VanillaRNN(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS
        )),
        ("BLSTM", BLSTM(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS
        )),
        ("RNN Attention", RNNAttention(
            dataset.vocab_size, EMBED_SIZE, HIDDEN_SIZE
        )),
    ]

    # Architecture summary
    print("--------------------------------")
    print("TASK-1: Model Architectures & Hyperparameters")
    print("--------------------------------")
    for name, model in models_config:
        n = count_params(model)
        print(f"  {name}: {n:,} trainable parameters")
    print("--------------------------------")
    print(f"  Hyperparameters: hidden_size={HIDDEN_SIZE}, embed_size={EMBED_SIZE},")
    print(f"  layers={NUM_LAYERS}, lr={LEARNING_RATE}, epochs={EPOCHS}")
    print("--------------------------------")

    training_names = load_training_names(dataset)
    results = {}
    loss_history = {}
    metrics = {
        "training": {},
        "generation": {},
        "per_temperature": {},
    }
    output_dir = os.path.dirname(os.path.abspath(__file__))
    #training the models
    for name, model in models_config:
        print(f"\nTraining {name} ")
        model = model.to(device)
        lr = LEARNING_RATE * 0.5 if "BLSTM" in name else LEARNING_RATE#learning rate for the model
        epochs = int(EPOCHS * 1.2) if "BLSTM" in name else EPOCHS#number of epochs for the model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses = []#list to store the losses
#training the model for the number of epochs
        for ep in range(epochs):
            loss = train_epoch(model, loader, criterion, optimizer, device, dataset.pad_idx)
            losses.append(round(loss, 4))
            if (ep + 1) % 25 == 0 or ep == 0:
                print(f"  Epoch {ep+1}/{epochs}  loss={loss:.4f}")
#storing the losses for the model
        loss_history[name] = losses
        metrics["training"][name] = {"losses": losses, "final_loss": losses[-1], "epochs": epochs}
        #generating names for the model
        top_k = 35 if "BLSTM" in name else 50
        gen_by_temp = {}
        default_gen = []
        #generating names for the model for the different temperatures
        for temp in TEMPERATURES:
            gen = generate_names(model, dataset, GEN_NAMES_PER_MODEL, temp, MAX_NAME_LEN, device, top_k=top_k)
            nr = novelty_rate(gen, training_names)#calculating the novelty rate for the model
            div = diversity(gen)#calculating the diversity for the model
            gen_by_temp[str(temp)] = {
                "novelty_rate": round(nr, 2),
                "diversity": round(div, 2),
                "count": len(gen),
                "samples": gen[:15],
            }
            if temp == 0.8:#storing the names for the default temperature(avg value)
                default_gen = gen

        metrics["per_temperature"][name] = gen_by_temp
        nr_default = novelty_rate(default_gen, training_names)
        div_default = diversity(default_gen)
        metrics["generation"][name] = {
            "novelty_rate": round(nr_default, 2),
            "diversity": round(div_default, 2),
            "total_generated": len(default_gen),
            "temp_tested": TEMPERATURES,
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
    print(f"\nsaved")

    print("\n" + "--------------------------------")
    print("Done!")
    print("--------------------------------")


if __name__ == "__main__":
    main()

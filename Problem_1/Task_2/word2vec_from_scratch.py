import os
import json
import time
from collections import Counter
import numpy as np

# Paths for input corpus and saving outputs
CORPUS_PATH = os.path.join("..", "Task_1", "task2.txt")
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

# Create directories if they don't exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Limiting dataset size and training settings for faster runs
MAX_SENTENCES = 120000
MIN_COUNT = 8
EPOCHS = 1

# Small set of hyperparameters to experiment with
EMBED_DIMS = [50, 100]
WINDOWS = [4, 6]
NEG_SAMPLES = [5, 10]


# Basic sigmoid function with clipping for numerical stability
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


# Softmax function used in CBOW full softmax training
def softmax(x):
    x = x - np.max(x)
    ex = np.exp(np.clip(x, -500, 500))
    return ex / (np.sum(ex) + 1e-10)


# Load corpus line by line and convert into tokenized sentences
def load_corpus(path, max_sentences=MAX_SENTENCES):
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_sentences:
                break
            toks = line.strip().split()
            if len(toks) >= 2:
                sentences.append(toks)
    return sentences


# Build vocabulary and keep only words above frequency threshold
def build_vocab(sentences, min_count=MIN_COUNT):
    counts = Counter(w for s in sentences for w in s)
    vocab = [w for w, c in counts.items() if c >= min_count]
    vocab.sort()

    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}

    # Frequency array used for negative sampling
    freq = np.array([counts[id2word[i]] for i in range(len(vocab))], dtype=np.float64)

    return word2id, id2word, freq


# Convert words in sentences to their corresponding ids
def encode_sentences(sentences, word2id):
    out = []
    for s in sentences:
        ids = [word2id[w] for w in s if w in word2id]
        if len(ids) >= 2:
            out.append(ids)
    return out


# Create a sampling table for efficient negative sampling
def sample_table(freq, power=0.75, table_size=2_000_000):
    prob = (freq ** power) / np.sum(freq ** power)
    table = np.zeros(table_size, dtype=np.int32)

    cumsum = np.cumsum(prob)
    j = 0

    for i in range(table_size):
        x = (i + 1) / table_size
        while j < len(cumsum) - 1 and x > cumsum[j]:
            j += 1
        table[i] = j

    return table


# Cosine similarity between two vectors
def cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-10)
    b = b / (np.linalg.norm(b) + 1e-10)
    return float(np.dot(a, b))


# Simple evaluation using a few word pairs
def evaluate_embeddings(emb, word2id):
    pairs = [("research", "phd"), ("student", "exam"), ("faculty", "department")]
    vals = []

    for w1, w2 in pairs:
        if w1 in word2id and w2 in word2id:
            vals.append(cosine(emb[word2id[w1]], emb[word2id[w2]]))

    return float(np.mean(vals)) if vals else 0.0


# Train CBOW using full softmax (no negative sampling)
def train_cbow_full_softmax(encoded, vocab_size, dim, window, lr=0.01, epochs=1):

    scale = 0.5 / dim
    W_in = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
    W_out = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)

    for ep in range(epochs):
        t0 = time.time()
        loss_sum, steps = 0.0, 0
        total = len(encoded)

        for sidx, sent in enumerate(encoded, start=1):
            for i, tgt in enumerate(sent):
                lo, hi = max(0, i - window), min(len(sent), i + window + 1)
                ctx = sent[lo:i] + sent[i + 1:hi]

                if not ctx:
                    continue

                # Average context vector
                v_ctx = np.mean(W_in[ctx], axis=0)

                # Compute probabilities using softmax
                logits = W_out @ v_ctx
                probs = softmax(logits)

                # Cross-entropy loss
                loss = -np.log(probs[tgt] + 1e-10)
                loss_sum += loss

                # Backpropagation
                grad = probs
                grad[tgt] -= 1.0

                W_out_old = W_out.copy()
                W_out -= lr * np.outer(grad, v_ctx)

                grad_ctx = W_out_old.T @ grad
                W_in[np.array(ctx)] -= (lr / len(ctx)) * grad_ctx

                steps += 1

            # Progress print for long training
            if sidx % 2000 == 0 or sidx == total:
                pct = 100.0 * sidx / total
                elapsed = time.time() - t0
                eta = (elapsed / max(sidx, 1)) * (total - sidx)

                print(f"\r  CBOW ep{ep+1}: {pct:6.2f}% | elapsed {elapsed:6.1f}s | eta {eta:6.1f}s", end="", flush=True)

        print()
        print(f"  CBOW epoch {ep+1}/{epochs} loss={loss_sum / max(steps, 1):.6f}, steps={steps}")

    return W_in


# Train Skip-gram using negative sampling
def train_skipgram_ns(encoded, vocab_size, freq, dim, window, neg_k, lr=0.01, epochs=1):

    scale = 0.5 / dim
    W_in = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)
    W_out = np.random.uniform(-scale, scale, (vocab_size, dim)).astype(np.float32)

    neg_table = sample_table(freq)

    for ep in range(epochs):
        t0 = time.time()
        loss_sum, steps = 0.0, 0
        total = len(encoded)

        for sidx, sent in enumerate(encoded, start=1):
            n = len(sent)

            for i, center in enumerate(sent):
                lo, hi = max(0, i - window), min(n, i + window + 1)
                ctx_words = sent[lo:i] + sent[i + 1:hi]

                if not ctx_words:
                    continue

                v_c = W_in[center]

                for ctx in ctx_words:

                    # Positive pair update
                    pos = sigmoid(np.dot(v_c, W_out[ctx]))
                    g_pos = (pos - 1.0) * lr

                    W_out[ctx] -= g_pos * v_c
                    W_in[center] -= g_pos * W_out[ctx]

                    loss_sum += -np.log(pos + 1e-10)

                    # Negative samples
                    neg_ids = []
                    while len(neg_ids) < neg_k:
                        nid = neg_table[np.random.randint(0, len(neg_table))]
                        if nid != ctx:
                            neg_ids.append(nid)

                    for nid in neg_ids:
                        negp = sigmoid(np.dot(v_c, W_out[nid]))
                        g_neg = negp * lr

                        W_out[nid] -= g_neg * v_c
                        W_in[center] -= g_neg * W_out[nid]

                        loss_sum += -np.log(1.0 - negp + 1e-10)

                    steps += 1

            # Progress tracking
            if sidx % 2000 == 0 or sidx == total:
                pct = 100.0 * sidx / total
                elapsed = time.time() - t0
                eta = (elapsed / max(sidx, 1)) * (total - sidx)

                print(f"\r  SG   ep{ep+1}: {pct:6.2f}% | elapsed {elapsed:6.1f}s | eta {eta:6.1f}s", end="", flush=True)

        print()
        print(f"  SG epoch {ep+1}/{epochs} loss={loss_sum / max(steps, 1):.6f}, steps={steps}")

    return W_in


# Save embeddings in readable text format
def save_txt_embeddings(path, emb, id2word):
    with open(path, "w", encoding="utf-8") as f:
        v, d = emb.shape
        f.write(f"{v} {d}\n")

        for i in range(v):
            f.write(f"{id2word[i]} {' '.join(f'{x:.6f}' for x in emb[i])}\n")


# Main execution flow
def main():

    print("Loading corpus...")
    sentences = load_corpus(CORPUS_PATH, MAX_SENTENCES)
    print(f"  Sentences used: {len(sentences)}")

    print("Building vocabulary...")
    word2id, id2word, freq = build_vocab(sentences, MIN_COUNT)
    print(f"  Vocabulary size: {len(word2id)}")

    encoded = encode_sentences(sentences, word2id)
    print(f"  Encoded sentences: {len(encoded)}")

    # Save mappings for later use
    np.save(os.path.join(MODELS_DIR, "word2id.npy"), word2id)
    np.save(os.path.join(MODELS_DIR, "id2word.npy"), id2word)

    results = {"cbow": [], "skipgram": []}

    # Run CBOW experiments
    for dim in EMBED_DIMS:
        for win in WINDOWS:
            print(f"\nTraining CBOW (dim={dim}, win={win})")

            emb = train_cbow_full_softmax(encoded, len(word2id), dim, win, lr=0.01, epochs=EPOCHS)

            base = os.path.join(MODELS_DIR, f"cbow_dim{dim}_win{win}")
            np.save(base + ".npy", emb)
            save_txt_embeddings(base + ".txt", emb, id2word)

            score = evaluate_embeddings(emb, word2id)
            results["cbow"].append({"dim": dim, "window": win, "score": score})

    # Run Skip-gram experiments
    for dim in EMBED_DIMS:
        for win in WINDOWS:
            for neg in NEG_SAMPLES:
                print(f"\nTraining Skip-gram (dim={dim}, win={win}, neg={neg})")

                emb = train_skipgram_ns(encoded, len(word2id), freq, dim, win, neg, lr=0.01, epochs=EPOCHS)

                base = os.path.join(MODELS_DIR, f"sg_dim{dim}_win{win}_neg{neg}")
                np.save(base + ".npy", emb)
                save_txt_embeddings(base + ".txt", emb, id2word)

                score = evaluate_embeddings(emb, word2id)
                results["skipgram"].append({"dim": dim, "window": win, "neg": neg, "score": score})

    # Save results for report
    with open(os.path.join(OUTPUTS_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
import os
import numpy as np
from collections import Counter
import json

# File paths for corpus, saved models and output results
CORPUS_PATH  = os.path.join("..", "Task_1", "task2.txt")
MODELS_DIR   = os.path.join("..", "Task_2", "models")
OUTPUTS_DIR  = "outputs"

RESULTS_JSON = os.path.join(OUTPUTS_DIR, "semantic_analysis_results.json")
REPORT_TXT   = os.path.join(OUTPUTS_DIR, "semantic_analysis_report.txt")

# Make sure required folders exist
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)


# Simple CBOW model implementation from scratch
class W2V:
    def __init__(self, vocab_size, embedding_dim, context_window, learning_rate=0.025):
        self.vocab_size     = vocab_size
        self.embedding_dim  = embedding_dim
        self.context_window = context_window
        self.learning_rate  = learning_rate

        # Initialize weights with small random values
        scale = np.sqrt(2.0 / (vocab_size + embedding_dim))
        self.W_input  = np.random.randn(vocab_size, embedding_dim) * scale
        self.W_output = np.random.randn(vocab_size, embedding_dim) * scale

        self.loss_history = []

    # Sigmoid with clipping to avoid overflow
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    # Training loop for CBOW
    def train(self, sentences, word2id, epochs=5):

        print(f"Training CBOW (dim={self.embedding_dim}, window={self.context_window})")

        for epoch in range(epochs):
            epoch_loss, count = 0.0, 0

            for sentence in sentences:
                word_ids = [word2id[w] for w in sentence if w in word2id]

                if len(word_ids) < 2:
                    continue

                for i, target_id in enumerate(word_ids):
                    lo  = max(0, i - self.context_window)
                    hi  = min(len(word_ids), i + self.context_window + 1)

                    ctx = word_ids[lo:i] + word_ids[i+1:hi]

                    if not ctx:
                        continue

                    # Average context embeddings
                    ctx_embed = np.mean(self.W_input[ctx], axis=0)

                    # Predict target using sigmoid
                    score = self.sigmoid(ctx_embed @ self.W_output[target_id])

                    error = score - 1.0
                    epoch_loss += -np.log(max(1e-10, score))

                    # Update weights
                    delta = error * self.learning_rate

                    self.W_output[target_id]    -= delta * ctx_embed
                    self.W_input[np.array(ctx)] -= delta * self.W_output[target_id]

                    count += 1

            avg_loss = epoch_loss / max(count, 1)
            self.loss_history.append(avg_loss)

            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.6f}")

        print("Training complete.\n")

    # Return learned embeddings
    def get_embeddings(self):
        return self.W_input.copy()


# Normalize vectors so cosine similarity becomes dot product
def normalize(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-10)


# Find top-k similar words using cosine similarity
def find_nearest_neighbors(word, word2id, id2word, emb_norm, k=5):
    if word not in word2id:
        return []

    idx = word2id[word]

    sims = emb_norm @ emb_norm[idx]
    sims[idx] = -2.0  # ignore the word itself

    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    return [(id2word[i], float(sims[i])) for i in top_idx]


# Solve analogy: A:B :: C:?
def word_analogy(a, b, c, word2id, id2word, emb_norm, k=5):

    for w in (a, b, c):
        if w not in word2id:
            return []

    # Standard analogy vector: B - A + C
    vec = emb_norm[word2id[b]] - emb_norm[word2id[a]] + emb_norm[word2id[c]]
    vec = vec / max(np.linalg.norm(vec), 1e-10)

    sims = emb_norm @ vec

    # Remove query words from results
    for w in (a, b, c):
        sims[word2id[w]] = -2.0

    top_idx = np.argpartition(sims, -k)[-k:]
    top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

    return [(id2word[i], float(sims[i])) for i in top_idx]


# Load full corpus into memory
def load_corpus(filepath):
    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            if words:
                lines.append(words)
    return lines


# Build vocabulary with minimum frequency threshold
def build_vocab(sentences, min_count=2):
    counts  = Counter(w for s in sentences for w in s)

    vocab   = {w for w, c in counts.items() if c >= min_count}

    word2id = {w: i for i, w in enumerate(sorted(vocab))}
    id2word = {i: w for w, i in word2id.items()}

    return word2id, id2word


# Try loading vocabulary saved during training phase
def try_load_saved_vocab():

    w2i_path = os.path.join(MODELS_DIR, "word2id.npy")
    i2w_path = os.path.join(MODELS_DIR, "id2word.npy")

    if os.path.exists(w2i_path) and os.path.exists(i2w_path):
        try:
            word2id = np.load(w2i_path, allow_pickle=True).item()
            id2word = np.load(i2w_path, allow_pickle=True).item()

            if isinstance(word2id, dict) and isinstance(id2word, dict):
                print(f"Loaded saved vocab (size={len(word2id)})")
                return word2id, id2word

        except Exception as e:
            print(f"Failed to load vocab: {e}")

    return None, None


# Try loading already trained embeddings
def try_load_pretrained(vocab_size):

    candidates = [
        os.path.join(MODELS_DIR, "best_sg_embeddings.npy"),
        os.path.join(MODELS_DIR, "best_cbow_embeddings.npy"),
    ]

    for path in candidates:
        if os.path.exists(path):
            emb = np.load(path)

            if emb.shape[0] == vocab_size:
                print(f"Loaded pretrained embeddings: {path}")
                return emb

    return None


# Main pipeline for semantic analysis
def main():

    print("\nSEMANTIC ANALYSIS START\n")

    # Load dataset
    if not os.path.exists(CORPUS_PATH):
        print("Corpus not found. Run preprocessing first.")
        return

    sentences = load_corpus(CORPUS_PATH)
    print(f"Sentences loaded: {len(sentences)}")

    # Get vocabulary
    word2id, id2word = try_load_saved_vocab()

    if word2id is None:
        word2id, id2word = build_vocab(sentences, min_count=3)
        print(f"Vocabulary rebuilt: {len(word2id)}")

    # Load or train embeddings
    embeddings = try_load_pretrained(len(word2id))

    if embeddings is None:
        print("Training new CBOW model...")

        model = W2V(len(word2id), embedding_dim=200, context_window=10)
        model.train(sentences, word2id, epochs=5)

        embeddings = model.get_embeddings()

    # Normalize embeddings
    emb_norm = normalize(embeddings)

    # Nearest neighbors test words
    test_words = ["research", "student", "phd", "exam"]

    nn_results = {}

    print("\nNearest Neighbors:\n")

    for word in test_words:
        nbrs = find_nearest_neighbors(word, word2id, id2word, emb_norm)

        nn_results[word] = nbrs

        print(word, "->", nbrs)

    # Analogy tests
    analogies = [
        ("ug", "pg", "btech"),
        ("research", "phd", "teaching"),
    ]

    analogy_results = {}

    print("\nAnalogies:\n")

    for a, b, c in analogies:
        preds = word_analogy(a, b, c, word2id, id2word, emb_norm)

        key = f"{a}:{b}::{c}:?"
        analogy_results[key] = preds

        print(key, "->", preds)

    # Save results
    results = {
        "nearest_neighbors": nn_results,
        "analogies": analogy_results
    }

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print("\nResults saved.")


if __name__ == "__main__":
    main()
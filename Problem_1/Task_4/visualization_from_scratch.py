import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import json
import os

# Lightweight Word2Vec implementation (CBOW + Skip-gram)
class FastWord2Vec:
    
    def __init__(self, vocab_size, embedding_dim, context_window, learning_rate=0.01, model_type='cbow'):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_window = context_window
        self.learning_rate = learning_rate
        self.model_type = model_type

        # Initialize embeddings with small random values
        self.W_input = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W_output = np.random.randn(vocab_size, embedding_dim) * 0.01

    # Sigmoid with clipping to avoid overflow
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # Train CBOW model
    def train_cbow(self, sentences, word2id, epochs=2):

        print(f"Training CBOW (dim={self.embedding_dim}, window={self.context_window})...")

        for epoch in range(epochs):
            for sent_idx, sentence in enumerate(sentences):

                # Print progress occasionally to track long runs
                if sent_idx % 500 == 0:
                    print(f"  Epoch {epoch+1}: sentence {sent_idx}/{len(sentences)}")

                # Convert words → ids and filter unknown ones
                word_ids = [word2id.get(word, -1) for word in sentence]
                word_ids = [wid for wid in word_ids if wid != -1]

                if len(word_ids) < 2:
                    continue

                for i in range(len(word_ids)):
                    start = max(0, i - self.context_window)
                    end = min(len(word_ids), i + self.context_window + 1)

                    context_ids = word_ids[start:i] + word_ids[i+1:end]
                    target_id = word_ids[i]

                    if not context_ids:
                        continue

                    # Average context embeddings
                    context_embed = np.mean(self.W_input[context_ids], axis=0)

                    # Predict target
                    output = self.sigmoid(np.dot(context_embed, self.W_output[target_id]))

                    error = output - 1
                    delta = error * self.learning_rate * 0.01

                    # Update weights
                    self.W_output[target_id] -= delta * context_embed
                    for ctx_id in context_ids:
                        self.W_input[ctx_id] -= delta * self.W_output[target_id]

        print("✔ CBOW Training complete!")

    # Train Skip-gram with negative sampling
    def train_skipgram(self, sentences, word2id, epochs=2, neg_samples=5):

        print(f"Training Skip-gram (dim={self.embedding_dim}, window={self.context_window})...")

        for epoch in range(epochs):
            for sent_idx, sentence in enumerate(sentences):

                if sent_idx % 500 == 0:
                    print(f"  Epoch {epoch+1}: sentence {sent_idx}/{len(sentences)}")

                word_ids = [word2id.get(word, -1) for word in sentence]
                word_ids = [wid for wid in word_ids if wid != -1]

                if len(word_ids) < 2:
                    continue

                for i in range(len(word_ids)):
                    target_id = word_ids[i]

                    start = max(0, i - self.context_window)
                    end = min(len(word_ids), i + self.context_window + 1)
                    context_ids = word_ids[start:i] + word_ids[i+1:end]

                    if not context_ids:
                        continue

                    target_embed = self.W_input[target_id]

                    # Positive updates
                    for ctx_id in context_ids:
                        ctx_embed = self.W_output[ctx_id]
                        output = self.sigmoid(np.dot(target_embed, ctx_embed))

                        error = output - 1
                        delta = error * self.learning_rate * 0.001

                        self.W_output[ctx_id] -= delta * target_embed
                        self.W_input[target_id] -= delta * ctx_embed

                    # Negative sampling updates
                    for _ in range(neg_samples):
                        neg_id = np.random.randint(0, self.vocab_size)

                        if neg_id != target_id:
                            neg_embed = self.W_output[neg_id]
                            output = self.sigmoid(np.dot(target_embed, neg_embed))

                            delta = output * self.learning_rate * 0.001
                            self.W_output[neg_id] -= delta * target_embed
                            self.W_input[target_id] -= delta * neg_embed

        print("✔ Skip-gram Training complete!")

    def get_embeddings(self):
        return self.W_input


# PCA implementation for dimensionality reduction
class PCAFromScratch:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):

        print(f"Fitting PCA ({self.n_components} components)...")

        # Center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Covariance matrix
        cov_matrix = np.cov(X_centered.T)

        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Pick top components
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]

        return self

    def transform(self, X):
        return np.dot(X - self.mean, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# Simplified t-SNE implementation
class TSNEFromScratch:

    def __init__(self, n_components=2, perplexity=30, n_iter=1000, learning_rate=200):
        self.n_components = n_components
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    # Compute pairwise distances
    def _pairwise_distances(self, X):
        n = X.shape[0]
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = np.sum((X[i] - X[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _compute_gaussian_kernel(self, distances, sigma):
        return np.exp(-distances / (2 * sigma ** 2))

    def fit_transform(self, X):

        print("Running t-SNE...")

        n = X.shape[0]

        # Normalize input to avoid overflow issues
        X = np.asarray(X, dtype=np.float64)
        X = X - np.mean(X, axis=0)
        X = X / max(np.max(np.abs(X)), 1e-10)

        Y = np.random.randn(n, self.n_components) * 0.0001

        distances = self._pairwise_distances(X)

        P = self._compute_gaussian_kernel(distances, sigma=1.0)
        P = P / max(np.sum(P), 1e-12)
        P = np.clip(P, 1e-12, 1)

        for iteration in range(self.n_iter):

            if iteration % 100 == 0:
                print(f"  Iteration {iteration}/{self.n_iter}")

            distances_2d = self._pairwise_distances(Y)

            Q = 1 / (1 + distances_2d)
            np.fill_diagonal(Q, 0)
            Q = Q / max(np.sum(Q), 1e-12)
            Q = np.clip(Q, 1e-12, 1)

            PQ = P - Q

            for i in range(n):
                grad = np.zeros(self.n_components)

                for j in range(n):
                    if i != j:
                        grad += 4 * PQ[i, j] * (Y[i] - Y[j])

                Y[i] -= self.learning_rate * grad

        print("✔ t-SNE complete!")

        return Y


# Load a subset of corpus for faster processing
def load_corpus_sample(filepath, max_lines=1000):
    lines = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            words = line.strip().split()
            if words:
                lines.append(words)

    return lines


# Build vocabulary from corpus
def build_vocab(sentences, min_count=3):
    word_counts = Counter()

    for sentence in sentences:
        word_counts.update(sentence)

    vocab = {word: count for word, count in word_counts.items() if count >= min_count}

    word2id = {word: idx for idx, word in enumerate(sorted(vocab.keys()))}
    id2word = {idx: word for word, idx in word2id.items()}

    return word2id, id2word


# Pick a small set of words for visualization
def select_visualization_words(word2id, id2word):

    words = ['research', 'student', 'phd', 'exam', 'learning', 'faculty', 'engineering']

    return [w for w in words if w in word2id]


# Plot embeddings using PCA and t-SNE
def visualize_embeddings(embeddings_cbow, embeddings_skipgram, word2id, id2word):

    viz_words = select_visualization_words(word2id, id2word)

    if len(viz_words) < 5:
        print("Not enough words to visualize")
        return

    indices = [word2id[w] for w in viz_words]

    embed_cbow = embeddings_cbow[indices]
    embed_skipgram = embeddings_skipgram[indices]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA (CBOW)
    pca = PCAFromScratch(2)
    cbow_2d = pca.fit_transform(embed_cbow)

    axes[0,0].scatter(cbow_2d[:,0], cbow_2d[:,1])
    axes[0,0].set_title("CBOW PCA")

    # PCA (Skip-gram)
    pca = PCAFromScratch(2)
    sg_2d = pca.fit_transform(embed_skipgram)

    axes[0,1].scatter(sg_2d[:,0], sg_2d[:,1])
    axes[0,1].set_title("Skip-gram PCA")

    # t-SNE (CBOW)
    tsne = TSNEFromScratch(n_iter=200, learning_rate=1)
    cbow_tsne = tsne.fit_transform(embed_cbow)

    axes[1,0].scatter(cbow_tsne[:,0], cbow_tsne[:,1])
    axes[1,0].set_title("CBOW t-SNE")

    # t-SNE (Skip-gram)
    tsne = TSNEFromScratch(n_iter=200, learning_rate=1)
    sg_tsne = tsne.fit_transform(embed_skipgram)

    axes[1,1].scatter(sg_tsne[:,0], sg_tsne[:,1])
    axes[1,1].set_title("Skip-gram t-SNE")

    plt.tight_layout()
    plt.savefig("embeddings_visualization.png")
    plt.close()

    print("✔ Visualization saved")


# Main execution
def main():

    print("Running Task-4...")

    corpus_path = os.path.join("..", "Task_1", "task2.txt")

    sentences = load_corpus_sample(corpus_path, max_lines=5000)

    word2id, id2word = build_vocab(sentences, min_count=2)
    vocab_size = len(word2id)

    print(f"Vocab size: {vocab_size}")

    # Train both models
    cbow = FastWord2Vec(vocab_size, 100, 5)
    cbow.train_cbow(sentences, word2id)

    skipgram = FastWord2Vec(vocab_size, 100, 5)
    skipgram.train_skipgram(sentences, word2id)

    # Visualize embeddings
    visualize_embeddings(
        cbow.get_embeddings(),
        skipgram.get_embeddings(),
        word2id,
        id2word
    )

    print("Done!")


if __name__ == "__main__":
    main()
# NLU Assignment 2: Word Embeddings and Sequence Models

This repository contains two main problems exploring fundamental Natural Language Understanding (NLU) techniques built from scratch and using standard libraries. 

- **Problem 1** focuses on scraping data, tokenizing, building Word2Vec (CBOW and Skip-gram) manually via Numpy, analyzing semantic similarities, and reducing dimensions manually (PCA, t-SNE) for visualization.
- **Problem 2** focuses on character-level sequence modeling (Vanilla RNN, BLSTM, RNN with Attention) for Indian Name Generation, along with quantitative and qualitative evaluation.

---

## 📂 Project Structure

### Problem 1: Word Embeddings Pipeline

#### `Problem_1/Task_1/` : Data Collection & Preprocessing
- **Script:** `Task1.py`
- **Description:** Scrapes text from academic URLs, parses local PDFs, and performs text preprocessing. This includes lowercasing, boilerplate removal, stopword removal, character/number filtering, and vocabulary normalization (e.g., standardizing academic terms like `B.Tech` or `UG`). 
- **Outputs:** Merges the corpus and creates cleaned sentence outputs (`task2.txt`) for embeddings. Generates a frequency-based Word Cloud.

#### `Problem_1/Task_2/` : Word2Vec Implementation (NumPy only)
- **Script:** `word2vec_from_scratch.py`
- **Description:** Implements **Continuous Bag of Words (CBOW)** using full softmax, and **Skip-gram** with negative sampling strictly from scratch using NumPy. 
- **Outputs:** Learns embedding matrices over the preprocessed corpus and exports `dict` / `npy` weights inside the `models/` directory for analytical downstream tasks.

#### `Problem_1/Task_3/` : Semantic Analysis
- **Script:** `semantic_analysis.py`
- **Description:** Loads the trained Word2Vec embeddings to answer semantic relation queries.
- **Tasks:**
  - **Nearest Neighbors:** Cosine similarity matching to identify top 5 closest words.
  - **Word Analogies:** Mathematical vector algebra (e.g. `B - A + C`) evaluating analogies such as `UG` is to `BTech` as `PG` is to `?`.

#### `Problem_1/Task_4/` : Visualization and Dimension Reduction
- **Script:** `visualization_from_scratch.py`
- **Description:** Projects high-dimensional text embeddings down to 2 dimensions for graphing.
- **Algorithms Implemented from Scratch:** 
  - **PCA** (Principal Component Analysis) using covariance matrix and eigenvectors.
  - **t-SNE** (t-Distributed Stochastic Neighbor Embedding) using gradient descent over pair-wise distances and Gaussian/Student-t distributions.
- **Outputs:** Generated scatter plots (`embeddings_visualization.png`) comparing CBOW vs Skip-Gram topologies.

---

### Problem 2: Sequence Generation Models

#### `Problem_2/Task 0/`
- **Dataset:** `TrainingNames.txt` - Provides raw characters sequences for training the Generative Language models.

#### `Problem_2/Task 1/` : Model Architectures & Generation
- **Script:** `sequence_models.py`
- **Description:** Defines PyTorch datasets and sequence models tailored for character-by-character text generation.
- **Models trained:**
  1. Vanilla RNN
  2. Bidirectional LSTM (BLSTM)
  3. RNN with Additive Causal Self-Attention
- **Outputs:** Trains network weights, generates synthetic names exploring temperature sampling and exports them as text lists.

#### `Problem_2/Task 2/` : Quantitative Evaluation
- **Script:** `evaluate.py`
- **Description:** Compares the dynamically generated outputs to the training set to evaluate how well the model learned character correlations.
- **Metrics:**
  - **Novelty Rate:** Measures percentage of unique generated sequences vs. memorized training examples.
  - **Diversity:** Analyzes the uniqueness within the generated batches themselves. 

#### `Problem_2/Task 3/` : Qualitative Analysis
- **Script:** `qualitative_analysis.py`
- **Description:** Reviews syllable realism visually summarizing failure modes such as constant repetitions, lack of vowels, or length irregularities mapping Indian name topologies.

---

## 🚀 How to Run

### Setting up Environment
First, ensure you have python `3.8+` with your standard environment initialized:
```bash
pip install numpy torch matplotlib wordcloud requests beautifulsoup4 PyPDF2
```

### Running Problem 1
Run scripts sequentially passing forward the derived datasets.
```bash
cd Problem_1

# Parse data and normalize
cd Task_1 && python3 Task1.py && cd ..

# Train neural embeddings representation
cd Task_2 && python3 word2vec_from_scratch.py && cd ..

# Run semantic searches and vector math
cd Task_3 && python3 semantic_analysis.py && cd ..

# Project visually
cd Task_4 && python3 visualization_from_scratch.py && cd ..
```

### Running Problem 2
```bash
cd Problem_2
cd "Task 1"
python3 sequence_models.py
cd "../Task 2"
python3 evaluate.py
cd "../Task 3"
python3 qualitative_analysis.py
```

### Note on Directories
File-dependent input/outputs generally link upwards using relative paths. E.g. `Problem_1/Task_2` points to corpus outputs populated backward in `Problem_1/Task_1`. Run the tasks chronologically to ensure outputs are populated for following tasks.

---  
*Assignment by: Rahul Ahuja (B23CH1037)*

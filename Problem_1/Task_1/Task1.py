#b23ch1037
#rahul ahuja
import os
import re
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Folder paths for input/output handling
PDF_FOLDER    = "./data"
DATA_TXT      = "./data_txt"
OUTPUT_FOLDER = "./data_txt"
MERGED_FILE   = "./data/clean_corpus_raw_task2.txt"
FINAL_OUTPUT  = "./task2.txt"

# List of web pages from which text will be scraped
URLS = [
    "https://iitj.ac.in/computer-science-engineering/en/faculty-achievements",
    "https://anandmishra22.github.io/",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/chemistry/en/chemistry",
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/",
    "https://www.iitj.ac.in/computer-science-engineering/"
]

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Basic cleaning: lowercase + remove numbers and special characters
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Remove unwanted website artifacts like links, emails, etc.
def remove_boilerplate(text):
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'(copyright|privacy policy|terms|disclaimer).*?(?=\n|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+|\d+ of \d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Convert academic terms into consistent format
def normalize_academic_terms(text):
    text = re.sub(r'\bB\.?\s*Tech\.?\b', ' btech ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bM\.?\s*Tech\.?\b', ' mtech ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bUG\b', ' ug ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPG\b', ' pg ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPh\.?\s*D\.?\b', ' phd ', text, flags=re.IGNORECASE)
    return text

# Clean repeated punctuation (e.g., !!! → .)
def remove_excessive_punctuation(text):
    text = re.sub(r'[!?.]{2,}', '.', text)
    text = re.sub(r'[-]{2,}', '-', text)
    return text

# Tokenize text and remove common stopwords
def tokenize_and_clean(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)

    stopwords = {
        'the','and','for','with','that','this','from','are','was','been','have','has','had',
        'can','could','would','should','will','shall','may','might','must',
        'about','above','after','again','against','all','am','an','any','as','at',
        'be','because','before','being','below','between','both','but','by',
        'do','does','doing','down','during','each','few','further',
        'he','her','here','him','his','how','i','if','in','into','is','it','its',
        'me','more','most','my','no','nor','not','of','off','on','only','or',
        'other','our','out','over','own','so','some','such','than','too','under',
        'up','very','we','were','what','when','where','which','while','who',
        'you','your'
    }

    tokens = [t for t in tokens if t not in stopwords and len(t) > 2]
    return tokens

# Read PDFs and convert them into text files
def pdf_to_txt():
    if not os.path.exists(PDF_FOLDER):
        print(f"  PDF folder '{PDF_FOLDER}' not found, skipping PDF extraction")
        return
    for file in os.listdir(PDF_FOLDER):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, file))
            text = ""
            for page in reader.pages:
                if page.extract_text():
                    text += page.extract_text() + "\n"

            with open(os.path.join(OUTPUT_FOLDER, file.replace(".pdf", ".txt")), "w", encoding="utf-8") as f:
                f.write(text)

# Fetch webpage content and save as text files
def url_to_txt():
    headers = {"User-Agent": "Mozilla/5.0"}

    for i, url in enumerate(URLS):
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts and styles to keep only meaningful text
        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n")

        with open(os.path.join(OUTPUT_FOLDER, f"url_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

# Combine all collected text files into one large corpus
def merge_files():
    all_text = ""
    count = 0

    for folder in [OUTPUT_FOLDER, DATA_TXT]:
        if not os.path.exists(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(".txt"):
                try:
                    with open(os.path.join(folder, file), "r", encoding="utf-8", errors="ignore") as f:
                        all_text += f.read() + "\n"
                    count += 1
                except Exception as e:
                    print(f"  Skip {file}: {e}")

    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f" Total documents merged: {count}")

# Main preprocessing pipeline
def preprocess():

    with open(MERGED_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    # Apply different cleaning steps
    text = remove_boilerplate(text)
    text = normalize_academic_terms(text)
    text = remove_excessive_punctuation(text)

    # Split text into sentences for Word2Vec-style processing
    sentences_raw = re.split(r'[.!?]\s+|\n+', text)
    sentence_tokens = []
    all_tokens = []

    for sent in sentences_raw:
        if sent.strip():
            sent_clean = basic_clean(sent)
            tok = tokenize_and_clean(sent_clean)
            if len(tok) >= 2:
                sentence_tokens.append(tok)
                all_tokens.extend(tok)

    # If sentence splitting fails, fallback to chunk-based splitting
    if len(sentence_tokens) < 100:
        text = basic_clean(text)
        all_tokens = tokenize_and_clean(text)
        chunk_size = 50
        sentence_tokens = [
            all_tokens[i:i+chunk_size]
            for i in range(0, len(all_tokens), chunk_size)
            if len(all_tokens[i:i+chunk_size]) >= 2
        ]

    # Save processed sentences to output file
    with open(FINAL_OUTPUT, "w", encoding="utf-8") as f:
        for sent in sentence_tokens:
            f.write(" ".join(sent) + "\n")

    tokens = all_tokens

    # Compute dataset statistics
    token_count = len(tokens)
    vocab_size = len(set(tokens))
    num_documents = len(sentence_tokens)
    freq = Counter(tokens)

    print("\n--- Dataset Statistics (Task-1) ---")
    print(f"Total documents (sentences): {num_documents}")
    print(f"Total tokens: {token_count}")
    print(f"Vocabulary size: {vocab_size}")

    print("\nTop 15 words:")
    for word, count in freq.most_common(15):
        print(word, ":", count)

    # Generate word cloud from most frequent words
    print("\nGenerating Word Cloud...")

    wc = WordCloud(
        width=1000,
        height=500,
        background_color='white'
    ).generate(" ".join(tokens))

    plt.figure(figsize=(12, 6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("Word Cloud - Most Frequent Words")
    plt.show()

    wc.to_file(os.path.join(OUTPUT_FOLDER, "wordcloud.png"))
    print(f"Word Cloud saved as {os.path.join(OUTPUT_FOLDER, 'wordcloud.png')}")

# Entry point of the script
if __name__ == "__main__":
    pdf_to_txt()
    url_to_txt()
    merge_files()
    preprocess()
import os
import glob
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
import json

nltk.download('punkt')
nltk.download('punkt_tab')

# --- Optional: Coreference Resolution ---
# If you choose to perform coreference resolution, you can uncomment the following code.
# Note: You'll need to install and configure a coreference resolution package such as neuralcoref.
#
# import spacy
# import neuralcoref
#
# def init_nlp():
#     nlp = spacy.load('en_core_web_sm')
#     neuralcoref.add_to_pipe(nlp)
#     return nlp
#
# nlp = init_nlp()
#
# def resolve_coreferences(text):
#     doc = nlp(text)
#     return doc._.coref_resolved
#
# For now, we'll use a dummy function that returns the text as is.

def resolve_coreferences(text):
    # Placeholder for coreference resolution
    return text

# --- Load and Clean Text Data ---
def load_and_clean_text(file_path):
    """
    Reads an HTML file, extracts the text, and cleans it.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Use BeautifulSoup to strip HTML tags
    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

def load_all_texts(directory):
    """
    Loads and cleans all HTML files from a specified directory.
    """
    all_texts = []
    for file in glob.glob(os.path.join(directory, "*.html")):
        text = load_and_clean_text(file)
        all_texts.append(text)
    return all_texts

# --- Chunking the Text ---
def chunk_text(text, max_chunk_words=500, overlap_sentences=1):
    """
    Splits text into chunks with a maximum number of words, allowing for overlapping sentences.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    overlap_buffer = []

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        # If adding the sentence would exceed the chunk size, finalize the current chunk.
        if current_word_count + sentence_word_count > max_chunk_words:
            chunks.append(" ".join(current_chunk))
            # Start the next chunk with the overlap buffer
            current_chunk = overlap_buffer + [sentence]
            current_word_count = sum(len(s.split()) for s in overlap_buffer) + sentence_word_count
            # Update the overlap buffer
            overlap_buffer = current_chunk[-overlap_sentences:]
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            # Update the overlap buffer
            overlap_buffer = current_chunk[-overlap_sentences:]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def load_processed_pages(json_file='processed_pages.json'):
    """
    Loads the processed pages from the JSON file.
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        pages = json.load(f)
    return pages

# --- Preprocessing Pipeline ---
def preprocess_processed_pages(json_file='processed_pages.json', max_chunk_words=500):
    """
    For each processed page, resolve coreferences and split the page data into chunks.
    Returns a list of chunks ready for embedding.
    """
    pages = load_processed_pages(json_file)
    all_chunks = []
    
    for idx, page in enumerate(pages):
        # Gather all text from excerpt and content
        text_parts = []
        if 'excerpt' in page:
            text_parts.append(page['excerpt'])
        if 'content' in page:
            text_parts.append(page['content'])
        combined_text = "\n".join(text_parts)

        # Optionally perform coreference resolution
        resolved_text = resolve_coreferences(combined_text)

        # Split text into chunks
        chunks = chunk_text(resolved_text, max_chunk_words)

        # Collect chunked results
        for chunk in chunks:
            all_chunks.append({
                'entry_id': idx,
                'chunk_text': chunk
            })
    
    return all_chunks

if __name__ == "__main__":
    # Set the directory containing your HTML files (scraped from Motive’s Help Center)
    chunks = preprocess_processed_pages('processed_pages.json', max_chunk_words=500)
    print(f"Total chunks prepared: {len(chunks)}")
    
    # Save the chunks to a JSON file for downstream processing
    with open('processed_text_chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

    # Print a sample chunk
    if chunks:
        sample = chunks[0]
        print("\nSample chunk:")
        # print("Page:", sample['page'])
        # print("Tags:", sample['tags'])
        print("Text preview:", sample['chunk_text'][:200], "...")
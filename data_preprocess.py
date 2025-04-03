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

# --- Chunking the Text ---
def chunk_text(text, max_chunk_words=500):
    """
    Splits text into chunks with a maximum number of words per chunk.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        if current_word_count + sentence_word_count > max_chunk_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# --- Deduplication ---
def deduplicate_chunks(chunks):
    """
    Deduplicate a list of chunk dictionaries based on the 'chunk_text' field.
    For duplicates, aggregate titles and URLs.
    
    Args:
        chunks (list): List of dictionaries, each with 'chunk_text', 'title', and 'url' keys.
    
    Returns:
        list: A list of deduplicated chunk dictionaries.
    """
    unique_chunks = {}
    for chunk in chunks:
        text = chunk.get("chunk_text", "").strip()
        if text in unique_chunks:
            # Aggregate titles and URLs from duplicates
            if chunk.get("title", "") not in unique_chunks[text]["titles"]:
                unique_chunks[text]["titles"].append(chunk.get("title", ""))
            if chunk.get("url", "") not in unique_chunks[text]["urls"]:
                unique_chunks[text]["urls"].append(chunk.get("url", ""))
        else:
            unique_chunks[text] = {
                "chunk_text": text,
                "titles": [chunk.get("title", "")],
                "urls": [chunk.get("url", "")]
            }
    
    # Convert the unique chunks dictionary to a list
    deduped_chunks = []
    for text, data in unique_chunks.items():
        deduped_chunks.append({
            "chunk_text": text,
            "titles": data["titles"],
            "urls": data["urls"]
        })
    return deduped_chunks

# --- Load Processed Pages ---
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
        title = page.get("title", "")
        url = page.get("url", "")
        # Gather all text from excerpt and content
        text_parts = []
        if 'excerpt' in page:
            excerpt = page.get("excerpt", "")
            text_parts.append(excerpt)
        if 'content' in page:
            content = page.get("content", "")
            text_parts.append(content)
            
        combined_text = "\n".join(text_parts)
        
        # Combine title and content/excerpt
        full_text = f"{title}\n{combined_text}"
        # Clean the text: remove extra whitespace, etc.
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        # Optionally perform coreference resolution
        resolved_text = resolve_coreferences(full_text)

        # Split text into chunks
        chunks = chunk_text(resolved_text, max_chunk_words)
                
        # Collect chunked results
        for chunk in chunks:
            all_chunks.append({
                # 'entry_id': idx,
                "title": title,
                "url": url,
                'chunk_text': chunk
            })
    
    return all_chunks

if __name__ == "__main__":
    # Set the directory containing your HTML files (scraped from Motive’s Help Center)
    chunks = preprocess_processed_pages('processed_pages.json', max_chunk_words=500)
    print(f"Total chunks prepared: {len(chunks)}")
    # Deduplicate chunks
    chunks = deduplicate_chunks(chunks)
    print(f"Total unique chunks after deduplication: {len(chunks)}")
    
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
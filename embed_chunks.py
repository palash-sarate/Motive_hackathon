import json
import numpy as np
import faiss
# from sentence_transformers import SentenceTransformer

# Load preprocessed text chunks from a JSON file.
def load_chunks(json_file="processed_text_chunks.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

# Generate embeddings for each text chunk.
# text-embedding-ada-002
# all-MiniLM-L6-v2
# all-MiniLM-L12-v2
# def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
#     model = SentenceTransformer(model_name)
#     texts = [chunk["chunk_text"] for chunk in chunks]
#     # The model returns a numpy array of shape (num_chunks, embedding_dim)
#     embeddings = model.encode(texts, convert_to_numpy=True)
#     return embeddings

# Build and index embeddings using FAISS.
def build_faiss_index(embeddings):
    embedding_dim = embeddings.shape[1]
    # Here we use a flat (brute-force) L2 index.
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings)
    return index

# Save the FAISS index and the metadata mapping.
def save_index_and_metadata(index, chunks, index_file="faiss_index.idx", metadata_file="chunk_metadata.json"):
    # Save the FAISS index to disk.
    faiss.write_index(index, index_file)
    # Save the mapping of index ids to the chunk metadata.
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)

def main():
    # Step 1: Load the processed text chunks.
    chunks = load_chunks("processed_text_chunks.json")
    print(f"Loaded {len(chunks)} chunks.")

    # Step 2: Generate embeddings.
    # embeddings = generate_embeddings(chunks)
    # print("Embeddings generated. Shape:", embeddings.shape)
    # # save embeddings to file
    # np.save("embeddings.npy", embeddings)
    
    # Load embeddings from file
    embeddings = np.load("embeddings_all-MiniLM-L12-v2.npy")
    print("Embeddings loaded. Shape:", embeddings.shape)
    
    # Step 3: Build FAISS index.
    index = build_faiss_index(embeddings)
    print("FAISS index built. Number of vectors indexed:", index.ntotal)

    # Step 4: Save index and metadata.
    save_index_and_metadata(index, chunks)
    print("Index and metadata saved. Ready for retrieval augmented generation.")

if __name__ == "__main__":
    main()

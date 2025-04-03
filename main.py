import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the FAISS index
index = faiss.read_index("faiss_index.idx")

# Load the metadata mapping
with open("chunk_metadata.json", "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to generate embedding for a query
def embed_query(query):
    return model.encode([query], convert_to_numpy=True)

# Function to retrieve top-k relevant chunks
def retrieve_top_k(query, k=5):
    query_embedding = embed_query(query)
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)
    # Fetch the corresponding chunks
    retrieved_chunks = [chunk_metadata[idx] for idx in indices[0]]
    return retrieved_chunks

# Function to generate a response using the retrieved chunks
def generate_response(user_query):
    retrieved_chunks = retrieve_top_k(user_query)
    # Combine the retrieved chunks into a single context
    context = " ".join([chunk["chunk_text"] for chunk in retrieved_chunks])
    # Here, integrate with your language model (e.g., GPT) to generate a response
    # For example:
    # response = language_model.generate_response(user_query, context)
    # return response

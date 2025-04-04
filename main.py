import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_core.prompts import PromptTemplate
import os
from langchain.embeddings import HuggingFaceEmbeddings

st_model_name = "all-MiniLM-L6-v2"

# Load the FAISS index
index = faiss.read_index(f"faiss_index_{st_model_name}.idx")

# Load the metadata mapping
with open(f"chunk_metadata_{st_model_name}.json", "r", encoding="utf-8") as f:
    chunk_metadata = json.load(f)

# Initialize the embedding model using LangChain's HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name=st_model_name)

# Function to generate embedding for a query
def embed_query(query):
    return np.array(embedding_model.embed_query(query)).reshape(1, -1)

# Function to retrieve top-k relevant chunks
def retrieve_top_k(query, k=5):
    query_embedding = embed_query(query)
    # Search the FAISS index
    distances, indices = index.search(query_embedding, k)
    # Fetch the corresponding chunks
    retrieved_chunks = [chunk_metadata[idx] for idx in indices[0]]
    return retrieved_chunks

# Initialize a Hugging Face model using LangChain integration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Choose a small efficient model
hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~1.1GB model

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
model = AutoModelForCausalLM.from_pretrained(
    hf_model_name, 
    device_map=device,
    torch_dtype=torch.float16  # Use half precision for efficiency
)

# Create text generation pipeline
text_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    device_map=device
)

# Create LangChain wrapper
llm = HuggingFacePipeline(pipeline=text_pipeline)

# Create response template
template = """
Context information:
{context}

Chat History:
{chat_history}

User Question: {question}

If you don't know the answer based on the provided context, say so. Don't make up information.
Answer:
"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=template,
)

# Stateful chat class with LangChain integration
class ConversationalBot:
    def __init__(self):
        self.chat_history = []
        
    def get_response(self, user_query):
        # Retrieve relevant chunks
        chunks = retrieve_top_k(user_query)
        context = " ".join([chunk["chunk_text"] for chunk in chunks])
        
        # Format chat history
        chat_history_text = ""
        for q, a in self.chat_history:
            chat_history_text += f"User: {q}\nAssistant: {a}\n"
        
        # Generate response using LangChain
        response = llm.invoke(
            PROMPT.format(
                context=context,
                chat_history=chat_history_text,
                question=user_query
            )
        )
        
        # Store conversation
        self.chat_history.append((user_query, response))
        if len(self.chat_history) > 5:  # Keep history manageable
            self.chat_history.pop(0)
            
        return {
            "response": response,
            "source_chunks": chunks
        }

# Example usage
if __name__ == "__main__":
    bot = ConversationalBot()
    
    # Interactive chat loop
    print("Welcome! Ask me anything about the documents (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        result = bot.get_response(user_input)
        print("\nBot:", result["response"])
        print("\nSources:", len(result["source_chunks"]), "chunks used")

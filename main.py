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
from huggingface_hub import login
import boto3
from pathlib import Path
from dotenv import load_dotenv
from langchain_aws import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

# set provider
provider = os.getenv("PROVIDER")
if not provider:
    raise ValueError("PROVIDER is not set in the environment variables.")

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN is not set in the environment variables.")
login(hf_token)

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")
if not aws_region:
    raise ValueError("AWS_REGION is not set in the environment variables.")
if not aws_access_key_id or not aws_secret_access_key:
    raise ValueError("AWS keys are not set in the environment variables.")

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
def retrieve_top_k(query, k=5, relevance_threshold=0.5):
    query_embedding = embed_query(query)
    # Search the FAISS index
    _, indices = index.search(query_embedding, k)
    # Fetch the corresponding chunks and filter by relevance
    retrieved_chunks = []
    for idx in indices[0]:
        chunk = chunk_metadata[idx]
        chunk_embedding = np.array(embedding_model.embed_query(chunk["chunk_text"])).reshape(1, -1)
        similarity = np.dot(query_embedding, chunk_embedding.T).item()
        if similarity >= relevance_threshold:
            retrieved_chunks.append(chunk)
    return retrieved_chunks

# Initialize a Hugging Face model using LangChain integration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Choose a huggingface model
# hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # ~1.1GB model
# hf_model_name = "Qwen/Qwen2.5-Omni-7B"
# hf_model_name = "mistralai/Mistral-7B-v0.1"
hf_model_name = "google/gemma-7b"

# Choose a Bedrock model
bedrock_model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
bedrock_provider = "Anthropic"

# Create LangChain wrapper based on provider
if provider.lower() == "huggingface":
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
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        device_map=device
    )
    llm = HuggingFacePipeline(pipeline=text_pipeline)
elif provider.lower() == "bedrock":
    session = boto3.Session()
    bedrock = session.client(
        service_name='bedrock', #creates a Bedrock client
        region_name=aws_region,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    # Create LangChain wrapper based on provider
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id = aws_access_key_id,
        aws_secret_access_key = aws_secret_access_key
    )
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"

    model_kwargs =  {
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    claude_3_client = ChatBedrock(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    )
else:
    raise ValueError(f"Unsupported provider: {provider}")


# Stateful chat class with LangChain integration
class ConversationalBot:
    def __init__(self):
        self.chat_history = []
        
    def get_response(self, user_query):
        # Retrieve relevant chunks
        chunks = retrieve_top_k(user_query)
        context = " ".join([chunk["chunk_text"] for chunk in chunks])
        if provider.lower() == "huggingface":
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

        elif provider.lower() == "bedrock":
            # Format chat history for ChatBedrock
            messages = []

            # System message with context
            system_message = f"""You are a helpful assistant created by Motive for answering questions about the Company named Motive, based on the provided context.

Context information:
{context}

Answer only based on the provided context but dont say something like based on the context. If you don't know the answer based on the context, say so. Don't make up information."""

            messages.append(("system", system_message))

            # Add chat history
            for q, a in self.chat_history:
                messages.append(("human", q))
                messages.append(("assistant", a))

            # Add current question
            messages.append(("human", user_query))

            # Create prompt template
            prompt = ChatPromptTemplate.from_messages(messages)

            # Create chain and execute
            chain = prompt | claude_3_client | StrOutputParser()
            response = chain.invoke({})

            # Store conversation
            self.chat_history.append((user_query, response))
            if len(self.chat_history) > 10:  # Keep history manageable
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

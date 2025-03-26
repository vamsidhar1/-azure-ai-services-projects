from fastapi import FastAPI, Query
import fitz  # PyMuPDF
import faiss
import numpy as np
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter

app = FastAPI()

OPENAI_API_KEY = "your-api-key"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Function to split text into chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_text(text)

# Function to get OpenAI embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

# Function to create and save FAISS index
def create_faiss_index(chunks):
    embeddings = np.array([get_embedding(chunk) for chunk in chunks], dtype=np.float32)
    dimension = embeddings.shape[1]  # Get embedding dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, "faiss_index")
    return index

# Load or create FAISS index
try:
    index = faiss.read_index("faiss_index")
except:
    print("No FAISS index found, creating a new one...")
    text = extract_text_from_pdf("example.pdf")  # Load your PDF
    chunks = split_text(text)
    index = create_faiss_index(chunks)

# Function to search FAISS index
def search_faiss(query, k=3):
    query_embedding = np.array([get_embedding(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, k)
    return indices[0]  # Returns top-k matching chunk indices

# Function to query OpenAI GPT-4 with retrieved context
def ask_gpt4(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI assistant that answers questions based on given context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )
    return response["choices"][0]["message"]["content"]

# FastAPI endpoint for querying the chatbot
@app.post("/ask")
async def ask_chatbot(query: str = Query(..., description="User's question")):
    top_chunks = search_faiss(query)
    context = " ".join([chunks[i] for i in top_chunks])
    answer = ask_gpt4(query, context)
    return {"question": query, "answer": answer}

# Run the API with: uvicorn filename:app --host 0.0.0.0 --port 8000

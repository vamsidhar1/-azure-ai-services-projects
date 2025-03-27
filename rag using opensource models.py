import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from fastapi import FastAPI, UploadFile, File
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import os

# Initialize FastAPI
app = FastAPI()

# Load Open-Source Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Open-Source LLM (Mistral-7B)
model_name = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Global Variables
chunks = []
index = None

def extract_text_from_pdf(pdf_path):
    """ Extracts text from a PDF file. """
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def chunk_text(text):
    """ Splits extracted text into manageable chunks. """
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def embed_text(chunks):
    """ Generates embeddings for text chunks using SentenceTransformers. """
    return embedding_model.encode(chunks, convert_to_numpy=True)

def build_faiss_index(embeddings):
    """ Creates a FAISS index for efficient retrieval. """
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_faiss(query, top_k=3):
    """ Searches FAISS for the most relevant chunks. """
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def ask_mistral(question, context):
    """ Uses Mistral-7B to generate an answer based on retrieved context. """
    prompt = f"Context: {context}\n\nQuestion: {question}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {key: val.to(llm_model.device) for key, val in inputs.items()}
    output = llm_model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """ API to upload a PDF, extract text, and build FAISS index. """
    global chunks, index

    # Save uploaded file
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Process PDF
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    embeddings = embed_text(chunks)
    index = build_faiss_index(embeddings)

    os.remove(file_path)  # Cleanup temp file
    return {"message": "PDF processed and indexed successfully!", "total_chunks": len(chunks)}

@app.get("/ask/")
async def ask(query: str):
    """ API to get an answer based on a user query. """
    if not chunks or index is None:
        return {"error": "No document has been uploaded yet!"}

    top_chunks = search_faiss(query)
    context = " ".join(top_chunks)
    answer = ask_mistral(query, context)

    return {"query": query, "answer": answer}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

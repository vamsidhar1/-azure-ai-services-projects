import os
import openai
import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import Vector
from azure.ai.formrecognizer import DocumentAnalysisClient
from fastapi import FastAPI, Request

# Load environment variables
AZURE_OPENAI_ENDPOINT = "https://YOUR_OPENAI_ENDPOINT.openai.azure.com"
AZURE_OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
AZURE_OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"
AZURE_OPENAI_CHAT_MODEL = "gpt-4-turbo"

AZURE_SEARCH_ENDPOINT = "https://YOUR_SEARCH_SERVICE.search.windows.net"
AZURE_SEARCH_API_KEY = "YOUR_SEARCH_API_KEY"
AZURE_SEARCH_INDEX_NAME = "documents-index"

AZURE_DOC_INTELLIGENCE_ENDPOINT = "https://YOUR_DOC_INTELLIGENCE.cognitiveservices.azure.com"
AZURE_DOC_INTELLIGENCE_API_KEY = "YOUR_DOC_INTELLIGENCE_API_KEY"

# Initialize Clients
search_client = SearchClient(endpoint=AZURE_SEARCH_ENDPOINT, index_name=AZURE_SEARCH_INDEX_NAME, credential=AzureKeyCredential(AZURE_SEARCH_API_KEY))
doc_intelligence_client = DocumentAnalysisClient(endpoint=AZURE_DOC_INTELLIGENCE_ENDPOINT, credential=AzureKeyCredential(AZURE_DOC_INTELLIGENCE_API_KEY))

app = FastAPI()

def get_embedding(text):
    """Generate vector embedding for given text using Azure OpenAI."""
    response = openai.Embedding.create(
        input=text,
        engine=AZURE_OPENAI_EMBEDDING_MODEL,
        api_key=AZURE_OPENAI_API_KEY
    )
    return response["data"][0]["embedding"]

def search_documents(query):
    """Perform a vector search in Azure AI Search."""
    query_vector = get_embedding(query)
    
    results = search_client.search(
        search_text="",  # Empty because we use vector search
        vectors=[Vector(value=query_vector, k=5, fields="vectorEmbedding")],
        select=["content", "metadata"]
    )

    return [doc["content"] for doc in results]

def process_document(file_path):
    """Extract text from a document using Azure Document Intelligence."""
    with open(file_path, "rb") as document:
        poller = doc_intelligence_client.begin_analyze_document("prebuilt-layout", document)
        result = poller.result()
    
    extracted_text = " ".join([line.content for page in result.pages for line in page.lines])
    return extracted_text

def generate_response(query):
    """Generate a response using Azure OpenAI GPT-4 and retrieved documents."""
    relevant_docs = search_documents(query)
    context = "\n".join(relevant_docs)

    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = openai.ChatCompletion.create(
        model=AZURE_OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": "You are an AI assistant."},
                  {"role": "user", "content": prompt}],
        api_key=AZURE_OPENAI_API_KEY
    )

    return response["choices"][0]["message"]["content"]

@app.post("/chat")
async def chat(request: Request):
    """API Endpoint for querying the RAG bot."""
    data = await request.json()
    query = data.get("query")
    response = generate_response(query)
    return {"response": response}

@app.post("/upload")
async def upload_file(file: bytes):
    """API Endpoint for document upload and processing."""
    with open("temp.pdf", "wb") as f:
        f.write(file)

    extracted_text = process_document("temp.pdf")
    
    # Convert to embedding and store in Azure AI Search
    embedding_vector = get_embedding(extracted_text)
    document_data = {
        "content": extracted_text,
        "vectorEmbedding": embedding_vector
    }
    search_client.upload_documents(documents=[document_data])

    return {"message": "Document processed and stored successfully"}

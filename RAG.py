import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
import faiss
import numpy as np
import openai

# OpenAI API key (replace with your actual key)
openai.api_key = 'your-openai-api-key'

# Step 1: Load PDF and Extract Text
def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

# Step 2: Chunk the Document
nltk.download('punkt')
nltk.download('punkt_tab')
def chunk_document(text):
    sentences = sent_tokenize(text)
    return sentences

# Step 3: Create Embeddings using OpenAI's text-embedding-ada-002
def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",  # Use OpenAI's ADA embedding model
            input=chunk
        )
        embeddings.append(response['data'][0]['embedding'])
    return embeddings

# Step 4: Create FAISS Index
def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

# Step 5: User Query and Embedding Generation using OpenAI
def query_to_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",  # Use OpenAI's ADA embedding model
        input=query
    )
    query_embedding = response['data'][0]['embedding']
    return np.array(query_embedding).astype('float32')

# Step 6: Search Similar Embedding
def search_similar_embeddings(query_embedding, index, k=3):
    distances, indices = index.search(query_embedding, k)
    return distances, indices

# Step 7: Retrieve Relevant Chunks
def retrieve_chunks(indices, chunks):
    return [chunks[i] for i in indices[0]]

# Step 8: Generate LLM Response
def generate_llm_response(prompt):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=200,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

# Example Usage
pdf_text = load_pdf("sample_document.pdf")
chunks = chunk_document(pdf_text)
embeddings = create_embeddings(chunks)
index = create_faiss_index(embeddings)

user_query = "What is the impact of climate change on agriculture?"
query_embedding = query_to_embedding(user_query)

distances, indices = search_similar_embeddings(query_embedding, index, k=3)
relevant_chunks = retrieve_chunks(indices, chunks)

# Format the prompt for the LLM
prompt = f"User Query: {user_query}\n\nRelevant Information:\n"
for idx, chunk in enumerate(relevant_chunks, 1):
    prompt += f"Chunk {idx}: {chunk}\n"

# Get LLM response
llm_response = generate_llm_response(prompt)
print("Final Response from LLM:", llm_response)

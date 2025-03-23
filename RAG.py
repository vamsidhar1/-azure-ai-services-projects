import fitz  # PyMuPDF
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
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
def chunk_document(text):
    sentences = sent_tokenize(text)
    return sentences

# Step 3: Create Embeddings
model = SentenceTransformer('all-mpnet-base-v2')
def create_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings

# Step 4: Create FAISS Index
def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

# Step 5: User Query and Embedding Generation
def query_to_embedding(query):
    query_embedding = model.encode([query])
    return query_embedding

# Step 6: Search Similar Embedding
def search_similar_embeddings(query_embedding, index, k=3):
    distances, indices = index.search(query_embedding.astype('float32'), k)
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

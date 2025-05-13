import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import openai
from utils import load_documents, split_into_chunks

def generate_embeddings(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    return model.encode(chunks)

def save_vector_store(embeddings, path='embeddings/index.faiss'):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, path)

def load_vector_store(path='embeddings/index.faiss'):
    return faiss.read_index(path)

def basic_faiss_search(query, chunks, index_path='embeddings/index.faiss', model_name='all-MiniLM-L6-v2', k=10):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])
    index = load_vector_store(index_path)
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def post_retrieval_filter(query, retrieved_chunks, top_n=5):
    keywords = set(query.lower().split())
    scored = []
    for chunk in retrieved_chunks:
        words = set(chunk.lower().split())
        overlap = len(keywords.intersection(words))
        scored.append((overlap, chunk))
    scored.sort(reverse=True)
    return [chunk for _, chunk in scored[:top_n]]

def hierarchical_search(query, chunks, index_path='embeddings/index.faiss', model_name='all-MiniLM-L6-v2', k=10, top_n=5):
    # Stage 1: Coarse retrieval
    coarse_results = basic_faiss_search(query, chunks, index_path, model_name, k)

    # Stage 2: Fine-grained re-ranking
    model = SentenceTransformer(model_name)
    query_emb = model.encode(query, convert_to_tensor=True)
    chunk_embs = model.encode(coarse_results, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_emb, chunk_embs)[0]

    scored_chunks = sorted(zip(cosine_scores, coarse_results), key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_n]]

def generate_answer(prompt):
    client = openai.OpenAI(
        api_key="ngu-CQo6jVAeHt", 
        base_url="https://ngullama.femtoid.com/v1"
    )
    response = client.chat.completions.create(
        model="qwen2.5-coder:7b",  
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

def rewrite_query(original_query):
    client = openai.OpenAI(
        api_key="ngu-CQo6jVAeHt",  
        base_url="https://ngullama.femtoid.com/v1"
    )
    response = client.chat.completions.create(
        model="qwen2.5-coder:7b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that rewrites user queries to improve document search relevance."},
            {"role": "user", "content": f"Rewrite the following query to make it more specific and effective for document retrieval:\n\n{original_query}"}
        ],
        max_tokens=100
    )
    return response.choices[0].message.content.strip()

# Retrieval-Augmented Generation (RAG) System

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using open-source tools with local vector storage. The system enhances LLM capabilities by retrieving relevant context from documents before generating responses.

---

## Learning Objectives
- Build a complete RAG pipeline with LangChain-style architecture  
- Load and process documents from PDF, DOCX, and TXT formats  
- Generate embeddings using Sentence Transformers  
- Store and retrieve vectors using FAISS  
- Apply prompt engineering to guide generation  
- Evaluate the system using standard IR metrics  
- Implement advanced RAG techniques: Query Rewriting, Post-Retrieval Filtering, and Hierarchical Retrieval

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <repo-name>
```
### 2. Install Dependencies
Make sure Python 3.10+ is installed.
```
pip install -r requirements.txt
```
### 3. Run the Pipeline
```
python main.py
```
## System Components
### Document Loading & Chunking
- Supports .pdf, .docx, .txt
- Chunked with overlap to preserve context

## Embedding Generation
- Uses all-MiniLM-L6-v2 from sentence-transformers
- Stored using FAISS for similarity search

## Retrieval Strategies
- Basic Similarity Search: FAISS k-NN search
- Post-Retrieval Filtering: Keyword-based overlap
- Hierarchical Retrieval: Coarse + fine-grained reranking using cosine similarity

## Prompting & Generation
- Custom prompt template with retrieved context
= Answer generation using OpenAI-compatible LLM (qwen2.5-coder:7b)

## Evaluation
- Measures precision, recall, and F1 score
- Sample queries are in test_dataset/queries.json

## Example Output
```
Test Case 1
Original Query: Tell me about AI.
Rewritten Query: What aspects of Artificial Intelligence would you like to know about?
Retrieved Chunks: [...]
Generated Answer: ...
Precision: 0.50 | Recall: 0.50 | F1 Score: 0.50
```

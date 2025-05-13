from rag_pipeline import rewrite_query, generate_answer, generate_embeddings, save_vector_store, hierarchical_search
from utils import load_documents, split_into_chunks
from prompt_template import create_prompt
from evaluation import precision_recall_f1
import json
import os

# Load and process documents
print("Loading and processing documents...")
docs = load_documents('documents')
chunks = []
for doc in docs:
    chunks.extend(split_into_chunks(doc))
print(f"Documents loaded. Total Chunks: {len(chunks)}")

# Generate and save embeddings
print("Generating and saving embeddings...")
embeddings = generate_embeddings(chunks)
save_vector_store(embeddings)
print("Embeddings saved successfully.\n")

# Load test queries and expected answers
if os.path.exists('test_dataset/queries.json'):
    with open('test_dataset/queries.json', 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
else:
    print("No test_dataset/queries.json found. Using default test cases.")
    test_cases = [
        {
            "query": "Tell me about AI.",
            "expected_answers": ["Artificial Intelligence", "smart machines"]
        },
        {
            "query": "What is machine learning?",
            "expected_answers": ["Machine learning", "learn from data"]
        }
    ]

total_precision, total_recall, total_f1 = 0, 0, 0

print("Starting Evaluation...\n")

# Run through each test case
for idx, case in enumerate(test_cases, start=1):
    original_query = case["query"]
    expected_answers = case["expected_answers"]

    print(f"\nTest Case {idx}")
    print(f"Original Query: {original_query}")

    # Query Rewriting
    rewritten_query = rewrite_query(original_query)
    print(f"Rewritten Query: {rewritten_query}")

    # Retrieval
    retrieved_chunks = hierarchical_search(rewritten_query, chunks)
    print(f"Retrieved Chunks: {retrieved_chunks}")

    # LLM Answer Generation
    context = "\n".join(retrieved_chunks)
    prompt = create_prompt(context, original_query)
    answer = generate_answer(prompt)
    print(f"\nGenerated Answer:\n{answer}")

    # Evaluation
    precision, recall, f1 = precision_recall_f1(expected_answers, retrieved_chunks)
    total_precision += precision
    total_recall += recall
    total_f1 += f1

    print(f"\nEvaluation Metrics for Query:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")
    print("-" * 60)

# Final Average Metrics
n = len(test_cases)
if n > 0:
    avg_precision = total_precision / n
    avg_recall = total_recall / n
    avg_f1 = total_f1 / n
else:
    avg_precision = avg_recall = avg_f1 = 0

print("\nFINAL AVERAGE EVALUATION METRICS:")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall:    {avg_recall:.2f}")
print(f"Average F1 Score:  {avg_f1:.2f}")

def create_prompt(context, question):
    return f"""You are an expert AI assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}
Answer:"""
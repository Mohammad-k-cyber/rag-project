"""
Simple RAG Pipeline - Local Implementation
"""
from ollama import Client
from langchain.text_splitter import CharacterTextSplitter
import json

client = Client(host='http://localhost:11434')

# Sample knowledge base
documents = [
    "Retrieval Augmented Generation combines retrieval and generation models.",
    "Vector databases store numerical representations of documents.",
    "Embeddings convert text to numerical vectors for similarity search.",
    "RAG improves accuracy by retrieving relevant context before generating.",
    "LLMs are large language models trained on massive text corpora.",
]

print("=== RAG PIPELINE DEMO ===\n")

# Step 1: Prepare documents
print("Step 1: Documents loaded")
for i, doc in enumerate(documents, 1):
    print(f"  {i}. {doc[:60]}...")

# Step 2: Simple retrieval (keyword matching for now)


def retrieve_relevant_docs(query, docs, top_k=2):
    """Simple keyword-based retrieval"""
    scores = []
    query_words = query.lower().split()

    for doc in docs:
        score = sum(1 for word in query_words if word in doc.lower())
        scores.append((doc, score))

    # Sort by score and return top-k
    ranked = sorted(scores, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked[:top_k] if score > 0]


# Step 3: Test RAG
queries = [
    "What is RAG?",
    "Tell me about embeddings",
    "How do large language models work?",
]

for query in queries:
    print(f"\n📝 Query: {query}")

    # Retrieve
    retrieved = retrieve_relevant_docs(query, documents)
    print(f"  Retrieved {len(retrieved)} documents:")
    for doc in retrieved:
        print(f"    - {doc[:50]}...")

    # Generate with context
    context = "\n".join(retrieved)
    prompt = f"""Context: {context}

Question: {query}
Answer:"""

    response = client.generate(model='mistral', prompt=prompt, stream=False)
    answer = response['response'][:150]
    print(f"  Answer: {answer}...")

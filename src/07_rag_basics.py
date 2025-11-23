from langchain_ibm import ChatWatsonx
from langchain_ibm import WatsonxEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate

# Initialize model and embeddings
model = ChatWatsonx(
    model_id="ibm/granite-3-8b-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params={"max_new_tokens": 256}
)

embeddings = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network"
)

# Sample documents (your knowledge base)
documents = [
    "RAG stands for Retrieval Augmented Generation",
    "RAG retrieves relevant documents before generating responses",
    "Vector databases store document embeddings for fast retrieval",
    "Embeddings convert text to numerical representations"
]

print("=== RAG PIPELINE ===\n")

# Step 1: Split documents
print("Step 1: Splitting documents...")
splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = splitter.split_text(" ".join(documents))
print(f"Created {len(chunks)} chunks\n")

# Step 2: Create vector store
print("Step 2: Creating vector store...")
vector_store = Chroma.from_texts(chunks, embeddings)
print("Vector store created\n")

# Step 3: Retrieve relevant documents
print("Step 3: Retrieving relevant documents...")
query = "What is RAG?"
retrieved = vector_store.similarity_search(query, k=2)
print(f"Retrieved {len(retrieved)} documents:")
for doc in retrieved:
    print(f"  - {doc.page_content}")
print()

# Step 4: Generate answer with context
print("Step 4: Generating answer with context...")
context = "\n".join([doc.page_content for doc in retrieved])
template = """
Context: {context}

Question: {question}

Answer based on the context:
"""
prompt = PromptTemplate.from_template(template)
formatted = prompt.format(context=context, question=query)
response = model.invoke(formatted)
print(f"Answer: {response.content}")
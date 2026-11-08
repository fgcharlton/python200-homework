from dotenv import load_dotenv
import os
import string
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.vector_stores.postgres import PGVectorStore

# Part 2: Mini-Project: Groundwork Coffee Co. Q&A Assistant
"""
Groundwork Coffee Co. is a small, community-focused coffee shop. They have put together a set of documents 
about their menu, story, hours, loyalty program, and catering services. Your job is to build a RAG-powered 
assistant that can answer questions about any of those documents accurately.
"""

# Step 1: Setup
# Load API key from .env
if load_dotenv():
    print("Successfully loaded api key")
else:
    print("No api key loaded check out .env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add an assert statement to verify that groundwork_docs/ exists
docs_dir = Path("resources/groundwork_docs")

assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

# Step 2: Load the Documents
# Load all documents from groundwork_docs/ using SimpleDirectoryReader
docs = SimpleDirectoryReader(str(docs_dir)).load_data()

# Print how many documents were loaded 
print(f"Loaded {len(docs)} documents from {docs_dir}")

# Print file name of each document
for doc in docs:
    print("Document written: ", doc.metadata["file_name"])

# Step 3: Build the Index and Query Engine
# Set the embedding model
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Chat model: Use gpt-4o-mini
Settings.llm = OpenAI(model="gpt-4o-mini")

# Build a VectorStoreIndex from the loaded documents
index = VectorStoreIndex.from_documents(docs)

# Create a query engine with similarity_top_k = 3
query_engine = index.as_query_engine(similarity_top_k=3)

# Confirm index is ready
print("Index built successfully. Ready to answer questions.")

# Step 4: Query the Assistant
# Run 5 queries below through your query engine. Print the question, answer from the model, and top retrieved node
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Document Name:", node_with_score.node.metadata["file_name"])
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# Did the assistant sound confident and accurate?
"""
Yes, the assistant sounded confident and accurate when answering each of the questions. 
It gave confident responses that fully answered each question.
"""

# Were there any answers that surprised me?
"""
I think the depth to which each question was answered surprised me. For example, with 
the "Do you offer any dairy-free milk options?" question, the model answered that fully
but also shared that there was no extra charge for dairy-free milk options. So, even 
though that wasn't asked directly, the model gave additional helpful context. 
"""
# Step 5: Find a Failure
query_engine = index.as_query_engine(similarity_top_k = 5)

questions = [
    "What restaurants are near the West Asheville location?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Document Name:", node_with_score.node.metadata["file_name"])
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# What you asked and why you expected it to be hard
"""
I asked "What restaurants are near the West Asheville location?". I expected this question to be hard
because there was no answer to it. The model knows the address of the location, but not the restaurants 
around it. 
"""

# What went wrong — wrong retrieval, missing information, the model guessed anyway?
"""
The model was unable to answer the question. However, it did not try to guess the answer. It simply
said the user should consult a local directory or map service to find any restaurants near the West
Asheville location. 
"""

# When the retrieval failed, did the model's tone change — did it become less certain, 
# or did it still sound confident even when it was wrong? What does this suggest about 
# trusting AI-generated responses?
"""
The model seemed slightly less confident, particularly around where to send the user to find the answer
that they were looking for. It said that "it may be best to consult a local directory or map service" to 
find the answer, which seems less confident and unsure than the answers above. However, it did not try to
guess restaurants that were around the location. 

This does show why it is important to not trust AI-generated responses outright. It is important to validate
all responses prior to trusting / using AI-generated responses. 
"""

# What you would change about the system to improve it?
"""
I think it would be helpful to give the system a personality since the model is primarily focused on a coffee shop.
If customers were to interact with it directly, it would be great to have it have a personality that aligns with the 
story of the coffee shop.
"""

# Step 6: Reflection
# The LlamaIndex lesson lists the steps that LlamaIndex automates: chunking, embedding, and indexing. How many 
# lines of code did your LlamaIndex implementation take in your project? Compare that with the keyword RAG code 
# from the Keyword-based RAG lesson, or with the code you would need to write each of those steps yourself. What 
# does that tell you about the value of using a framework?
"""
For the set of 5 questions, there were around 30 lines of code using LlamaIndex, which is significantly less than the keyword
RAG code. LlamaIndex is powerful because of how much work is wrapped into so little code. It is a fully maintained framework 
that bundles chunking, embedding, vector storage, retrieval, ranking, and context injection into a clean and consistent interface.
This is extremely efficient especially as it pertains to writing code and debugging if needed. 
"""

# You have now built a system that answers questions from real documents. Describe a different use case — not a 
# coffee shop — where this approach would add genuine value to a business or organization.
"""
This approach could be especially useful in a multitude of businesses or organizations. For example, I could see this be extremely
helpful for a grocery store chain. Customers often use an app or webpage to order groceries, so having a chat bot that is able to quickly pull
from a multitude of documents could be extremely helpful. It could also aid in the customer's experience, so they aren't spending 
additional time searching the app / webpage for the information that they need. 
"""

# What is one failure mode that RAG cannot fully prevent, even when retrieval is working correctly?
"""
RAGs cannot prevent incorrect source document data. If the information is incorrect in the documents that a RAG is pulling from, then the RAG
cannot prevent that. It is important to ensure that all documentation is up-to-date and correct before allowing the RAG to pull it. If documentation
is not correct, then the model will feed incorrect information to the user, which could lead to some big consequences if the user doesn't notice that 
it is incorrect. 
"""
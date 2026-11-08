from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# --- Concepts ---

# Q1
"""
Three teams at a software company are each building a different AI project. Add a comment block to 
your code that identifies the best approach — prompt engineering, fine-tuning, or RAG — for each 
scenario, and gives a 1-2 sentence explanation of your reasoning.

Scenario A: A legal team wants an assistant that can answer questions about their internal policy 
library — hundreds of PDFs that are updated every quarter.
RAG would be the best approach because it would retrieve the relevant information from the documents
and inject the information into the LLM prompt. There are large amounts of information the model would
need to access, making RAG an effective approach. 

Scenario B: A startup wants their model to write product copy in a very specific brand voice — a 
dry, minimalist style that does not appear much online. They have 3,000 examples their in-house 
writers produced over the years.
Fine-tuning would be the best approach for several reasons, most notably due to wanting the model to 
adopt a very specific brand voice as well as having many examples to lean on. Fine-tuning is the best 
approach when you want the model to adopt a particular style, internalize complex patterns, and 
when you have many examples. 

Scenario C: A data analyst needs to ask an LLM questions about a single two-page report she just 
received. She does not need this to work for any other document.
Since the analyst is only asking the LLM questions about a single two-page report, the best approach
would be prompt engineering. The information is short and stable, making prompt engineering an effective approach. 
"""

# Q2
"""
AI hallucinations (responses that sound confident but are wrong) can be particularly difficult to detect. 
Add a comment to your code answering this:

Why is a confidently wrong answer more harmful than one that says "I am not sure"? Give one example of a real 
situation where a confident hallucination could cause harm.
Confidently wrong answers could be difficult for the user to idenify, leading to them use that information. 
AI hallucinations appear plausible, but are incorrect or nonsensical. I have used AI in the past to gather information
about a specific subject matter, but noticed some hallucinations from the AI (such as incorrect information and citations). 
While no serious harm was caused, if I hadn't noticed, this could have caused me to use this incorrect information in my
job which would lead to misinformation around public health topics.

Think about the tone of the response as well as its content — why does the way the model expresses an answer 
affect how much we trust it?
A confident answer may give the user security that the answer is correct, leading them to use it. When you ask a human 
a question, typically, the more sure they seem when answering allows you to also believe they correct. The same logic
can be applied to AI. 
"""

# Q3
"""
The steps below make up a complete RAG pipeline, but they are out of order. Copy the list into your code as a comment, 
arrange them in the correct order, and add a one-sentence description of what happens at each step.

Incorrect Order
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]

Correct Order
steps = [
    "Extract text from source documents",       Pull the raw text out of source document (PDFs, webpages, etc.)
    "Split text into chunks",                   Break raw text into smaller chunks that can be embedded and fit into a prompt
    "Convert text chunks into embeddings",      Turn each chunk into a vector that captures its meaning, so chunks can be numerically compared
    "Receive the user's query",                 Get user's question
    "Embed the user's query",                   Convert user's question into a vector using the same model, so it's in the same space as the chunks
    "Retrieve the most relevant chunks",        Compare query vector against chunk vectors, pull the closest matches
    "Inject retrieved chunks into the prompt",  Put chunks into prompt so the model can read it
    "Generate a response from the LLM",         Have the model answer the user's query from the retrieved context 
]
"""

# --- Keyword RAG ---
import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# Q1: Run simple_keyword_retrieval and verbose=True
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# Explain which document was selected and why.
"""
loyalty.txt was selected. Three of the documents tied with an overlap
of 1: hours.txt matched on "weekends", then hiring.txt and loyalty.txt 
matched on "your". "Your" is not in the stopword list, so it survived filtering.
Since there was a tie, Python reviews in alphabetically descending order, so 
loyalty.txt wins the ties. So, the retrieval is correct as the code is written,
but a poor answer to the question.
"""

# Q2: Run the same function iwth this second query using the same documents.
query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# Which document was selected? Whether keyword RAG got this right - and why or why not? What kind of retrieval would do better here?
"""
No overlap keywords were found, so no document was selected. The RAG did get this right, because there was no keywords to match
together. However, it did not get it right in terms of actually answering the question. A semantic RAG would be better, which
would search by meaning rather than exact words. 
"""

# Q3: 

# Predict which document will be selected for the query below
"""
I predict that, again, the RAG will be unable to select a document. Based on my context, it would best be
addressed by the loyalty.txt file. However, since the words differ between the question and text in the 
loyalty.txt program then no document would be selected. 
"""
query = "How do I sign up for rewards?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print(results[0][0])

# Was your prediction correct? If the result surprised you, add a comment explaining what happened.
"""
Yes, my prediction was correct. Since the question is worded differently than the loyalty.txt file,
the question wasn't able to be addressed. A semantic RAG would be needed for the model to understand
the context versus just matching words. 
"""

# --- Semantic RAG Concepts --- 

# Q1
"""
What is a vector embedding? (1-2 sentences)
Embeddings are dense numerical vectors that represent the meaning of the text. This allows chunks with 
similar meanings to lang near each other in the vector space, allowing for the semantic search to
search by meaning rather than exact word match.

Two text chunks have cosine similarity scores of 0.85 and 0.30 with a given query. Which chunk is 
more relevant, and what does that number tell you about the relationship between the texts?
The chunk with a score of 0.85 is more relevant because it is closer to 1. Cosine similarity helps put
a numerical value to the closeness between two vectors, and the closer to 1 indicates a highly similar
meaning. 

Why can semantic search find a relevant chunk even when none of the exact words from the query 
appear in the chunk?
Semantic RAGs search by meaning rather than exact words, allowing it to find relevant chunks even if
there are no exact words from the query in the chunks.
"""

# Q2 
"""
Keyword RAG and semantic RAG handle the same problem differently. Copy this table into your code as 
a comment and fill in the right column:
| Feature                    | Keyword RAG                       | Semantic RAG |
|----------------------------|-----------------------------------|--------------|
| What is compared?          | Exact word overlap                | Vector Embeddings            |
| What is retrieved?         | Full document                     | Chunks                       |
| Can it handle synonyms?    | No                                | Yes                          |
| Storage format             | Plain text dictionary             | VectorStoreIndex             |
| Relevance score            | Number of overlapping keywords    | Cosine Similarity            |
"""

# --- LlamaIndex ---
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Load documents directly from PDFs in the folder
docs = SimpleDirectoryReader("brightleaf_pdfs").load_data()

# Embedding model: Use text-embedding-3-small
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Chat model: Use gpt-4o-mini
Settings.llm = OpenAI(model="gpt-4o-mini")

# Q1
# Build a vector index automatically (handles chunking and embeddings)
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k = 3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

"""
Do the retrieved chunks look relevant to the question?
Yes, all retrieved chunks do seem relevant to each question. 

Does the model's response sound confident and specific, or does it hedge with phrases like "based on 
the context" or "I'm not sure"? Note what you observe about the tone.
The model seems both confident and specific when answering the questions. For both questions, the model
seems to speak matter-of-factly about the answer. 

Did anything unexpected get retrieved?
No, it doesn't appear that anything unexpected was retrieved. It retrieves the answer, the most relevant chunks,
node ID, similarity scores, and the text snippet. 
"""

# Q2: Re-run one of the queries from Q1 twice

# Once with similarity_top_k = 1
query_engine = index.as_query_engine(similarity_top_k = 1)

questions = [
    "What employee benefits does BrightLeaf offer?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Once with similarity_top_k = 5
query_engine = index.as_query_engine(similarity_top_k = 5)

questions = [
    "What employee benefits does BrightLeaf offer?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# How did the response change (if at all)? 
"""
In the first response (similarity_top_k = 1), the response seemed the list the benefits almost as if they were listed in the PDF.
In the second response (similarity_top_k = 5), the response seemed to be more synthesized and flow versus just a list. 
"""

# Is more retrieved context always better?
"""
No, more retrieved context is not always better. You want the most relevant pieces, not necessarily all the pieces.
Having too much context can muddy the water, making the synthesized information unclear.
"""

# Q3
# Try a query that the pipeline may struggle with -- something vague, something that spans multiple documents, or something where 
# the information might not be in the documents at all. Print the response and all retrieved chunks.
query_engine = index.as_query_engine(similarity_top_k = 3)

questions = [
    "What is the salary for a Data Analyst position at BrightLeaf on average?",
]

for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Add a comment explaining what you expected, what actually happened, and what you would change about the system to handle 
# this kind of query better.

"""
I didn't expect the model to be able to answer this question. However, I thought that the model may try to make something up.
It didn't hallucinate, it simply stated that the provided information does not include any specific salary details for that position
at BrightLeaf. I think if I could change one thing, it may provide additional details that it does have access to like benefits.
The model could state "While I don't have information on salary, BrightLeaf offers competitive x, y, z benefits" or something similar. 
"""

# Q4 
# Import and instantiate a FaithfulnessEvaluator and a RelevancyEvaluator, both using gpt-4o-mini as the judge LLM. (If you 
# need a reminder of the import and setup, the lesson's "RAG Evaluation using LlamaIndex" section shows the same pattern.) 
# Run them on this query:

q = "What employee benefits does BrightLeaf offer?"

# Print both scores. Then run the evaluators again on a query you expect to produce a lower-quality response — for example, 
# a question about something that is clearly not in the Brightleaf documents.
# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluator
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

# Get reponse to related query
response = query_engine.query(q)

# Evaluate faithfullness and relevancy for related query
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation For Related Query: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result For Related Query: " + str(relevancy_result.score))

# Unrelated query 
q2 = "What is the salary for a Data Analyst position at BrightLeaf on average?"

# Get reponse to related query
response2 = query_engine.query(q2)

# Evaluate faithfullness and relevancy for related query
faithfulness_result = faithfulness_evaluator.evaluate_response(query=q2, response=response2)
print("Faithfulness Evaluation For Unrelated Query: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q2, response=response2)
print("Relevancy Result For Unrelated Query: " + str(relevancy_result.score))

# What does a faithfulness score of 1.0 mean? What would a score of 0.0 indicate?
"""
A faithfulness score of 1.0 indicates that a response was faithful to the retrieved contexts.
Essentially, the response did not contain any hallucinations or lying. A low score would mean that
the response was not faithful to the retrieved contexts, and that it may contain hallucinations 
and lying. 
"""

# What does a relevancy score measure, and how is it different from faithfulness?
"""
The relevancy score measures how relevant the retrieved contexts were to the query. So, was the 
response on-topic, or off-topic and rambiling. This is different from faithfulness because it
assess if the response actually answers the question queried (not if the response was faithful 
to the retrieved documents).
"""
# Did the scores change between your two queries? If so, why do you think that happened?
"""
Faithfulness did not change between queries, which makes sense. As we saw above in Q3, the model
did not try to make up any answers. It stuck to only what it had within the documents, and admitted
when it did not have the necessary documents to give an answer. This indicates high faithfulness.
The relevancy score did change. It was 1.0 for the question that the model had documents to answer, while
0.0 for the question that the model didn't have documents to answer. Therefore, the second question received
a relevancy score of 0.0 because the answer did not answer (and therefore wasn't relevant) to the question.
"""
# What is the "LLM-as-a-judge" approach, and why is it used for RAG evaluation instead of a simple accuracy metric?
"""
Evaluation of a model is carried out using the ""LLM-as-a-judge" approach, which simply means that the evaluation
metrics are computed using an external LLM (different from the RAG framework). We use this approach because we are
trying to assess accuracy in the natural language space rather than the numerical or categorical space (like we do for ML 
models).
"""
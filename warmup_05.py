# Part 1: Warmyp Exercises

# --- Completions API ---
# Q1: Set up the OpenAI client and make first chat completion call
from dotenv import load_dotenv
from openai import OpenAI

if load_dotenv():
    print("Successfully loaded api key")

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Chat Responses: ", response.choices[0].message.content)
print("Chat Model: ", response.model)
print("Chat Total Tokens: ", response.usage.total_tokens)

# Q2: Run the same prompt three times with three different temperature settings: 0, 0.7, and 1.5
prompt = "Suggest a creative name for a data engineering consultancy."

temperatures = [0, 0.7, 1.5]

def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role":"user", "content": prompt}],
        temperature=temperature,
    )
    return response.choices[0].message.content

for temperature in temperatures: 
    print(temperature, ":", get_completion(prompt, "gpt-4o-mini", temperature))

# What do you notice about how the outputs differ?
# As the temperature increases, I see more ideas for a data engineering consultancy name. As you get up to 1.5, there are multiple name suggestions for the data engineering consultancy. 

# Which temperature would you use if you need a consistent, reproducible output?
# A temperature of 0 is more deterministic, so you would receive more consistent, reproducible responses. 

# Q3: Use n=3 with temperature=1 to get three different completions in a single API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

# Print all three
print("Chat Response 1: ", response.choices[0].message.content)
print("Chat Response 2: ", response.choices[1].message.content)
print("Chat Response 3: ", response.choices[2].message.content)

# Q4: Set max_tokens=15 and send a prompt that would normally produce a long response
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

# Print result
print("Chat Response Max Tokens 15: ", response.choices[0].message.content)

# What happened, and why might you want to use max_tokems in a real application?
# The chat cut the response off as it was answering. The response ended mid-sentence and did not give much information about neural networks.
# Max_tokens is a useful way to keep costs under control.

# --- System Messages and Personas ---
# Q1: Use a system message to give the model a personality, then ask it a question.
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(model='gpt-4o-mini',
                                   messages=messages)

print("Patient, Encouraging Python Tutor\n", response.choices[0].message.content)

# Change the personality, but ask the model the same prompt
messages = [
    {"role": "system", "content": "You are a impatient, discouraging Python tutor. You always explain things complexly and end with a confusing statement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(model='gpt-4o-mini',
                                   messages=messages)

print("Impatient, Discouraging Python Tutor\n", response.choices[0].message.content)

# What changed between tutors?
# The first tutor (patient, encouraging) explained list comprehension in a clear, simple, and encouraging way. 
# The second tutor (impatient, discouraging) explain list comprehensions in a complex, convoluted way. 

# Q2: Build the following conversation manually (no loop, no user input) and send it in a single API call
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(model='gpt-4o-mini',
                                   messages=messages)

# Print the model's response
print("Stateless - But Knows User's Name", response.choices[0].message.content)

# Why does the model know Jordan's name, even though it's stateless?
# Because the model has the chat's history despite it being stateless. There is no need to remember previous conversation because the entire conversation is included.

# --- Prompt Engineering --- 
# Q1: Zero-Shot
prompt = "Label each review as positive, negative, or mixed. Responde with only one word."

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

for i, review in enumerate(reviews, start=1): 
    full_prompt = f"{prompt}\n\nReview: {review}"
    label = get_completion(full_prompt, "gpt-4o-mini")
    print(f"Review {i}: {label}")

# Q2: One-Shot
example = """
Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed
"""

for i, review in enumerate(reviews, start=1): 
    full_prompt = f"{prompt}\n\nExample: {example}\n\nReview: {review}"
    label = get_completion(full_prompt, "gpt-4o-mini")
    print(f"Review {i}: {label}")

# Did adding one example change the format or consistency of the output compared to Q1?
# no adding one example did not change the consistency of the output compared to Q1. The model still labeled the reviews the same.

# Q3: Few-Shot
multi_example = """
Example 1:
Review: "Fast shipping but the item arrived damaged."
Sentiment: Mixed

Example 2: 
Review: "Fast shipping, great price, and excellent customer service."
Sentiment: Positive

Example 3:
Review: "Slow shipping, terrible price, and rude customer service."
Sentiment: Negative
"""

for i, review in enumerate(reviews, start=1): 
    full_prompt = f"{prompt}\n\nExample: {multi_example}\n\nReview: {review}"
    label = get_completion(full_prompt, "gpt-4o-mini")
    print(f"Review {i}: {label}")

# Compare all three approaches: When would you choose each one?
# All three approaches labeled each review the same (i.e., Positive, Negative, Mixed).
# It is helpful to use one-shot or few-shot when the task is nuanced or you need a specific output format.
# Zero-shot can be used when fast, less complex tasks are needed. Zero-shot is less reliable with complext tasks. 

# Q4: Chain of Thought 
prompt = """
Show your step-by-step reasoning. Before giving the final answer, explain your reasoning in 3 to 4 brief steps.

Then give the final answer on its own line labelled: Final answer: <value>

A data engineer earns $85,000 per year. She gets a 12 percent raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""

response = get_completion(prompt, temperature=0)
print(response)

# Why does asking the model to reason step by step tend to improve accuracy on problems like this?
# For tricky problems, asking the model to show its work step-by-step helps spot mistakes and makes results more reliable. 

# Q5: Structured Output 
import json

prompt = """
Analyze the sentiment of this customer review and respond only with valid JSON. Do not wrap the JSON in Markdown code fences or add any text before or after it.
Return keys: sentiment (positive/negative/mixed), confidence (0 to 1 scale), reason (one short sentence).

Review: I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited.
"""

# Print raw response
response = get_completion(prompt, temperature=0)
print("Raw response:", response)

# Print the JSON with a try/except block
try:
    result = json.loads(response)
    print("Parsed sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")

# Q6: Delimiters
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

# Print results
response = get_completion(prompt, temperature=0)
print("Delimiter Response:\n", response)

# Second prompt using a passage that is not a set of instructions
user_text2 = "I like pasta."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""

response = get_completion(prompt, temperature=0)
print("No Delimiter Response:\n", response)

# What problem do delimiters help prevent?
# When a prompt has multiple sections, the model can lose track of where one part ends and the next begins. 
# Delimiters clearly separate the sections so the model doesn't get confused. 

# --- Local Models with Ollama ---
# Q1: Run the following prompt using Ollama 

# Ollama Response
"""A large language model is an AI system designed to understand and generate human 
language, trained on vast amounts of text to learn patterns and context. It can 
perform tasks like translation, content creation, or answering questions, making 
it versatile for various applications."""

# Run same prompt in OpenAI
prompt = """
Explain what a large language model is in two sentences.
"""

response = get_completion(prompt, temperature=0)
print("OpenAI Response:\n", response)

# OpenAI Reponse
"""
A large language model is an artificial intelligence system designed to understand and 
generate human-like text by analyzing vast amounts of written data. It uses deep learning 
techniques, particularly neural networks, to predict and produce coherent and contextually 
relevant language based on the input it receives.
"""
# What differences did you notice between the two responses? 
# In my opinion, the Ollama response is a clear, simple explanation of AI. The OpenAI response seems a bit more complex, and 
# mentions specific concepts (e.g., neural networks) that may not be as readily known to all users. Overall, both versions 
# give a good idea of what AI is. 

# What is one advantage and one disadvantage of running a model locally?
# Running a local model, with tools like Ollama, can help avoid users paying a per-token API fees.
# However, the downloads for models can be slow sometimes.

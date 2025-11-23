from ollama import Client
import time

client = Client(host='http://localhost:11434')

# Use mistral model - perfect for your GPU
MODEL = 'mistral'


def measure_time(func):
    """Decorator to measure execution time"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        return result, elapsed
    return wrapper


@measure_time
def generate_response(prompt):
    response = client.generate(model=MODEL, prompt=prompt, stream=False)
    return response['response']


print("=== ZERO-SHOT PROMPTING ===")
answer, elapsed = generate_response("What is Retrieval Augmented Generation?")
print(f"Response: {answer}")
print(f"Time: {elapsed:.2f}s\n")

print("=== ONE-SHOT PROMPTING ===")
one_shot = """
Example: What is Machine Learning?
Answer: Machine Learning is a subset of AI that enables computers to learn from data.

Now answer: What is Deep Learning?
"""
answer, elapsed = generate_response(one_shot)
print(f"Response: {answer}")
print(f"Time: {elapsed:.2f}s\n")

print("=== FEW-SHOT PROMPTING ===")
few_shot = """
Examples:
Q: What is NLP? A: Natural Language Processing enables computers to understand human language.
Q: What is Computer Vision? A: Computer Vision enables machines to see and analyze images.

Q: What is Reinforcement Learning?
"""
answer, elapsed = generate_response(few_shot)
print(f"Response: {answer}")
print(f"Time: {elapsed:.2f}s\n")

print("=== CHAIN-OF-THOUGHT ===")
cot = """
A store has 20 apples. They sell 7 apples and receive 5 more.
How many apples are there now?

Think step by step:
"""
answer, elapsed = generate_response(cot)
print(f"Response: {answer}")
print(f"Time: {elapsed:.2f}s")

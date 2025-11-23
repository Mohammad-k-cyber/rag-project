from ollama import Client
import json
import re

client = Client(host='http://localhost:11434')
MODEL = 'mistral'

print("=== JSON OUTPUT PARSING ===")
template = """
Analyze this review and respond in JSON format:
{
  "sentiment": "positive/negative/neutral",
  "score": "1-10",
  "summary": "one sentence"
}

Review: "Great product, arrived quickly but packaging was poor"
JSON Response:
"""

response = client.generate(model=MODEL, prompt=template, stream=False)
response_text = response['response']

print(f"Raw Response: {response_text}\n")

# Try to extract JSON
try:
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        parsed = json.loads(json_str)
        print("Parsed JSON:")
        print(json.dumps(parsed, indent=2))
except json.JSONDecodeError:
    print("Could not parse JSON (normal for smaller models)")
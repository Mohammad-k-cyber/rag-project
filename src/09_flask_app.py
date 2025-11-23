from flask import Flask, request, jsonify
from ollama import Client
import time

app = Flask(__name__)
client = Client(host='http://localhost:11434')
MODEL = 'mistral'

@app.route('/ask', methods=['POST'])
def ask():
    """Answer a question"""
    data = request.json
    question = data.get('question', 'What is AI?')
    
    start = time.time()
    response = client.generate(
        model=MODEL,
        prompt=f"Answer briefly: {question}",
        stream=False
    )
    elapsed = time.time() - start
    
    return jsonify({
        "answer": response['response'],
        "time_seconds": round(elapsed, 2),
        "model": MODEL
    })

@app.route('/summarize', methods=['POST'])
def summarize():
    """Summarize text"""
    data = request.json
    text = data.get('text', '')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    response = client.generate(
        model=MODEL,
        prompt=f"Summarize in 2 sentences:\n{text}",
        stream=False
    )
    
    return jsonify({
        "summary": response['response'],
        "model": MODEL
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "model": MODEL,
        "host": "http://localhost:11434"
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        "api": "RAG Learning API",
        "endpoints": {
            "/ask (POST)": "Ask a question",
            "/summarize (POST)": "Summarize text",
            "/health (GET)": "Health check"
        },
        "example": {
            "curl": "curl -X POST http://localhost:5000/ask -H 'Content-Type: application/json' -d '{\"question\": \"What is RAG?\"}'",
            "python": "requests.post('http://localhost:5000/ask', json={'question': 'What is RAG?'})"
        }
    })

if __name__ == '__main__':
    print(f"🚀 Starting Flask app with {MODEL} model")
    app.run(debug=True, host='0.0.0.0', port=5000)
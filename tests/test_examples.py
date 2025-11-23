"""
Simple tests for RAG examples
"""
import pytest
from ollama import Client


def test_ollama_connection():
    """Test that Ollama is running"""
    client = Client(host='http://localhost:11434')
    response = client.generate(
        model='mistral',
        prompt='Hi',
        stream=False
    )
    assert 'response' in response


def test_zero_shot_prompting():
    """Test zero-shot prompting"""
    client = Client(host='http://localhost:11434')
    response = client.generate(
        model='mistral',
        prompt='What is AI?',
        stream=False
    )
    assert len(response['response']) > 0
    assert 'response' in response


def test_few_shot_prompting():
    """Test few-shot prompting"""
    client = Client(host='http://localhost:11434')
    prompt = """
    Examples:
    Q: What is ML? A: Machine Learning is AI.
    Q: What is DL? A: Deep Learning uses neural networks.
    Q: What is NLP?
    """
    response = client.generate(
        model='mistral',
        prompt=prompt,
        stream=False
    )
    assert len(response['response']) > 0

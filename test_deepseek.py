#!/usr/bin/env python3
"""
Test script to verify DeepSeek API key is working
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Check if API key is loaded
api_key = os.environ.get("DEEPSEEK_API_KEY")
if api_key:
    print("✓ API key loaded from .env file")
    print(f"  Key starts with: {api_key[:10]}...")
else:
    print("✗ No DEEPSEEK_API_KEY found in environment")
    print("  Add DEEPSEEK_API_KEY=your_key_here to your .env file")
    exit(1)

# Test 2: Try to import and initialize the DeepSeek client (via the OpenAI SDK)
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    print("✓ DeepSeek client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize DeepSeek client: {e}")
    exit(1)

# Test 3: Try a simple API call
try:
    response = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "user", "content": "Say 'Hello, API is working!' in one sentence."}
        ],
        max_tokens=20
    )
    print("✓ API call successful!")
    print(f"  Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ API call failed: {e}")
    exit(1)

print("\n✓ All tests passed! Your DeepSeek API key is configured correctly.")

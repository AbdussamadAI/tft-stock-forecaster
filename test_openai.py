#!/usr/bin/env python3
"""
Test script to verify OpenAI API key is working
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test 1: Check if API key is loaded
api_key = os.environ.get("OPENAI_API_KEY")
if api_key:
    print("✓ API key loaded from .env file")
    print(f"  Key starts with: {api_key[:10]}...")
else:
    print("✗ No API key found in environment")
    exit(1)

# Test 2: Try to import and initialize OpenAI client
try:
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize OpenAI client: {e}")
    exit(1)

# Test 3: Try a simple API call
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
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

print("\n✓ All tests passed! Your OpenAI API key is configured correctly.")

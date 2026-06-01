"""Quick test runner for ai_chat.

Usage:
  # Ensure GROQ_API_KEY is set in env (or in .env)
  python tools/test_ai_chat.py

This script intentionally does not print secrets; it only verifies that `ai_chat` executes.
"""
import os
import sys

from core.ai import ai_chat

if __name__ == '__main__':
    if not os.getenv('GROQ_API_KEY'):
        print('GROQ_API_KEY not set. Set it in your env or .env and rerun.')
        sys.exit(1)

    history = [{"role": "user", "content": "Say hi and your name."}]
    try:
        resp = ai_chat(history, memory="")
        print('ai_chat response:')
        print(resp)
    except Exception as e:
        print('ai_chat raised an exception:')
        raise

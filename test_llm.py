"""
Run this standalone to test your LLM connection BEFORE running the game.
  python test_llm.py
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ── Test Groq ─────────────────────────────────────────────────────────────────
print("=" * 50)
print("Testing Groq...")
groq_key = os.getenv("GROQ_API_KEY")
print(f"  Key found: {'YES - ' + groq_key[:12] + '...' if groq_key else 'NO - add GROQ_API_KEY to .env'}")

if groq_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Reply with exactly: CANDIDATES: UP, DOWN, LEFT\nREASONING: test"}],
            max_tokens=50,
        )
        print(f"  Response: {response.choices[0].message.content.strip()}")
        print("  Groq: OK")
    except ImportError:
        print("  groq not installed — run: pip install groq")
    except Exception as e:
        print(f"  Groq ERROR: {e}")

# ── Test Gemini ───────────────────────────────────────────────────────────────
print()
print("Testing Gemini (new SDK)...")
gem_key = os.getenv("GEMINI_API_KEY")
print(f"  Key found: {'YES - ' + gem_key[:12] + '...' if gem_key else 'NO - add GEMINI_API_KEY to .env'}")

if gem_key:
    try:
        from google import genai
        client = genai.Client(api_key=gem_key)
        # List available models first
        models = [m.name for m in client.models.list()]
        flash  = [m for m in models if "flash" in m.lower()]
        print(f"  Available flash models: {flash[:5]}")
        if flash:
            response = client.models.generate_content(
                model=flash[0],
                contents="Reply with exactly: CANDIDATES: UP, DOWN, LEFT\nREASONING: test"
            )
            print(f"  Response: {response.text.strip()[:100]}")
            print(f"  Gemini OK using {flash[0]}")
        else:
            print("  No flash models available for your key")
    except ImportError:
        print("  google-genai not installed — run: pip install google-genai")
    except Exception as e:
        print(f"  Gemini ERROR: {e}")

print()
print("=" * 50)

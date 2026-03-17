"""
minimax_client.py - Direct Cloud API client for MiniMax models.

MiniMax provides an OpenAI-compatible Chat Completions API at
https://api.minimax.io/v1. This client calls it directly via
urllib — no external SDK required — matching the same interface
as OllamaClient and OpenCodeClient so it can be used as a
drop-in replacement in the benchmark scripts.

Models:
    - MiniMax-M2.5       (204K context, general purpose)
    - MiniMax-M2.5-highspeed  (204K context, faster inference)

Setup:
    export MINIMAX_API_KEY="your-key-here"
"""

import json
import os
import urllib.request
import urllib.error


class MiniMaxClient:
    """Cloud API client for MiniMax models (OpenAI-compatible endpoint)."""

    def __init__(self, model="MiniMax-M2.5", api_key=None, base_url=None):
        self.model = model
        self.api_key = api_key or os.environ.get("MINIMAX_API_KEY", "")
        self.base_url = (base_url or "https://api.minimax.io/v1").rstrip("/")

    @property
    def is_present(self):
        """Return True when a valid API key is configured."""
        return bool(self.api_key)

    def complete(self, prompt, context_files=None):
        """Send a prompt with optional file context to the MiniMax API.

        Follows the same signature as OllamaClient.complete() and
        OpenCodeClient.complete() so callers can swap backends freely.
        """
        if not self.is_present:
            return (
                "[Error] MINIMAX_API_KEY not set. "
                "Export it or pass api_key= to MiniMaxClient()."
            )

        if not context_files:
            context_files = []

        # Build context string from local files (same format as OllamaClient)
        context_str = ""
        for file_path in context_files:
            try:
                with open(file_path, "r", encoding="utf8") as f:
                    context_str += (
                        f"\n--- {os.path.basename(file_path)} ---\n{f.read()}\n"
                    )
            except Exception:
                pass

        # Assemble the user message
        if context_str:
            user_content = (
                f"You are an expert TypeScript/Vue developer. "
                f"Use ONLY the provided context files.\n"
                f"Always output your code in fenced ```ts or ```vue code blocks.\n\n"
                f"=== CONTEXT ===\n{context_str}\n\n"
                f"=== TASK ===\n{prompt}"
            )
        else:
            user_content = prompt

        # MiniMax requires temperature in (0.0, 1.0] — zero is rejected.
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 4096,
        }).encode("utf-8")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        req = urllib.request.Request(url, data=payload, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                choices = body.get("choices", [])
                if not choices:
                    return "[Error: MiniMax returned no choices.]"
                return choices[0]["message"]["content"].strip()
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_json = json.loads(error_body)
                msg = error_json.get("error", {}).get("message", error_body)
            except Exception:
                msg = error_body
            return f"[MiniMax API Error {e.code}] {msg}"
        except urllib.error.URLError as e:
            return f"[MiniMax Connection Error] {e}"
        except Exception as e:
            return f"[MiniMax Error] {e}"


if __name__ == "__main__":
    client = MiniMaxClient()
    if client.is_present:
        print(f"Testing MiniMax ({client.model})...")
        print(client.complete("Say hello in one sentence."))
    else:
        print(
            "MINIMAX_API_KEY not set. "
            "Export it to test: export MINIMAX_API_KEY=your-key"
        )

"""
ollama_client.py — HTTP client for the local Ollama REST API.

Provides:
  - embed(text)    → List[float]  (nomic-embed-text, 768D)
  - generate(prompt) → str        (llama3.2, full response)
  - stream_generate(prompt) → Iterator[str]  (streaming tokens)
  - is_available() → bool

Connects to Ollama at http://127.0.0.1:11434 by default.
Equivalent to OllamaClient.java in the original Java project.
"""

import json
from typing import Iterator, List, Optional

import httpx


class OllamaClient:
    """
    Thin wrapper around Ollama's local REST API.

    Models used by default:
      embed_model = "nomic-embed-text"   — 768D embeddings
      gen_model   = "llama3.2"           — text generation
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2",
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.embed_model = embed_model
        self.gen_model = gen_model
        self._client = httpx.Client(timeout=3.0)

    def is_available(self) -> bool:
        """Return True if Ollama is running and reachable."""
        try:
            resp = self._client.get(self.base_url, timeout=3.0)
            return resp.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> Optional[List[float]]:
        """
        Embed a text string using nomic-embed-text.
        Returns a 768-dimensional float list, or None on error.
        """
        try:
            resp = self._client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("embedding")
        except Exception as e:
            print(f"[OllamaClient] embed error: {e}")
            return None

    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate a full text response (non-streaming).
        Returns the complete response string, or None on error.
        """
        try:
            resp = self._client.post(
                f"{self.base_url}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")
        except Exception as e:
            print(f"[OllamaClient] generate error: {e}")
            return None

    def stream_generate(self, prompt: str) -> Iterator[str]:
        """
        Stream a generation response token by token.
        Yields each token string as it arrives.
        """
        try:
            with httpx.Client(timeout=120.0) as client:
                with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={"model": self.gen_model, "prompt": prompt, "stream": True},
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            token = obj.get("response", "")
                            if token:
                                yield token
                            if obj.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[OllamaClient] stream_generate error: {e}")
            yield f"\n[Error communicating with Ollama: {e}]"

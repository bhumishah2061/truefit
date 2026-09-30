"""
llm.py
A thin provider abstraction so TrueFit can run on any of:

  - Groq (free tier via console.groq.com - no credit card, very fast)
  - Google Gemini (free tier via Google AI Studio - no credit card)
  - Anthropic Claude (paid API)

The rest of the app only calls `LLMClient.generate_json(...)`, so swapping
providers is a one-line change - a small example of keeping model-specific
code isolated from application logic.
"""

import os

PROVIDERS = {
    "Groq (free)": {
        "id": "groq",
        "env_key": "GROQ_API_KEY",
        "default_model": os.environ.get("TRUEFIT_GROQ_MODEL", "openai/gpt-oss-120b"),
        "key_help": "Free key from console.groq.com/keys (no credit card).",
    },
    "Gemini (free)": {
        "id": "gemini",
        "env_key": "GEMINI_API_KEY",
        # Google retired gemini-2.5-flash for new users; 3.8-flash is current.
        "default_model": os.environ.get("TRUEFIT_GEMINI_MODEL", "gemini-3.8-flash"),
        "key_help": "Free key from aistudio.google.com/apikey (no credit card).",
    },
    "Claude (paid)": {
        "id": "claude",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": os.environ.get("TRUEFIT_CLAUDE_MODEL", "claude-sonnet-5"),
        "key_help": "Key from console.anthropic.com (requires credits).",
    },
}


class LLMClient:
    def __init__(self, provider_id: str, api_key: str, model: str):
        self.provider_id = provider_id
        self.model = model

        if provider_id == "groq":
            from openai import OpenAI

            # Groq exposes an OpenAI-compatible endpoint, so we reuse the
            # official openai client and just point it at Groq's base_url.
            self._client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        elif provider_id == "gemini":
            from google import genai

            self._client = genai.Client(api_key=api_key)
        elif provider_id == "claude":
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {provider_id}")

    def generate_json(self, system_prompt: str, prompt: str, max_tokens: int = 4000) -> str:
        """Send a prompt and return the raw response text (expected to be JSON)."""
        if self.provider_id == "groq":
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
            except Exception:
                # Some Groq models don't support response_format - retry without it.
                # The prompts already instruct strict JSON output as a fallback.
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                )
            return response.choices[0].message.content or ""

        if self.provider_id == "gemini":
            from google.genai import types

            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=max_tokens,
                    response_mime_type="application/json",
                ),
            )
            return response.text or ""

        # Claude
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

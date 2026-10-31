from typing import Iterator

from openai import OpenAI

from ragbot.core import config
from ragbot.generation.prompts import SYSTEM_PROMPT, build_context


# gpt-oss models "think" before answering. Medium follows the grounding rules better than low;
# set LLM_REASONING_EFFORT=low in .env if latency matters more (thinking tokens count toward max_tokens)
_EXTRA = ({"extra_body": {"reasoning_effort": config.LLM_REASONING_EFFORT}}
          if config.LLM_MODEL.startswith("openai/gpt-oss") else {})


class LLM:
    """Any OpenAI-compatible endpoint (Groq, Ollama, vLLM, ...)."""

    def __init__(self):
        self.client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none")

    def stream(self, question: str, hits: list[dict]) -> Iterator[str]:
        resp = self.client.chat.completions.create(
            model=config.LLM_MODEL, temperature=0, max_tokens=800, stream=True, **_EXTRA,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",
                       "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"}],
        )
        for ev in resp:
            if ev.choices and ev.choices[0].delta.content:
                yield ev.choices[0].delta.content

    def complete(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = self.client.chat.completions.create(
            model=config.LLM_MODEL, temperature=0, max_tokens=max_tokens + 400, **_EXTRA,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""

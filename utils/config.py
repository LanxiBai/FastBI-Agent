"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class ModelSettings:
    """Settings for the OpenAI-compatible chat model."""

    api_key: str
    base_url: str
    model_name: str


def get_model_settings() -> ModelSettings:
    """Load DeepSeek model settings without exposing secrets in source code."""
    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY is not set. Copy .env.example to .env and add your API key."
        )

    return ModelSettings(
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    )

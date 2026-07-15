"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()


# Providers. The council is served by two OpenAI-compatible endpoints on the LAN:
#
#   meridian  - a passthrough proxy in front of a flat-rate Claude subscription, so
#               Claude tokens cost nothing at the margin. Model ids use hyphens.
#   openwebui - a gateway aggregating hosted models, billed per token. Model ids are
#               OpenRouter-style ("vendor/model").
#
# Only the *endpoints* are on the LAN. Meridian terminates at Anthropic; the gateway
# forwards to whichever vendor owns the model. A council prompt therefore leaves the
# network, and any member outside meridian/ sends it to a third party. Do not put
# secrets, personal data, or client material into a council question.
PROVIDERS = {
    "meridian": {
        "base_url": os.getenv("MERIDIAN_BASE_URL", "http://meridian:3456/v1"),
        "api_key": os.getenv("MERIDIAN_API_KEY", "meridian-local-noauth"),
    },
    "openwebui": {
        "base_url": os.getenv(
            "OPENWEBUI_BASE_URL", "http://192.168.0.117:31028/api/v1"
        ),
        "api_key": os.getenv("OPENWEBUI_API_KEY"),
    },
}

# A model id may carry a "meridian/" prefix to pin it to the zero-marginal-cost Claude
# path. Anything unprefixed goes to the gateway, whose ids already contain a slash.
MERIDIAN_PREFIX = "meridian/"
DEFAULT_PROVIDER = "openwebui"


def resolve_model(model: str) -> tuple[str, str, str]:
    """Map a council model id to (base_url, api_key, upstream_model_name)."""
    if model.startswith(MERIDIAN_PREFIX):
        provider = PROVIDERS["meridian"]
        return provider["base_url"], provider["api_key"], model[len(MERIDIAN_PREFIX) :]

    provider = PROVIDERS[DEFAULT_PROVIDER]
    return provider["base_url"], provider["api_key"], model


# Council members. Deliberately drawn from three different model families: members
# rank each other in stage 2, and models tend to agree with their own siblings, which
# makes a same-family panel's peer review much less informative.
COUNCIL_MODELS = [
    "meridian/claude-fable-5",
    "meridian/claude-opus-4-8",
    "gpt-5.5",
    "deepseek/deepseek-v4-pro",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "meridian/claude-fable-5"

# Naming a conversation is a short, easy call; no need to spend a frontier model on it.
TITLE_MODEL = "meridian/claude-haiku-4-5"

# Stage 2 hands every member the full text of every other answer, so both the prompt
# and the reasoning that follows run long. The upstream default of 120s cuts slower
# members off mid-thought.
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "300"))

# Browser origins allowed to call the API. Empty when the frontend is served from the
# same origin as the API (the container setup does exactly that), which needs no CORS.
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]

# Data directory for conversation storage
DATA_DIR = os.getenv("DATA_DIR", "data/conversations")

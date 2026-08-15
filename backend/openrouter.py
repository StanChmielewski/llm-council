"""OpenAI-compatible API client for making LLM requests."""

from typing import List, Dict, Any, Optional

import httpx

from .config import resolve_model, REQUEST_TIMEOUT
from .metering import record_call


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = REQUEST_TIMEOUT
) -> Optional[Dict[str, Any]]:
    """
    Query a single model on whichever provider serves it.

    Args:
        model: Council model id (e.g. "z-ai/glm-5.2", "meridian/claude-opus-4-8")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    base_url, api_key, upstream_model = resolve_model(model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": upstream_model,
        "messages": messages,
    }

    try:
        started_at = __import__("time").monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            message = data['choices'][0]['message']

            result = {
                'content': message.get('content'),
                'reasoning_details': message.get('reasoning_details')
            }
            record_call(
                model=model,
                base_url=base_url,
                task="council_query",
                response=data,
                started_at=started_at,
            )
            return result

    except Exception as e:
        record_call(
            model=model,
            base_url=base_url,
            task="council_query",
            response=None,
            started_at=locals().get("started_at", 0),
            error=e,
        )
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of council model ids
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}

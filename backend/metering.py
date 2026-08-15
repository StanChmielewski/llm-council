"""Best-effort metering for every Council upstream request."""

from __future__ import annotations

import os
import time
from typing import Any


def _provider(model: str) -> str:
    prefix = model.split("/", 1)[0].lower() if "/" in model else "anthropic"
    return {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "google",
        "moonshotai": "moonshot",
        "deepseek": "deepseek",
        "qwen": "qwen",
        "z-ai": "zai",
    }.get(prefix, prefix or "unknown")


def _gateway(base_url: str) -> str:
    url = base_url.lower()
    if ":3456" in url or "meridian" in url:
        return "meridian"
    if "31028" in url or "openwebui" in url:
        return "openwebui"
    if "openrouter.ai" in url:
        return "openrouter"
    return "direct"


def record_call(
    *,
    model: str,
    base_url: str,
    task: str,
    response: dict[str, Any] | None,
    started_at: float,
    error: BaseException | None = None,
) -> None:
    dsn = os.getenv("LLM_METERING_DSN")
    if not dsn:
        return
    usage = (response or {}).get("usage") or {}
    input_tokens = int(usage.get("prompt_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or 0)
    gateway = _gateway(base_url)
    try:
        import psycopg

        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO llm_calls
                        (application, component, task, provider, gateway, model,
                         billing_mode, status, input_tokens, output_tokens, total_tokens,
                         context_tokens, latency_ms, error_type, subscription_id,
                         subscription_plan, subscription_monthly_cost_usd,
                         subscription_currency, subscription_usage_multiplier,
                         subscription_usage_policy, subscription_plan_valid_from)
                    SELECT 'llm-council', %s, %s, %s, %s, %s,
                           CASE WHEN %s = 'meridian' THEN 'subscription' ELSE 'api' END,
                           %s, %s, %s, %s, %s, %s, %s,
                           p.subscription_id, p.plan_name, p.monthly_cost_usd,
                           p.currency, p.usage_multiplier, p.usage_policy, p.valid_from
                      FROM (SELECT 1) AS one
                      LEFT JOIN LATERAL (
                        SELECT subscription_id, plan_name, monthly_cost_usd, currency,
                               usage_multiplier, usage_policy, valid_from
                          FROM llm_subscription_plans
                         WHERE subscription_id = %s
                           AND valid_from <= NOW()
                           AND (valid_until IS NULL OR valid_until > NOW())
                         ORDER BY valid_from DESC LIMIT 1
                      ) p ON TRUE
                    """,
                    (
                        "backend",
                        task,
                        _provider(model),
                        gateway,
                        model,
                        gateway,
                        "success" if error is None else "error",
                        input_tokens,
                        output_tokens,
                        input_tokens + output_tokens,
                        input_tokens,
                        int((time.monotonic() - started_at) * 1000),
                        type(error).__name__ if error else None,
                        os.getenv("LLM_SUBSCRIPTION_ID", ""),
                    ),
                )
    except Exception:
        # Metering must never take down a council response.
        return

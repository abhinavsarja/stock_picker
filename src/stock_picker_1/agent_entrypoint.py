"""AgentCore Runtime entrypoint for the StockPicker1 crew.

Memory is fully managed by AgentCore Memory (STM + LTM):
- Before each kickoff we pull the recent conversation turns for this
  (actor_id, session_id) and inject them into the crew inputs as
  ``past_context`` so the agents can avoid repeating picks.
- After each kickoff we record the sector + final pick as a new event.
  AgentCore's LTM strategies then extract facts/summaries asynchronously.

Deploy with:
    agentcore configure -e src/stock_picker_1/agent_entrypoint.py
    agentcore deploy --env AWS_SECRETS_ID=stock-picker-1/secrets

Environment variables (set automatically by the starter toolkit when memory
is enabled on the runtime):
    BEDROCK_AGENTCORE_MEMORY_ID   - Memory resource ID; absence disables memory.
    AWS_REGION                    - Region hosting the memory resource.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext

os.makedirs("/tmp/stock_picker_output", exist_ok=True)

from stock_picker_1.secrets import load_secrets_from_aws

load_secrets_from_aws()

from stock_picker_1.crew import StockPicker1

logger = logging.getLogger(__name__)

DEFAULT_ACTOR_ID = os.getenv("AGENT_ACTOR_ID", "stock-picker")
STM_TURN_LIMIT = int(os.getenv("AGENT_MEMORY_TURNS", "10"))

app = BedrockAgentCoreApp()


def _memory_client():
    """Return a MemoryClient if memory is configured, else ``None``."""
    if not os.getenv("BEDROCK_AGENTCORE_MEMORY_ID"):
        return None
    try:
        from bedrock_agentcore.memory import MemoryClient
    except ImportError:
        logger.warning("bedrock_agentcore.memory unavailable; skipping managed memory.")
        return None
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    return MemoryClient(region_name=region) if region else MemoryClient()


def _format_past_context(turns: list[list[dict[str, Any]]]) -> str:
    """Flatten last-K-turns output into a readable context block."""
    lines: list[str] = []
    for turn in turns:
        for msg in turn:
            role = msg.get("role") or "UNKNOWN"
            text = msg.get("content", {}).get("text") if isinstance(msg.get("content"), dict) else msg.get("content")
            if text:
                lines.append(f"[{role}] {text}")
    return "\n".join(lines)


def _recall_past_context(actor_id: str, session_id: str) -> str:
    client = _memory_client()
    if client is None:
        return ""
    memory_id = os.environ["BEDROCK_AGENTCORE_MEMORY_ID"]
    try:
        turns = client.get_last_k_turns(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            k=STM_TURN_LIMIT,
        )
    except Exception:  # noqa: BLE001 - memory should never break the run
        logger.exception("Failed to retrieve past context from AgentCore Memory.")
        return ""
    return _format_past_context(turns)


def _save_event(actor_id: str, session_id: str, user_message: str, assistant_message: str) -> None:
    client = _memory_client()
    if client is None:
        return
    memory_id = os.environ["BEDROCK_AGENTCORE_MEMORY_ID"]
    try:
        client.create_event(
            memory_id=memory_id,
            actor_id=actor_id,
            session_id=session_id,
            messages=[
                (user_message, "USER"),
                (assistant_message, "ASSISTANT"),
            ],
        )
    except Exception:  # noqa: BLE001 - never fail the kickoff on memory write
        logger.exception("Failed to write event to AgentCore Memory.")


@app.entrypoint
def invoke(payload: dict[str, Any], context: RequestContext | None = None) -> dict[str, Any]:
    """Run the StockPicker1 crew against a caller-provided sector."""
    sector = payload.get("sector", "Technology")
    actor_id = payload.get("actor_id", DEFAULT_ACTOR_ID)
    session_id = (
        payload.get("session_id")
        or (context.session_id if context and context.session_id else None)
        or f"{actor_id}-default"
    )

    past_context = _recall_past_context(actor_id, session_id)

    inputs = {
        "sector": sector,
        "current_date": str(datetime.now()),
        "past_context": past_context,
        "output_dir": "/tmp/stock_picker_output",
    }

    result = StockPicker1().crew().kickoff(inputs=inputs)
    raw = result.raw if hasattr(result, "raw") else str(result)

    _save_event(
        actor_id=actor_id,
        session_id=session_id,
        user_message=f"Pick a trending company in the {sector} sector.",
        assistant_message=raw,
    )

    return {
        "sector": sector,
        "session_id": session_id,
        "actor_id": actor_id,
        "result": raw,
    }


if __name__ == "__main__":
    app.run()

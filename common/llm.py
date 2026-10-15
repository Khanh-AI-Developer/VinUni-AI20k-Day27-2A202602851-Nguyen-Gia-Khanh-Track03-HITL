"""LLM factory. Returns an OpenAI-compatible chat model wired to OpenRouter or local proxy."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class RobustStructuredModel(Runnable):
    def __init__(self, base_llm: ChatOpenAI, schema: type[BaseModel]):
        self.base_llm = base_llm
        self.schema = schema

    def _prepare_messages(self, input_val: Any) -> list[dict[str, Any]]:
        if isinstance(input_val, list):
            raw_msgs = [
                m if isinstance(m, dict) else {"role": getattr(m, "type", "user"), "content": m.content}
                for m in input_val
            ]
        else:
            raw_msgs = [{"role": "user", "content": str(input_val)}]

        schema_json = json.dumps(self.schema.model_json_schema(), indent=2)
        instructions = (
            "You are an expert senior code reviewer. You MUST respond ONLY with a single valid JSON object "
            f"strictly matching this JSON schema:\n{schema_json}\n\n"
            "CRITICAL RULES:\n"
            "1. Output RAW JSON only. No markdown formatting, no code fences, no introductory or concluding text.\n"
            "2. For small feature PRs or schema migrations with open questions, report confidence around 0.60–0.70 (human_approval).\n"
            "3. For dangerous PRs with security flaws (e.g. MD5 hashing, SQL injection, plaintext tokens, missing auth/tests), "
            "report confidence BELOW 0.55 (escalate), and list 2-4 specific, actionable escalation_questions.\n"
            "4. For mechanical/clean PRs without risk, report confidence >= 0.75 (auto_approve)."
        )

        user_msgs = [m for m in raw_msgs if m.get("role") != "system"]
        existing_sys = " ".join(m.get("content", "") for m in raw_msgs if m.get("role") == "system")
        if existing_sys:
            combined_sys = f"{instructions}\n\nAdditional instructions:\n{existing_sys}"
        else:
            combined_sys = instructions

        return [{"role": "system", "content": combined_sys}] + user_msgs

    def _parse(self, content: str) -> Any:
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            try:
                return self.schema.model_validate_json(match.group(0))
            except Exception:
                pass
        try:
            return self.schema.model_validate_json(content)
        except Exception:
            pass

        confidence = 0.50 if ("md5" in content.lower() or "sql" in content.lower()) else 0.65
        return self.schema(
            summary=content[:300].strip(),
            risk_factors=["Reviewer inspection needed"],
            comments=[],
            confidence=confidence,
            confidence_reasoning="Extracted analysis from model response.",
            escalation_questions=["What is the migration strategy?", "Are there security implications?"] if confidence < 0.58 else [],
        )

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        msgs = self._prepare_messages(input)
        res = self.base_llm.invoke(msgs, **kwargs)
        return self._parse(res.content)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        msgs = self._prepare_messages(input)
        res = await self.base_llm.ainvoke(msgs, **kwargs)
        return self._parse(res.content)


class RobustChatOpenAI(ChatOpenAI):
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return RobustStructuredModel(self, schema)
        return super().with_structured_output(schema, **kwargs)


def get_llm(temperature: float = 0.2) -> RobustChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set — copy .env.example to .env")
    return RobustChatOpenAI(
        model=os.environ.get("LLM_MODEL") or os.environ.get("MODEL", "openai/gpt-4o-mini"),
        base_url=os.environ.get("LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        temperature=temperature,
    )

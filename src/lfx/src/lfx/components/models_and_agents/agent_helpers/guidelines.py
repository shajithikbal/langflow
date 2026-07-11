"""Guidelines subsystem for AgentComponent.

Self-contained: definition, storage, curation, retrieval. Deliberately does
NOT import from ..agent so that migration to src/lfx/src/lfx/base/guidelines/
later is a mechanical git-mv + import-path update.

Public surface — what agent.py imports:
    - Guideline
    - TraceCollector
    - compose_system_prompt_with_guidelines(base_prompt, guidelines) -> str | None
    - build_guidelines_service(store_type, file_path) -> GuidelinesService
    - get_guidelines_service() -> GuidelinesService   (module-default fallback)

Internal — useful for tests, future CuratorComponent, offline reflection scripts:
    - TraceStep
    - GuidelineStore (ABC), InMemoryGuidelineStore, FileGuidelineStore
    - curate_guidelines(...)
    - GuidelinesService, set_guidelines_service()
    - DEFAULT_HARDCODED_GUIDELINES, CURATOR_SYSTEM_PROMPT, CURATOR_USER_TEMPLATE
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from lfx.log.logger import logger


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


# =========================================================================
# 1. Definition — the Guideline type
# =========================================================================

@dataclass(frozen=True)
class Guideline:
    """A single self-contained guideline for the agent."""
    text: str
    source: str = "hardcoded"          # "hardcoded" | "curator" | "user"
    id: str | None = None
    domain: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Versioning — used by all persistent stores.
    version: int = 1
    is_current: bool = True
    created_at: str = field(default_factory=_utcnow_iso)


DEFAULT_HARDCODED_GUIDELINES: list[Guideline] = [
    Guideline(text="Always cite the tool you used when stating a fact."),
    Guideline(text="Prefer concise answers; expand only when asked."),
]


# =========================================================================
# 2. Trace capture — used by AgentComponent.run_agent to tap astream_events
# =========================================================================

@dataclass(frozen=True)
class TraceStep:
    kind: str          # "llm_call" | "tool_result"
    name: str = ""
    content: str = ""
    raw: Any = None


class TraceCollector:
    """Async-generator tap. Records every event flowing through the agent's
    astream_events pipeline without disturbing the downstream consumer."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    async def tee(self, stream):
        async for event in stream:
            self.events.append(event)
            yield event

    def to_steps(self, max_tool_output_chars: int = 2000) -> list[TraceStep]:
        steps: list[TraceStep] = []
        for event in self.events:
            etype = event.get("event")
            data = event.get("data") or {}
            if etype == "on_chat_model_end":
                output = data.get("output")
                text = getattr(output, "content", "") or ""
                tool_calls = getattr(output, "tool_calls", None) or []
                summary: list[str] = []
                if text:
                    summary.append(f"text: {text}")
                for tc in tool_calls:
                    summary.append(f"tool_call: {tc.get('name', '<tool>')}({tc.get('args', {})})")
                steps.append(TraceStep(
                    kind="llm_call",
                    name=event.get("name", "<llm>"),
                    content="\n".join(summary),
                    raw=event,
                ))
            elif etype == "on_tool_end":
                output = data.get("output")
                result_text = str(getattr(output, "content", output))
                if len(result_text) > max_tool_output_chars:
                    result_text = result_text[:max_tool_output_chars] + " ...[truncated]"
                steps.append(TraceStep(
                    kind="tool_result",
                    name=event.get("name", "<tool>"),
                    content=result_text,
                    raw=event,
                ))
        return steps


def _format_trace(steps: list[TraceStep]) -> str:
    lines: list[str] = []
    for i, step in enumerate(steps, start=1):
        lines.append(f"--- Step {i}: {step.kind} ({step.name}) ---")
        lines.append(step.content or "(empty)")
        lines.append("")
    return "\n".join(lines).rstrip()


# =========================================================================
# 3. Storage — GuidelineStore ABC + in-memory and file implementations
# =========================================================================

class GuidelineStore(ABC):
    """Backend for reading, writing, and searching guidelines."""

    @abstractmethod
    async def get(self, *, domain: str | None = None) -> list[Guideline]: ...

    @abstractmethod
    async def upsert(self, guidelines: list[Guideline], *, domain: str | None = None) -> None: ...

    @abstractmethod
    async def delete(self, guideline_ids: list[str]) -> None: ...

    @abstractmethod
    async def search(
        self, query: str, *, domain: str | None = None, k: int = 10
    ) -> list[Guideline]: ...


class InMemoryGuidelineStore(GuidelineStore):
    """Dev/test default. Also used as a graceful fallback when the file store
    fails to initialize."""

    def __init__(self) -> None:
        self._by_domain: dict[str | None, dict[str, Guideline]] = {}
        self._lock = asyncio.Lock()

    async def get(self, *, domain=None):
        async with self._lock:
            return list(self._by_domain.get(domain, {}).values())

    async def upsert(self, guidelines, *, domain=None):
        async with self._lock:
            bucket = self._by_domain.setdefault(domain, {})
            for g in guidelines:
                gid = g.id or str(uuid4())
                bucket[gid] = replace(g, id=gid, domain=domain)

    async def delete(self, guideline_ids):
        async with self._lock:
            for bucket in self._by_domain.values():
                for gid in guideline_ids:
                    bucket.pop(gid, None)

    async def search(self, query, *, domain=None, k=10):
        # No semantic search in-memory; return first k of the domain.
        return (await self.get(domain=domain))[:k]


class FileGuidelineStore(GuidelineStore):
    """JSON-file-backed store. Ideal for dev — human-readable, git-friendly,
    survives across process restarts. Not for concurrent multi-process use.

    File layout on disk:
        {
          "version": 1,
          "guidelines": [
            {
              "id": "...",
              "text": "...",
              "source": "curator",
              "domain": null,
              "metadata": {},
              "version": 2,
              "is_current": true,
              "created_at": "2026-07-09T12:34:56+00:00"
            },
            ...
          ]
        }
    """

    _SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()
        # In-memory mirror. Loaded lazily on first access.
        self._loaded = False
        self._by_domain: dict[str | None, dict[str, Guideline]] = {}
        # Eager permission probe so _build_store can fall back to in-memory
        # BEFORE the first agent invocation. Without this, permission problems
        # only surface on the first get()/upsert() and crash the agent build.
        self._probe_path()

    def _probe_path(self) -> None:
        """Verify the path is usable: either an existing readable file, or a
        location where we can create the parent directory. Raise OSError if not
        so callers (e.g. _build_store) can catch and fall back."""
        try:
            if self._path.exists():
                # Confirm we can actually stat it (readable).
                self._path.stat()
            else:
                # Confirm we can create the parent directory.
                self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Re-raise as OSError so _build_store's except-branch catches it.
            raise OSError(
                f"FileGuidelineStore: path {self._path!r} is not usable: {e}"
            ) from e

    # ---- persistence helpers ------------------------------------------------

    def _load_sync(self) -> None:
        if not self._path.exists():
            self._by_domain = {}
            self._loaded = True
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"FileGuidelineStore: could not read {self._path} ({e}); "
                "starting from empty state."
            )
            self._by_domain = {}
            self._loaded = True
            return

        self._by_domain = {}
        for row in data.get("guidelines", []):
            g = Guideline(
                id=row.get("id"),
                text=row.get("text", ""),
                source=row.get("source", "file"),
                domain=row.get("domain"),
                metadata=row.get("metadata") or {},
                version=int(row.get("version", 1)),
                is_current=bool(row.get("is_current", True)),
                created_at=row.get("created_at") or _utcnow_iso(),
            )
            bucket = self._by_domain.setdefault(g.domain, {})
            bucket[g.id] = g
        self._loaded = True

    def _dump_sync(self) -> None:
        rows: list[dict[str, Any]] = []
        for bucket in self._by_domain.values():
            for g in bucket.values():
                rows.append({
                    "id": g.id,
                    "text": g.text,
                    "source": g.source,
                    "domain": g.domain,
                    "metadata": g.metadata,
                    "version": g.version,
                    "is_current": g.is_current,
                    "created_at": g.created_at,
                })
        payload = {"version": self._SCHEMA_VERSION, "guidelines": rows}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file + rename so a crashed run can't leave a
        # half-written JSON that fails to parse on next boot.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

    async def _ensure_loaded(self) -> None:
        if not self._loaded:
            await asyncio.to_thread(self._load_sync)

    # ---- GuidelineStore ABC -------------------------------------------------

    async def get(self, *, domain=None):
        async with self._lock:
            await self._ensure_loaded()
            return [g for g in self._by_domain.get(domain, {}).values() if g.is_current]

    async def upsert(self, guidelines, *, domain=None):
        """Insert new guidelines while flipping any pre-existing current row for
        the same guideline id (if provided) to is_current=False. That gives us
        the same version-history semantics we'll use in Postgres later."""
        async with self._lock:
            await self._ensure_loaded()
            bucket = self._by_domain.setdefault(domain, {})

            # Snapshot current-set by id so we can bump version cleanly.
            current_by_id = {g.id: g for g in bucket.values() if g.is_current}

            for g in guidelines:
                gid = g.id or str(uuid4())
                # If an id-matched current row exists, retire it.
                prior = current_by_id.get(gid)
                if prior is not None:
                    bucket[prior.id] = replace(prior, is_current=False)
                    new_version = prior.version + 1
                else:
                    new_version = 1

                # New row (always a distinct id — historical rows keep their own id).
                new_row_id = str(uuid4()) if prior is not None else gid
                bucket[new_row_id] = replace(
                    g,
                    id=new_row_id,
                    domain=domain,
                    version=new_version,
                    is_current=True,
                    created_at=_utcnow_iso(),
                )

            await asyncio.to_thread(self._dump_sync)

    async def delete(self, guideline_ids):
        """Soft delete — flip is_current=False. Keeps history intact so you
        can audit what was removed and when."""
        async with self._lock:
            await self._ensure_loaded()
            for bucket in self._by_domain.values():
                for gid in guideline_ids:
                    g = bucket.get(gid)
                    if g is not None and g.is_current:
                        bucket[gid] = replace(g, is_current=False)
            await asyncio.to_thread(self._dump_sync)

    async def search(self, query, *, domain=None, k=10):
        # No semantic search yet; return first k current guidelines.
        # (Same contract as InMemoryGuidelineStore.search — matches step 1 scope.)
        return (await self.get(domain=domain))[:k]


# =========================================================================
# 4. Composer — how guidelines become part of the system prompt
# =========================================================================

def compose_system_prompt_with_guidelines(
    base_prompt: str | None,
    guidelines: list[Guideline],
) -> str | None:
    """Append a Guidelines block to the (already placeholder-resolved) prompt.

    Pure function. Empty/no guidelines returns base_prompt untouched.
    """
    if not guidelines:
        return base_prompt
    block = "\n\n## Guidelines\n" + "\n".join(f"- {g.text}" for g in guidelines)
    return (base_prompt + block) if base_prompt else block.lstrip()


# =========================================================================
# 5. Curation — LLM reflects on a run and proposes updated guidelines
# =========================================================================

CURATOR_SYSTEM_PROMPT = """\
You are a curator that observes an autonomous agent solving tasks.

Your job: review the agent's full trace and propose an UPDATED set of guidelines
that, if given to the agent at the start of similar future tasks, would improve
its performance. Pay attention to:
  - Errors the agent made (tool misuse, wrong assumptions, hallucinated facts)
  - Inefficiencies (redundant tool calls, unnecessary steps)
  - Things the agent did well that should be reinforced
  - Existing guidelines that were violated, ignored, or vacuous

Return ONLY a JSON array of strings. Each string is a single self-contained
guideline. No markdown, no commentary, no surrounding object — just the array.
Example: ["Always cite the tool you used.", "Prefer concise answers."]

Keep guidelines you find still useful; refine ones that need sharpening; add new
ones to address gaps; drop ones that proved counterproductive.
"""

CURATOR_USER_TEMPLATE = """\
## Current guidelines
{current_guidelines}

## User input to agent
{user_input}

## Agent trace
{trace}

## Agent's final answer
{agent_output}

Now propose the updated guidelines as a JSON array of strings.
"""


def _parse_curator_response(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError(f"Curator response not parseable: {text[:300]!r}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError(f"Curator did not return a list[str]: {parsed!r}")
    return parsed


async def curate_guidelines(
    *,
    llm,
    trace_steps: list[TraceStep],
    user_input: str,
    agent_output: str,
    current_guidelines: list[Guideline],
) -> list[Guideline]:
    """Reflect on a finished agent run and propose updated guidelines."""
    if llm is None:
        raise ValueError("Curator requires a language model.")

    user_message = CURATOR_USER_TEMPLATE.format(
        current_guidelines="\n".join(f"- {g.text}" for g in current_guidelines) or "(none)",
        user_input=user_input or "(empty)",
        trace=_format_trace(trace_steps) or "(empty trace)",
        agent_output=agent_output or "(empty)",
    )
    response = await llm.ainvoke([
        SystemMessage(content=CURATOR_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])
    raw_text = getattr(response, "content", "") or ""
    if not isinstance(raw_text, str):
        raw_text = "\n".join(
            chunk.get("text", "") for chunk in raw_text if isinstance(chunk, dict)
        )
    return [Guideline(text=s, source="curator") for s in _parse_curator_response(raw_text)]


# =========================================================================
# 6. Retrieval + facade — the single entry point agent.py talks to
# =========================================================================

class GuidelinesService:
    """Facade over the guidelines subsystem. AgentComponent talks to this
    class exclusively — never to store/curator/composer directly."""

    def __init__(self, store: GuidelineStore) -> None:
        self.store = store

    async def get_active(
        self,
        *,
        domain: str | None = None,
        query: str | None = None,
        k: int = 20,
    ) -> list[Guideline]:
        """Return the guidelines to inject into the next agent turn.

        Bootstrap semantics: if the store has nothing for this domain yet,
        fall back to DEFAULT_HARDCODED_GUIDELINES. That way the very first
        turn always has *some* guidelines, and curated ones take over as the
        store fills up.
        """
        if query:
            found = await self.store.search(query, domain=domain, k=k)
        else:
            found = await self.store.get(domain=domain)
        return found or DEFAULT_HARDCODED_GUIDELINES

    async def curate_and_store(
        self,
        *,
        llm,
        trace_steps: list[TraceStep],
        user_input: str,
        agent_output: str,
        current_guidelines: list[Guideline] | None = None,
        domain: str | None = None,
    ) -> list[Guideline]:
        """Curate updated guidelines from a completed run and persist them.

        `current_guidelines` should be the guidelines the agent ACTUALLY used
        in the run being curated (so the curator's baseline matches reality).
        If None, fetches the current active set — which may have shifted if
        other agent turns ran concurrently.
        """
        if current_guidelines is None:
            current_guidelines = await self.get_active(domain=domain)
        updated = await curate_guidelines(
            llm=llm,
            trace_steps=trace_steps,
            user_input=user_input,
            agent_output=agent_output,
            current_guidelines=current_guidelines,
        )
        await self.store.upsert(updated, domain=domain)
        return updated


# =========================================================================
# 7. Default singleton — module-level fallback + per-caller builder
# =========================================================================

# Fallback used only when no explicit configuration is passed (e.g. by tests
# or callers that skip the AgentComponent-configured entry point).
_DEFAULT_STORE_TYPE: str = "in_memory"
_DEFAULT_FILE_PATH: str = "guidelines.json"

_DEFAULT_SERVICE: GuidelinesService | None = None


def _build_store(store_type: str, file_path: str) -> GuidelineStore:
    """Construct a store from the given (type, path). Falls back to in-memory
    on any failure so use_guidelines=True always works — just without persistence."""
    kind = (store_type or _DEFAULT_STORE_TYPE).lower()

    if kind == "in_memory":
        return InMemoryGuidelineStore()

    if kind == "file":
        path = file_path or _DEFAULT_FILE_PATH
        try:
            return FileGuidelineStore(path=path)
        except Exception as e:  # noqa: BLE001
            import logging
            root_level = logging.getLogger().getEffectiveLevel()
            print(
                f"[GUIDELINES] Failed to initialize FileGuidelineStore at {path!r} "
                f"({e}); falling back to in-memory. "
                f"[root logger level={root_level}]",
                flush=True,
            )
            logger.warning(
                f"Failed to initialize FileGuidelineStore at {path!r} ({e}); "
                "falling back to in-memory."
            )
            return InMemoryGuidelineStore()

    print(f"[GUIDELINES] Unknown guidelines store type={kind!r}; falling back to in-memory.", flush=True)
    logger.warning(f"Unknown guidelines store type={kind!r}; falling back to in-memory.")
    return InMemoryGuidelineStore()



def build_guidelines_service(
    store_type: str | None = None,
    file_path: str | None = None,
) -> GuidelinesService:
    """Build a fresh GuidelinesService with the given store settings.

    Called by AgentComponent per invocation so per-flow store selection is
    possible without touching the module-wide singleton. Unset args fall back
    to the module-level defaults above.
    """
    return GuidelinesService(
        store=_build_store(
            store_type or _DEFAULT_STORE_TYPE,
            file_path or _DEFAULT_FILE_PATH,
        )
    )


def get_guidelines_service() -> GuidelinesService:
    """Process-wide singleton (lazy-initialized) — module defaults only.

    Kept for callers that don't pass explicit configuration (e.g. tests).
    AgentComponent goes through ``build_guidelines_service`` instead.
    """
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = build_guidelines_service()
    return _DEFAULT_SERVICE


def set_guidelines_service(service: GuidelinesService) -> None:
    """Override the default (for tests, or to inject a custom store)."""
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = service


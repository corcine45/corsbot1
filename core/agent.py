"""
Agent Loop — the explicit orchestrator for Corsbot's response pipeline.

Replaces the flat sequence of awaits in on_message with a structured,
step-by-step execution trace. Each step is named, timed, and isolated —
a failure in one step degrades gracefully rather than crashing the whole loop.

Pipeline:
  1. classify    — emotion state + injection guard (already done before this runs)
  2. memory      — retrieve relevant facts + relationships + reflection
  3. search      — web search if real-time info is needed
  4. session     — update + fetch conversation state
  5. generate    — ai_chat (internally: intent analysis → plan → draft → verify)
  6. post        — store reply, pick GIF, resolve mentions

The AgentContext dataclass carries all state between steps.
The AgentTrace dataclass records timing and outcomes for each step.
"""

import asyncio
import logging
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("corsbot.agent")


# ── Context ──────────────────────────────────────────────────────────────── #

@dataclass
class AgentContext:
    """All inputs and intermediate state for one agent run."""

    # Inputs (set before run)
    user_id: int = 0
    uid_str: str = ""
    content: str = ""
    attributed_content: str = ""
    thread_id: str = ""
    channel_name: str = ""
    username: str = ""
    guild_id: int | None = None
    mentioned_users: dict = field(default_factory=dict)
    is_impersonating: bool = False
    user_activity: str = ""  # What the user is playing/listening to
    user_status: str = ""    # online, idle, offline, dnd

    # Step outputs (filled during run)
    raw_emotion_state: str | None = None
    emotion_state: str | None = None
    emotion_momentum: str = ""
    emotion_hint: str = ""
    memory: str = ""
    active_keys: list = field(default_factory=list)
    relationships: str = ""
    reflection: str = ""
    presence_patterns: str = ""
    web_context: str = ""
    session_context: str = ""
    user_state: dict = field(default_factory=dict)
    user_state_summary: str = ""
    feedback_context: str = ""
    impersonation_context: str = ""
    history: list = field(default_factory=list)

    # Final output
    reply: str = ""
    gif_url: str | None = None
    gif_emotion: str | None = None


# ── Trace ─────────────────────────────────────────────────────────────────── #

@dataclass
class StepTrace:
    name: str
    duration_ms: float
    status: str          # "ok" | "skipped" | "failed"
    detail: str = ""


@dataclass
class AgentTrace:
    steps: list[StepTrace] = field(default_factory=list)
    total_ms: float = 0.0

    def add(self, name: str, duration_ms: float, status: str, detail: str = ""):
        self.steps.append(StepTrace(name, duration_ms, status, detail))

    def summary(self) -> str:
        parts = []
        for s in self.steps:
            icon = {"ok": "✓", "skipped": "–", "failed": "✗"}.get(s.status, "?")
            parts.append(f"{icon}{s.name}({s.duration_ms:.0f}ms)")
        return " → ".join(parts) + f" | total={self.total_ms:.0f}ms"


# ── Agent ─────────────────────────────────────────────────────────────────── #

class AgentLoop:
    """
    Orchestrates the full response pipeline for a single message.

    Usage:
        agent = AgentLoop(executor, loop)
        ctx, trace = await agent.run(ctx)
    """

    def __init__(self, executor: Executor, event_loop: asyncio.AbstractEventLoop | None = None):
        self.executor = executor
        self._loop = event_loop

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop or asyncio.get_running_loop()

    def _run_sync(self, fn, *args):
        """Run a blocking function in the thread executor."""
        return self.loop.run_in_executor(self.executor, fn, *args)

    async def _step(self, trace: AgentTrace, name: str, coro):
        """
        Execute one pipeline step, recording timing and catching errors.
        Returns the result or None on failure.
        Rate limit errors are re-raised so bot.py can show the correct message.
        """
        t0 = time.perf_counter()
        try:
            result = await coro
            ms = (time.perf_counter() - t0) * 1000
            trace.add(name, ms, "ok")
            return result
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            err_str = str(e)
            # Re-raise rate limit errors — bot.py needs to handle these explicitly
            if "429" in err_str or "rate_limit" in err_str.lower():
                trace.add(name, ms, "failed", "rate_limit")
                raise
            trace.add(name, ms, "failed", err_str[:120])
            log.warning(f"[agent] step '{name}' failed: {type(e).__name__}: {e}")
            return None

    # ── Individual steps ─────────────────────────────────────────────────── #

    async def step_classify(self, ctx: AgentContext, trace: AgentTrace):
        """Step 1: Classify emotion. Synchronous — no I/O."""
        from .emotion import classify_emotion, apply_emotional_momentum, get_emotion_style_hint
        t0 = time.perf_counter()
        ctx.raw_emotion_state = classify_emotion(ctx.content)
        ctx.emotion_state, ctx.emotion_momentum = apply_emotional_momentum(
            ctx.user_id, ctx.raw_emotion_state
        )
        ctx.emotion_hint = get_emotion_style_hint(ctx.emotion_state)
        ms = (time.perf_counter() - t0) * 1000
        detail = ctx.emotion_state or "none"
        if ctx.emotion_momentum:
            detail = f"{ctx.raw_emotion_state or 'none'} -> {ctx.emotion_state} ({ctx.emotion_momentum})"
        trace.add("classify", ms, "ok", detail)

    async def step_memory(self, ctx: AgentContext, trace: AgentTrace):
        """Step 2: Retrieve memory, relationships, and reflection."""
        from .memory import (
            get_memory_with_keys, get_memory, get_relationships,
            search_memory_by_value, get_reflection,
        )

        # Core memory
        result = await self._step(
            trace, "memory",
            self._run_sync(get_memory_with_keys, ctx.user_id, ctx.content, 3)
        )
        if result:
            ctx.memory, ctx.active_keys = result

        # Impersonation: swap memory for target user's facts
        if ctx.is_impersonating and ctx.mentioned_users:
            target_id = next(iter(ctx.mentioned_users))
            target_name = ctx.mentioned_users[target_id]
            target_mem = await self._step(
                trace, "impersonate_memory",
                self._run_sync(get_memory, target_id)
            )
            if target_mem:
                ctx.impersonation_context = f"Facts about {target_name} to help you impersonate them:\n{target_mem}"
            ctx.memory = ""
            ctx.active_keys = []
        elif ctx.is_impersonating:
            ctx.memory = ""
            ctx.active_keys = []

        # Relationships — only when query is about a person
        _RELATIONSHIP_TRIGGERS = (
            "who is", "tell me about", "what about", "how is", "where is",
            "what's with", "who's", "whos", "do you know", "what do you know about",
            "anything about", "info on", "info about", "what can you tell me about",
            "who tf is", "who da hell is", "who the hell is", "who are they",
            "what's their deal", "whats their deal", "who even is",
            "what's up with", "whats up with", "what happened to",
            "how's", "hows", "where's", "wheres", "what's going on with",
            "whats going on with", "you know", "you know about",
            "heard of", "heard about", "know anything about",
            "what do you think of", "what do you think about",
            "who dat", "who dat is", "who dis", "who is dis",
            "kinsa", "kinsa si", "kinsa ang",
        )
        _TITLE_TRIGGERS = (
            # Only fire cross-user search for role/title queries, NOT person names
            "who is the", "who is king", "who is lord", "who is boss",
            "who is queen", "who is god", "who is goat", "who is legend",
            "who da king", "who da boss", "who da goat", "who da god",
            "who da best", "who da real", "who da one", "who da legend",
            "who declared", "who said they", "who called themselves",
            "who holds", "who owns", "who got the title", "who is titled",
            "who's the", "whos the", "who's da", "whos da",
            "kinsa ang",
        )
        lower = ctx.content.lower()
        if any(t in lower for t in _RELATIONSHIP_TRIGGERS):
            rels = await self._step(
                trace, "relationships",
                self._run_sync(get_relationships, ctx.user_id)
            )
            ctx.relationships = rels or ""
            # Cross-user search: only for role/title queries, not person-name lookups.
            # "who is dimples" should NOT scan all users' memory — that surfaces
            # stored facts as if they're confirmed truth.
            if any(t in lower for t in _TITLE_TRIGGERS):
                cross = await self._step(
                    trace, "cross_user_search",
                    self._run_sync(search_memory_by_value, ctx.content)
                )
                if cross:
                    ctx.relationships = (ctx.relationships + "\n" + cross).strip()
        else:
            trace.add("relationships", 0, "skipped")

        # Reflection
        reflection = await self._step(
            trace, "reflection",
            self._run_sync(get_reflection, ctx.uid_str)
        )
        ctx.reflection = reflection or ""

        from .presence import get_presence_patterns
        presence_patterns = await self._step(
            trace, "presence_patterns",
            self._run_sync(get_presence_patterns, ctx.uid_str)
        )
        ctx.presence_patterns = presence_patterns or ""

    async def step_search(self, ctx: AgentContext, trace: AgentTrace):
        """Step 3: Web search if the message needs real-time info."""
        from .search import needs_web_search, web_search, build_search_query

        if not needs_web_search(ctx.content) or ctx.mentioned_users:
            trace.add("search", 0, "skipped")
            return

        query = build_search_query(ctx.content)
        result = await self._step(
            trace, "search",
            self._run_sync(web_search, query)
        )
        ctx.web_context = result or ""
        if ctx.web_context:
            log.debug(f"[agent] search fetched context for: {query!r}")

    async def step_session(self, ctx: AgentContext, trace: AgentTrace):
        """Step 4: Update and fetch conversation state."""
        from .session import add_message, should_refresh, analyze_state, get_state_prompt

        t0 = time.perf_counter()
        add_message(ctx.user_id, ctx.content)
        if should_refresh(ctx.user_id):
            await self._run_sync(analyze_state, ctx.user_id)
        ctx.session_context = get_state_prompt(ctx.user_id)
        ms = (time.perf_counter() - t0) * 1000
        trace.add("session", ms, "ok", ctx.session_context[:60] if ctx.session_context else "empty")

    async def step_user_state(self, ctx: AgentContext, trace: AgentTrace):
        """Build one unified runtime profile for generation."""
        from .user_state import build_user_state_from_context, compress_user_state

        t0 = time.perf_counter()
        ctx.user_state = build_user_state_from_context(ctx)
        ctx.user_state_summary = compress_user_state(ctx.user_state, ctx.username)
        ms = (time.perf_counter() - t0) * 1000
        detail = f"{len(ctx.user_state_summary)} chars" if ctx.user_state_summary else "empty"
        trace.add("user_state", ms, "ok", detail)

    async def step_generate(self, ctx: AgentContext, trace: AgentTrace) -> str | None:
        """
        Step 5: Generate the reply.
        Internally runs: intent analysis → plan → draft → self-critique.
        """
        from .ai import ai_chat

        result = await self._step(
            trace, "generate",
            self._run_sync(
                ai_chat,
                ctx.history, ctx.memory,
                ctx.username, ctx.user_id,
                ctx.relationships, ctx.web_context,
                ctx.impersonation_context, ctx.feedback_context,
                ctx.channel_name, ctx.session_context,
                ctx.reflection, ctx.emotion_hint, ctx.emotion_state,
                ctx.user_activity, ctx.user_status, ctx.user_state_summary,
            )
        )
        return result

    async def step_post(self, ctx: AgentContext, trace: AgentTrace):
        """Step 6: Pick GIF based on message emotion."""
        from .emotion import pick_gif_for_message

        t0 = time.perf_counter()
        gif_url, gif_emotion = await self._run_sync(
            pick_gif_for_message, ctx.content, ctx.reply
        )
        ctx.gif_url = gif_url
        ctx.gif_emotion = gif_emotion
        ms = (time.perf_counter() - t0) * 1000
        trace.add("post", ms, "ok", gif_emotion or "no gif")

    # ── Main run ─────────────────────────────────────────────────────────── #

    async def run(self, ctx: AgentContext) -> tuple[AgentContext, AgentTrace]:
        """
        Execute the full pipeline. Returns the populated context and trace.
        Steps are run sequentially. A failed step leaves its output empty
        and the pipeline continues — degraded but not crashed.
        """
        trace = AgentTrace()
        t_start = time.perf_counter()

        await self.step_classify(ctx, trace)
        await self.step_memory(ctx, trace)
        await self.step_search(ctx, trace)
        await self.step_session(ctx, trace)
        await self.step_user_state(ctx, trace)

        reply = await self.step_generate(ctx, trace)
        ctx.reply = reply or ""

        if ctx.reply:
            await self.step_post(ctx, trace)
        else:
            trace.add("post", 0, "skipped", "no reply")

        trace.total_ms = (time.perf_counter() - t_start) * 1000
        log.info(f"[agent] user={ctx.uid_str} {trace.summary()}")
        return ctx, trace

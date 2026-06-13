"""
Agent Loop — the explicit orchestrator for Corsbot's response pipeline.

Pipeline:
  1. classify     — emotion state classification + momentum tracking
  2. memory       — retrieve facts, relationships, reflection, presence
  3. search       — web search if real-time info is needed
  4. session      — update + fetch conversation state
  5. user_state   — build unified runtime user profile
  6. reason       — synthesize what's known before drafting (prevents hallucination)
  7. intent       — classify what the user actually needs
  8. plan         — turn intent + reasoning into a concrete response plan
  9. draft        — generate reply with plan injected as guidance
  10. safety      — verify draft against intent/plan, rewrite if it fails
  11. post        — pick GIF based on emotion

Steps 6-10 only run for non-trivial routes (default, empathy, search).
Banter/fast routes skip planning and go straight to draft.

Each step is named, timed, and isolated — a failure degrades gracefully.
"""

import asyncio
import logging
import time
from concurrent.futures import Executor
from dataclasses import dataclass, field
from typing import Any

from core.logger import get_logger

log = get_logger("corsbot.agent")


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
    user_status: str = ""  # online, idle, offline, dnd

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
    conversation_summary: str = ""  # rolling summary of older messages
    task_mode: str = ""
    task_clues: list[str] = field(default_factory=list)
    search_query: str = ""
    user_state: dict = field(default_factory=dict)
    user_state_summary: str = ""
    feedback_context: str = ""
    impersonation_context: str = ""
    history: list = field(default_factory=list)
    route: Any = None

    # Final output
    reply: str = ""
    response_plan: str = ""  # plan produced by step_generate, stored for debugging
    server_members: str = ""  # compact member list for small servers
    gif_url: str | None = None
    gif_emotion: str | None = None


# ── Trace ─────────────────────────────────────────────────────────────────── #


@dataclass
class StepTrace:
    name: str
    duration_ms: float
    status: str  # "ok" | "skipped" | "failed"
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

    def __init__(
        self, executor: Executor, event_loop: asyncio.AbstractEventLoop | None = None
    ):
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
        from .emotion import (
            apply_emotional_momentum,
            classify_emotion,
            get_emotion_style_hint,
        )

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
            get_memory,
            get_memory_with_keys,
            get_reflection,
            get_relationships,
            search_memory_by_value,
        )
        from .presence import get_presence_patterns

        mem_result, reflection, presence_patterns = await asyncio.gather(
            self._step(
                trace,
                "memory",
                self._run_sync(get_memory_with_keys, ctx.user_id, ctx.content, 3),
            ),
            self._step(
                trace, "reflection", self._run_sync(get_reflection, ctx.uid_str)
            ),
            self._step(
                trace,
                "presence_patterns",
                self._run_sync(get_presence_patterns, ctx.uid_str),
            ),
        )

        if mem_result:
            ctx.memory, ctx.active_keys = mem_result
        ctx.reflection = reflection or ""
        ctx.presence_patterns = presence_patterns or ""

        # Impersonation: swap memory for target user's facts
        if ctx.is_impersonating and ctx.mentioned_users:
            target_id = next(iter(ctx.mentioned_users))
            target_name = ctx.mentioned_users[target_id]
            target_mem = await self._step(
                trace, "impersonate_memory", self._run_sync(get_memory, target_id)
            )
            if target_mem:
                ctx.impersonation_context = f"Facts about {target_name} to help you impersonate them:\n{target_mem}"
            ctx.memory = ""
            ctx.active_keys = []
        elif ctx.is_impersonating:
            # Fictional character — clear user memory, inject character roleplay hint
            ctx.memory = ""
            ctx.active_keys = []
            lower = ctx.content.lower()
            from handlers import MessageHandler

            for kw in MessageHandler.IMPERSONATE_KEYWORDS:
                if kw in lower:
                    after = lower.split(kw, 1)[-1].strip()
                    words = after.split()
                    char_name = words[0].rstrip(".,!?") if words else ""
                    if char_name:
                        ctx.impersonation_context = (
                            f"You are now roleplaying as {char_name}. "
                            f"Use your knowledge of this character — their personality, speech style, "
                            f"catchphrases, and mannerisms. Stay fully in character. "
                            f"Do NOT mix them up with other characters."
                        )
                    break

        # Relationships — only retrieve when the user is explicitly asking about a person
        from .memory import get_relationship_names

        _RELATIONSHIP_TRIGGERS = (
            "who is",
            "tell me about",
            "what about",
            "how is",
            "where is",
            "what's with",
            "who's",
            "whos",
            "do you know",
            "what do you know about",
            "anything about",
            "info on",
            "info about",
            "what can you tell me about",
            "who tf is",
            "who da hell is",
            "who the hell is",
            "who are they",
            "what's their deal",
            "whats their deal",
            "who even is",
            "what's up with",
            "whats up with",
            "what happened to",
            "how's",
            "hows",
            "where's",
            "wheres",
            "what's going on with",
            "whats going on with",
            "you know",
            "you know about",
            "heard of",
            "heard about",
            "know anything about",
            "what do you think of",
            "what do you think about",
            "who dat",
            "who dat is",
            "who dis",
            "who is dis",
            "kinsa",
            "kinsa si",
            "kinsa ang",
        )
        _TITLE_TRIGGERS = (
            "who is the",
            "who is king",
            "who is lord",
            "who is boss",
            "who is queen",
            "who is god",
            "who is goat",
            "who is legend",
            "who da king",
            "who da boss",
            "who da goat",
            "who da god",
            "who da best",
            "who da real",
            "who da one",
            "who da legend",
            "who declared",
            "who said they",
            "who called themselves",
            "who holds",
            "who owns",
            "who got the title",
            "who is titled",
            "who's the",
            "whos the",
            "who's da",
            "whos da",
            "kinsa ang",
        )
        lower = ctx.content.lower()
        has_trigger = any(t in lower for t in _RELATIONSHIP_TRIGGERS)

        if has_trigger:
            rels = await self._step(
                trace, "relationships", self._run_sync(get_relationships, ctx.user_id)
            )
            ctx.relationships = rels or ""
            # Cross-user search: only for role/title queries, not person-name lookups.
            # "who is dimples" should NOT scan all users' memory — that surfaces
            # stored facts as if they're confirmed truth.
            if any(t in lower for t in _TITLE_TRIGGERS):
                cross = await self._step(
                    trace,
                    "cross_user_search",
                    self._run_sync(search_memory_by_value, ctx.content),
                )
                if cross:
                    ctx.relationships = (ctx.relationships + "\n" + cross).strip()
        else:
            trace.add("relationships", 0, "skipped")

    async def step_search(self, ctx: AgentContext, trace: AgentTrace):
        """Step 3: Web search if the message needs real-time info."""
        from .search import (
            build_search_query,
            build_song_search_query_from_clues,
            collect_song_clues,
            extract_song_artist_hint,
            is_song_followup_message,
            is_song_identification_turn,
            merge_song_clue_lines,
            needs_web_search,
            web_search,
        )
        from .session import (
            clear_song_task_state,
            get_song_task_state,
            update_song_task_state,
        )

        active_song = get_song_task_state(ctx.user_id)
        song_guess_mode = (
            is_song_identification_turn(ctx.content, ctx.history)
            or (active_song.active and is_song_followup_message(ctx.content))
        )

        if (
            not needs_web_search(ctx.content) and not song_guess_mode
        ) or ctx.mentioned_users:
            if active_song.active and not song_guess_mode:
                clear_song_task_state(ctx.user_id)
            trace.add("search", 0, "skipped")
            return

        if song_guess_mode:
            ctx.task_mode = "song_guess"
            new_clues = collect_song_clues(ctx.content, ctx.history)
            ctx.task_clues = (
                merge_song_clue_lines(active_song.clues, new_clues, max_lines=8)
                if active_song.active
                else new_clues
            )
            artist_hint = active_song.artist_hint or extract_song_artist_hint(
                ctx.content, ctx.history
            )
            query = build_song_search_query_from_clues(ctx.task_clues, artist_hint)
            if not query:
                query = build_search_query(ctx.content)
            if ctx.task_clues or artist_hint:
                update_song_task_state(
                    ctx.user_id,
                    ctx.task_clues,
                    artist_hint=artist_hint,
                )
            else:
                clear_song_task_state(ctx.user_id)
        else:
            query = build_search_query(ctx.content)
            if active_song.active:
                clear_song_task_state(ctx.user_id)
        ctx.search_query = query
        t0 = time.perf_counter()
        try:
            result = await web_search(query)
            ctx.web_context = result or ""
            ms = (time.perf_counter() - t0) * 1000
            trace.add("search", ms, "ok", query[:60])
            if ctx.web_context:
                log.debug(f"[agent] search fetched context for: {query!r}")
        except Exception as e:
            ms = (time.perf_counter() - t0) * 1000
            trace.add("search", ms, "error", str(e))
            log.warning("search_failed", query=query, error=str(e))

    async def step_session(
        self, ctx: AgentContext, trace: AgentTrace, fetch_context: bool = True
    ):
        """Step 4: Update conversation state + fetch rolling summary."""
        from .session import (
            add_message,
            analyze_state,
            get_state_prompt,
            should_refresh,
        )
        from .summarizer import (
            get_history_with_summary,
            should_summarize,
            summarize_thread,
        )

        t0 = time.perf_counter()
        add_message(ctx.user_id, ctx.content)

        if not fetch_context:
            ms = (time.perf_counter() - t0) * 1000
            trace.add("session", ms, "ok", "store-only")
            return

        # Trigger summarization in background if threshold reached
        if should_summarize(ctx.thread_id):
            asyncio.ensure_future(
                self.loop.run_in_executor(
                    self.executor, summarize_thread, ctx.thread_id
                )
            )

        if should_refresh(ctx.user_id):
            await self._run_sync(analyze_state, ctx.user_id)

        ctx.session_context = get_state_prompt(ctx.user_id)

        # Replace raw history with summary + recent messages
        summary, recent = await self._run_sync(get_history_with_summary, ctx.thread_id)
        ctx.conversation_summary = summary
        if recent:
            ctx.history = recent  # override the full history passed in

        ms = (time.perf_counter() - t0) * 1000
        summary_note = f"summary={len(summary)}chars" if summary else "no_summary"
        trace.add("session", ms, "ok", f"{summary_note} recent={len(ctx.history)}msgs")

    async def step_user_state(self, ctx: AgentContext, trace: AgentTrace):
        """Build one unified runtime profile for generation."""
        from .user_state import build_user_state_from_context, compress_user_state

        t0 = time.perf_counter()
        ctx.user_state = build_user_state_from_context(ctx)
        ctx.user_state_summary = compress_user_state(ctx.user_state, ctx.username)
        ms = (time.perf_counter() - t0) * 1000
        detail = (
            f"{len(ctx.user_state_summary)} chars"
            if ctx.user_state_summary
            else "empty"
        )
        trace.add("user_state", ms, "ok", detail)

    async def step_generate(self, ctx: AgentContext, trace: AgentTrace) -> str | None:
        """
        Steps 5-9: reason → plan → draft → safety_check → rewrite → respond.

        Each sub-step is traced individually. Failures degrade gracefully —
        a failed reason/plan still produces a draft, a failed safety check
        still returns the original draft.
        """
        from .ai import (
            _analyze_and_plan,
            _reason,
            _should_plan,
            _verify_reply,
            ai_chat,
            route_message,
        )

        last_user_msg = next(
            (e["content"] for e in reversed(ctx.history) if e["role"] == "user"),
            ctx.content,
        )
        route = ctx.route or route_message(
            last_user_msg,
            ctx.emotion_state,
            bool(ctx.web_context),
            history_len=len(ctx.history),
        )
        do_plan = (
            _should_plan(route, last_user_msg, ctx.emotion_state)
            or ctx.task_mode == "song_guess"
        )

        task_hint = ""
        if ctx.task_mode == "song_guess":
            lines = "\n".join(f"- {line}" for line in ctx.task_clues[:6])
            task_hint = (
                "Active task: identify a song from lyric clues across multiple user turns. "
                "Stay on that task instead of treating each short line literally. "
                "Use the clue bundle, any artist hint, and any web results. "
                "If uncertain, give one best guess only and ask for one more lyric line."
            )
            if lines:
                task_hint += f"\nRecent clue lines:\n{lines}"

        # ── Steps 5-7: Reason + analyze/plan (parallel where possible) ──
        reasoning = ""
        analysis = ""
        plan = ""
        if do_plan:
            has_context = any(
                [ctx.memory, ctx.relationships, ctx.web_context, ctx.reflection]
            )
            analysis_input = last_user_msg
            if task_hint:
                analysis_input += "\n\n" + task_hint

            t0 = time.perf_counter()
            try:
                coros = []
                if has_context:
                    coros.append(
                        self._run_sync(
                            _reason,
                            last_user_msg,
                            ctx.memory,
                            ctx.relationships,
                            ctx.web_context,
                            ctx.reflection,
                            ctx.emotion_state,
                        )
                    )
                else:
                    coros.append(asyncio.sleep(0))

                coros.append(
                    self._run_sync(
                        _analyze_and_plan,
                        analysis_input,
                        ctx.emotion_state,
                        ctx.session_context,
                        ctx.emotion_hint,
                        ctx.reflection,
                    )
                )

                results = await asyncio.gather(*coros, return_exceptions=True)

                if has_context:
                    reasoning = results[0] if isinstance(results[0], str) else ""
                    plan_result = results[1] if isinstance(results[1], tuple) else ("", "")
                else:
                    plan_result = results[1] if isinstance(results[1], tuple) else ("", "")

                analysis, plan = plan_result
                if reasoning and plan:
                    plan = f"Reasoning:\n{reasoning}\n\n{plan}"

                ms = (time.perf_counter() - t0) * 1000
                trace.add(
                    "reason", ms, "ok", reasoning[:60] if reasoning else "skipped"
                )
                trace.add("intent", ms, "ok", analysis[:60] if analysis else "empty")
                trace.add("plan", ms, "ok", plan[:60] if plan else "empty")
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000
                trace.add("reason", ms, "failed", str(e)[:60])
                trace.add("intent", ms, "failed", str(e)[:60])
                trace.add("plan", ms, "failed", str(e)[:60])
        else:
            trace.add("reason", 0, "skipped")
            trace.add("intent", 0, "skipped")
            trace.add("plan", 0, "skipped")

        # ── Step 8: Draft ────────────────────────────────────────────────
        # Generate the actual reply, injecting the plan as guidance.
        response_plan = plan
        if task_hint:
            response_plan = (task_hint + ("\n\n" + plan if plan else "")).strip()
        ctx.response_plan = response_plan

        result = await self._step(
            trace,
            "draft",
            self._run_sync(
                ai_chat,
                ctx.history,
                ctx.memory,
                ctx.username,
                ctx.user_id,
                ctx.relationships,
                ctx.web_context,
                ctx.impersonation_context,
                ctx.feedback_context,
                ctx.channel_name,
                ctx.session_context,
                ctx.reflection,
                ctx.emotion_hint,
                ctx.emotion_state,
                ctx.user_activity,
                ctx.user_status,
                ctx.user_state_summary,
                response_plan,
                ctx.conversation_summary,
                ctx.server_members,
            ),
        )
        if not result:
            return None

        # ── Step 9: Human check + rewrite ────────────────────────────────
        # Skip for low-stakes default-route replies — saves one LLM round trip.
        skip_safety = route.route == "default" and (
            len(last_user_msg.split()) < 12
            and "emotional_weight: high" not in (analysis or "").lower()
            and "response_type: empathy" not in (analysis or "").lower()
        )
        if do_plan and analysis and not skip_safety:
            t0 = time.perf_counter()
            try:
                verified = await self._run_sync(
                    _verify_reply,
                    result,
                    analysis,
                    response_plan,
                )
                ms = (time.perf_counter() - t0) * 1000
                rewritten = verified != result
                trace.add("safety", ms, "ok", "rewritten" if rewritten else "pass")
                result = verified
            except Exception as e:
                ms = (time.perf_counter() - t0) * 1000
                trace.add("safety", ms, "failed", str(e)[:60])
                log.warning("safety_check_failed", user_id=ctx.user_id, error=str(e))
        else:
            trace.add("safety", 0, "skipped")

        return result

    async def step_post(self, ctx: AgentContext, trace: AgentTrace | None = None):
        """Pick GIF based on message emotion. Call after the text reply is sent."""
        from .emotion import pick_gif_for_message

        t0 = time.perf_counter()
        try:
            gif_url, gif_emotion = await pick_gif_for_message(ctx.content, ctx.reply)
            ctx.gif_url = gif_url
            ctx.gif_emotion = gif_emotion
        except Exception as e:
            ctx.gif_url = None
            ctx.gif_emotion = None
            log.warning("gif_failed", user_id=ctx.user_id, error=str(e))
        ms = (time.perf_counter() - t0) * 1000
        if trace is not None:
            trace.add("post", ms, "ok", ctx.gif_emotion or "no gif")

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
        await self.step_search(ctx, trace)

        from .ai import route_message

        ctx.route = route_message(
            ctx.content,
            ctx.emotion_state,
            bool(ctx.web_context),
            history_len=len(ctx.history),
        )
        trace.add("route", 0, "ok", ctx.route.route)

        if ctx.route.route == "fast" and not ctx.is_impersonating:
            trace.add("memory", 0, "skipped", "fast route")
            await self.step_session(ctx, trace, fetch_context=False)
            trace.add("user_state", 0, "skipped", "fast route")
        else:
            await asyncio.gather(
                self.step_memory(ctx, trace),
                self.step_session(ctx, trace),
            )
            await self.step_user_state(ctx, trace)

        reply = await self.step_generate(ctx, trace)
        ctx.reply = reply or ""

        if ctx.reply:
            from .session import add_bot_message

            await self._run_sync(add_bot_message, ctx.user_id, ctx.reply)
        else:
            trace.add("post", 0, "skipped", "no reply")

        trace.total_ms = (time.perf_counter() - t_start) * 1000
        log.info(
            "agent_run_complete",
            user_id=ctx.user_id,
            guild_id=ctx.guild_id,
            latency_ms=round(trace.total_ms),
            emotion=ctx.emotion_state,
            memory_keys=ctx.active_keys,
            web_search=bool(ctx.web_context),
            gif=bool(ctx.gif_url),
            trace=trace.summary(),
        )
        return ctx, trace

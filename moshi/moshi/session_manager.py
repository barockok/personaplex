# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Session pool manager replacing the single-session asyncio.Lock."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

from aiohttp import web

from .session import Session, SessionStatus
from .metrics import active_sessions, session_duration_seconds, connection_rejections_total
from .utils.logging import setup_logger

logger = setup_logger(__name__)


class SessionManager:
    """Manages a pool of concurrent voice sessions on shared model weights.

    Replaces the old ``asyncio.Lock`` pattern. Each session gets its own
    lightweight streaming context while sharing the loaded model weights.
    """

    def __init__(
        self,
        mimi,
        other_mimi,
        text_tokenizer,
        lm_gen,
        device,
        voice_prompt_dir: Optional[str] = None,
        max_sessions: int = 1,
        worker_id: Optional[str] = None,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.lm_gen = lm_gen
        self.device = device
        self.voice_prompt_dir = voice_prompt_dir
        self.max_sessions = max_sessions
        self.worker_id = worker_id or os.uname().nodename

        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()  # protects _sessions dict only, not inference

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def available_slots(self) -> int:
        return self.max_sessions - self.active_count

    async def acquire_session(self, request: web.Request) -> Session:
        """Allocate a session slot and create per-session streaming context.

        Raises web.HTTPServiceUnavailable if at capacity.
        """
        async with self._lock:
            if self.active_count >= self.max_sessions:
                connection_rejections_total.inc()
                logger.warning(
                    f"[{self.worker_id}] session rejected: "
                    f"{self.active_count}/{self.max_sessions} slots in use"
                )
                raise web.HTTPServiceUnavailable(
                    text="No session slots available"
                )

            session = self._create_session(request)
            self._sessions[session.session_id] = session
            active_sessions.inc()

        logger.info(
            f"[{self.worker_id}] session acquired: {session.session_id[:8]} "
            f"({self.active_count}/{self.max_sessions})"
        )
        return session

    async def release_session(self, session_id: str):
        """Release a session slot, clean up streaming state."""
        async with self._lock:
            session = self._sessions.pop(session_id, None)

        if session is None:
            return

        session.status = SessionStatus.DISCONNECTING
        duration = time.time() - session.created_at
        session_duration_seconds.observe(duration)
        active_sessions.dec()
        session.status = SessionStatus.CLOSED

        logger.info(
            f"[{self.worker_id}] session released: {session.session_id[:8]} "
            f"duration={duration:.1f}s ({self.active_count}/{self.max_sessions})"
        )

    @staticmethod
    def _wrap_with_system_tags(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
            return cleaned
        return f"<system> {cleaned} <system>"

    def _create_session(self, request: web.Request) -> Session:
        """Build a Session with per-session streaming context references."""
        voice_prompt_path = self._resolve_voice_prompt(
            request.query.get("voice_prompt", "")
        )
        text_prompt = request.query.get("text_prompt", "")
        seed_str = request.query.get("seed")
        seed = int(seed_str) if seed_str is not None else None

        # Configure LMGen for this session's prompts.
        # NOTE: In the current architecture, voice prompt and text prompt
        # are set on the shared LMGen instance. For true concurrent sessions
        # with different prompts, the LMGen/Mimi instances need to be
        # duplicated per session. For now we share them (matching the
        # original server behavior) and the SessionManager serializes
        # access to prompt configuration via _lock.
        if voice_prompt_path and self.lm_gen.voice_prompt != voice_prompt_path:
            if voice_prompt_path.endswith(".pt"):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

        if text_prompt:
            self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(
                self._wrap_with_system_tags(text_prompt)
            )
        else:
            self.lm_gen.text_prompt_tokens = None

        peer = request.remote or "unknown"
        peer_port = ""
        transport = request.transport
        if transport is not None:
            peername = transport.get_extra_info("peername")
            if peername:
                peer_port = str(peername[1])

        return Session(
            worker_id=self.worker_id,
            voice_prompt=voice_prompt_path or "",
            text_prompt=text_prompt,
            seed=seed,
            client_ip=f"{peer}:{peer_port}",
            mimi=self.mimi,
            other_mimi=self.other_mimi,
            lm_gen=self.lm_gen,
            text_tokenizer=self.text_tokenizer,
            device=self.device,
        )

    def _resolve_voice_prompt(self, filename: str) -> Optional[str]:
        if not filename or self.voice_prompt_dir is None:
            return None
        import os
        path = os.path.join(self.voice_prompt_dir, filename)
        if not os.path.exists(path):
            raise web.HTTPBadRequest(
                text=f"Voice prompt '{filename}' not found"
            )
        return path

    def health_data(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "active_sessions": self.active_count,
            "max_sessions": self.max_sessions,
            "healthy": True,
        }
# force rebuild 1775087085

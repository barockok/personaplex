# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Per-connection voice session with isolated streaming state."""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import random

import aiohttp
from aiohttp import web
import numpy as np
import sphn
import torch

from .utils.logging import setup_logger, ColorizedLog

logger = setup_logger(__name__)


def _seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class SessionStatus(enum.Enum):
    CONNECTING = "connecting"
    ACTIVE = "active"
    DISCONNECTING = "disconnecting"
    CLOSED = "closed"


@dataclass
class Session:
    """Represents one active voice conversation.

    Each session holds its own streaming context (Mimi encoder/decoder
    state, LMGen state, audio buffers) on top of shared model weights.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    worker_id: str = ""
    voice_prompt: str = ""
    text_prompt: str = ""
    seed: Optional[int] = None
    status: SessionStatus = SessionStatus.CONNECTING
    created_at: float = field(default_factory=time.time)
    client_ip: str = ""

    # Per-session model components (set by SessionManager.acquire_session)
    mimi: object = field(default=None, repr=False)
    other_mimi: object = field(default=None, repr=False)
    lm_gen: object = field(default=None, repr=False)
    text_tokenizer: object = field(default=None, repr=False)
    device: object = field(default=None, repr=False)

    def _frame_size(self) -> int:
        return int(self.mimi.sample_rate / self.mimi.frame_rate)

    async def handle_chat(self, ws: web.WebSocketResponse):
        """Run the voice streaming loop for this session.

        Extracted from the original ServerState.handle_chat, but
        operates on this session's own streaming context.
        """
        clog = ColorizedLog.randomize()
        clog.log("info", f"[{self.session_id[:8]}] session starting")

        self.status = SessionStatus.ACTIVE
        frame_size = self._frame_size()

        if self.seed is not None and self.seed != -1:
            _seed_all(self.seed)

        opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()

        close = False

        async def is_alive():
            if close or ws.closed:
                return False
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    return False
            except asyncio.TimeoutError:
                return True
            except aiohttp.ClientConnectionError:
                return False
            return True

        # Process system prompts (voice + text)
        await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
        self.mimi.reset_streaming()
        clog.log("info", f"[{self.session_id[:8]}] system prompts done")

        if not await is_alive():
            return

        # Send handshake
        await ws.send_bytes(b"\x00")
        clog.log("info", f"[{self.session_id[:8]}] handshake sent")

        async def recv_loop():
            nonlocal close
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.ERROR:
                        clog.log("error", f"[{self.session_id[:8]}] {ws.exception()}")
                        break
                    elif message.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                    ):
                        break
                    elif message.type != aiohttp.WSMsgType.BINARY:
                        continue
                    data = message.data
                    if not isinstance(data, bytes) or len(data) == 0:
                        continue
                    kind = data[0]
                    if kind == 1:  # audio
                        opus_reader.append_bytes(data[1:])
            finally:
                close = True

        async def opus_loop():
            all_pcm_data = None
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))
                while all_pcm_data.shape[-1] >= frame_size:
                    chunk = all_pcm_data[:frame_size]
                    all_pcm_data = all_pcm_data[frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)
                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                        if tokens is None:
                            continue
                        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await ws.send_bytes(msg)

        async def send_loop():
            while True:
                if close:
                    return
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await ws.send_bytes(b"\x01" + msg)

        tasks = [
            asyncio.create_task(recv_loop(), name="recv"),
            asyncio.create_task(opus_loop(), name="opus"),
            asyncio.create_task(send_loop(), name="send"),
        ]
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc:
                clog.log("error", f"[{self.session_id[:8]}] {task.get_name()} crashed: {exc}")
            else:
                clog.log("info", f"[{self.session_id[:8]}] {task.get_name()} finished")
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        clog.log("info", f"[{self.session_id[:8]}] session ended")

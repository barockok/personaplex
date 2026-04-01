# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.


# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json as json_mod
import os
from pathlib import Path
import random
import secrets
import sys
import tarfile
from typing import Literal, Optional

import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
import torch

from .client_utils import make_log, colorize
from .metrics import metrics_handler
from .models import loaders, MimiModel, LMModel, LMGen
from .session_manager import SessionManager
from .utils.connection import create_ssl_context, get_lan_ip
from .utils.logging import setup_logger, ColorizedLog


logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


def torch_auto_device(requested: Optional[DeviceString] = None) -> torch.device:
    """Return a torch.device based on the requested string or availability."""
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def wrap_with_system_tags(text: str) -> str:
    """Add system tags as the model expects if they are missing."""
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


async def handle_chat(request: web.Request) -> web.WebSocketResponse:
    """Handle a voice session WebSocket connection.

    Acquires a session slot from the SessionManager, delegates to
    the Session's handle_chat method, and releases the slot on
    completion or error.
    """
    session_manager: SessionManager = request.app["session_manager"]

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    session = await session_manager.acquire_session(request)
    try:
        await session.handle_chat(ws)
    finally:
        await session_manager.release_session(session.session_id)
        await ws.close()

    return ws


async def handle_health(request: web.Request) -> web.Response:
    """Return worker health status as JSON."""
    session_manager: SessionManager = request.app["session_manager"]
    return web.json_response(session_manager.health_data())


def _get_voice_prompt_dir(voice_prompt_dir: Optional[str], hf_repo: str) -> Optional[str]:
    """Download and extract voice prompts if not provided locally."""
    if voice_prompt_dir is not None:
        return voice_prompt_dir

    logger.info("retrieving voice prompts")
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"

    if not voices_dir.exists():
        logger.info(f"extracting {voices_tgz} to {voices_dir}")
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)

    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a 'voices/' directory")

    return str(voices_dir)


def _get_static_path(static: Optional[str]) -> Optional[str]:
    if static is None:
        logger.info("retrieving the static content")
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    elif static != "none":
        return static
    return None


def _build_mock_session_manager(args) -> SessionManager:
    """Build a SessionManager with mock backends (no GPU, no model download)."""
    from .mock_backend import MockMimiModel, MockLMGen, MockTextTokenizer

    mimi = MockMimiModel()
    other_mimi = MockMimiModel()
    lm_gen = MockLMGen()
    text_tokenizer = MockTextTokenizer()

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    return SessionManager(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        device=torch.device("cpu"),
        voice_prompt_dir=None,
        max_sessions=args.max_sessions,
        worker_id=args.worker_id,
    )


def _build_real_session_manager(args) -> SessionManager:
    """Build a SessionManager with real model weights."""
    args.voice_prompt_dir = _get_voice_prompt_dir(
        args.voice_prompt_dir,
        args.hf_repo,
    )
    if args.voice_prompt_dir is not None:
        assert os.path.exists(args.voice_prompt_dir), \
            f"Directory missing: {args.voice_prompt_dir}"
    logger.info(f"voice_prompt_dir = {args.voice_prompt_dir}")

    args.device = torch_auto_device(args.device)
    seed_all(42424242)

    # Download config.json to increment download counter
    hf_hub_download(args.hf_repo, "config.json")

    logger.info("loading mimi")
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, args.device)
    other_mimi = loaders.get_mimi(args.mimi_weight, args.device)
    logger.info("mimi loaded")

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    logger.info("loading moshi")
    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(args.moshi_weight, device=args.device, cpu_offload=args.cpu_offload)
    lm.eval()
    logger.info("moshi loaded")

    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=args.device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    session_manager = SessionManager(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm_gen=lm_gen,
        device=args.device,
        voice_prompt_dir=args.voice_prompt_dir,
        max_sessions=args.max_sessions,
        worker_id=args.worker_id,
    )

    # Warmup
    logger.info("warming up the model")
    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    for _ in range(4):
        chunk = torch.zeros(1, 1, frame_size, dtype=torch.float32, device=args.device)
        codes = mimi.encode(chunk)
        _ = other_mimi.encode(chunk)
        for c in range(codes.shape[-1]):
            tokens = lm_gen.step(codes[:, :, c: c + 1])
            if tokens is None:
                continue
            _ = mimi.decode(tokens[:, 1:9])
            _ = other_mimi.decode(tokens[:, 1:9])
    if args.device.type == "cuda":
        torch.cuda.synchronize()

    return session_manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action="store_true",
                        help="Activate a gradio tunnel.")
    parser.add_argument("--gradio-tunnel-token",
                        help="Provide a custom (secret) token here to keep getting the same URL.")

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults PersonaPlex.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload LM model layers to CPU when GPU memory is insufficient.")
    parser.add_argument("--voice-prompt-dir", type=str,
                        help="Directory containing voice prompt files.")
    parser.add_argument("--ssl", type=str,
                        help="Directory with key.pem and cert.pem for HTTPS.")

    # New multi-session flags
    parser.add_argument("--max-sessions", type=int, default=1,
                        help="Maximum concurrent voice sessions (default: 1).")
    parser.add_argument("--worker-id", type=str, default=None,
                        help="Worker identifier for health/metrics (default: hostname).")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock inference backend (no GPU, no model download).")

    args = parser.parse_args()

    # Build session manager
    if args.mock:
        logger.info("starting in mock mode (no GPU required)")
        session_manager = _build_mock_session_manager(args)
    else:
        session_manager = _build_real_session_manager(args)

    logger.info(
        f"session manager ready: worker_id={session_manager.worker_id}, "
        f"max_sessions={session_manager.max_sessions}"
    )

    # Static content
    if not args.mock:
        static_path: Optional[str] = _get_static_path(args.static)
        assert static_path is None or os.path.exists(static_path), \
            f"Static path does not exist: {static_path}."
        logger.info(f"static_path = {static_path}")
    else:
        static_path = None

    # Gradio tunnel
    setup_tunnel = None
    tunnel_token = ""
    if args.gradio_tunnel:
        try:
            from gradio import networking
        except ImportError:
            logger.error("Cannot find gradio. Install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    # Build app
    app = web.Application()
    app["session_manager"] = session_manager
    app.router.add_get("/api/chat", handle_chat)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/metrics", metrics_handler)

    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        logger.info(f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )

    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info(f"Access the Web UI directly at {protocol}://{host_ip}:{args.port}")

    if setup_tunnel is not None:
        tunnel = setup_tunnel("localhost", args.port, tunnel_token, None)
        logger.info(f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")

    web.run_app(app, port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()

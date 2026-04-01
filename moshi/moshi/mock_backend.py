# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Mock inference backend for GPU-free testing of session management.

Provides lightweight stand-ins for MimiModel and LMGen that run on CPU
without loading any model weights. Useful for testing session lifecycle,
concurrency, routing, and metrics without requiring a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


MOCK_SAMPLE_RATE = 24000
MOCK_FRAME_RATE = 12  # 12.5 in real model, use 12 for simplicity
MOCK_FRAME_SIZE = MOCK_SAMPLE_RATE // MOCK_FRAME_RATE
MOCK_NUM_CODEBOOKS = 17
MOCK_DEP_Q = 8


@dataclass
class _MockStreamingState:
    offset: int = 0

    def reset(self):
        self.offset = 0


class MockMimiModel:
    """Lightweight stand-in for MimiModel.

    Encodes audio to fixed codebook tokens and decodes tokens back to
    silence. Supports the same streaming interface (streaming_forever,
    reset_streaming) as the real model.
    """

    def __init__(self):
        self.sample_rate = MOCK_SAMPLE_RATE
        self.frame_rate = MOCK_FRAME_RATE
        self._streaming_state: Optional[_MockStreamingState] = None

    def streaming_forever(self, batch_size: int):
        self._streaming_state = _MockStreamingState()

    def reset_streaming(self):
        if self._streaming_state is not None:
            self._streaming_state.reset()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return fixed codebook indices (all zeros)."""
        B = x.shape[0]
        # Real model returns [B, K, T] where K = num_codebooks - dep_q - 1
        n_codes = MOCK_NUM_CODEBOOKS - MOCK_DEP_Q - 1
        return torch.zeros(B, n_codes, 1, dtype=torch.long)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Return silence audio."""
        B = codes.shape[0]
        return torch.zeros(B, 1, MOCK_FRAME_SIZE, dtype=torch.float32)


class MockLMGen:
    """Lightweight stand-in for LMGen.

    Returns fixed tokens on each step. Supports the same streaming
    interface and voice/text prompt configuration as the real model.
    """

    def __init__(self):
        self._streaming_state: Optional[_MockStreamingState] = None
        self.lm_model = _MockLMModel()
        self.voice_prompt: Optional[str] = None
        self.text_prompt_tokens: Optional[list[int]] = None
        self._frame_rate = MOCK_FRAME_RATE
        self._sample_rate = MOCK_SAMPLE_RATE

    def streaming_forever(self, batch_size: int):
        self._streaming_state = _MockStreamingState()

    def reset_streaming(self):
        if self._streaming_state is not None:
            self._streaming_state.reset()

    def step(self, input_tokens: torch.Tensor) -> Optional[torch.Tensor]:
        """Return fixed output tokens [B, dep_q+1, 1]."""
        state = self._streaming_state
        if state is None:
            raise RuntimeError("Call streaming_forever() before step().")
        state.offset += 1
        if state.offset <= 1:
            return None
        B = input_tokens.shape[0]
        # text token (0=EPAD) + dep_q audio tokens (all 1s)
        tokens = torch.ones(B, MOCK_DEP_Q + 1, 1, dtype=torch.long)
        tokens[:, 0, :] = 0  # EPAD text token
        return tokens

    def load_voice_prompt(self, path: str):
        self.voice_prompt = path

    def load_voice_prompt_embeddings(self, path: str):
        self.voice_prompt = path

    async def step_system_prompts_async(self, mimi, is_alive=None):
        """Mock system prompt processing — returns immediately."""
        pass


class _MockLMModel:
    """Minimal LMModel stand-in for MockLMGen."""

    def __init__(self):
        self.dep_q = MOCK_DEP_Q
        self.device = torch.device("cpu")


class MockTextTokenizer:
    """Stand-in for sentencepiece tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]  # Fixed token sequence

    def id_to_piece(self, token_id: int) -> str:
        return "▁mock"

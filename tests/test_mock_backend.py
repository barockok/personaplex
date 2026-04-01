# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for MockBackend classes in moshi.moshi.mock_backend."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy transitive imports so we can import mock_backend without GPU
# deps. Same pattern as test_session_manager.py.
# ---------------------------------------------------------------------------
_HEAVY_STUBS = [
    "einops",
    "safetensors",
    "safetensors.torch",
    "huggingface_hub",
    "sounddevice",
    "sentencepiece",
    "sphn",
]
for _mod_name in _HEAVY_STUBS:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

import torch

from moshi.moshi.mock_backend import (
    MOCK_DEP_Q,
    MOCK_FRAME_RATE,
    MOCK_FRAME_SIZE,
    MOCK_SAMPLE_RATE,
    MockLMGen,
    MockMimiModel,
    MockTextTokenizer,
)


# ---------------------------------------------------------------------------
# 1. test_mock_mimi_encode_decode
# ---------------------------------------------------------------------------
class TestMockMimiEncodeDecode:
    def test_encode_shape(self):
        mimi = MockMimiModel()
        mimi.streaming_forever(batch_size=1)

        x = torch.zeros(1, 1, MOCK_FRAME_SIZE)
        codes = mimi.encode(x)

        assert codes.dim() == 3
        assert codes.shape[0] == 1  # batch
        assert codes.shape[2] == 1  # time frames

    def test_decode_shape(self):
        mimi = MockMimiModel()
        mimi.streaming_forever(batch_size=1)

        n_codes = codes_dim = mimi.encode(torch.zeros(1, 1, MOCK_FRAME_SIZE)).shape[1]
        codes = torch.zeros(1, n_codes, 1, dtype=torch.long)
        audio = mimi.decode(codes)

        assert audio.dim() == 3
        assert audio.shape[0] == 1  # batch
        assert audio.shape[2] == MOCK_FRAME_SIZE  # frame size


# ---------------------------------------------------------------------------
# 2. test_mock_mimi_reset_streaming
# ---------------------------------------------------------------------------
class TestMockMimiResetStreaming:
    def test_reset_sets_offset_to_zero(self):
        mimi = MockMimiModel()
        mimi.streaming_forever(batch_size=1)

        # Advance internal state by encoding a few frames
        for _ in range(5):
            mimi.encode(torch.zeros(1, 1, MOCK_FRAME_SIZE))

        mimi.reset_streaming()
        assert mimi._streaming_state is not None
        assert mimi._streaming_state.offset == 0


# ---------------------------------------------------------------------------
# 3. test_mock_lm_gen_step
# ---------------------------------------------------------------------------
class TestMockLMGenStep:
    def test_first_step_returns_none_second_returns_tensor(self):
        lm = MockLMGen()
        lm.streaming_forever(batch_size=1)

        input_tokens = torch.zeros(1, MOCK_DEP_Q + 1, 1, dtype=torch.long)

        result_first = lm.step(input_tokens)
        assert result_first is None

        result_second = lm.step(input_tokens)
        assert result_second is not None
        assert isinstance(result_second, torch.Tensor)
        assert result_second.shape == (1, MOCK_DEP_Q + 1, 1)


# ---------------------------------------------------------------------------
# 4. test_mock_lm_gen_step_without_streaming_raises
# ---------------------------------------------------------------------------
class TestMockLMGenStepWithoutStreaming:
    def test_raises_runtime_error(self):
        lm = MockLMGen()
        input_tokens = torch.zeros(1, MOCK_DEP_Q + 1, 1, dtype=torch.long)

        with pytest.raises(RuntimeError, match="streaming_forever"):
            lm.step(input_tokens)


# ---------------------------------------------------------------------------
# 5. test_mock_text_tokenizer
# ---------------------------------------------------------------------------
class TestMockTextTokenizer:
    def test_encode_returns_list_of_ints(self):
        tok = MockTextTokenizer()
        result = tok.encode("hello world")
        assert isinstance(result, list)
        assert all(isinstance(t, int) for t in result)
        assert len(result) > 0

    def test_id_to_piece_returns_string(self):
        tok = MockTextTokenizer()
        piece = tok.id_to_piece(42)
        assert isinstance(piece, str)
        assert len(piece) > 0


# ---------------------------------------------------------------------------
# 6. test_mock_no_cuda_required
# ---------------------------------------------------------------------------
class TestMockNoCudaRequired:
    def test_all_mocks_run_on_cpu(self):
        """Verify none of the mock classes require CUDA by exercising them
        entirely on CPU without errors."""
        # MockMimiModel
        mimi = MockMimiModel()
        mimi.streaming_forever(batch_size=1)
        codes = mimi.encode(torch.zeros(1, 1, MOCK_FRAME_SIZE))
        audio = mimi.decode(codes)
        mimi.reset_streaming()

        # MockLMGen
        lm = MockLMGen()
        lm.streaming_forever(batch_size=1)
        assert lm.lm_model.device == torch.device("cpu")
        lm.step(torch.zeros(1, MOCK_DEP_Q + 1, 1, dtype=torch.long))
        lm.step(torch.zeros(1, MOCK_DEP_Q + 1, 1, dtype=torch.long))
        lm.reset_streaming()

        # MockTextTokenizer
        tok = MockTextTokenizer()
        tok.encode("test")
        tok.id_to_piece(0)


# ---------------------------------------------------------------------------
# 7. test_multiple_mock_contexts_concurrent
# ---------------------------------------------------------------------------
class TestMultipleMockContextsConcurrent:
    def test_independent_streaming_states(self):
        mimi_a = MockMimiModel()
        mimi_b = MockMimiModel()

        mimi_a.streaming_forever(batch_size=1)
        mimi_b.streaming_forever(batch_size=1)

        # Advance mimi_a several steps
        for _ in range(3):
            mimi_a.encode(torch.zeros(1, 1, MOCK_FRAME_SIZE))

        # mimi_b should still be at offset 0 (encode doesn't advance offset
        # in the current impl, so we verify states are separate objects)
        assert mimi_a._streaming_state is not mimi_b._streaming_state

        # Reset one, the other should be unaffected
        mimi_a.reset_streaming()
        assert mimi_a._streaming_state.offset == 0

        # Verify mimi_b still works independently
        codes_b = mimi_b.encode(torch.zeros(1, 1, MOCK_FRAME_SIZE))
        assert codes_b.shape == (1, codes_b.shape[1], 1)

    def test_independent_lm_gen_states(self):
        lm_a = MockLMGen()
        lm_b = MockLMGen()

        lm_a.streaming_forever(batch_size=1)
        lm_b.streaming_forever(batch_size=1)

        input_tokens = torch.zeros(1, MOCK_DEP_Q + 1, 1, dtype=torch.long)

        # Advance lm_a past the initial None step
        lm_a.step(input_tokens)  # returns None (offset=1)
        lm_a.step(input_tokens)  # returns tensor (offset=2)

        # lm_b should still return None on first step
        result_b = lm_b.step(input_tokens)
        assert result_b is None

        # And tensor on second step
        result_b2 = lm_b.step(input_tokens)
        assert result_b2 is not None
        assert result_b2.shape == (1, MOCK_DEP_Q + 1, 1)

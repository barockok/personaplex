"""Tests for the Session dataclass and SessionStatus enum."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch

from moshi.moshi.session import Session, SessionStatus
from moshi.moshi.mock_backend import MockMimiModel, MockLMGen, MockTextTokenizer


def _make_session(**kwargs) -> Session:
    """Helper to build a Session with mock model components."""
    defaults = dict(
        mimi=MockMimiModel(),
        other_mimi=MockMimiModel(),
        lm_gen=MockLMGen(),
        text_tokenizer=MockTextTokenizer(),
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return Session(**defaults)


class TestSessionInitialStatus:
    def test_session_initial_status(self):
        session = _make_session()
        assert session.status is SessionStatus.CONNECTING


class TestSessionStatusTransitions:
    def test_session_status_transitions(self):
        session = _make_session()

        assert session.status is SessionStatus.CONNECTING

        session.status = SessionStatus.ACTIVE
        assert session.status is SessionStatus.ACTIVE

        session.status = SessionStatus.DISCONNECTING
        assert session.status is SessionStatus.DISCONNECTING

        session.status = SessionStatus.CLOSED
        assert session.status is SessionStatus.CLOSED


class TestSessionUniqueIds:
    def test_session_unique_ids(self):
        s1 = _make_session()
        s2 = _make_session()
        assert s1.session_id != s2.session_id


class TestSessionStateIsolation:
    def test_session_state_isolation(self):
        s1 = _make_session(voice_prompt="voice_a", seed=42)
        s2 = _make_session(voice_prompt="voice_b", seed=99)

        # Different configuration values
        assert s1.voice_prompt != s2.voice_prompt
        assert s1.seed != s2.seed

        # Independent mock model instances (not shared objects)
        assert s1.mimi is not s2.mimi
        assert s1.other_mimi is not s2.other_mimi
        assert s1.lm_gen is not s2.lm_gen
        assert s1.text_tokenizer is not s2.text_tokenizer


class TestSessionStateIsolationSameVoicePrompt:
    def test_session_state_isolation_same_voice_prompt(self):
        shared_prompt = "same_voice"
        s1 = _make_session(voice_prompt=shared_prompt)
        s2 = _make_session(voice_prompt=shared_prompt)

        assert s1.voice_prompt == s2.voice_prompt

        # Even with the same prompt, model instances must be independent
        assert s1.mimi is not s2.mimi
        assert s1.other_mimi is not s2.other_mimi
        assert s1.lm_gen is not s2.lm_gen

        # Mutating streaming state on one session must not affect the other
        s1.mimi.streaming_forever(batch_size=1)
        s1.mimi._streaming_state.offset = 10

        assert s2.mimi._streaming_state is None


class TestSessionResourceCleanup:
    def test_session_resource_cleanup(self):
        session = _make_session()
        ref = weakref.ref(session)

        session.status = SessionStatus.CLOSED

        # Drop the only strong reference
        del session
        gc.collect()

        assert ref() is None, "Session was not garbage collected — possible circular reference"

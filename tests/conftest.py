"""Shared offline test fakes. Tests must never touch the network."""

from __future__ import annotations

import socket
from collections.abc import Sequence
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class NetworkBlockedError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(*args, **kwargs):
        raise NetworkBlockedError("Network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)
    monkeypatch.setattr(socket, "create_connection", guard)
    monkeypatch.setattr(socket, "getaddrinfo", guard)
    import requests

    monkeypatch.setattr(requests.Session, "request", guard)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("OPENAI_API_KEY", "BRAVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


class FakeToolCallingChatModel(BaseChatModel):
    """Chat model returning scripted AIMessages in order; supports bind_tools."""

    responses: list[AIMessage]
    index: int = 0
    calls: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        msg = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


@pytest.fixture
def fake_embeddings():
    return DeterministicFakeEmbedding(size=32)


@pytest.fixture
def fake_chat_model_factory():
    def make(responses: list[AIMessage]) -> FakeToolCallingChatModel:
        return FakeToolCallingChatModel(responses=responses, calls=[])

    return make

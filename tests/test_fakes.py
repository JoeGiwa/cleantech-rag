import pytest
import requests
from langchain_core.messages import AIMessage, HumanMessage


def test_fake_model_tool_call_then_answer(fake_chat_model_factory):
    call = {"name": "search", "args": {"q": "solar"}, "id": "1", "type": "tool_call"}
    model = fake_chat_model_factory([AIMessage(content="", tool_calls=[call]), AIMessage("done")])
    bound = model.bind_tools([])
    first = bound.invoke([HumanMessage("hi")])
    assert first.tool_calls[0]["name"] == "search"
    assert bound.invoke([HumanMessage("hi")]).content == "done"


def test_fake_embeddings(fake_embeddings):
    assert len(fake_embeddings.embed_query("a")) == 32


def test_real_http_call_fails():
    with pytest.raises(Exception, match="blocked"):
        requests.get("http://example.com", timeout=1)

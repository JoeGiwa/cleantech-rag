from langchain_core.messages import AIMessage, HumanMessage


def test_last_response_repeats_when_exhausted(fake_chat_model_factory):
    model = fake_chat_model_factory([AIMessage("only")])
    assert [model.invoke([HumanMessage("x")]).content for _ in range(3)] == ["only"] * 3


def test_calls_not_shared_between_instances(fake_chat_model_factory):
    a = fake_chat_model_factory([AIMessage("a")])
    b = fake_chat_model_factory([AIMessage("b")])
    a.invoke([HumanMessage("x")])
    assert len(a.calls) == 1 and b.calls == []


def test_tool_call_id_and_args_preserved(fake_chat_model_factory):
    call = {"name": "t", "args": {"k": 1}, "id": "abc", "type": "tool_call"}
    m = fake_chat_model_factory([AIMessage("", tool_calls=[call])]).bind_tools([])
    out = m.invoke([HumanMessage("x")])
    assert out.tool_calls[0]["id"] == "abc" and out.tool_calls[0]["args"] == {"k": 1}


def test_fake_embeddings_deterministic(fake_embeddings):
    assert fake_embeddings.embed_query("a") == fake_embeddings.embed_query("a")
    assert fake_embeddings.embed_query("a") != fake_embeddings.embed_query("b")
    assert fake_embeddings.embed_documents([]) == []

import math
import socket

import pytest

from cleantech_rag.evaluation import metrics as m


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def blocked(*a, **k):
        raise AssertionError("network access in test")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def test_precision_divides_by_k():
    assert m.precision_at_k(["a", "b", "c"], ["a"], k=5) == pytest.approx(1 / 5)


def test_precision_retrieved_divides_by_retrieved():
    assert m.precision_at_k_retrieved(["a", "b", "c"], ["a"], k=5) == pytest.approx(1 / 3)


def test_precision_retrieved_uses_k_when_more_retrieved():
    assert m.precision_at_k_retrieved(list("abcdefg"), ["a", "g"], k=5) == pytest.approx(1 / 5)


def test_precision_empty_retrieval():
    assert m.precision_at_k([], ["a"]) == 0.0
    assert m.precision_at_k_retrieved([], ["a"]) == 0.0


def test_precision_truncates_to_k():
    assert m.precision_at_k(list("abcdef"), ["f"], k=5) == 0.0


def test_duplicates_not_double_counted():
    assert m.precision_at_k(["a", "a", "a"], ["a"], k=5) == pytest.approx(1 / 5)
    assert m.precision_at_k_retrieved(["a", "a", "b"], ["a"], k=5) == pytest.approx(1 / 2)
    assert m.recall_at_k(["a", "a"], ["a", "b"], k=5) == pytest.approx(1 / 2)
    assert m.mrr(["x", "x", "a"], ["a"]) == pytest.approx(1 / 2)
    assert m.ndcg_at_k(["a", "a"], ["a"], k=5) == pytest.approx(1.0)


def test_recall():
    assert m.recall_at_k(["a", "b", "c"], ["a", "c", "z", "y"], k=5) == pytest.approx(2 / 4)
    assert m.recall_at_k(["a"], [], k=5) == 0.0


def test_mrr():
    assert m.mrr(["x", "y", "a"], ["a", "b"]) == pytest.approx(1 / 3)
    assert m.mrr(["x"], ["a"]) == 0.0


def test_ndcg_hand_computed():
    # hits at ranks 1 and 3; 2 relevant docs
    dcg = 1 / math.log2(2) + 1 / math.log2(4)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    assert m.ndcg_at_k(["a", "x", "b"], ["a", "b"], k=5) == pytest.approx(dcg / idcg)
    assert m.ndcg_at_k(["a", "b"], ["a", "b"], k=5) == pytest.approx(1.0)
    assert m.ndcg_at_k(["x"], ["a"], k=5) == 0.0
    assert m.ndcg_at_k(["a"], [], k=5) == 0.0


def test_tokenize():
    assert m.tokenize("Solar, WIND! 2023.") == ["solar", "wind", "2023"]


def test_bleu_identical_is_one():
    text = "The solar industry grew quickly in 2023 across Europe."
    assert m.bleu_score(text, text) == pytest.approx(1.0)


def test_bleu_overlap_positive_and_less_than_one():
    ref = "the solar industry grew quickly in 2023 across europe"
    pred = "the solar industry grew fast in 2023 across the world"
    assert 0.0 < m.bleu_score(pred, ref) < 1.0


def test_bleu_empty_and_disjoint():
    assert m.bleu_score("", "abc") == 0.0
    assert m.bleu_score("alpha beta gamma delta", "one two three four") < 0.1


def test_rouge_hand_computed():
    # unigrams: pred 3, ref 3, overlap 2 -> P=R=F=2/3 ; bigrams: overlap 1 of 2 -> 1/2
    s = m.rouge_scores("the cat sat", "the cat ran")
    assert s["rouge1"] == pytest.approx(2 / 3)
    assert s["rouge2"] == pytest.approx(1 / 2)
    assert s["rougeL"] == pytest.approx(2 / 3)


def test_rouge_identical():
    s = m.rouge_scores("wind power grows", "wind power grows")
    assert s == {"rouge1": 1.0, "rouge2": 1.0, "rougeL": 1.0}


def test_semantic_similarity_fake_encoder():
    vecs = {"a": [1.0, 0.0], "b": [1.0, 1.0], "c": [0.0, 1.0]}
    enc = lambda texts: [vecs[t] for t in texts]  # noqa: E731
    assert m.semantic_similarity("a", "a", enc) == pytest.approx(1.0)
    assert m.semantic_similarity("a", "c", enc) == pytest.approx(0.0)
    assert m.semantic_similarity("a", "b", enc) == pytest.approx(1 / math.sqrt(2))


def test_semantic_similarity_zero_vector():
    enc = lambda texts: [[0.0, 0.0], [1.0, 0.0]]  # noqa: E731
    assert m.semantic_similarity("x", "y", enc) == 0.0


def test_missing_dependency_message(monkeypatch):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(m.MissingEvalDependencyError, match="uv sync --extra eval"):
        m.default_encoder()

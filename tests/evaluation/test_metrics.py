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


# ---------------------------------------------------------------- edge cases


def test_k_zero_and_negative():
    assert m.precision_at_k(["a"], ["a"], k=0) == 0.0
    assert m.precision_at_k(["a"], ["a"], k=-1) == 0.0
    assert m.precision_at_k_retrieved(["a"], ["a"], k=0) == 0.0
    assert m.recall_at_k(["a"], ["a"], k=0) == 0.0
    assert m.ndcg_at_k(["a"], ["a"], k=0) == 0.0


def test_k_larger_than_list():
    assert m.precision_at_k(["a", "b"], ["a"], k=10) == pytest.approx(1 / 10)
    assert m.recall_at_k(["a", "b"], ["a"], k=10) == 1.0
    assert m.ndcg_at_k(["a", "b"], ["a"], k=10) == pytest.approx(1.0)


def test_empty_inputs():
    assert m.precision_at_k(["a"], [], k=5) == 0.0
    assert m.precision_at_k_retrieved(["a"], [], k=5) == 0.0
    assert m.recall_at_k([], ["a"], k=5) == 0.0
    assert m.mrr([], ["a"]) == 0.0
    assert m.mrr(["a"], []) == 0.0
    assert m.ndcg_at_k([], ["a"], k=5) == 0.0
    assert m.ndcg_at_k([], [], k=5) == 0.0


def test_duplicates_in_relevant_not_double_counted():
    assert m.recall_at_k(["a"], ["a", "a"], k=5) == 1.0
    assert m.ndcg_at_k(["a"], ["a", "a"], k=5) == pytest.approx(1.0)


def test_duplicates_do_not_consume_top_k_slots():
    # after dedupe the top 2 are a, b; "c" is rank 3 and must be excluded at k=2
    assert m.precision_at_k(["a", "a", "b", "c"], ["c"], k=2) == 0.0
    assert m.precision_at_k_retrieved(["a", "a", "b"], ["b"], k=2) == pytest.approx(1 / 2)


def test_mrr_rank_and_ndcg_rank_cutoff():
    assert m.mrr(["x", "a"], ["a", "b"]) == pytest.approx(1 / 2)
    assert m.ndcg_at_k(["x", "x2", "a"], ["a"], k=2) == 0.0


def test_precision_retrieved_all_hits_and_accepts_tuples():
    assert m.precision_at_k_retrieved(("a", "b"), ("a", "b"), k=5) == 1.0
    assert m.precision_at_k(("a", "b"), ("a", "b"), k=2) == 1.0


def test_unhashable_ids_raise_typeerror():
    with pytest.raises(TypeError):
        m.precision_at_k([["a"]], ["a"])


def test_int_ids_work_but_do_not_match_str_ids():
    assert m.precision_at_k([1, 2], [1], k=2) == pytest.approx(1 / 2)  # type: ignore[list-item]
    # documents current behaviour: no implicit str/int coercion
    assert m.precision_at_k(["1"], [1], k=1) == 0.0  # type: ignore[list-item]


def test_tokenize_empty_and_punctuation_only():
    assert m.tokenize("") == []
    assert m.tokenize("...!!") == []


def test_bleu_empty_both_and_punctuation_only():
    assert m.bleu_score("", "") == 0.0
    assert m.bleu_score("abc", "") == 0.0
    assert m.bleu_score("!!!", "the cat") == 0.0


def test_bleu_short_text_nonzero_with_smoothing():
    assert m.bleu_score("the cat sat", "the cat ran") > 0.0


def test_bleu_case_and_punctuation_insensitive():
    # <4 tokens: smoothing keeps it below 1, but case/punctuation must not change the score
    assert m.bleu_score("Solar, Wind!", "solar wind") == pytest.approx(
        m.bleu_score("solar wind", "solar wind")
    )


def test_rouge_empty_and_disjoint():
    s = m.rouge_scores("", "the cat")
    assert s == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    s = m.rouge_scores("alpha beta", "gamma delta")
    assert s == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def test_rouge_rougeL_hand_computed_order_matters():
    # LCS of "a b c d" vs "a c b d" is 3 (a, b, d or a, c, d) -> F = 3/4
    s = m.rouge_scores("alpha beta gamma delta", "alpha gamma beta delta")
    assert s["rouge1"] == pytest.approx(1.0)
    assert s["rougeL"] == pytest.approx(3 / 4)


def test_semantic_similarity_encoder_receives_pred_then_ref():
    seen = []

    def enc(texts):
        seen.append(list(texts))
        return [[1.0, 0.0], [1.0, 0.0]]

    assert m.semantic_similarity("P", "R", enc) == pytest.approx(1.0)
    assert seen == [["P", "R"]]


def test_semantic_similarity_empty_strings_and_opposite_vectors():
    enc = lambda texts: [[1.0, 0.0], [-1.0, 0.0]]  # noqa: E731
    assert m.semantic_similarity("", "", enc) == pytest.approx(-1.0)


def test_semantic_similarity_dimension_mismatch_raises():
    enc = lambda texts: [[1.0, 0.0], [1.0]]  # noqa: E731
    with pytest.raises(ValueError):
        m.semantic_similarity("a", "b", enc)


def test_semantic_similarity_does_not_load_default_when_encoder_given(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("default encoder must not be built")

    monkeypatch.setattr(m, "default_encoder", boom)
    enc = lambda texts: [[1.0], [1.0]]  # noqa: E731
    assert m.semantic_similarity("a", "b", enc) == pytest.approx(1.0)


def _block_import(monkeypatch, prefix):
    import builtins

    real = builtins.__import__

    def fake(name, *a, **k):
        if name == prefix or name.startswith(prefix + "."):
            raise ImportError(name)
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    # importlib.import_module bypasses builtins.__import__, so patch it too
    import importlib

    real_im = importlib.import_module

    def fake_im(name, *a, **k):
        if name == prefix or name.startswith(prefix + "."):
            raise ImportError(name)
        return real_im(name, *a, **k)

    monkeypatch.setattr(importlib, "import_module", fake_im)


@pytest.mark.parametrize(
    ("prefix", "call"),
    [
        ("nltk", lambda: m.bleu_score("a b", "a b")),
        ("rouge_score", lambda: m.rouge_scores("a b", "a b")),
        ("sentence_transformers", lambda: m.semantic_similarity("a", "b")),
    ],
)
def test_missing_dependency_raises_clear_error(monkeypatch, prefix, call):
    _block_import(monkeypatch, prefix)
    with pytest.raises(m.MissingEvalDependencyError, match="uv sync --extra eval"):
        call()


def test_missing_dependency_error_is_importerror():
    assert issubclass(m.MissingEvalDependencyError, ImportError)


def test_importing_metrics_does_not_import_heavy_deps():
    import subprocess
    import sys

    code = (
        "import sys, cleantech_rag.evaluation.metrics;"
        "heavy=('torch','sentence_transformers','nltk','rouge_score');"
        "bad=[n for n in heavy if n in sys.modules];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"

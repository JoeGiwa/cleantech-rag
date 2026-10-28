import logging

import pytest

from cleantech_rag.evaluation.datasets import (
    load_complex_qa,
    load_simple_qa,
    parse_complex_qa,
    validate,
)

TXT = """# Title

**1. Query:** Question one?
**Desired Output:** Answer one.
**Referenced Articles:** 111, 112

**2. Query:** Question two?
**Desired Output:** Answer two.
**Referenced Articles:** 222

**3. Query:** Question three?
**Desired Output:** Answer three.
**Referenced Articles:** 333, 334 (extra note 999)
"""

MALFORMED = """**1. Query:** Q1?
**Desired Output:** A1.
**Referenced Articles:** 111

**2. Query:** Q2 has no ids
**Desired Output:** A2.

**3. Query:** Q3?
**Desired Output:** A3.
**Referenced Articles:** 333
"""


def test_each_entry_gets_own_ids_as_str():
    qa = parse_complex_qa(TXT)
    assert [q.article_ids for q in qa] == [["111", "112"], ["222"], ["333", "334"]]
    assert all(isinstance(i, str) for q in qa for i in q.article_ids)
    assert qa[0].question == "Question one?" and qa[0].answer == "Answer one."


def test_malformed_entry_does_not_steal_next_ids(caplog):
    with caplog.at_level(logging.WARNING):
        qa = parse_complex_qa(MALFORMED)
    assert [q.query_num for q in qa] == [1, 3]
    assert qa[0].article_ids == ["111"] and qa[1].article_ids == ["333"]
    assert "query 2" in caplog.text


def test_malformed_strict_raises():
    with pytest.raises(ValueError):
        parse_complex_qa(MALFORMED, strict=True)


def test_validate_counts():
    qa = parse_complex_qa(
        TXT + "\n**4. Query:** Q4?\n**Desired Output:** A.\n**Referenced Articles:** 222\n"
    )
    v = validate(qa, corpus_ids={"111", "112", "222", "333"})
    assert v == {
        "num_queries": 4,
        "distinct_id_sets": 3,
        "queries_sharing_id_sets": 2,
        "gold_ids_missing_from_corpus": 1,
    }


def test_load_files(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text(TXT, encoding="utf-8")
    assert len(load_complex_qa(p)) == 3
    c = tmp_path / "s.csv"
    c.write_text(
        "example_id;question_id;question;relevant_text;answer;article_url\n"
        "1;2;Why?;ctx;Because.;http://x/1\n",
        encoding="utf-8",
    )
    (s,) = load_simple_qa(c)
    assert s.question == "Why?" and s.article_url == "http://x/1" and s.example_id == "1"

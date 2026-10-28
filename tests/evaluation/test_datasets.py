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
        "queries_without_ids": 0,
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


def _entry(n, q, a, refs):
    return f"**{n}. Query:** {q}\n**Desired Output:** {a}\n**Referenced Articles:** {refs}\n"


def test_empty_and_headerless_input():
    assert parse_complex_qa("") == []
    assert parse_complex_qa("# Title\nno queries here\n") == []


def test_malformed_last_entry_is_skipped():
    text = _entry(1, "Q1?", "A1.", "111") + "\n**2. Query:** Q2?\n**Desired Output:** A2.\n"
    qa = parse_complex_qa(text)
    assert [q.query_num for q in qa] == [1]
    assert qa[0].article_ids == ["111"]


def test_malformed_first_entry_does_not_take_next_ids():
    text = "**1. Query:** Q1?\n**Desired Output:** A1.\n\n" + _entry(2, "Q2?", "A2.", "222")
    qa = parse_complex_qa(text)
    assert [(q.query_num, q.article_ids) for q in qa] == [(2, ["222"])]


def test_all_malformed_strict_and_lenient():
    text = "**1. Query:** Q1?\n**Desired Output:** A1.\n"
    assert parse_complex_qa(text) == []
    with pytest.raises(ValueError, match="query 1"):
        parse_complex_qa(text, strict=True)


def test_crlf_line_endings():
    text = TXT.replace("\n", "\r\n")
    qa = parse_complex_qa(text)
    assert [q.article_ids for q in qa] == [["111", "112"], ["222"], ["333", "334"]]
    assert qa[0].question == "Question one?"
    assert qa[0].answer == "Answer one."


def test_extra_whitespace_in_ids_line_and_header():
    text = (
        "**1. Query:**    Q1?  \n**Desired Output:**   A1.  \n"
        "**Referenced Articles:**   111 ,   112  ,113   \n"
    )
    (q,) = parse_complex_qa(text)
    assert q.article_ids == ["111", "112", "113"]
    assert q.question == "Q1?" and q.answer == "A1."


def test_trailing_parenthetical_ignored():
    (q,) = parse_complex_qa(_entry(1, "Q?", "A.", "10, 20 (see also 30, 40) "))
    assert q.article_ids == ["10", "20"]


def test_duplicate_ids_deduped_preserving_order():
    (q,) = parse_complex_qa(_entry(1, "Q?", "A.", "5, 3, 5, 3, 9"))
    assert q.article_ids == ["5", "3", "9"]


def test_header_numbering_gaps_and_unordered():
    text = (
        _entry(1, "Q1?", "A1.", "1") + _entry(7, "Q7?", "A7.", "7") + _entry(3, "Q3?", "A3.", "3")
    )
    qa = parse_complex_qa(text)
    assert [q.query_num for q in qa] == [1, 7, 3]
    assert [q.article_ids for q in qa] == [["1"], ["7"], ["3"]]


def test_missing_ids_line_in_middle_many_entries():
    text = (
        _entry(1, "Q1?", "A1.", "1")
        + "**2. Query:** Q2?\n**Desired Output:** A2.\n\n"
        + "**3. Query:** Q3?\n**Desired Output:** A3.\n\n"
        + _entry(4, "Q4?", "A4.", "4")
    )
    qa = parse_complex_qa(text)
    assert {q.query_num: q.article_ids for q in qa} == {1: ["1"], 4: ["4"]}


def test_multiline_question_and_answer():
    text = (
        "**1. Query:** Line a\nline b\n**Desired Output:** Ans a\nans b\n"
        "**Referenced Articles:** 1\n"
    )
    (q,) = parse_complex_qa(text)
    assert q.question == "Line a\nline b"
    assert q.answer == "Ans a\nans b"


def test_missing_desired_output_still_gets_ids():
    (q,) = parse_complex_qa("**1. Query:** Q only\n**Referenced Articles:** 42\n")
    assert q.article_ids == ["42"] and q.answer == "" and q.question == "Q only"


def test_empty_ids_line_is_not_silently_kept():
    # An entry whose IDs line has no IDs should be treated as malformed (skipped).
    qa = parse_complex_qa("**1. Query:** Q?\n**Desired Output:** A.\n**Referenced Articles:**\n")
    assert qa == []


def test_load_complex_qa_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_complex_qa(tmp_path / "nope.txt")


def test_validate_empty_and_without_corpus():
    assert validate([]) == {
        "num_queries": 0,
        "distinct_id_sets": 0,
        "queries_sharing_id_sets": 0,
        "queries_without_ids": 0,
    }
    assert "gold_ids_missing_from_corpus" not in validate(parse_complex_qa(TXT))


def test_validate_empty_corpus_and_all_present():
    qa = parse_complex_qa(TXT)
    assert validate(qa, corpus_ids=set())["gold_ids_missing_from_corpus"] == 5
    full = {"111", "112", "222", "333", "334"}
    assert validate(qa, corpus_ids=full)["gold_ids_missing_from_corpus"] == 0


def test_validate_id_set_order_insensitive_and_no_content():
    text = _entry(1, "SECRET_Q", "SECRET_A", "1, 2") + _entry(2, "Q2", "A2", "2, 1")
    v = validate(parse_complex_qa(text))
    assert v["distinct_id_sets"] == 1 and v["queries_sharing_id_sets"] == 2
    assert all(isinstance(k, str) and isinstance(x, int) for k, x in v.items())
    assert "SECRET" not in repr(v)


def test_validate_does_not_log_or_print_content(capsys, caplog):
    qa = parse_complex_qa(_entry(1, "SECRET_Q", "SECRET_A", "1"))
    with caplog.at_level(logging.DEBUG):
        validate(qa, corpus_ids={"1"})
    out = capsys.readouterr()
    assert "SECRET" not in out.out + out.err + caplog.text


def test_simple_qa_ids_stay_str_and_blank_cells(tmp_path):
    c = tmp_path / "s.csv"
    c.write_text(
        "example_id;question_id;question;relevant_text;answer;article_url\n007;2;Why?;;Because.;\n",
        encoding="utf-8",
    )
    (s,) = load_simple_qa(c)
    assert s.example_id == "007" and s.question_id == "2"
    assert s.relevant_text == "" and s.article_url == ""


def _simple(urls):
    from cleantech_rag.evaluation.datasets import SimpleQA

    return [SimpleQA(str(i), str(i), "q", "t", "a", u) for i, u in enumerate(urls)]


@pytest.fixture
def mapping(tmp_path):
    from cleantech_rag.evaluation.datasets import load_url_index

    corpus = tmp_path / "c.csv"
    corpus.write_text(
        "Unnamed: 0,title,url\n1,t,http://a/1\n2,t,http://b/2/\n3,t,http://dup\n4,t,http://dup\n",
        encoding="utf-8",
    )
    return load_url_index(corpus)


def _map(urls, index):
    from cleantech_rag.evaluation.datasets import map_article_ids

    return map_article_ids(_simple(urls), index)


def test_map_mapped_url(mapping):
    mapped, unmapped = _map(["http://a/1"], mapping)
    assert mapped[0].article_ids == ["1"] and unmapped == 0


def test_map_normalizes_whitespace_and_trailing_slash(mapping):
    mapped, _ = _map([" http://a/1/ ", "http://b/2"], mapping)
    assert [m.article_ids for m in mapped] == [["1"], ["2"]]


def test_map_duplicate_url_keeps_all_ids_in_order(mapping):
    mapped, _ = _map(["http://dup"], mapping)
    assert mapped[0].article_ids == ["3", "4"]


def test_map_ids_are_str(mapping):
    mapped, _ = _map(["http://a/1", "http://dup"], mapping)
    assert all(isinstance(i, str) for m in mapped for i in m.article_ids)


def test_map_unmapped_kept_empty_counted_and_warned_without_content(mapping, caplog):
    with caplog.at_level(logging.WARNING):
        mapped, unmapped = _map(["http://a/1", "http://nope"], mapping)
    assert mapped[1].article_ids == [] and unmapped == 1
    assert "1 of 2" in caplog.text and "nope" not in caplog.text


def test_validate_excludes_empty_id_sets(mapping):
    mapped, _ = _map(["http://a/1", "http://nope", "http://nope2", "http://dup"], mapping)
    assert validate(mapped, {"1", "2", "3"}) == {
        "num_queries": 4,
        "distinct_id_sets": 2,
        "queries_sharing_id_sets": 0,
        "queries_without_ids": 2,
        "gold_ids_missing_from_corpus": 1,
    }

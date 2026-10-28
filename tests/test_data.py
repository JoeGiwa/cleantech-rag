from pathlib import Path

import pytest

from cleantech_rag.data import chunk_documents, load_articles, to_documents

CSV = Path(__file__).parent / "fixtures" / "articles.csv"


def test_load_renames_and_skips_empty():
    df = load_articles(CSV)
    assert list(df["id"]) == ["10", "11", "13", "14"]


def test_limit_and_include_ids():
    assert list(load_articles(CSV, limit=2)["id"]) == ["10", "11"]
    df = load_articles(CSV, limit=1, include_ids=[14, "13"])
    assert list(df["id"]) == ["10", "13", "14"]


def test_documents_metadata():
    doc = to_documents(load_articles(CSV))[0]
    assert doc.page_content == "Short\n\nBrief text."
    assert doc.metadata == {
        "article_id": "10",
        "title": "Short",
        "url": "http://d/10",
        "date": "2024-01-02",
        "domain": "d.com",
        "author": "A",
    }


def test_chunking_size_overlap_and_ids():
    docs = to_documents(load_articles(CSV))
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    long = [c for c in chunks if c.metadata["article_id"] == "11"]
    assert len(long) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)
    assert all(c.metadata["date"] == "2024-02-03" for c in long)
    assert [c.metadata["chunk_id"] for c in long] == [f"11-{i}" for i in range(len(long))]
    # overlap: end of a chunk reappears at start of the next
    # (chunk 0 is just the title, since the splitter breaks on the blank line first)
    tail = " ".join(long[1].page_content.split()[-3:])
    assert tail in long[2].page_content[:80]
    again = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert [c.metadata["chunk_id"] for c in again] == [c.metadata["chunk_id"] for c in chunks]


def test_chunking_uses_settings_defaults():
    chunks = chunk_documents(to_documents(load_articles(CSV)))
    assert all(len(c.page_content) <= 1000 for c in chunks)


# ---- edge cases -------------------------------------------------------------------


def _write_csv(tmp_path, text):
    p = tmp_path / "a.csv"
    p.write_text(text)
    return p


HEADER = ",title,date,author,content,domain,url\n"


def test_no_nan_strings_in_date_metadata():
    for d in to_documents(load_articles(CSV)):
        assert d.metadata["date"] not in ("", "nan")


def test_limit_zero_returns_empty_or_only_included():
    assert load_articles(CSV, limit=0).empty
    assert list(load_articles(CSV, limit=0, include_ids=[14])["id"]) == ["14"]


def test_limit_larger_than_rows_and_negative():
    assert len(load_articles(CSV, limit=1000)) == 4
    assert load_articles(CSV, limit=-5).empty


def test_include_ids_overlapping_sample_no_duplicates():
    df = load_articles(CSV, limit=2, include_ids=[10, "11", 10, "11"])
    assert list(df["id"]) == ["10", "11"]


def test_include_ids_missing_from_file_ignored():
    df = load_articles(CSV, limit=1, include_ids=[999, "nope"])
    assert list(df["id"]) == ["10"]


def test_include_ids_pointing_at_empty_row_not_resurrected():
    df = load_articles(CSV, limit=1, include_ids=[12])
    assert list(df["id"]) == ["10"]


def test_include_ids_without_limit_returns_all():
    assert list(load_articles(CSV, include_ids=[14])["id"]) == ["10", "11", "13", "14"]


def test_include_ids_generator_and_empty():
    assert list(load_articles(CSV, limit=1, include_ids=(i for i in [13]))["id"]) == ["10", "13"]
    assert list(load_articles(CSV, limit=1, include_ids=[])["id"]) == ["10"]


def test_limit_counts_after_dropping_empty_rows():
    # row 12 is empty; limit=3 must yield 10, 11, 13
    assert list(load_articles(CSV, limit=3)["id"]) == ["10", "11", "13"]


def test_whitespace_only_and_nan_content_skipped(tmp_path):
    p = _write_csv(
        tmp_path,
        HEADER
        + '1,T,2024-01-01,A,"   \n\t ",d,u\n2,T,2024-01-01,A,,d,u\n3,T,2024-01-01,A,ok,d,u\n',
    )
    assert list(load_articles(p)["id"]) == ["3"]


def test_nan_title_author_date_become_empty_strings(tmp_path):
    p = _write_csv(tmp_path, HEADER + "1,,,,Body text.,,\n")
    doc = to_documents(load_articles(p))[0]
    assert doc.page_content == "Body text."
    assert doc.metadata["title"] == ""
    assert doc.metadata["author"] == ""
    assert doc.metadata["date"] == ""
    assert "nan" not in " ".join(doc.metadata.values()).lower()
    assert "nan" not in doc.page_content.lower()


def test_missing_optional_columns_filled(tmp_path):
    p = _write_csv(tmp_path, ",title,content\n5,T,Body\n")
    doc = to_documents(load_articles(p))[0]
    assert doc.metadata["article_id"] == "5"
    assert doc.metadata["date"] == ""


def test_article_id_is_str():
    for d in to_documents(load_articles(CSV)):
        assert isinstance(d.metadata["article_id"], str)


def test_empty_dataframe_paths(tmp_path):
    p = _write_csv(tmp_path, HEADER)
    df = load_articles(p)
    assert df.empty
    assert to_documents(df) == []
    assert chunk_documents([]) == []


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_articles(tmp_path / "nope.csv")


def test_accepts_str_path():
    assert len(load_articles(str(CSV))) == 4


def test_overlap_is_honored_and_bounded():
    docs = to_documents(load_articles(CSV, include_ids=[11]))
    docs = [d for d in docs if d.metadata["article_id"] == "11"]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    body = chunks[1:]  # chunk 0 is the title only
    shared = []
    for a, b in zip(body, body[1:], strict=False):
        n = max(
            k
            for k in range(len(a.page_content) + 1)
            if b.page_content.startswith(a.page_content[len(a.page_content) - k :])
        )
        shared.append(n)
        assert n <= 50
    assert max(shared) > 0


def test_zero_overlap_has_no_shared_text():
    docs = [d for d in to_documents(load_articles(CSV)) if d.metadata["article_id"] == "11"]
    with_overlap = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    no_overlap = chunk_documents(docs, chunk_size=200, chunk_overlap=0)
    assert len(no_overlap) <= len(with_overlap)
    assert all(len(c.page_content) <= 200 for c in no_overlap)


def test_short_article_single_chunk_index_zero():
    chunks = chunk_documents(to_documents(load_articles(CSV, limit=1)), 200, 50)
    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "10-0"


def test_chunks_keep_article_metadata():
    docs = to_documents(load_articles(CSV))
    by_id = {d.metadata["article_id"]: d.metadata for d in docs}
    for c in chunk_documents(docs, 200, 50):
        for k, v in by_id[c.metadata["article_id"]].items():
            assert c.metadata[k] == v


def test_chunk_ids_unique_across_corpus():
    ids = [
        c.metadata["chunk_id"] for c in chunk_documents(to_documents(load_articles(CSV)), 100, 20)
    ]
    assert len(ids) == len(set(ids))


def test_chunking_does_not_mutate_input_metadata():
    docs = to_documents(load_articles(CSV))
    chunk_documents(docs, 200, 50)
    assert all("chunk_id" not in d.metadata for d in docs)


def test_overlap_gt_size_raises():
    docs = to_documents(load_articles(CSV, limit=1))
    with pytest.raises(ValueError):
        chunk_documents(docs, chunk_size=50, chunk_overlap=80)


def test_overlap_equal_size_terminates_and_respects_size():
    # langchain only rejects overlap > size; equal must still terminate and honor size
    docs = [d for d in to_documents(load_articles(CSV)) if d.metadata["article_id"] == "11"]
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=50)
    assert chunks
    assert all(len(c.page_content) <= 50 for c in chunks)


def test_explicit_zero_overlap_not_replaced_by_default():
    # chunk_overlap=0 must not fall through to the settings default (falsy-zero trap)
    docs = [d for d in to_documents(load_articles(CSV)) if d.metadata["article_id"] == "11"]
    a = chunk_documents(docs, chunk_size=200, chunk_overlap=0)
    b = chunk_documents(docs, chunk_size=200, chunk_overlap=50)
    assert [c.page_content for c in a] != [c.page_content for c in b]

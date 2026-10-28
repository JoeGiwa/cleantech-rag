from pathlib import Path

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

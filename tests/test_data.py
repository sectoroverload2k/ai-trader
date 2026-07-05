from ai_trader.data import _news_item_to_dict, _paginate_news


class _FakeArticle:
    def __init__(self, headline):
        self.created_at = "2024-01-01T00:00:00Z"
        self.headline = headline
        self.summary = "summary " + headline
        self.url = "http://example.com"


def test_news_item_to_dict_maps_fields():
    d = _news_item_to_dict(_FakeArticle("hello"))
    assert d["headline"] == "hello"
    assert d["summary"] == "summary hello"
    assert d["url"] == "http://example.com"
    assert d["timestamp"] == "2024-01-01T00:00:00Z"


def test_paginate_follows_tokens_until_exhausted():
    pages = {
        None: ([{"headline": "a"}], "t1"),
        "t1": ([{"headline": "b"}], "t2"),
        "t2": ([{"headline": "c"}], None),  # last page, no next token
    }
    out = _paginate_news(lambda tok: pages[tok], max_items=1000)
    assert [x["headline"] for x in out] == ["a", "b", "c"]


def test_paginate_respects_max_items():
    def fetch(_tok):
        return [{"headline": "x"}] * 50, "more"

    out = _paginate_news(fetch, max_items=120)
    assert len(out) == 120  # stops even though the source is endless


def test_paginate_stops_on_empty_page():
    out = _paginate_news(lambda tok: ([], None), max_items=100)
    assert out == []

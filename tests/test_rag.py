import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic import NOT_FOUND, agentic_answer, build_context  # noqa: E402
from rag_eval import QUESTIONS, retrieval_metrics  # noqa: E402
from rag_utils import load_chunks  # noqa: E402


def test_every_chunk_keeps_source_and_header():
    chunks = load_chunks()
    assert len({c.metadata["kaynak"] for c in chunks}) == 6
    assert all(" / " in c.page_content.splitlines()[0] for c in chunks)


def test_small_chunks_split_retention_rule():
    # 120 karakterde 90 gün kuralı ile 2 yıl kuralı ayrı parçalara düşer
    small = load_chunks(chunk_size=120, chunk_overlap=18)
    with_90 = [c.page_content for c in small if "90 gün" in c.page_content]
    assert with_90 and not any("2 yıl" in c for c in with_90)
    big = load_chunks(chunk_size=350, chunk_overlap=50)
    assert any("90 gün" in c.page_content and "2 yıl" in c.page_content for c in big)


def test_retrieval_metrics_perfect_and_wrong():
    gold = dict(QUESTIONS)
    perfect = retrieval_metrics(lambda q: [gold[q], "x.md"])
    assert perfect == {"hit@1": 1.0, "hit@3": 1.0, "MRR": 1.0}
    second = retrieval_metrics(lambda q: ["x.md", gold[q], gold[q]])
    assert second["hit@1"] == 0 and second["hit@3"] == 1 and second["MRR"] == 0.5


def _doc(src):
    return SimpleNamespace(metadata={"kaynak": src}, page_content="içerik")


def test_agent_rewrites_then_answers():
    scores = iter([0.05, 0.9])
    result = agentic_answer(
        "izin?", search=lambda q: [(_doc("yillik_izin_politikasi.md"), next(scores))],
        answer=lambda q, hits: "yanıt", rewrite=lambda q, prev: "yıllık izin talebi", threshold=0.5)
    assert result["bulundu"] and result["yanit"] == "yanıt"
    assert [t["sorgu"] for t in result["iz"]] == ["izin?", "yıllık izin talebi"]


def test_agent_gives_up_after_limit():
    calls = []
    result = agentic_answer(
        "yemek kartı?", search=lambda q: calls.append(q) or [(_doc("a.md"), 0.01)],
        answer=lambda q, h: "uydurma", rewrite=lambda q, prev: prev + "!", threshold=0.5, max_rewrites=2)
    assert result["yanit"] == NOT_FOUND and not result["bulundu"]
    assert len(calls) == 3


def test_build_context_labels_sources():
    ctx = build_context([(SimpleNamespace(metadata={"kaynak": "a.md"}, page_content="metin"), 1.0)])
    assert ctx.startswith("[a.md]")

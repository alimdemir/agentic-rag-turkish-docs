"""Getir -> yeniden sırala -> (gerekirse sorguyu yeniden yaz) -> yanıtla döngüsü.

Model ve getirici bağımsızdır; Colab not defterinde E5 + bge-reranker + Qwen3 ile,
testlerde sahte fonksiyonlarla çalışır.
"""
from __future__ import annotations

from typing import Callable

NOT_FOUND = "Belgelerde bu soruyu yanıtlayacak bilgi bulunamadı."

Hit = tuple[object, float]  # (Document, reranker skoru)


def agentic_answer(question: str,
                   search: Callable[[str], list[Hit]],
                   answer: Callable[[str, list[Hit]], str],
                   rewrite: Callable[[str, str], str],
                   threshold: float,
                   max_rewrites: int = 2) -> dict:
    """En iyi reranker skoru eşiğin altındaysa sorgu yeniden yazılır.

    `max_rewrites` sınırı dolunca model tahmin yürütmez; bilginin bulunamadığını söyler.
    """
    trace = []
    query = question
    for attempt in range(max_rewrites + 1):
        hits = search(query)
        best = hits[0][1] if hits else float("-inf")
        trace.append({
            "deneme": attempt + 1,
            "sorgu": query,
            "en_iyi_skor": round(float(best), 3),
            "kaynaklar": [getattr(d, "metadata", {}).get("kaynak") for d, _ in hits[:3]],
        })
        if best >= threshold:
            return {"yanit": answer(question, hits[:3]), "iz": trace, "bulundu": True}
        if attempt < max_rewrites:
            query = rewrite(question, query)
    return {"yanit": NOT_FOUND, "iz": trace, "bulundu": False}


def build_context(hits: list[Hit]) -> str:
    """Modele verilecek bağlam: her parça kaynak adıyla etiketlenir (atıf için)."""
    return "\n\n".join(f"[{d.metadata['kaynak']}]\n{d.page_content}" for d, _ in hits)

"""LangChain Deep Agents ile belge ajanı: arama aracı, sistem istemi ve iz (trace) çıktısı."""
from __future__ import annotations

import json
import textwrap
from typing import Callable

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool


def build_agent(model, search_tool, **kwargs):
    """Deep agent: planlama (write_todos) ara katmanı açık, arama aracı ve Türkçe sistem istemiyle."""
    from deepagents import create_deep_agent
    from langchain.agents.middleware import TodoListMiddleware

    return create_deep_agent(model=model, tools=[search_tool], system_prompt=SYSTEM_PROMPT,
                             middleware=[TodoListMiddleware()], **kwargs)

SYSTEM_PROMPT = """Sen bir şirketin iç belgelerini yanıtlayan asistansın. Türkçe yanıt ver.
Kurallar:
1. Soru birden fazla konu içeriyorsa önce write_todos ile her konu için ayrı bir adım yaz ve adımları tamamladıkça güncelle.
2. Her konu için belge_ara aracını ayrı ayrı çağır; toplamda en fazla 4 arama yap.
3. Yanıtı yalnızca belge_ara'nın döndürdüğü parçalara dayandır; her bilginin sonunda köşeli parantez içinde kaynak dosya adını yaz.
4. Araç "bulunamadı" derse o konu için bilginin belgelerde olmadığını açıkça söyle; tahmin yürütme.
5. Dosya sistemi araçlarını ve alt ajanları kullanma; yalnız write_todos ve belge_ara yeterli."""


def make_search_tool(search: Callable[[str], list], threshold: float, k: int = 3):
    """`search(sorgu)` -> [(Document, skor), ...] (skora göre azalan) sarmalayan LangChain aracı."""

    @tool
    def belge_ara(sorgu: str) -> str:
        """Şirket içi belgelerde arama yapar (yıllık izin, KVKK ve mülakat kayıtları, aday değerlendirme,
        uzaktan çalışma, GPU sunucusu, oryantasyon). Tek bir konu için kısa bir Türkçe sorgu ver."""
        hits = list(search(sorgu))[:k]
        best = hits[0][1] if hits else 0.0
        if best < threshold:
            return f"İlgili parça bulunamadı (en iyi skor {best:.3f})."
        return "\n\n".join(f"[{d.metadata['kaynak']}] (skor {s:.3f})\n{d.page_content}"
                           for d, s in hits if s >= threshold)

    return belge_ara


def trace_lines(messages, width: int = 140) -> list[str]:
    """Ajan mesajlarını okunur bir iz hâline getirir (planlama, araç çağrıları, yanıt)."""
    out = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc["name"] == "write_todos":
                    out.append("📝 write_todos")
                    for t in tc["args"].get("todos", []):
                        out.append(f"     [{t.get('status', '?'):<11}] {t.get('content', '')}")
                else:
                    out.append(f"🔧 {tc['name']}({json.dumps(tc['args'], ensure_ascii=False)})")
        elif isinstance(m, ToolMessage):
            if m.name == "write_todos":
                continue
            body = " ".join(str(m.content).split())
            out.append(textwrap.shorten("   ↳ " + body, width=width, placeholder=" ..."))
        elif isinstance(m, AIMessage) and m.content:
            out.append("")
            out.append("💬 YANIT:")
            out.extend(textwrap.wrap(str(m.content), width))
    return out

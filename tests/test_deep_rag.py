import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

deepagents = pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage  # noqa: E402

from deep_rag import build_agent, make_search_tool, trace_lines  # noqa: E402


class ScriptedModel(GenericFakeChatModel):
    """Araç çağrısı üreten sahte model (bind_tools'u yok sayar)."""

    def bind_tools(self, tools, **kwargs):
        return self


def _doc(src, text="içerik"):
    return SimpleNamespace(metadata={"kaynak": src}, page_content=text)


def test_search_tool_threshold():
    tool = make_search_tool(lambda q: [(_doc("a.md", "90 gün"), 0.9), (_doc("b.md"), 0.01)], threshold=0.1)
    out = tool.invoke({"sorgu": "saklama"})
    assert "[a.md]" in out and "b.md" not in out
    low = make_search_tool(lambda q: [(_doc("a.md"), 0.01)], threshold=0.1)
    assert "bulunamadı" in low.invoke({"sorgu": "yemek kartı"})


def test_deep_agent_plans_searches_and_answers():
    script = iter([
        AIMessage(content="", tool_calls=[{"name": "write_todos", "id": "t1", "args": {"todos": [
            {"content": "Kayıt saklama süresini ara", "status": "in_progress"},
            {"content": "Yanıtı yaz", "status": "pending"}]}}]),
        AIMessage(content="", tool_calls=[{"name": "belge_ara", "id": "t2", "args": {"sorgu": "kayıt saklama süresi"}}]),
        AIMessage(content="Kayıtlar en fazla 90 gün saklanır [kvkk.md]."),
    ])
    model = ScriptedModel(messages=script)
    search = make_search_tool(lambda q: [(_doc("kvkk.md", "en fazla 90 gün"), 0.95)], threshold=0.1)
    agent = build_agent(model, search)
    result = agent.invoke({"messages": [{"role": "user", "content": "Kayıtlar ne kadar saklanır?"}]})
    lines = trace_lines(result["messages"])
    assert any(l.startswith("📝 write_todos") for l in lines)
    assert any("belge_ara" in l for l in lines)
    assert any("[kvkk.md]" in l for l in lines)
    assert result["todos"][0]["content"] == "Kayıt saklama süresini ara"

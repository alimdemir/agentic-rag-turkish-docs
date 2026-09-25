"""Belge yükleme, parçalama ve embedding sınıfları."""
from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOC_DIR = Path(__file__).parent / "belgeler"


def load_chunks(chunk_size: int = 350, chunk_overlap: int = 50, doc_dir: Path = DOC_DIR) -> list[Document]:
    """Markdown belgeleri önce başlıklara, sonra karakter sınırına göre böler.

    Her parçanın başına `belge / bölüm` başlığı eklenir; böylece parça tek başına
    okunduğunda da nereye ait olduğu anlaşılır. Kaynak dosya metadata'da tutulur.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "belge"), ("##", "bolum")], strip_headers=True)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n\n", "\n", ". ", " "])
    chunks = []
    for path in sorted(Path(doc_dir).glob("*.md")):
        sections = header_splitter.split_text(path.read_text(encoding="utf-8"))
        for sec in sections:
            sec.metadata["kaynak"] = path.name
        for i, ch in enumerate(splitter.split_documents(sections)):
            ch.metadata["parca_no"] = i
            ch.page_content = f"{ch.metadata.get('belge', '')} / {ch.metadata.get('bolum', '')}\n{ch.page_content}"
            chunks.append(ch)
    return chunks


class WordLlamaEmbeddings(Embeddings):
    """Paketle gelen hafif WordLlama modeli; internetsiz (kapalı) ortamda çalışır."""

    def __init__(self):
        import wordllama
        from wordllama import WordLlama
        # cache_dir paketin kendi klasörü: tokenizer'ı Hugging Face'ten indirmeye çalışmasın
        self.model = WordLlama.load(cache_dir=Path(wordllama.__file__).parent, disable_download=True)

    def embed_documents(self, texts):
        return self.model.embed(texts, norm=True).tolist()

    def embed_query(self, text):
        return self.model.embed([text], norm=True)[0].tolist()


class E5Embeddings(Embeddings):
    """intfloat/multilingual-e5 ailesi. E5, sorgu ve pasajlarda ön ek bekler."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-base", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts):
        return self.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode([f"query: {text}"], normalize_embeddings=True)[0].tolist()

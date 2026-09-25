"""Küçük getirme test kümesi ve ölçütler (hit@1, hit@3, MRR)."""
from __future__ import annotations

from typing import Callable, Iterable

# (soru, doğru belge)
QUESTIONS: list[tuple[str, str]] = [
    ("Mülakat video kayıtları ne kadar süre saklanıyor?", "kvkk_mulakat_kayitlari.md"),
    ("Yıllık izin talebi kaç gün önceden yapılmalı?", "yillik_izin_politikasi.md"),
    ("Aday değerlendirme formunda hangi yetkinlikler puanlanıyor?", "aday_degerlendirme_sureci.md"),
    ("Uzaktan çalışırken hangi saatlerde ulaşılabilir olmak gerekiyor?", "uzaktan_calisma_yonergesi.md"),
    ("A100 sunucusunu demo için kim açabilir?", "gpu_sunucu_kullanimi.md"),
    ("Yeni başlayan biri ilk hafta neler yapıyor?", "yeni_calisan_oryantasyonu.md"),
    ("Aday kaydedilmek istemezse ne olur?", "kvkk_mulakat_kayitlari.md"),
    ("Kullanılmayan izin günleri sonraki yıla aktarılır mı?", "yillik_izin_politikasi.md"),
    ("Kişisel bilgisayarıma aday verisi indirebilir miyim?", "uzaktan_calisma_yonergesi.md"),
    ("GPU belleği yetmezse ilk ne yapılmalı?", "gpu_sunucu_kullanimi.md"),
    ("Duygu analizi sonucu işe alım kararında tek başına kullanılır mı?", "aday_degerlendirme_sureci.md"),
    ("Beş yılı aşkın kıdemde kaç gün izin var?", "yillik_izin_politikasi.md"),
]

# Belgelerde yanıtı olmayan sorular: sistem "bilgi bulunamadı" demeli
UNANSWERABLE: list[str] = [
    "Şirketin yemek kartı limiti ne kadar?",
    "Doğum izni kaç hafta?",
    "Ofiste evcil hayvan kabul ediliyor mu?",
]


def retrieval_metrics(retrieve: Callable[[str], Iterable[str]],
                      questions: list[tuple[str, str]] = QUESTIONS, k: int = 3) -> dict[str, float]:
    """`retrieve(soru)` sıralı kaynak adları döndürmeli (tekrarlar ilk geçtiği sırada sayılır)."""
    hit1 = hitk = mrr = 0.0
    for question, gold in questions:
        ranked = list(dict.fromkeys(retrieve(question)))[:k]
        hit1 += ranked[:1] == [gold]
        if gold in ranked:
            hitk += 1
            mrr += 1 / (ranked.index(gold) + 1)
    n = len(questions)
    return {"hit@1": hit1 / n, f"hit@{k}": hitk / n, "MRR": mrr / n}

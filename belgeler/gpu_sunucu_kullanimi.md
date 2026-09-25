# GPU Sunucusu Kullanım Kuralları

## Kaynak
Model denemeleri ve canlı demolar için A100 80 GB GPU içeren bulut sunucusu kullanılır. Sunucu saatlik ücretlendirildiği için sürekli açık tutulmaz.

## Açma ve kapatma
Sunucuyu yalnızca yapay zekâ ekibinin sorumlu mühendisi veya onun yetkilendirdiği kişi açabilir. Demo öncesinde sunucu en erken 15 dakika önce başlatılır, demo bittiğinde kapatılır ve kullanım süresi ekip kanalına yazılır.

## Bellek kullanımı
Aynı GPU üzerinde birden fazla model çalıştırılacaksa her modelin tahmini VRAM ihtiyacı önceden not edilir. Bellek taşması yaşanırsa önce batch boyutu ve eşzamanlı istek sayısı azaltılır.

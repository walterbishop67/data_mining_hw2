# Proje Dokümantasyonu

Bu teslim klasörü, bilgisayar bilimleri makale özetlerinden Top-5 dergi önerisi üretmek ve veri setindeki konu alanlarını kümelendirmek için hazırlanmıştır.

## Teslim Dosyaları

| Dosya/Klasör | Görev |
|---|---|
| `app.py` | Streamlit arayüzünü başlatır |
| `src/` | Veri yükleme, metin temizleme, model, öneri ve dashboard kodları |
| `notebooks/20210808053_Final_Project.ipynb` | Final notebook |
| `report/20210808053_IEEE_PROJECT_REPORT.tex` | IEEE rapor kaynağı |
| `CompSciencePub.sqlite` | Kodun kullandığı SQLite veri tabanı |
| `exports/20210808053/journal_recommender_pipeline.pkl` | Kayıtlı final dergi öneri modeli |
| `exports/20210808053/journal_recommender_meta.json` | Model değerlendirme metrikleri |
| `exports/20210808053/step9_clustered_dataset.csv` | Kayıtlı topic clustering çıktısı |

## Ana Fikir

1. SQLite veri tabanından article, abstract, journal, keyword, keyword plus ve subject alanları alınır.
2. Metinler HTML ve gereksiz karakterlerden temizlenir.
3. Baseline modüller TF-IDF + cosine similarity ile içerik tabanlı öneri üretir.
4. Final model, title, abstract, keywords ve subjects kanallarını ayrı TF-IDF temsilleriyle işler.
5. `SGDClassifier(loss="log_loss")` çok sınıflı journal prediction için eğitilir.
6. Topic clustering için TF-IDF + KMeans kullanılır.

## Final Model

Final recommender `src/final_project/training.py` içindedir. Model dört kanal kullanır:

- `title_channel`
- `abstract_channel`
- `keywords_channel`
- `subjects_channel`

Her kanal ayrı `TfidfVectorizer` ile temsil edilir. Özellikler `ColumnTransformer` ile birleştirilir ve `SGDClassifier(loss="log_loss")` ile dergi olasılıkları üretilir. Streamlit arayüzü en yüksek olasılıklı 5 dergiyi gösterir.

Kullanıcı sadece abstract girebilir. Başlık, keyword ve subject alanları opsiyoneldir. Bu tasarım ödevin abstract gereksinimini karşılar ve ek metadata varsa daha güçlü tahmin üretir.

## Kayıtlı Sonuçlar

`journal_recommender_meta.json` içindeki kayıtlı değerlendirme:

| Metrik | Değer |
|---|---:|
| Kullanılan makale sayısı | 22,966 |
| Dergi sınıfı sayısı | 406 |
| Holdout Top-1 accuracy | 0.7061 |
| Holdout Top-5 accuracy | 0.9323 |

Top-5 accuracy, ödevin ana çıktısı Top-5 dergi önerisi olduğu için temel başarı metriğidir.

Kaydedilen model, test ayrımını temiz tutmak için yalnızca stratified train split üzerinde fit edilmiştir. Holdout test verisi metrik hesaplama dışında modele tekrar verilmez.

## Çalıştırma

```bash
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Yeniden Üretim

Ara CSV dosyaları temiz teslim klasöründe tutulmaz. Modeli ve cluster çıktısını yeniden üretmek gerekirse:

```bash
py -m src.final_project.enrichment
py -m src.final_project.training
py -m src.final_project.topic_modeling
```

## Akademik Not

Final modelde keyword ve subject bilgileri kullanıldığı için model güçlüdür, fakat gerçek kullanıcı sadece abstract verdiğinde bu ek kanallar boş kalabilir. Bu durum raporda sınırlılık olarak tartışılır; proje hem abstract-only baseline mantığını hem de zengin metadata ile final classifier yaklaşımını içerir.

# Dataset Splitter

Bir CSV veri kümesini model eğitimi ve değerlendirme için `train.csv` ve `test.csv` olarak böler. Aynı dosya, seçenekler ve Python sürümüyle aynı kayıtlar seçilir. Python 3.10 veya üzeri yeterlidir.

## Kullanım

```sh
python dataset_splitter.py examples/observations.csv split --test-size 0.25 --seed 42 --stratify label
```

Örnekte `split/` klasörüne 9 eğitim ve 3 test kaydı yazılır. Her iki çıktı da başlıkları ve kaynak satır sırasını korur. `split.json` tohum, kayıt sayıları ve katmanlı bölme seçildiyse sınıf dağılımını gösterir. Çıktı klasörünün üst dizini hazır olmalı. Mevcut bir klasör veya dosya üzerine yazılmaz.

- `--test-size` ondalık (`0.25`) veya kesir (`1/4`) alır. Kayıt sayısı en yakın tam sayıya yuvarlanır, yarım değerler yukarı yuvarlanır. Eğitim ve test bölümlerinin her biri en az bir kayıt içermelidir.
- `--stratify` kullanıldığında her sınıftan iki bölüme de en az bir kayıt ayrılır. Kalan test yerleri sınıfların kalan kayıt sayılarıyla orantılı dağıtılır. Sınıflar bu koşulu karşılayamıyorsa komut çıktı oluşturmadan durur.
- `--delimiter ';'` veya `--delimiter '\t'` ayraç seçer. Girdi kodlaması `--encoding cp1254` gibi değiştirilebilir. Çıktılar UTF-8’dir.
- Araç satır düzeyinde bölme yapar ve tabloyu bellekte tutar. Aynı kişiye veya zamana ait kayıtları birlikte tutmak gereken veri kümeleri için grup ya da zaman temelli bölme gerekir.

## Testler

```sh
python -m unittest discover -s tests -v
```

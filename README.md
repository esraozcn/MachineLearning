# MachineLearning

GOOGLE COLAB LİNKLERİ:

### Naive Bayes

https://colab.research.google.com/drive/1PGyGG1Wn7bmlGJxAxKdKVEnZ92W7MM96?authuser=1#scrollTo=jQiRz0T8BYza

### K-Nearest Neighbors (KNN)

https://colab.research.google.com/drive/1wPQUSFJyjQJRn83-f1nFMVeDpEpA4fiR?authuser=1#scrollTo=StKC7fUXLamc

### Lineer Regression

https://colab.research.google.com/drive/1wasqVrOToYlVtRWs4PoLGLPXBfrB4b5H?authuser=1#scrollTo=eXCiDVAFDkzR

YOUTUBE LİNKİ:

# Veri Analizi ve Modelleme Projesi

Bu proje, bir veri seti üzerinde Naive Bayes, K-Nearest Neighbors (KNN) ve Lineer Regresyon modellerini kullanarak sınıflandırma ve regresyon analizi gerçekleştirmektedir. Proje, veri ön işleme, model oluşturma ve performans değerlendirmesi adımlarını içermektedir.

## Proje Amacı

Bu projenin amacı, ekte belirttiğim(data_science_job.csv) veri seti üzerinde farklı makine öğrenimi algoritmalarını uygulayarak en iyi performansı gösteren modeli belirlemektir. Proje, veri analizi ve modelleme süreçlerini kapsamaktadır.

## Kullanılan Yöntemler

### Naive Bayes

1. **Veri Yükleme**: Veri seti yüklendi.
2. **Kategorik Değişkenleri Sayısal Hale Getirme**: Kategorik değişkenler sayısal hale getirildi.
3. **Özellikler ve Hedef Değişkeni Ayırma**: Özellikler (X) ve hedef değişken (y) belirlendi.
4. **Eksik Değerleri Doldurma**: Eksik değerler ortalama ile dolduruldu.
5. **Eğitim ve Test Setlerine Ayırma**: Veri seti eğitim ve test setlerine ayrıldı.
6. **Model Oluşturma**: Naive Bayes modeli oluşturuldu ve eğitildi.
7. **Tahmin Yapma**: Test seti üzerinde tahmin yapıldı.
8. **Performans Değerlendirme**: Doğruluk, kesinlik, hatırlama ve F1 skoru hesaplandı.

### K-Nearest Neighbors (KNN)

1. **Veri Yükleme**: Veri seti yüklendi.
2. **Kategorik Değişkenleri Sayısal Hale Getirme**: Kategorik değişkenler sayısal hale getirildi.
3. **Özellikler ve Hedef Değişkeni Ayırma**: Özellikler (X) ve hedef değişken (y) belirlendi.
4. **Eksik Değerleri Doldurma**: Eksik değerler dolduruldu.
5. **Eğitim ve Test Setlerine Ayırma**: Veri seti eğitim ve test setlerine ayrıldı.
6. **Model Oluşturma**: KNN modeli oluşturuldu ve eğitildi.
7. **Tahmin Yapma**: Test seti üzerinde tahmin yapıldı.
8. **Performans Değerlendirme**: Doğruluk, kesinlik, hatırlama ve F1 skoru hesaplandı.

### Lineer Regresyon

1. **Veri Yükleme**: Veri seti yüklendi.
2. **Kategorik Değişkenleri Sayısal Hale Getirme**: Kategorik değişkenler sayısal hale getirildi.
3. **Özellikler ve Hedef Değişkeni Ayırma**: Özellikler (X) ve hedef değişken (y) belirlendi.
4. **Eksik Değerleri Doldurma**: Eksik değerler dolduruldu.
5. **Eğitim ve Test Setlerine Ayırma**: Veri seti eğitim ve test setlerine ayrıldı.
6. **Model Oluşturma**: Lineer regresyon modeli oluşturuldu ve eğitildi.
7. **Tahmin Yapma**: Test seti üzerinde tahmin yapıldı.
8. **Performans Değerlendirme**: Mean Squared Error (MSE) ve R² skoru hesaplandı.

## Sonuçlar

- **Naive Bayes**: Model, belirli bir doğruluk oranı ile çalıştı. Kesinlik, hatırlama ve F1 skoru gibi metrikler değerlendirildi.
- Doğruluk: 0.7643528183716075
- **K-Nearest Neighbors (KNN)**: Model, benzer şekilde belirli bir doğruluk oranı ile çalıştı. Performans metrikleri hesaplandı.
- Doğruluk: 0.7291231732776617
- **Lineer Regresyon**: Sürekli bir hedef değişken için MSE ve R² skoru hesaplandı. Modelin performansı değerlendirildi.
- Doğruluk: 0.7661795407098121

### Hangi Model Daha Uygun?

- **Kategorik Hedef Değişkenler**: Naive Bayes veya KNN modelleri daha uygun.
- **Sürekli Hedef Değişkenler**: Lineer Regresyon en uygun modeldir.


##Sınıflandırma Raporu

-**Naive Bayes**:
              precision    recall  f1-score   support

         0.0       0.82      0.88      0.85      2880
         1.0       0.53      0.43      0.47       952

    accuracy                           0.76      3832
   macro avg       0.68      0.65      0.66      3832
weighted avg       0.75      0.76      0.76      3832

-**KNN**:
              precision    recall  f1-score   support

         0.0       0.77      0.90      0.83      2880
         1.0       0.41      0.21      0.28       952

    accuracy                           0.73      3832
   macro avg       0.59      0.55      0.55      3832
weighted avg       0.68      0.73      0.69      3832

-**Lineer Regression**:
              precision    recall  f1-score   support

         0.0       0.79      0.93      0.86      2880
         1.0       0.56      0.27      0.36       952

    accuracy                           0.77      3832
   macro avg       0.68      0.60      0.61      3832
weighted avg       0.74      0.77      0.73      3832


Kapsamlı Sonuç Analizi



###NAIVE BAYES
**Güçlü Yönler**:
1. Hızlı ve verimli bir modeldir, büyük veri setlerinde iyi performans gösterir.
2. Özellikler arasında bağımsızlık varsayımı geçerli olduğunda etkili sonuçlar verir.

**Zayıf Yönler**:
1. Özellikler arasında güçlü bir ilişki varsa, modelin performansı düşebilir.
2. Sınıflar arasında belirgin bir ayrım yoksa, tahminlerde zayıf kalabilir.

**Uygunluk**: Naive Bayes, veri setinizdeki özelliklerin bağımsızlık varsayımını sağladığı durumlarda iyi sonuçlar verebilir. Eğer veri setinizdeki özellikler arasında bağımsızlık varsa, bu model uygun bir seçimdir.

---

### K-Nearest Neighbors (KNN)
**Güçlü Yönler**:
1. Basit ve anlaşılır bir modeldir, sınıflar arasında belirgin bir ayrım olduğunda iyi performans gösterir.
2. Özelliklerin etkisini doğrudan gözlemleme imkanı sunar.

**Zayıf Yönler**:
1. Büyük veri setlerinde yavaş çalışabilir, çünkü her tahmin için tüm eğitim verilerini kullanır.
2. Gürültülü verilerde ve çok sayıda özellik olduğunda performansı düşebilir.

**Uygunluk**: KNN, veri setinizdeki sınıflar arasında belirgin bir ayrım olduğunda iyi sonuçlar verebilir. Özelliklerin ölçeklendirilmesi önemlidir; bu nedenle, verilerinizi normalleştirmek veya standartlaştırmak gerekebilir.

---

### Lineer Regresyon

**Güçlü Yönler**:
1. Sürekli bir hedef değişkeni tahmin etmek için etkili bir modeldir.
2. Modelin sonuçları kolayca yorumlanabilir ve anlaşılabilir.

**Zayıf Yönler**:
1. Modelin varsayımları (doğrusal ilişki, normal dağılım, homoscedasticity) sağlanmadığında performansı düşebilir.
2. Aykırı değerlere karşı hassastır.

**Uygunluk**: Lineer Regresyon, sürekli bir hedef değişkeni varsa ve modelin varsayımları sağlanıyorsa en uygun modeldir.

---

### Genel Değerlendirme

Naive Bayes ve KNN modelleri, veri setimdeki kategorik hedef değişken için uygun seçeneklerdir. Her iki modelin performansını karşılaştırarak en iyi sonucu veren modeli seçebiliriz.(Ben Lineer Regression'u tercih ederim.)
Lineer Regresyon, sürekli bir hedef değişken için uygundur ve bu durumda en iyi sonuçları verebilir. Ancak, hedef değişkeniniz kategorikse, bu model uygun değildir.

### Sonuç

Bu projede, Naive Bayes, KNN ve Lineer Regresyon modellerini kullanarak veri setim üzerinde kapsamlı bir analiz gerçekleştirdim. Her modelin performansını değerlendirerek, veri setinize en uygun olanı belirlemek için gerekli adımları attım. Elde edilen sonuçlar, model seçiminde ve veri analizi süreçlerinde rehberlik edecektir.






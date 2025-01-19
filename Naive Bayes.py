#Naive Bayes
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Veri setini yükleme
data = pd.read_csv('data_science_job.csv')  

# Veri setinin ilk birkaç satırını görüntüleme
print(data.head())

# Veri setinin genel bilgilerini görüntüleme
print(data.info())

# Temel istatistikleri görüntüleme
print(data.describe())

from sklearn.preprocessing import LabelEncoder

# Kategorik değişkenleri sayısal hale getirme
label_encoder = LabelEncoder()

data['gender'] = label_encoder.fit_transform(data['gender'].fillna('Unknown'))
data['relevent_experience'] = label_encoder.fit_transform(data['relevent_experience'])
data['enrolled_university'] = label_encoder.fit_transform(data['enrolled_university'].fillna('Unknown'))
data['education_level'] = label_encoder.fit_transform(data['education_level'].fillna('Unknown'))
data['major_discipline'] = label_encoder.fit_transform(data['major_discipline'].fillna('Unknown'))
data['company_size'] = label_encoder.fit_transform(data['company_size'].fillna('Unknown'))
data['company_type'] = label_encoder.fit_transform(data['company_type'].fillna('Unknown'))

# training_hours dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['training_hours'], bins=30, kde=True)
plt.title('Eğitim Saatleri Dağılımı')
plt.xlabel('Eğitim Saatleri')
plt.ylabel('Frekans')
plt.show()


# experience dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['experience'], bins=30, kde=True)
plt.title('Deneyim Dağılımı')
plt.xlabel('Deneyim (Yıl)')
plt.ylabel('Frekans')
plt.show()

# city_development_index dağılımı
plt.figure(figsize=(10, 6))
sns.histplot(data['city_development_index'], bins=30, kde=True)
plt.title('Şehir Gelişim İndeksi Dağılımı')
plt.xlabel('Şehir Gelişim İndeksi')
plt.ylabel('Frekans')
plt.show()

from sklearn.impute import SimpleImputer  
# Özellikler ve hedef değişkeni ayırma
X = data[['training_hours', 'experience', 'city_development_index']]  # Özellikler
y = data['target']  # Hedef değişken

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='mean')
X[['training_hours', 'experience', 'city_development_index']] = imputer.fit_transform(X[['training_hours', 'experience', 'city_development_index']])

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes modelini oluşturma
model = GaussianNB()
model.fit(X_train, y_train)  # Modeli eğitme

# Test seti ile tahmin yapma
y_pred = model.predict(X_test)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy}')

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sınıflandırma raporunu al
report = classification_report(y_test, y_pred, output_dict=True)

# Sınıflar ve metrikler
classes = list(report.keys())[:-3]  # Son üç anahtar (accuracy, macro avg, weighted avg) hariç
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Sınıflandırma raporunu yazdırma
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# Karmaşıklık matrisini hesaplama
conf_matrix = confusion_matrix(y_test, y_pred)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Sınıflandırma raporunu al
report = classification_report(y_test, y_pred, output_dict=True)

# Sınıflar ve metrikler
classes = list(report.keys())[:-3]  # Son üç anahtar (accuracy, macro avg, weighted avg) hariç
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Görselleştirme
x = np.arange(len(classes))  # sınıf sayısı kadar x ekseni
width = 0.25  # çubuk genişliği

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width, precision, width, label='Kesinlik (Precision)', color='b')
bars2 = ax.bar(x, recall, width, label='Hatırlama (Recall)', color='g')
bars3 = ax.bar(x + width, f1_score, width, label='F1 Skoru', color='r')

# Eksen ayarları
ax.set_xlabel('Sınıflar')
ax.set_ylabel('Değerler')
ax.set_title('Sınıflandırma Raporu Metrikleri')
ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45)
ax.legend()

plt.show()
#Lineer Regression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
data = pd.read_csv('data_science_job.csv')  

# Kategorik değişkenleri sayısal hale getirme (gerekirse)
data = pd.get_dummies(data, columns=['city', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'company_size', 'company_type'], drop_first=True)

# Özellikler ve hedef değişkeni ayırma
X = data[['training_hours', 'experience', 'city_development_index']]  # Özellikler
y = data['target']  # Hedef değişken

# Eksik değerleri doldurma
imputer = SimpleImputer(strategy='mean')
X[['training_hours', 'experience', 'city_development_index']] = imputer.fit_transform(X[['training_hours', 'experience', 'city_development_index']])


# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Lineer regresyon modelini oluşturma
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)  # Modeli eğitme

# Test seti ile tahmin yapma
y_pred = linear_model.predict(X_test)

# Model performansını değerlendirme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R^2 Score: {r2}')

# Gerçek ve tahmin edilen değerleri görselleştirme
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Y = X çizgisi
plt.xlabel('Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler')
plt.title('Gerçek ve Tahmin Edilen Değerler')
plt.show()

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Hedef değişkeni sınıflandırma (örneğin, 0.5 eşik değeri)
y_pred_class = (y_pred >= 0.5).astype(int)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred_class)
print(f'Doğruluk: {accuracy}')

# Sınıflandırma raporunu al
report = classification_report(y_test, y_pred_class, output_dict=True)

# Sınıflar ve metrikler
classes = list(report.keys())[:-3]  # Son üç anahtar (accuracy, macro avg, weighted avg) hariç
precision = [report[cls]['precision'] for cls in classes]
recall = [report[cls]['recall'] for cls in classes]
f1_score = [report[cls]['f1-score'] for cls in classes]

# Sınıflandırma raporunu yazdırma
print("Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_class))

# Karmaşıklık matrisini hesaplama
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Karmaşıklık matrisini görselleştirme
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.title('Karmaşıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()


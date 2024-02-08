
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
file_path = 'cars.csv'
cars_data = pd.read_csv(file_path)

# Определение целевой переменной
target = cars_data['ID_MARK']
# Предварительная обработка данных
numeric_columns = ['Год от', 'Год до']
cars_numeric = cars_data[numeric_columns].copy()

# Заполнение пропущенных значений в 'Год до' средними значениями
cars_numeric['Год до'] = cars_numeric['Год до'].fillna(cars_numeric['Год от'].mean())


# Стандартизация числовых данных
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(cars_numeric)

# Применение PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(numeric_features_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Визуализация пространства главных компонент
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', data=principal_df)
plt.title('PCA of Cars Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Добавление категориальных признаков с One-Hot-кодированием для сравнения
categorical_columns = cars_data.select_dtypes(include=['object']).columns
cars_categorical = cars_data[categorical_columns]

# Применение One-Hot-кодирования к категориальным признакам
encoder = OneHotEncoder()
cars_categorical_encoded = encoder.fit_transform(cars_categorical)

# Объединение PCA-преобразованных числовых признаков с One-Hot-кодированными категориальными признаками
combined_features = np.concatenate([principal_components, cars_categorical_encoded.toarray()], axis=1)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(combined_features, target, test_size=0.3, random_state=42)

# Обучение модели на объединенных данных
model_combined = LogisticRegression()
model_combined.fit(X_train, y_train)
predictions_combined = model_combined.predict(X_test)
accuracy_combined = accuracy_score(y_test, predictions_combined)

print("точность модели:", accuracy_combined)

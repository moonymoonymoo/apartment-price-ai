import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv("data.csv")

# Кодируем категориальные признаки
for col in ["district", "building_type"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Убираем целевую переменную (price)
X = df.drop("price", axis=1)

# Запускаем кластеризацию
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X)

# Сохраняем результат
df.to_csv("clustered_data.csv", index=False)
print("Кластеры присвоены. Данные сохранены в clustered_data.csv")

# Визуализация (по площади и цене)
plt.figure(figsize=(8, 5))
scatter = plt.scatter(df["square"], df["price"], c=df["cluster"], cmap="viridis", s=60, edgecolors='k')
plt.xlabel("Площадь (м²)")
plt.ylabel("Цена (₸)")
plt.title("Кластеризация квартир по площади и цене")
plt.colorbar(scatter, label="Кластер")
plt.grid(True)
plt.tight_layout()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

# Модели
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Загрузка данных
df = pd.read_csv("data.csv")

for col in ['district', 'building_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsRegressor(),
    "SVM": SVR(),
    "Gradient Boosting": GradientBoostingRegressor()
}

print("MODEL COMPARISON:\n")

# Обучение и оценка каждой модели
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"{name}:\n MAE = {mae:.2f}, R² = {r2:.2f}\n")
    except Exception as e:
        print(f"{name}: ERROR -> {e}\n")

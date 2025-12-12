# Импорт библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import xgboost as xgb

# 1. Загрузка данных
df = pd.read_csv('wine-dataset.csv')  # укажите путь к файлу


# 2. Предварительный анализ
print("Первые 5 строк:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())
print("\nСтатистики:")
print(df.describe())

# 3. Обработка пропусков
df.dropna(inplace=True)  # удаление строк с пропусками

# 4. Анализ целевой переменной
print("Распределение классов (quality):")
print(df['quality'].value_counts())

# 5. Выбор признаков и целевой переменной
X = df.drop('quality', axis=1)  # все признаки кроме quality
y = df['quality']  # целевая переменная

# 6. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Нормализация признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Обучение моделей

# Модель 1: Логистическая регрессия
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Модель 2: Случайный лес
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Модель 3: XGBoost
xgb_model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

# Модель 4: SVM
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

# 9. Оценка моделей
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"{model_name}:")
    print(f"  Точность (Accuracy): {acc:.4f}")
    print(f"  F1‑мера (weighted): {f1:.4f}")
    print("  Матрица ошибок:")
    print(confusion_matrix(y_true, y_pred))
    print("\n" + "-"*50 + "\n")

# Оценка всех моделей
evaluate_model(y_test, y_pred_lr, "Логистическая регрессия")
evaluate_model(y_test, y_pred_rf, "Случайный лес")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_svm, "SVM")

# 10. Детальный отчёт для лучшей модели (например, Random Forest)
print("Детальный отчёт (Random Forest):")
print(classification_report(y_test, y_pred_rf))

# 11. Визуализация матрицы ошибок (для Random Forest)
plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_test, y_pred_rf),
    annot=True, fmt='d', cmap='Blues',
    xticklabels=sorted(y.unique()),
    yticklabels=sorted(y.unique())
)
plt.title("Матрица ошибок (Random Forest)")
plt.xlabel("Предсказанное значение")
plt.ylabel("Истинное значение")
plt.show()

# 12. Важность признаков (для Random Forest)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title("Важность признаков (Random Forest)")
plt.xlabel("Важность")
plt.show()
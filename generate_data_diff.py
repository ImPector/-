--- generate_data.py (原始)


+++ generate_data.py (修改后)
#!/usr/bin/env python3
"""
Генерация данных для трех сценариев множественной линейной регрессии
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

# Создаем директорию для данных
os.makedirs('data', exist_ok=True)

# =============================================================================
# Сценарий А: Независимые признаки разного масштаба
# =============================================================================
print("Генерация сценария А...")

n_samples = 500

# Признаки разного масштаба
X1_A = np.random.normal(0, 1, n_samples)  # масштаб ~1
X2_A = np.random.uniform(0, 1000, n_samples)  # масштаб ~1000
X3_A = np.random.exponential(10, n_samples)  # масштаб ~10

# Целевая переменная (истинная модель)
y_A = 5 * X1_A + 0.01 * X2_A + 2 * X3_A + np.random.normal(0, 5, n_samples)

# Создаем DataFrame
df_A = pd.DataFrame({
    'feature_small_scale': X1_A,
    'feature_large_scale': X2_A,
    'feature_exponential': X3_A,
    'non_informative': np.random.randn(n_samples),  # неинформативный признак
    'target': y_A
})

# Добавляем пропуски (5%)
missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
df_A.loc[missing_indices[:10], 'feature_small_scale'] = np.nan
df_A.loc[missing_indices[10:20], 'feature_large_scale'] = np.nan

# Добавляем выбросы (3%)
outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
df_A.loc[outlier_indices[:5], 'feature_small_scale'] = df_A.loc[outlier_indices[:5], 'feature_small_scale'] * 10
df_A.loc[outlier_indices[5:10], 'feature_large_scale'] = df_A.loc[outlier_indices[5:10], 'feature_large_scale'] * 5

# Добавляем некорректные значения (2%)
incorrect_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
df_A.loc[incorrect_indices[:5], 'feature_exponential'] = -999  # некорректное значение
df_A.loc[incorrect_indices[5:], 'target'] = np.inf  # бесконечность

df_A.to_csv('data/scenario_A.csv', index=False)
print(f"Сценарий А сохранен: {df_A.shape}")

# =============================================================================
# Сценарий Б: Числовые и категориальные признаки
# =============================================================================
print("Генерация сценария Б...")

n_samples = 600

# Числовые признаки
X1_B = np.random.normal(50, 15, n_samples)
X2_B = np.random.uniform(0, 100, n_samples)

# Категориальные признаки
categories = ['A', 'B', 'C', 'D']
cat1_B = np.random.choice(categories, n_samples, p=[0.4, 0.3, 0.2, 0.1])
cat2_B = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2])

# Целевая переменная (с учетом категорий)
cat1_effect = {'A': 0, 'B': 5, 'C': 10, 'D': 15}
cat2_effect = {'Low': -5, 'Medium': 0, 'High': 8}

y_B = (0.5 * X1_B +
       0.3 * X2_B +
       np.array([cat1_effect[c] for c in cat1_B]) +
       np.array([cat2_effect[c] for c in cat2_B]) +
       np.random.normal(0, 8, n_samples))

df_B = pd.DataFrame({
    'numeric_feature_1': X1_B,
    'numeric_feature_2': X2_B,
    'categorical_1': cat1_B,
    'categorical_2': cat2_B,
    'non_informative_cat': np.random.choice(['X', 'Y', 'Z'], n_samples),  # неинформативный
    'target': y_B
})

# Добавляем пропуски
missing_B = np.random.choice(n_samples, size=int(n_samples * 0.06), replace=False)
df_B.loc[missing_B[:15], 'numeric_feature_1'] = np.nan
df_B.loc[missing_B[15:30], 'categorical_1'] = np.nan

# Добавляем выбросы
outlier_B = np.random.choice(n_samples, size=int(n_samples * 0.04), replace=False)
df_B.loc[outlier_B[:10], 'numeric_feature_2'] = df_B.loc[outlier_B[:10], 'numeric_feature_2'] * 8

# Добавляем некорректные значения
df_B.loc[np.random.choice(n_samples, 10), 'categorical_1'] = 'INVALID'
df_B.loc[np.random.choice(n_samples, 5), 'target'] = -9999

df_B.to_csv('data/scenario_B.csv', index=False)
print(f"Сценарий Б сохранен: {df_B.shape}")

# =============================================================================
# Сценарий В: Числовые признаки с мультиколлинеарностью
# =============================================================================
print("Генерация сценария В...")

n_samples = 550

# Базовый признак
X_base = np.random.normal(100, 20, n_samples)

# Сильно коррелированные признаки (мультиколлинеарность)
X1_V = X_base + np.random.normal(0, 2, n_samples)  # высокая корреляция с X_base
X2_V = X_base * 1.05 + np.random.normal(0, 3, n_samples)  # высокая корреляция с X1_V
X3_V = 0.9 * X1_V + 0.1 * X2_V + np.random.normal(0, 1, n_samples)  # линейная комбинация

# Независимый признак
X4_V = np.random.uniform(-50, 50, n_samples)

# Целевая переменная
y_V = (2 * X1_V +
       1.5 * X4_V +
       np.random.normal(0, 10, n_samples))

df_V = pd.DataFrame({
    'collinear_1': X1_V,
    'collinear_2': X2_V,
    'collinear_3': X3_V,
    'independent': X4_V,
    'non_informative': np.random.randn(n_samples),  # неинформативный
    'target': y_V
})

# Добавляем пропуски
missing_V = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
df_V.loc[missing_V[:12], 'collinear_1'] = np.nan
df_V.loc[missing_V[12:22], 'independent'] = np.nan

# Добавляем выбросы
outlier_V = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
df_V.loc[outlier_V[:8], 'collinear_2'] = df_V.loc[outlier_V[:8], 'collinear_2'] * 7

# Добавляем некорректные значения
df_V.loc[np.random.choice(n_samples, 8), 'collinear_3'] = np.nan
df_V.loc[np.random.choice(n_samples, 5), 'target'] = 999999  # явный выброс в target

df_V.to_csv('data/scenario_C.csv', index=False)
print(f"Сценарий В сохранен: {df_V.shape}")

print("\n=== Все данные сгенерированы и сохранены в папке data/ ===")
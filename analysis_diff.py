#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Настройка стилей для графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("=" * 80)
print("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ ДАННЫХ И РЕГРЕССИОННОЕ МОДЕЛИРОВАНИЕ")
print("=" * 80)


# =============================================================================
# ФУНКЦИИ ДЛЯ ОБРАБОТКИ И АНАЛИЗА
# =============================================================================

def load_and_inspect(filepath, scenario_name):
    """Загрузка данных и первичный осмотр"""
    print(f"\n{'='*60}")
    print(f"СЦЕНАРИЙ {scenario_name}: {filepath}")
    print('='*60)

    df = pd.read_csv(filepath)
    print(f"\n1. ЗАГРУЗКА ДАННЫХ")
    print(f"   Размерность: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"\n   Первые 5 строк:")
    print(df.head())
    return df


def primary_data_preparation(df, scenario_name):
    """Первичная подготовка и обработка данных"""
    print(f"\n2. ПЕРВИЧНАЯ ПОДГОТОВКА ДАННЫХ")

    # Копия для работы
    df_clean = df.copy()

    # Проверка типов данных
    print(f"\n   Типы данных:")
    print(df_clean.dtypes)

    # Проверка пропусков
    print(f"\n   Пропуски:")
    missing = df_clean.isnull().sum()
    missing_pct = (missing / len(df_clean) * 100).round(2)
    missing_info = pd.DataFrame({'count': missing, 'percent': missing_pct})
    print(missing_info[missing_info['count'] > 0])

    # Поиск некорректных значений (бесконечности, очень большие значения)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df_clean[col]).sum()
        if inf_count > 0:
            print(f"   ⚠ Найдено {inf_count} бесконечных значений в '{col}'")
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)

        # Поиск явно некорректных значений (например, -999, 999999)
        if (df_clean[col] == -999).sum() > 0:
            print(f"   ⚠ Найдены некорректные значения -999 в '{col}'")
            df_clean.loc[df_clean[col] == -999, col] = np.nan

        if (df_clean[col] == 999999).sum() > 0:
            print(f"   ⚠ Найдены подозрительные значения 999999 в '{col}'")
            df_clean.loc[df_clean[col] == 999999, col] = np.nan

    # Обработка категориальных некорректных значений
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        invalid_vals = df_clean[col][df_clean[col].isin(['INVALID', 'N/A', ''])]
        if len(invalid_vals) > 0:
            print(f"   ⚠ Найдены некорректные категории в '{col}': {invalid_vals.unique()}")
            df_clean.loc[df_clean[col].isin(['INVALID', 'N/A', '']), col] = np.nan

    # Заполнение пропусков медианой для числовых и модой для категориальных
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            print(f"   ✓ Заполнены пропуски в '{col}' медианой: {median_val:.2f}")

    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"   ✓ Заполнены пропуски в '{col}' модой: {mode_val}")

    # Обработка выбросов в целевой переменной (если есть target)
    if 'target' in df_clean.columns:
        Q1 = df_clean['target'].quantile(0.25)
        Q3 = df_clean['target'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask = (df_clean['target'] < lower_bound) | (df_clean['target'] > upper_bound)
        outlier_count = outlier_mask.sum()

        if outlier_count > 0:
            print(f"\n   ⚠ Найдено {outlier_count} выбросов в целевой переменной 'target'")
            print(f"   Границы допустимых значений: [{lower_bound:.2f}, {upper_bound:.2f}]")
            # Замена выбросов на границы
            df_clean.loc[df_clean['target'] < lower_bound, 'target'] = lower_bound
            df_clean.loc[df_clean['target'] > upper_bound, 'target'] = upper_bound
            print(f"   ✓ Выбросы заменены на границы интервала")

    print(f"\n   Итоговая размерность после очистки: {df_clean.shape}")
    return df_clean


def detect_outliers_iqr(df, column):
    """Обнаружение выбросов методом IQR"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound


def univariate_analysis(df, scenario_name):
    """Одномерный описательный анализ"""
    print(f"\n3. ОДНОМЕРНЫЙ ОПИСАТЕЛЬНЫЙ АНАЛИЗ")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    print(f"\n   Статистики числовых признаков:")
    print(df[numeric_cols].describe().round(2))

    # Анализ распределения
    print(f"\n   Анализ распределения:")
    for col in numeric_cols:
        skewness = df[col].skew().round(2)
        kurtosis = df[col].kurtosis().round(2)
        print(f"   {col}: асимметрия={skewness}, эксцесс={kurtosis}")

    # Обнаружение выбросов
    print(f"\n   Выбросы (метод IQR):")
    for col in numeric_cols:
        outliers, lb, ub = detect_outliers_iqr(df, col)
        outlier_count = len(outliers)
        outlier_pct = round((outlier_count / len(df) * 100), 2)
        if outlier_count > 0:
            print(f"   {col}: {outlier_count} выбросов ({outlier_pct}%), границы: [{lb:.2f}, {ub:.2f}]")

    return numeric_cols


def exploratory_data_analysis(df, scenario_name):
    """Разведочный анализ данных"""
    print(f"\n4. РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Корреляционная матрица
    print(f"\n   Корреляционная матрица:")
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix.round(2))

    # Выявление сильных корреляций
    print(f"\n   Сильные корреляции (>0.7):")
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j],
                                   corr_matrix.iloc[i, j]))
                print(f"   {corr_matrix.columns[i]} ↔ {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.2f}")

    if not strong_corr:
        print("   Сильных корреляций не обнаружено")

    # Анализ категориальных признаков (если есть)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"\n   Категориальные признаки:")
        for col in cat_cols:
            print(f"   {col}: {df[col].nunique()} уникальных значений")
            print(f"      Распределение: {dict(df[col].value_counts())}")

    return corr_matrix, strong_corr


def prepare_features(df, target_col='target'):
    """Подготовка признаков для моделирования"""
    df_prep = df.copy()

    # Разделение на числовые и категориальные
    numeric_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_prep.select_dtypes(include=['object']).columns.tolist()

    # Убираем target из признаков
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in cat_cols:
        cat_cols.remove(target_col)

    # One-Hot Encoding для категориальных признаков
    if cat_cols:
        df_encoded = pd.get_dummies(df_prep, columns=cat_cols, drop_first=True)
        feature_cols = [c for c in df_encoded.columns if c != target_col]
    else:
        df_encoded = df_prep
        feature_cols = numeric_cols

    X = df_encoded[feature_cols]
    y = df_encoded[target_col]

    return X, y, feature_cols


def build_regression_model(X, y, scenario_name, use_ridge=False):
    """Построение и оценка регрессионной модели"""
    print(f"\n5. ПОСТРОЕНИЕ РЕГРЕССИОННОЙ МОДЕЛИ")

    # Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"   Обучающая выборка: {X_train.shape[0]} образцов")
    print(f"   Тестовая выборка: {X_test.shape[0]} образцов")

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Выбор модели
    if use_ridge:
        model = Ridge(alpha=1.0)
        model_name = "Ridge Regression"
    else:
        model = LinearRegression()
        model_name = "Linear Regression"

    print(f"   Используемая модель: {model_name}")

    # Обучение модели
    model.fit(X_train_scaled, y_train)

    # Предсказания
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Оценка качества
    print(f"\n   Метрики качества:")

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"   MSE (train/test): {train_mse:.2f} / {test_mse:.2f}")
    print(f"   RMSE (train/test): {train_rmse:.2f} / {test_rmse:.2f}")
    print(f"   MAE (train/test): {train_mae:.2f} / {test_mae:.2f}")
    print(f"   R² (train/test): {train_r2:.4f} / {test_r2:.4f}")

    # Коэффициенты модели
    print(f"\n   Коэффициенты модели:")
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    print(coef_df.to_string(index=False))

    # Кросс-валидация
    print(f"\n   Кросс-валидация (5-fold):")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train,
                                cv=cv, scoring='r2')
    print(f"   R² по фолдам: {cv_scores.round(4)}")
    print(f"   Средний R²: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    metrics = {
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    }

    return model, scaler, metrics, coef_df


def analyze_scenario(filepath, scenario_name, use_ridge=False):
    """Полный анализ одного сценария"""
    print("\n" + "="*80)
    print(f" АНАЛИЗ СЦЕНАРИЯ {scenario_name}")
    print("="*80)

    # 2. Загрузка данных
    df = load_and_inspect(filepath, scenario_name)

    # 3. Первичная подготовка
    df_clean = primary_data_preparation(df, scenario_name)

    # 4. Одномерный анализ
    numeric_cols = univariate_analysis(df_clean, scenario_name)

    # 5. Разведочный анализ
    corr_matrix, strong_corr = exploratory_data_analysis(df_clean, scenario_name)

    # 6. Построение модели
    X, y, feature_cols = prepare_features(df_clean)
    model, scaler, metrics, coef_df = build_regression_model(
        X, y, scenario_name, use_ridge=use_ridge
    )

    return metrics, coef_df, strong_corr


# =============================================================================
# ОСНОВНОЙ АНАЛИЗ ВСЕХ СЦЕНАРИЕВ
# =============================================================================

if __name__ == "__main__":

    # Словарь для хранения результатов
    all_results = {}

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ А: Независимые признаки разного масштаба
    # -------------------------------------------------------------------------
    metrics_A, coef_A, corr_A = analyze_scenario(
        'data/scenario_A.csv',
        'А (разный масштаб)',
        use_ridge=False
    )
    all_results['A'] = {'metrics': metrics_A, 'coefficients': coef_A, 'correlations': corr_A}

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ Б: Числовые и категориальные признаки
    # -------------------------------------------------------------------------
    metrics_B, coef_B, corr_B = analyze_scenario(
        'data/scenario_B.csv',
        'Б (категориальные)',
        use_ridge=False
    )
    all_results['B'] = {'metrics': metrics_B, 'coefficients': coef_B, 'correlations': corr_B}

    # -------------------------------------------------------------------------
    # СЦЕНАРИЙ В: Мультиколлинеарность (используем Ridge)
    # -------------------------------------------------------------------------
    metrics_C, coef_C, corr_C = analyze_scenario(
        'data/scenario_C.csv',
        'В (мультиколлинеарность)',
        use_ridge=True  # Ridge лучше справляется с мультиколлинеарностью
    )
    all_results['C'] = {'metrics': metrics_C, 'coefficients': coef_C, 'correlations': corr_C}

    # ==========================================================================
    # СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # ==========================================================================
    print("\n" + "="*80)
    print(" СРАВНЕНИЕ РЕЗУЛЬТАТОВ ВСЕХ СЦЕНАРИЕВ")
    print("="*80)

    comparison_df = pd.DataFrame({
        'Сценарий А': metrics_A,
        'Сценарий Б': metrics_B,
        'Сценарий В': metrics_C
    })

    print("\nСравнительная таблица метрик:")
    print(comparison_df.round(4))

    print("\n\nКлючевые выводы:")
    print("-" * 60)

    # Анализ сценария А
    print("\n1. СЦЕНАРИЙ А (разный масштаб признаков):")
    print(f"   - R² тест: {metrics_A['test_r2']:.4f}")
    print(f"   - Кросс-валидация: {metrics_A['cv_r2_mean']:.4f} (+/- {metrics_A['cv_r2_std']*2:.4f})")
    print("   - Масштабирование признаков критически важно для корректной работы")

    # Анализ сценария Б
    print("\n2. СЦЕНАРИЙ Б (категориальные признаки):")
    print(f"   - R² тест: {metrics_B['test_r2']:.4f}")
    print(f"   - Кросс-валидация: {metrics_B['cv_r2_mean']:.4f} (+/- {metrics_B['cv_r2_std']*2:.4f})")
    print("   - One-Hot Encoding позволил использовать категориальные признаки")

    # Анализ сценария В
    print("\n3. СЦЕНАРИЙ В (мультиколлинеарность):")
    print(f"   - R² тест: {metrics_C['test_r2']:.4f}")
    print(f"   - Кросс-валидация: {metrics_C['cv_r2_mean']:.4f} (+/- {metrics_C['cv_r2_std']*2:.4f})")
    print("   - Ridge регуляризация помогла стабилизировать коэффициенты")
    if corr_C:
        print(f"   - Обнаружено сильных корреляций: {len(corr_C)}")

    print("\n" + "="*80)
    print(" АНАЛИЗ ЗАВЕРШЕН")
    print("="*80)

"""
Лабораторная работа по факторному анализу
Факторный анализ многомерных данных
ИСПОЛЬЗОВАНИЕ БИБЛИОТЕКИ factor-analyzer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from scipy import stats
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
import warnings
warnings.filterwarnings('ignore')

# Настройка для отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

# Загрузка данных
print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА ПО ФАКТОРНОМУ АНАЛИЗУ")
print("=" * 80)

# Загружаем датасет Wine
wine_data = load_wine()
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)

print("\n1. ОПИСАНИЕ ИСХОДНЫХ ДАННЫХ")
print("-" * 80)
print(f"Количество наблюдений: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1]}")
print("\nНазвания признаков:")
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\nПервые 5 строк данных:")
print(df.head())

print("\n2. ДЕСКРИПТИВНЫЙ АНАЛИЗ")
print("-" * 80)
print("\nОписательная статистика:")
print(df.describe())

# Проверка на пропущенные значения
print(f"\nПропущенные значения:")
print(df.isnull().sum().sum())

# Проверка нормальности распределения
print("\n3. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ")
print("-" * 80)
print("\nТест Шапиро-Уилка (выборочно для нескольких переменных):")
normality_results = []
for col in df.columns[:5]:  # Проверяем первые 5 переменных
    stat, p_value = stats.shapiro(df[col])
    normality_results.append({
        'Переменная': col,
        'Статистика': round(stat, 4),
        'p-value': round(p_value, 4),
        'Нормальность': 'Да' if p_value > 0.05 else 'Нет'
    })
    
normality_df = pd.DataFrame(normality_results)
print(normality_df.to_string(index=False))

# Визуализация распределений
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()
for i, col in enumerate(df.columns[:9]):
    axes[i].hist(df[col], bins=20, edgecolor='black', alpha=0.7)
    axes[i].set_title(f'{col}')
    axes[i].set_xlabel('Значение')
    axes[i].set_ylabel('Частота')
plt.tight_layout()
plt.savefig('распределения_переменных.png', dpi=300, bbox_inches='tight')
print("\nГрафик распределений сохранен в файл 'распределения_переменных.png'")

# Корреляционный анализ
print("\n4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ")
print("-" * 80)
correlation_matrix = df.corr()
print("\nКорреляционная матрица (первые 5x5):")
print(correlation_matrix.iloc[:5, :5].round(3))

# Визуализация корреляционной матрицы
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица всех переменных', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('корреляционная_матрица.png', dpi=300, bbox_inches='tight')
print("\nКорреляционная матрица сохранена в файл 'корреляционная_матрица.png'")

# Стандартизация переменных
print("\n5. СТАНДАРТИЗАЦИЯ ПЕРЕМЕННЫХ")
print("-" * 80)
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print("Переменные стандартизированы (среднее = 0, стандартное отклонение = 1)")
print("\nПроверка стандартизации:")
print(f"Средние значения: {df_scaled.mean().round(6).tolist()}")
print(f"Стандартные отклонения: {df_scaled.std().round(6).tolist()}")

# Критерии КМО и Бартлетта
print("\n6. КРИТЕРИИ КМО И БАРТЛЕТТА")
print("-" * 80)

# Используем готовые функции из библиотеки factor-analyzer
kmo_all, kmo_model = calculate_kmo(df_scaled)
bartlett_chi2, bartlett_p = calculate_bartlett_sphericity(df_scaled)
bartlett_df = len(df.columns) * (len(df.columns) - 1) / 2

print(f"КМО (Kaiser-Meyer-Olkin): {kmo_model:.4f}")
print(f"Интерпретация КМО: ", end="")
if kmo_model >= 0.9:
    print("Отлично")
elif kmo_model >= 0.8:
    print("Хорошо")
elif kmo_model >= 0.7:
    print("Средне")
elif kmo_model >= 0.6:
    print("Посредственно")
else:
    print("Неприемлемо")

print(f"\nТест сферичности Бартлетта:")
print(f"Хи-квадрат: {bartlett_chi2:.4f}")
print(f"Степени свободы: {int(bartlett_df)}")
print(f"p-value: {bartlett_p:.4f}")
print(f"Интерпретация: {'Данные подходят для факторного анализа' if bartlett_p < 0.05 else 'Данные не подходят для факторного анализа'}")

# Определение количества факторов
print("\n7. ОПРЕДЕЛЕНИЕ КОЛИЧЕСТВА ФАКТОРОВ")
print("-" * 80)

# Используем FactorAnalyzer для получения собственных значений
fa_temp = FactorAnalyzer(rotation=None, method='principal')
fa_temp.fit(df_scaled)
eigenvalues, _ = fa_temp.get_eigenvalues()

print("\nСобственные значения:")
eigenval_df = pd.DataFrame({
    'Фактор': range(1, len(eigenvalues) + 1),
    'Собственное значение': eigenvalues.round(4),
    'Доля дисперсии (%)': (eigenvalues / len(eigenvalues) * 100).round(2),
    'Накопленная доля (%)': (np.cumsum(eigenvalues) / len(eigenvalues) * 100).round(2)
})
print(eigenval_df.to_string(index=False))

# Критерий Кайзера (факторы с собственным значением > 1)
kaiser_factors = np.sum(eigenvalues > 1)
print(f"\nКритерий Кайзера: количество факторов = {kaiser_factors} (собственные значения > 1)")

# График каменистой осыпи (Scree Plot)
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Критерий Кайзера (λ=1)')
plt.xlabel('Номер фактора', fontsize=12)
plt.ylabel('Собственное значение', fontsize=12)
plt.title('График каменистой осыпи (Scree Plot)', fontsize=14, pad=20)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('график_каменистой_осыпи.png', dpi=300, bbox_inches='tight')
print("\nГрафик каменистой осыпи сохранен в файл 'график_каменистой_осыпи.png'")

# Определение количества факторов по каменистой осыпи
scree_factors = kaiser_factors
for i in range(len(eigenvalues) - 1):
    if eigenvalues[i] - eigenvalues[i + 1] < 0.1:  # Если разница мала
        scree_factors = i + 1
        break

print(f"Критерий каменистой осыпи: рекомендуется {scree_factors} факторов")

# Выбираем оптимальное количество факторов
n_factors = max(kaiser_factors, 3)  # Минимум 3 фактора для демонстрации
print(f"\nВыбранное количество факторов для анализа: {n_factors}")

# Факторный анализ
print("\n8. ФАКТОРНЫЙ АНАЛИЗ (МЕТОД ГЛАВНЫХ КОМПОНЕНТ)")
print("-" * 80)

# Используем FactorAnalyzer из библиотеки factor-analyzer
# Факторный анализ без вращения
fa = FactorAnalyzer(n_factors=n_factors, rotation=None, method='principal')
fa.fit(df_scaled)

# Матрица факторных нагрузок до вращения
loadings_before = pd.DataFrame(
    fa.loadings_,
    index=df.columns,
    columns=[f'Фактор {i+1}' for i in range(n_factors)]
)

# Общности
communalities = fa.get_communalities()

# Факторный анализ с вращением Varimax
fa_rotated = FactorAnalyzer(n_factors=n_factors, rotation='varimax', method='principal')
fa_rotated.fit(df_scaled)

# Матрица факторных нагрузок после вращения
loadings_after = pd.DataFrame(
    fa_rotated.loadings_,
    index=df.columns,
    columns=[f'Фактор {i+1}' for i in range(n_factors)]
)

# Получаем факторы для визуализации
factors = fa_rotated.transform(df_scaled)

# Собственные значения для выбранных факторов
eigenvals_sel, _ = fa_rotated.get_eigenvalues()
variance_explained = eigenvals_sel[:n_factors]
total_variance = len(df.columns)
variance_ratio = variance_explained / total_variance

print("\nМатрица факторных нагрузок ДО вращения:")
print(loadings_before.round(3))

# Общности
communalities_df = pd.DataFrame({
    'Переменная': df.columns,
    'Общность': communalities
})
print("\nОбщности (Communalities):")
print(communalities_df.round(3).to_string(index=False))

# Применение вращения Varimax
print("\n9. ПРИМЕНЕНИЕ ВРАЩЕНИЯ VARIMAX")
print("-" * 80)
print("Вращение Varimax выполнено библиотекой factor-analyzer")

print("\nМатрица факторных нагрузок ПОСЛЕ вращения (Varimax):")
print(loadings_after.round(3))

# Визуализация факторных нагрузок
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# До вращения
sns.heatmap(loadings_before, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=axes[0], cbar_kws={"shrink": 0.8})
axes[0].set_title('Матрица факторных нагрузок ДО вращения', fontsize=12, pad=15)

# После вращения
sns.heatmap(loadings_after, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, ax=axes[1], cbar_kws={"shrink": 0.8})
axes[1].set_title('Матрица факторных нагрузок ПОСЛЕ вращения (Varimax)', fontsize=12, pad=15)

plt.tight_layout()
plt.savefig('матрица_факторных_нагрузок.png', dpi=300, bbox_inches='tight')
print("\nМатрица факторных нагрузок сохранена в файл 'матрица_факторных_нагрузок.png'")

# Диаграммы рассеивания для факторов
print("\n10. ДИАГРАММЫ РАССЕИВАНИЯ ДЛЯ ФАКТОРОВ")
print("-" * 80)
factors_df = pd.DataFrame(
    factors,
    columns=[f'Фактор {i+1}' for i in range(n_factors)]
)

# Создаем диаграммы рассеивания для пар факторов
if n_factors >= 2:
    fig, axes = plt.subplots(1, min(3, n_factors - 1), figsize=(15, 5))
    if n_factors == 2:
        axes = [axes]
    for i in range(min(3, n_factors - 1)):
        axes[i].scatter(factors_df.iloc[:, i], factors_df.iloc[:, i+1], 
                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[i].set_xlabel(f'Фактор {i+1}')
        axes[i].set_ylabel(f'Фактор {i+2}')
        axes[i].set_title(f'Диаграмма рассеивания: Фактор {i+1} vs Фактор {i+2}')
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('диаграммы_рассеивания_факторов.png', dpi=300, bbox_inches='tight')
    print("Диаграммы рассеивания сохранены в файл 'диаграммы_рассеивания_факторов.png'")

# Интерпретация факторов
print("\n11. ИНТЕРПРЕТАЦИЯ ФАКТОРОВ")
print("-" * 80)
print("\nИнтерпретация факторов на основе факторных нагрузок (|нагрузка| > 0.5):")
for i in range(n_factors):
    factor_name = f'Фактор {i+1}'
    high_loadings = loadings_after[factor_name].abs() > 0.5
    variables = loadings_after[high_loadings][factor_name].sort_values(ascending=False)
    print(f"\n{factor_name}:")
    for var, loading in variables.items():
        print(f"  {var}: {loading:.3f}")

# Доля объясненной дисперсии
variance_df = pd.DataFrame({
    'Фактор': [f'Фактор {i+1}' for i in range(n_factors)],
    'Собственное значение': variance_explained.round(4),
    'Доля дисперсии (%)': (variance_ratio * 100).round(2),
    'Накопленная доля (%)': (np.cumsum(variance_ratio) * 100).round(2)
})
print("\n\nДоля объясненной дисперсии:")
print(variance_df.to_string(index=False))

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН")
print("=" * 80)


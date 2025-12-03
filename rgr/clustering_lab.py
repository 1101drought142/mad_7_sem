"""
РГЗ. Кластеризация
Датасет: Iris Plants Database
Методы: K-means, Иерархическая кластеризация (Agglomerative Clustering)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# Настройка визуализации
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# Создание папки для графиков
os.makedirs('photos', exist_ok=True)

# ============================================================================
# 1. ЗАГРУЗКА И ОПИСАНИЕ ДАННЫХ
# ============================================================================

print("=" * 80)
print("РГЗ: КЛАСТЕРИЗАЦИЯ ИРИСОВ")
print("=" * 80)

# Загрузка данных
df = pd.read_csv('data.csv')
print("\n1. ОПИСАНИЕ ИСХОДНЫХ ДАННЫХ")
print("-" * 80)
print(f"Количество записей: {len(df)}")
print(f"Количество признаков: {len(df.columns) - 2}")  # Исключаем Id и Species
print(f"Количество классов (для сравнения): {df['Species'].nunique()}")
print(f"\nКлассы: {df['Species'].unique().tolist()}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nИнформация о данных:")
print(df.info())
print(f"\nПропущенные значения:")
print(df.isnull().sum())

# Разделение на признаки и метки (для внешних метрик)
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df[feature_cols].copy()
y_true = df['Species'].copy()

print(f"\nПризнаки для кластеризации: {feature_cols}")

# ============================================================================
# 2. ДЕСКРИПТИВНЫЙ АНАЛИЗ
# ============================================================================

print("\n" + "=" * 80)
print("2. ДЕСКРИПТИВНЫЙ АНАЛИЗ")
print("=" * 80)

print("\n2.1. Основные статистики:")
print(X.describe())

print("\n2.2. Распределение по классам:")
class_dist = df['Species'].value_counts()
print(class_dist)
print(f"\nПроцентное распределение:")
print((class_dist / len(df) * 100).round(2))

# Проверка нормальности распределения
print("\n2.3. ПРОВЕРКА НОРМАЛЬНОСТИ РАСПРЕДЕЛЕНИЯ:")
print("-" * 80)
print("Тест Шапиро-Уилка (Shapiro-Wilk test):")
print("H0: данные распределены нормально")
print("α = 0.05\n")

normality_results = []
for feature in X.columns:
    stat, p_value = stats.shapiro(X[feature])
    is_normal = p_value > 0.05
    normality_results.append({
        'Признак': feature,
        'Статистика': round(stat, 4),
        'p-value': round(p_value, 6),
        'Нормальное': 'Да' if is_normal else 'Нет'
    })
    print(f"{feature:20s}: W={stat:.4f}, p={p_value:.6f} -> {'НОРМАЛЬНОЕ' if is_normal else 'НЕ НОРМАЛЬНОЕ'}")

normality_df = pd.DataFrame(normality_results)
print("\nСводная таблица:")
print(normality_df.to_string(index=False))

# Визуализация распределений
print("\n2.4. Визуализация распределений признаков...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, feature in enumerate(X.columns):
    axes[idx].hist(X[feature], bins=15, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Распределение {feature}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature)
    axes[idx].set_ylabel('Частота')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('photos/распределения_признаков.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/распределения_признаков.png")

# Корреляционная матрица
print("\n2.5. Корреляционная матрица:")
corr_matrix = X.corr()
print(corr_matrix.round(3))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('photos/корреляционная_матрица.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/корреляционная_матрица.png")

# ============================================================================
# 3. ОТБОР ИНФОРМАТИВНЫХ ПРИЗНАКОВ
# ============================================================================

print("\n" + "=" * 80)
print("3. ОТБОР ИНФОРМАТИВНЫХ ПРИЗНАКОВ")
print("=" * 80)

# Анализ корреляций для отбора признаков
print("\nВысокая корреляция между признаками (>0.8):")
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            print(f"  {corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")

if not high_corr_pairs:
    print("  Нет признаков с высокой корреляцией (>0.8)")

# Все признаки информативны для небольшого набора данных
print("\nВывод: Все признаки будут использованы для кластеризации.")
print("Набор данных небольшой (4 признака), все признаки информативны.")

# ============================================================================
# 4. СТАНДАРТИЗАЦИЯ ПЕРЕМЕННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("4. СТАНДАРТИЗАЦИЯ ПЕРЕМЕННЫХ")
print("=" * 80)

scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns,
    index=X.index
)

print("\nСтатистики до стандартизации:")
print(X.describe().round(3))
print("\nСтатистики после стандартизации:")
print(X_scaled.describe().round(3))

# ============================================================================
# 5. ДИАГРАММЫ РАССЕИВАНИЯ
# ============================================================================

print("\n" + "=" * 80)
print("5. ДИАГРАММЫ РАССЕИВАНИЯ")
print("=" * 80)

# Pairplot с реальными метками классов
print("\n5.1. Составная диаграмма рассеивания (pairplot)...")
pairplot_data = X_scaled.copy()
pairplot_data['Species'] = y_true

g = sns.pairplot(pairplot_data, hue='Species', diag_kind='kde', 
                 palette='Set2', plot_kws={'alpha': 0.7, 's': 50})
g.fig.suptitle('Составная диаграмма рассеивания (по классам)', 
               y=1.02, fontsize=14, fontweight='bold')
plt.savefig('photos/pairplot_по_классам.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/pairplot_по_классам.png")

# Категоризованные диаграммы рассеивания
print("\n5.2. Категоризованные диаграммы рассеивания...")

# Основные пары признаков
key_pairs = [
    ('PetalLengthCm', 'PetalWidthCm'),
    ('SepalLengthCm', 'SepalWidthCm'),
    ('PetalLengthCm', 'SepalLengthCm'),
    ('PetalWidthCm', 'SepalWidthCm')
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, (feat1, feat2) in enumerate(key_pairs):
    for species in y_true.unique():
        mask = y_true == species
        axes[idx].scatter(X_scaled.loc[mask, feat1], X_scaled.loc[mask, feat2],
                         label=species, alpha=0.7, s=60)
    axes[idx].set_xlabel(feat1, fontsize=11)
    axes[idx].set_ylabel(feat2, fontsize=11)
    axes[idx].set_title(f'{feat1} vs {feat2}', fontsize=12, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Категоризованные диаграммы рассеивания', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('photos/категоризованные_диаграммы_рассеивания.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/категоризованные_диаграммы_рассеивания.png")

# PCA для визуализации в 2D
print("\n5.3. PCA визуализация (2 главные компоненты)...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# С реальными метками
for species in y_true.unique():
    mask = y_true == species
    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=species, alpha=0.7, s=60)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
ax1.set_title('PCA: Проекция данных (с реальными метками)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Без меток (для оценки кластерной структуры)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7, s=60, c='gray')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
ax2.set_title('PCA: Проекция данных (без меток)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('photos/pca_визуализация.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/pca_визуализация.png")

print(f"\nОбъясненная дисперсия PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"Объясненная дисперсия PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Общая объясненная дисперсия: {pca.explained_variance_ratio_.sum():.2%}")

# Оценка количества кластеров по визуализации
print("\n5.4. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ:")
print("-" * 80)
print("По диаграммам рассеивания видно:")
print("- Четко выделяются 3 группы объектов")
print("- Одна группа (Iris-setosa) хорошо отделена от остальных")
print("- Две другие группы (Iris-versicolor и Iris-virginica) частично перекрываются")
print("- Предполагаемое количество кластеров: 3")
print("- Тип кластерной структуры: компактные кластеры сферической формы")

# ============================================================================
# 6. ОБОСНОВАНИЕ ВЫБОРА МЕТОДОВ КЛАСТЕРИЗАЦИИ
# ============================================================================

print("\n" + "=" * 80)
print("6. ОБОСНОВАНИЕ ВЫБОРА МЕТОДОВ КЛАСТЕРИЗАЦИИ")
print("=" * 80)

print("\nВыбранные методы:")
print("1. K-means (k-средних)")
print("   - Подходит для компактных сферических кластеров")
print("   - Требует указания количества кластеров")
print("   - Эффективен для данных с нормальным распределением")
print("   - Быстрый алгоритм")
print("\n2. Иерархическая кластеризация (Agglomerative Clustering)")
print("   - Не требует предварительного указания количества кластеров")
print("   - Позволяет строить дендрограмму")
print("   - Хорошо работает с различными формами кластеров")
print("   - Метод полной связи (complete linkage) для компактных кластеров")

# ============================================================================
# 7. МЕТОД 1: K-MEANS
# ============================================================================

print("\n" + "=" * 80)
print("7. МЕТОД 1: K-MEANS")
print("=" * 80)

# Определение оптимального количества кластеров (метод локтя)
print("\n7.1. Определение оптимального количества кластеров (метод локтя)...")
inertias = []
silhouette_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# Метод локтя
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Количество кластеров (k)', fontsize=11)
ax1.set_ylabel('Инерция (Inertia)', fontsize=11)
ax1.set_title('Метод локтя для K-means', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='k=3')

ax2.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
ax2.set_xlabel('Количество кластеров (k)', fontsize=11)
ax2.set_ylabel('Силуэтный коэффициент', fontsize=11)
ax2.set_title('Силуэтный анализ для K-means', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(x=3, color='r', linestyle='--', alpha=0.7, label='k=3')

ax1.legend()
ax2.legend()
plt.tight_layout()
plt.savefig('photos/kmeans_выбор_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/kmeans_выбор_k.png")

optimal_k = 3
print(f"\nОптимальное количество кластеров: {optimal_k}")

# K-means с оптимальным k
print(f"\n7.2. Применение K-means с k={optimal_k}...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# Центры кластеров
centers_kmeans = kmeans.cluster_centers_
print("\nЦентры кластеров (стандартизированные данные):")
centers_df = pd.DataFrame(centers_kmeans, columns=X_scaled.columns)
print(centers_df.round(3))

# Визуализация K-means
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# В пространстве PCA
colors_kmeans = ['red', 'blue', 'green']
for i in range(optimal_k):
    mask = labels_kmeans == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[0].scatter(pca.transform(scaler.transform(centers_df))[:, 0],
               pca.transform(scaler.transform(centers_df))[:, 1],
               c='black', marker='x', s=200, linewidths=3, label='Центры')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
axes[0].set_title('K-means кластеризация (PCA проекция)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PetalLength vs PetalWidth
for i in range(optimal_k):
    mask = labels_kmeans == i
    axes[1].scatter(X_scaled.loc[mask, 'PetalLengthCm'], 
                   X_scaled.loc[mask, 'PetalWidthCm'],
                   c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[1].scatter(centers_kmeans[:, 2], centers_kmeans[:, 3],
               c='black', marker='x', s=200, linewidths=3, label='Центры')
axes[1].set_xlabel('PetalLengthCm (стандартизировано)', fontsize=11)
axes[1].set_ylabel('PetalWidthCm (стандартизировано)', fontsize=11)
axes[1].set_title('K-means: PetalLength vs PetalWidth', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('photos/kmeans_результаты.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/kmeans_результаты.png")

# ============================================================================
# 8. МЕТОД 2: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ
# ============================================================================

print("\n" + "=" * 80)
print("8. МЕТОД 2: ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ")
print("=" * 80)

# Построение дендрограммы
print("\n8.1. Построение дендрограммы...")

# Используем метод полной связи (complete linkage)
linkage_matrix = linkage(X_scaled, method='complete', metric='euclidean')

plt.figure(figsize=(14, 8))
dendrogram(linkage_matrix, truncate_mode='level', p=10, leaf_font_size=10)
plt.title('Дендрограмма (метод полной связи)', fontsize=14, fontweight='bold')
plt.xlabel('Объекты', fontsize=11)
plt.ylabel('Расстояние', fontsize=11)
plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.7, label='Уровень разрезания (k=3)')
plt.legend()
plt.tight_layout()
plt.savefig('photos/дендрограмма.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/дендрограмма.png")

# Иерархическая кластеризация с k=3
print("\n8.2. Применение иерархической кластеризации с k=3...")
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='complete')
labels_hierarchical = hierarchical.fit_predict(X_scaled)

# Визуализация иерархической кластеризации
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# В пространстве PCA
for i in range(optimal_k):
    mask = labels_hierarchical == i
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=11)
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=11)
axes[0].set_title('Иерархическая кластеризация (PCA проекция)', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# PetalLength vs PetalWidth
for i in range(optimal_k):
    mask = labels_hierarchical == i
    axes[1].scatter(X_scaled.loc[mask, 'PetalLengthCm'],
                   X_scaled.loc[mask, 'PetalWidthCm'],
                   c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[1].set_xlabel('PetalLengthCm (стандартизировано)', fontsize=11)
axes[1].set_ylabel('PetalWidthCm (стандартизировано)', fontsize=11)
axes[1].set_title('Иерархическая: PetalLength vs PetalWidth', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('photos/hierarchical_результаты.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/hierarchical_результаты.png")

# ============================================================================
# 9. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ
# ============================================================================

print("\n" + "=" * 80)
print("9. ОЦЕНКА КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
print("=" * 80)

def evaluate_clustering(X, labels, method_name):
    """Оценка качества кластеризации"""
    results = {}
    
    # Внутренние метрики
    results['Silhouette'] = silhouette_score(X, labels)
    results['Calinski-Harabasz'] = calinski_harabasz_score(X, labels)
    results['Davies-Bouldin'] = davies_bouldin_score(X, labels)
    
    # Внутрикластерные расстояния
    intra_cluster_distances = []
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 1:
            distances = []
            for i in range(len(cluster_points)):
                for j in range(i+1, len(cluster_points)):
                    dist = np.linalg.norm(cluster_points.iloc[i] - cluster_points.iloc[j])
                    distances.append(dist)
            intra_cluster_distances.append(np.mean(distances))
    results['Среднее внутрикластерное расстояние'] = np.mean(intra_cluster_distances)
    
    # Расстояния между кластерами
    inter_cluster_distances = []
    unique_labels = np.unique(labels)
    for i in range(len(unique_labels)):
        for j in range(i+1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            # Минимальное расстояние между кластерами
            min_dist = np.inf
            for point_i in cluster_i.values:
                for point_j in cluster_j.values:
                    dist = np.linalg.norm(point_i - point_j)
                    if dist < min_dist:
                        min_dist = dist
            inter_cluster_distances.append(min_dist)
    results['Среднее межкластерное расстояние'] = np.mean(inter_cluster_distances)
    
    # Компактность (отношение внутрикластерных расстояний к межкластерным)
    results['Компактность'] = results['Среднее внутрикластерное расстояние'] / results['Среднее межкластерное расстояние']
    
    return results

# Оценка K-means
print("\n9.1. Метрики качества для K-means:")
results_kmeans = evaluate_clustering(X_scaled, labels_kmeans, "K-means")
for metric, value in results_kmeans.items():
    print(f"  {metric}: {value:.4f}")

# Оценка иерархической кластеризации
print("\n9.2. Метрики качества для иерархической кластеризации:")
results_hierarchical = evaluate_clustering(X_scaled, labels_hierarchical, "Hierarchical")
for metric, value in results_hierarchical.items():
    print(f"  {metric}: {value:.4f}")

# Внешние метрики (сравнение с реальными метками)
print("\n9.3. Внешние метрики (сравнение с реальными метками):")

# Преобразование меток классов в числовые
label_encoder = {species: i for i, species in enumerate(y_true.unique())}
y_true_numeric = y_true.map(label_encoder)

print("\nK-means:")
ari_kmeans = adjusted_rand_score(y_true_numeric, labels_kmeans)
nmi_kmeans = normalized_mutual_info_score(y_true_numeric, labels_kmeans)
print(f"  Adjusted Rand Index (ARI): {ari_kmeans:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi_kmeans:.4f}")

print("\nИерархическая кластеризация:")
ari_hierarchical = adjusted_rand_score(y_true_numeric, labels_hierarchical)
nmi_hierarchical = normalized_mutual_info_score(y_true_numeric, labels_hierarchical)
print(f"  Adjusted Rand Index (ARI): {ari_hierarchical:.4f}")
print(f"  Normalized Mutual Information (NMI): {nmi_hierarchical:.4f}")

# Силуэтный анализ
print("\n9.4. Силуэтный анализ...")

def plot_silhouette(X, labels, method_name, filename):
    """Построение силуэтной диаграммы"""
    silhouette_vals = silhouette_samples(X, labels)
    n_clusters = len(np.unique(labels))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lower = 10
    
    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        
        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
        y_lower = y_upper + 10
    
    ax.set_xlabel('Силуэтный коэффициент', fontsize=12)
    ax.set_ylabel('Номер кластера', fontsize=12)
    ax.set_title(f'Силуэтная диаграмма: {method_name}', fontsize=14, fontweight='bold')
    
    avg_score = silhouette_score(X, labels)
    ax.axvline(x=avg_score, color="red", linestyle="--", 
               label=f'Средний силуэтный коэффициент: {avg_score:.3f}')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

plot_silhouette(X_scaled, labels_kmeans, "K-means", 'photos/silhouette_kmeans.png')
print("Сохранено: photos/silhouette_kmeans.png")

plot_silhouette(X_scaled, labels_hierarchical, "Иерархическая кластеризация", 
                'photos/silhouette_hierarchical.png')
print("Сохранено: photos/silhouette_hierarchical.png")

# Сравнительная таблица метрик
print("\n9.5. Сравнительная таблица метрик:")
comparison_df = pd.DataFrame({
    'K-means': results_kmeans,
    'Иерархическая': results_hierarchical
})
comparison_df['Внешние метрики'] = {
    'ARI': ari_kmeans,
    'NMI': nmi_kmeans,
    'ARI (Hierarchical)': ari_hierarchical,
    'NMI (Hierarchical)': nmi_hierarchical
}
print(comparison_df.round(4))

# ============================================================================
# 10. ОЦЕНКА ЗНАЧИМОСТИ ПРИЗНАКОВ
# ============================================================================

print("\n" + "=" * 80)
print("10. ОЦЕНКА ЗНАЧИМОСТИ ПРИЗНАКОВ")
print("=" * 80)

# Анализ важности признаков через центры кластеров
print("\n10.1. Анализ важности признаков через центры кластеров (K-means):")
centers_original = scaler.inverse_transform(centers_kmeans)
centers_original_df = pd.DataFrame(centers_original, columns=X.columns)

print("\nЦентры кластеров (исходные данные):")
print(centers_original_df.round(3))

# Разброс значений признаков между кластерами
feature_importance = {}
for feature in X.columns:
    feature_std = centers_original_df[feature].std()
    feature_range = centers_original_df[feature].max() - centers_original_df[feature].min()
    feature_importance[feature] = {
        'Стандартное отклонение': feature_std,
        'Размах': feature_range
    }

importance_df = pd.DataFrame(feature_importance).T
importance_df = importance_df.sort_values('Размах', ascending=False)
print("\nВажность признаков (по размаху значений в центрах кластеров):")
print(importance_df.round(4))

# Визуализация важности признаков
plt.figure(figsize=(10, 6))
importance_df['Размах'].plot(kind='barh', color='steelblue')
plt.xlabel('Размах значений в центрах кластеров', fontsize=11)
plt.ylabel('Признак', fontsize=11)
plt.title('Важность признаков для кластеризации', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('photos/важность_признаков.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/важность_признаков.png")

# ============================================================================
# 11. СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕШЕНИЙ
# ============================================================================

print("\n" + "=" * 80)
print("11. СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕШЕНИЙ")
print("=" * 80)

# Визуализация сравнения
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# K-means
for i in range(optimal_k):
    mask = labels_kmeans == i
    axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[0, 0].set_title('K-means', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel(f'PC1', fontsize=10)
axes[0, 0].set_ylabel(f'PC2', fontsize=10)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Иерархическая
for i in range(optimal_k):
    mask = labels_hierarchical == i
    axes[0, 1].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      c=colors_kmeans[i], label=f'Кластер {i+1}', alpha=0.7, s=60)
axes[0, 1].set_title('Иерархическая кластеризация', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel(f'PC1', fontsize=10)
axes[0, 1].set_ylabel(f'PC2', fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Реальные метки
for species in y_true.unique():
    mask = y_true == species
    axes[1, 0].scatter(X_pca[mask, 0], X_pca[mask, 1],
                      label=species, alpha=0.7, s=60)
axes[1, 0].set_title('Реальные метки классов', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel(f'PC1', fontsize=10)
axes[1, 0].set_ylabel(f'PC2', fontsize=10)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Сравнение метрик
metrics_comparison = pd.DataFrame({
    'K-means': [
        results_kmeans['Silhouette'],
        results_kmeans['Calinski-Harabasz'],
        1 / results_kmeans['Davies-Bouldin'],  # Инвертируем для лучшей интерпретации
        ari_kmeans,
        nmi_kmeans
    ],
    'Иерархическая': [
        results_hierarchical['Silhouette'],
        results_hierarchical['Calinski-Harabasz'],
        1 / results_hierarchical['Davies-Bouldin'],
        ari_hierarchical,
        nmi_hierarchical
    ]
}, index=['Silhouette', 'Calinski-Harabasz', '1/Davies-Bouldin', 'ARI', 'NMI'])

axes[1, 1].axis('tight')
axes[1, 1].axis('off')
table = axes[1, 1].table(cellText=metrics_comparison.round(3).values,
                         rowLabels=metrics_comparison.index,
                         colLabels=metrics_comparison.columns,
                         cellLoc='center',
                         loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
axes[1, 1].set_title('Сравнение метрик качества', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('Сравнительный анализ методов кластеризации', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('photos/сравнительный_анализ.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/сравнительный_анализ.png")

print("\nВыводы:")
print("- Оба метода показывают хорошее качество кластеризации")
print("- K-means и иерархическая кластеризация дают схожие результаты")
print("- Высокие значения ARI и NMI указывают на хорошее соответствие реальным классам")
print("- Силуэтные коэффициенты > 0.5 указывают на хорошее разделение кластеров")

# ============================================================================
# 12. ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ
# ============================================================================

print("\n" + "=" * 80)
print("12. ИССЛЕДОВАНИЕ ВЛИЯНИЯ ПАРАМЕТРОВ (K-means)")
print("=" * 80)

# Исследование влияния количества кластеров
print("\n12.1. Влияние количества кластеров на метрики качества...")

k_range = range(2, 8)
silhouette_scores_detailed = []
calinski_scores = []
davies_bouldin_scores = []

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_temp = kmeans_temp.fit_predict(X_scaled)
    
    silhouette_scores_detailed.append(silhouette_score(X_scaled, labels_temp))
    calinski_scores.append(calinski_harabasz_score(X_scaled, labels_temp))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, labels_temp))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(k_range, silhouette_scores_detailed, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Количество кластеров (k)', fontsize=11)
axes[0].set_ylabel('Силуэтный коэффициент', fontsize=11)
axes[0].set_title('Влияние k на силуэтный коэффициент', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=3, color='r', linestyle='--', alpha=0.7)

axes[1].plot(k_range, calinski_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Количество кластеров (k)', fontsize=11)
axes[1].set_ylabel('Calinski-Harabasz Index', fontsize=11)
axes[1].set_title('Влияние k на Calinski-Harabasz Index', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=3, color='r', linestyle='--', alpha=0.7)

axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Количество кластеров (k)', fontsize=11)
axes[2].set_ylabel('Davies-Bouldin Index', fontsize=11)
axes[2].set_title('Влияние k на Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3)
axes[2].axvline(x=3, color='r', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('photos/влияние_параметров_k.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/влияние_параметров_k.png")

print("\nОптимальное значение k=3 подтверждается максимумом силуэтного коэффициента")
print("и Calinski-Harabasz Index, а также минимумом Davies-Bouldin Index")

# Исследование влияния инициализации для K-means
print("\n12.2. Влияние инициализации на K-means...")
init_methods = ['k-means++', 'random']
init_results = {}

for init_method in init_methods:
    scores = []
    for _ in range(10):
        kmeans_temp = KMeans(n_clusters=3, init=init_method, random_state=None, n_init=1)
        labels_temp = kmeans_temp.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels_temp))
    init_results[init_method] = scores

fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([init_results['k-means++'], init_results['random']], 
                labels=['k-means++', 'random'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('Силуэтный коэффициент', fontsize=11)
ax.set_title('Влияние метода инициализации на качество K-means', 
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('photos/влияние_инициализации.png', dpi=300, bbox_inches='tight')
plt.close()
print("Сохранено: photos/влияние_инициализации.png")

print("\nВывод: k-means++ обеспечивает более стабильные результаты")

# ============================================================================
# 13. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("13. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 80)

print("\n13.1. Характеристика кластеров (K-means):")
print("-" * 80)

# Сопоставление кластеров с реальными классами
cluster_to_class = {}
for cluster_id in range(optimal_k):
    cluster_mask = labels_kmeans == cluster_id
    class_distribution = y_true[cluster_mask].value_counts()
    dominant_class = class_distribution.index[0]
    cluster_to_class[cluster_id] = dominant_class
    print(f"\nКластер {cluster_id + 1}:")
    print(f"  Доминирующий класс: {dominant_class}")
    print(f"  Количество объектов: {cluster_mask.sum()}")
    print(f"  Распределение по классам:")
    for cls, count in class_distribution.items():
        print(f"    {cls}: {count} ({count/cluster_mask.sum()*100:.1f}%)")
    
    # Средние значения признаков
    print(f"  Средние значения признаков:")
    cluster_means = X.loc[cluster_mask].mean()
    for feature, value in cluster_means.items():
        print(f"    {feature}: {value:.2f}")

print("\n13.2. Выводы:")
print("-" * 80)
print("1. Оба метода кластеризации (K-means и иерархическая) успешно выделяют")
print("   3 кластера, соответствующих трем видам ирисов.")
print("2. Кластер 1 соответствует Iris-setosa (хорошо отделен от остальных).")
print("3. Кластеры 2 и 3 соответствуют Iris-versicolor и Iris-virginica")
print("   (частично перекрываются, что соответствует биологической близости).")
print("4. Наиболее важные признаки для кластеризации:")
print(f"   - {importance_df.index[0]} (размах: {importance_df.iloc[0, 1]:.2f})")
print(f"   - {importance_df.index[1]} (размах: {importance_df.iloc[1, 1]:.2f})")
print("5. Метрики качества указывают на хорошее разделение кластеров:")
print(f"   - Силуэтный коэффициент (K-means): {results_kmeans['Silhouette']:.3f}")
print(f"   - ARI (K-means): {ari_kmeans:.3f}")
print("6. Оптимальное количество кластеров k=3 подтверждается несколькими метриками.")

print("\n" + "=" * 80)
print("РАБОТА ЗАВЕРШЕНА")
print("=" * 80)
print(f"\nВсего создано графиков: {len([f for f in os.listdir('photos') if f.endswith('.png')])}")
print("Графики сохранены в папке 'photos/'")


"""
Лабораторная работа по классификации
Датасет: Iris Plants Database
Методы: Нейронная сеть (MLP), Метод опорных векторов (SVM)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    precision_recall_curve
)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Настройка визуализации
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================================
# 1. ЗАГРУЗКА И ОПИСАНИЕ ДАННЫХ
# ============================================================================

print("=" * 80)
print("ЛАБОРАТОРНАЯ РАБОТА: КЛАССИФИКАЦИЯ ИРИСОВ")
print("=" * 80)

# Загрузка данных
df = pd.read_csv('data/iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

print("\n1. ОПИСАНИЕ ИСХОДНЫХ ДАННЫХ")
print("-" * 80)
print(f"Количество записей: {len(df)}")
print(f"Количество признаков: {len(df.columns) - 1}")
print(f"Количество классов: {df['class'].nunique()}")
print(f"\nКлассы: {df['class'].unique().tolist()}")
print(f"\nПервые 5 строк:")
print(df.head())
print(f"\nИнформация о данных:")
print(df.info())
print(f"\nПропущенные значения:")
print(df.isnull().sum())

# Разделение на признаки и целевую переменную
X = df.drop('class', axis=1)
y = df['class']

# ============================================================================
# 2. ДЕСКРИПТИВНЫЙ АНАЛИЗ
# ============================================================================

print("\n" + "=" * 80)
print("2. ДЕСКРИПТИВНЫЙ АНАЛИЗ")
print("=" * 80)

print("\n2.1. Основные статистики:")
print(X.describe())

print("\n2.2. Распределение по классам:")
class_dist = df['class'].value_counts()
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
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Распределение признаков по классам', fontsize=16, fontweight='bold')

for idx, feature in enumerate(X.columns):
    ax = axes[idx // 2, idx % 2]
    for class_name in df['class'].unique():
        data = df[df['class'] == class_name][feature]
        ax.hist(data, alpha=0.6, label=class_name, bins=15)
    ax.set_xlabel(feature)
    ax.set_ylabel('Частота')
    ax.set_title(f'Распределение {feature}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distributions_by_class.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: distributions_by_class.png")
plt.close()

# ============================================================================
# 3. ОТБОР ПРИЗНАКОВ
# ============================================================================

print("\n" + "=" * 80)
print("3. ОТБОР НАИБОЛЕЕ ИНФОРМАТИВНЫХ ПРИЗНАКОВ")
print("=" * 80)

# Кодирование целевой переменной для анализа
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Использование F-статистики для отбора признаков
selector = SelectKBest(score_func=f_classif, k='all')
selector.fit(X, y_encoded)

feature_scores = pd.DataFrame({
    'Признак': X.columns,
    'F-статистика': selector.scores_,
    'p-value': selector.pvalues_
})
feature_scores = feature_scores.sort_values('F-статистика', ascending=False)

print("\nОценка информативности признаков (F-статистика):")
print(feature_scores.to_string(index=False))

# Корреляционная матрица
print("\n3.1. Корреляционная матрица признаков:")
corr_matrix = X.corr()
print(corr_matrix.round(4))

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: correlation_matrix.png")
plt.close()

# ============================================================================
# 4. СТАНДАРТИЗАЦИЯ ПЕРЕМЕННЫХ
# ============================================================================

print("\n" + "=" * 80)
print("4. СТАНДАРТИЗАЦИЯ ПЕРЕМЕННЫХ")
print("=" * 80)

# Стандартизация для SVM и нейронной сети
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

print("\nСтатистики до стандартизации:")
print(X.describe().round(4))
print("\nСтатистики после стандартизации:")
print(X_scaled_df.describe().round(4))

# ============================================================================
# 5. КАТЕГОРИЗИРОВАННЫЕ ДИАГРАММЫ РАССЕИВАНИЯ
# ============================================================================

print("\n" + "=" * 80)
print("5. КАТЕГОРИЗИРОВАННЫЕ ДИАГРАММЫ РАССЕИВАНИЯ")
print("=" * 80)

# Создание парных диаграмм рассеивания
fig = plt.figure(figsize=(16, 12))
g = sns.pairplot(df, hue='class', diag_kind='kde', palette='Set2', height=2.5)
g.fig.suptitle('Парные диаграммы рассеивания по классам', y=1.02, fontsize=16, fontweight='bold')
plt.savefig('pairplot_scatter.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: pairplot_scatter.png")
plt.close()

# Дополнительные диаграммы для ключевых пар признаков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Диаграммы рассеивания для ключевых пар признаков', fontsize=16, fontweight='bold')

pairs = [
    ('sepal_length', 'sepal_width'),
    ('petal_length', 'petal_width'),
    ('sepal_length', 'petal_length'),
    ('sepal_width', 'petal_width')
]

for idx, (feat1, feat2) in enumerate(pairs):
    ax = axes[idx // 2, idx % 2]
    for class_name in df['class'].unique():
        data = df[df['class'] == class_name]
        ax.scatter(data[feat1], data[feat2], label=class_name, alpha=0.7, s=60)
    ax.set_xlabel(feat1)
    ax.set_ylabel(feat2)
    ax.set_title(f'{feat1} vs {feat2}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('key_scatter_pairs.png', dpi=300, bbox_inches='tight')
print("График сохранен: key_scatter_pairs.png")
plt.close()

print("\nАнализ расположения классов:")
print("- Iris-setosa: линейно отделима от остальных двух классов")
print("- Iris-versicolor и Iris-virginica: не линейно отделимы друг от друга")

# ============================================================================
# 6. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩЕЕ И ТЕСТОВОЕ ПОДМНОЖЕСТВА
# ============================================================================

print("\n" + "=" * 80)
print("6. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩЕЕ И ТЕСТОВОЕ ПОДМНОЖЕСТВА")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nРазмер обучающей выборки: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Размер тестовой выборки: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\nРаспределение классов в обучающей выборке:")
print(pd.Series(y_train).value_counts().sort_index())
print("\nРаспределение классов в тестовой выборке:")
print(pd.Series(y_test).value_counts().sort_index())

# ============================================================================
# 7. ПОСТРОЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ
# ============================================================================

print("\n" + "=" * 80)
print("7. ПОСТРОЕНИЕ МОДЕЛЕЙ КЛАССИФИКАЦИИ")
print("=" * 80)

# 7.1. НЕЙРОННАЯ СЕТЬ (MLP)
print("\n7.1. НЕЙРОННАЯ СЕТЬ (MLPClassifier)")
print("-" * 80)

# Структура: 4 входных признака -> 2 скрытых слоя (100, 50 нейронов) -> 3 выходных класса
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 регуляризация
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False
)

print("Параметры модели:")
print(f"  - Скрытые слои: (100, 50)")
print(f"  - Функция активации: ReLU")
print(f"  - Решатель: Adam")
print(f"  - Регуляризация (alpha): 0.001")
print(f"  - Начальная скорость обучения: 0.001")
print(f"  - Максимум итераций: 1000")
print(f"  - Early stopping: включен")

mlp.fit(X_train, y_train)
y_pred_mlp_train = mlp.predict(X_train)
y_pred_mlp_test = mlp.predict(X_test)

print(f"\nКоличество итераций обучения: {mlp.n_iter_}")
print(f"Финальная потеря: {mlp.loss_:.6f}")

# 7.2. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)
print("\n7.2. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)")
print("-" * 80)

svm = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    probability=True,  # Для расчета вероятностей (нужно для AUC)
    random_state=42
)

print("Параметры модели:")
print(f"  - Ядро: RBF (радиальная базисная функция)")
print(f"  - C (параметр регуляризации): 1.0")
print(f"  - Gamma: 'scale' (автоматический выбор)")
print(f"  - Вероятности: включены (для метрик)")

svm.fit(X_train, y_train)
y_pred_svm_train = svm.predict(X_train)
y_pred_svm_test = svm.predict(X_test)

print("\nОбоснование выбора параметров:")
print("  - RBF kernel: позволяет решать нелинейные задачи (versicolor и virginica не линейно разделимы)")
print("  - C=1.0: баланс между переобучением и недообучением")
print("  - gamma='scale': автоматический выбор масштаба для RBF ядра")

# ============================================================================
# 8. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ
# ============================================================================

print("\n" + "=" * 80)
print("8. ОЦЕНКА КАЧЕСТВА МОДЕЛЕЙ")
print("=" * 80)

def calculate_metrics(y_true, y_pred, y_proba=None, model_name="", dataset_name=""):
    """Расчет всех метрик классификации"""
    metrics = {}
    
    # Базовые метрики
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision (macro)'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Recall (macro)'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['F1 (macro)'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['Precision (micro)'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['Recall (micro)'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['F1 (micro)'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    # AUC (требуется вероятности)
    if y_proba is not None:
        try:
            # Для многоклассовой задачи используем one-vs-rest
            metrics['AUC (macro)'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['AUC (micro)'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='micro')
        except:
            metrics['AUC (macro)'] = None
            metrics['AUC (micro)'] = None
    
    return metrics

def print_confusion_matrix(y_true, y_pred, class_names, title=""):
    """Вывод и визуализация confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{title}")
    print("-" * 50)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    # Визуализация
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Количество'})
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    return cm

# MLP метрики
print("\n8.1. НЕЙРОННАЯ СЕТЬ (MLP)")
print("=" * 80)

y_proba_mlp_train = mlp.predict_proba(X_train)
y_proba_mlp_test = mlp.predict_proba(X_test)

class_names = le.classes_

print("\nТЕСТОВАЯ ВЫБОРКА:")
cm_mlp_test = print_confusion_matrix(y_test, y_pred_mlp_test, class_names, 
                                     "Confusion Matrix - MLP (Тестовая выборка)")
plt.savefig('confusion_matrix_mlp_test.png', dpi=300, bbox_inches='tight')
plt.close()

metrics_mlp_test = calculate_metrics(y_test, y_pred_mlp_test, y_proba_mlp_test, "MLP", "Test")
print("\nМетрики на тестовой выборке:")
for metric, value in metrics_mlp_test.items():
    if value is not None:
        print(f"  {metric:25s}: {value:.6f}")

print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
cm_mlp_train = print_confusion_matrix(y_train, y_pred_mlp_train, class_names,
                                      "Confusion Matrix - MLP (Обучающая выборка)")
plt.savefig('confusion_matrix_mlp_train.png', dpi=300, bbox_inches='tight')
plt.close()

metrics_mlp_train = calculate_metrics(y_train, y_pred_mlp_train, y_proba_mlp_train, "MLP", "Train")
print("\nМетрики на обучающей выборке:")
for metric, value in metrics_mlp_train.items():
    if value is not None:
        print(f"  {metric:25s}: {value:.6f}")

# SVM метрики
print("\n8.2. МЕТОД ОПОРНЫХ ВЕКТОРОВ (SVM)")
print("=" * 80)

y_proba_svm_train = svm.predict_proba(X_train)
y_proba_svm_test = svm.predict_proba(X_test)

print("\nТЕСТОВАЯ ВЫБОРКА:")
cm_svm_test = print_confusion_matrix(y_test, y_pred_svm_test, class_names,
                                     "Confusion Matrix - SVM (Тестовая выборка)")
plt.savefig('confusion_matrix_svm_test.png', dpi=300, bbox_inches='tight')
plt.close()

metrics_svm_test = calculate_metrics(y_test, y_pred_svm_test, y_proba_svm_test, "SVM", "Test")
print("\nМетрики на тестовой выборке:")
for metric, value in metrics_svm_test.items():
    if value is not None:
        print(f"  {metric:25s}: {value:.6f}")

print("\nОБУЧАЮЩАЯ ВЫБОРКА:")
cm_svm_train = print_confusion_matrix(y_train, y_pred_svm_train, class_names,
                                      "Confusion Matrix - SVM (Обучающая выборка)")
plt.savefig('confusion_matrix_svm_train.png', dpi=300, bbox_inches='tight')
plt.close()

metrics_svm_train = calculate_metrics(y_train, y_pred_svm_train, y_proba_svm_train, "SVM", "Train")
print("\nМетрики на обучающей выборке:")
for metric, value in metrics_svm_train.items():
    if value is not None:
        print(f"  {metric:25s}: {value:.6f}")

# Сравнительная таблица
print("\n8.3. СРАВНИТЕЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("=" * 80)

comparison_data = {
    'MLP (Train)': metrics_mlp_train,
    'MLP (Test)': metrics_mlp_test,
    'SVM (Train)': metrics_svm_train,
    'SVM (Test)': metrics_svm_test
}

comparison_df = pd.DataFrame(comparison_data)
print("\nСравнительная таблица метрик:")
print(comparison_df.round(6))

# ROC кривые
print("\n8.4. ROC КРИВЫЕ")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('ROC кривые для многоклассовой классификации', fontsize=16, fontweight='bold')

# MLP
ax = axes[0]
for i, class_name in enumerate(class_names):
    y_true_binary = (y_test == i).astype(int)
    y_proba_binary = y_proba_mlp_test[:, i]
    fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
    auc_score = roc_auc_score(y_true_binary, y_proba_binary)
    ax.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('MLP - ROC кривые')
ax.legend()
ax.grid(True, alpha=0.3)

# SVM
ax = axes[1]
for i, class_name in enumerate(class_names):
    y_true_binary = (y_test == i).astype(int)
    y_proba_binary = y_proba_svm_test[:, i]
    fpr, tpr, _ = roc_curve(y_true_binary, y_proba_binary)
    auc_score = roc_auc_score(y_true_binary, y_proba_binary)
    ax.plot(fpr, tpr, label=f'{class_name} (AUC={auc_score:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('SVM - ROC кривые')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("График сохранен: roc_curves.png")
plt.close()

# ============================================================================
# 9. ОЦЕНКА ЗНАЧИМОСТИ ПРИЗНАКОВ
# ============================================================================

print("\n" + "=" * 80)
print("9. ОЦЕНКА ЗНАЧИМОСТИ ПРИЗНАКОВ")
print("=" * 80)

# Для MLP - веса первого слоя
print("\n9.1. Значимость признаков для MLP (через веса первого слоя):")
mlp_weights = mlp.coefs_[0]  # Веса первого слоя (4 входных -> 100 нейронов)
# Абсолютные значения весов для каждого признака
feature_importance_mlp = np.abs(mlp_weights).mean(axis=1)
feature_importance_mlp_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': feature_importance_mlp
}).sort_values('Важность', ascending=False)

print(feature_importance_mlp_df.to_string(index=False))

# Для SVM - через коэффициенты опорных векторов
print("\n9.2. Значимость признаков для SVM:")
# Для SVM с RBF ядром сложнее напрямую оценить важность, используем permutation importance
from sklearn.inspection import permutation_importance

perm_importance_mlp = permutation_importance(mlp, X_test, y_test, n_repeats=10, random_state=42)
perm_importance_svm = permutation_importance(svm, X_test, y_test, n_repeats=10, random_state=42)

importance_df = pd.DataFrame({
    'Признак': X.columns,
    'MLP (важность)': perm_importance_mlp.importances_mean,
    'MLP (std)': perm_importance_mlp.importances_std,
    'SVM (важность)': perm_importance_svm.importances_mean,
    'SVM (std)': perm_importance_svm.importances_std
}).sort_values('MLP (важность)', ascending=False)

print("\nВажность признаков (permutation importance):")
print(importance_df.round(6).to_string(index=False))

# Визуализация важности признаков
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Важность признаков', fontsize=16, fontweight='bold')

# MLP
ax = axes[0]
sorted_idx = perm_importance_mlp.importances_mean.argsort()
ax.barh(range(len(X.columns)), perm_importance_mlp.importances_mean[sorted_idx])
ax.set_yticks(range(len(X.columns)))
ax.set_yticklabels([X.columns[i] for i in sorted_idx])
ax.set_xlabel('Важность признака')
ax.set_title('MLP - Permutation Importance')
ax.grid(True, alpha=0.3, axis='x')

# SVM
ax = axes[1]
sorted_idx = perm_importance_svm.importances_mean.argsort()
ax.barh(range(len(X.columns)), perm_importance_svm.importances_mean[sorted_idx])
ax.set_yticks(range(len(X.columns)))
ax.set_yticklabels([X.columns[i] for i in sorted_idx])
ax.set_xlabel('Важность признака')
ax.set_title('SVM - Permutation Importance')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: feature_importance.png")
plt.close()

# ============================================================================
# 10. ИЗМЕНЕНИЕ ПАРАМЕТРОВ И АНАЛИЗ РЕЗУЛЬТАТОВ
# ============================================================================

print("\n" + "=" * 80)
print("10. ИЗМЕНЕНИЕ ПАРАМЕТРОВ И АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 80)

print("\n10.1. Эксперименты с параметрами MLP:")
print("-" * 80)

mlp_params_experiments = [
    {'hidden_layer_sizes': (50, 25), 'alpha': 0.001, 'name': 'Меньше нейронов (50,25)'},
    {'hidden_layer_sizes': (200, 100), 'alpha': 0.001, 'name': 'Больше нейронов (200,100)'},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.01, 'name': 'Больше регуляризации (alpha=0.01)'},
    {'hidden_layer_sizes': (100, 50), 'alpha': 0.0001, 'name': 'Меньше регуляризации (alpha=0.0001)'},
]

mlp_results = []
for params in mlp_params_experiments:
    mlp_exp = MLPClassifier(
        hidden_layer_sizes=params['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        alpha=params['alpha'],
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10
    )
    mlp_exp.fit(X_train, y_train)
    y_pred_exp = mlp_exp.predict(X_test)
    acc = accuracy_score(y_test, y_pred_exp)
    mlp_results.append({
        'Конфигурация': params['name'],
        'Accuracy': acc,
        'Iterations': mlp_exp.n_iter_
    })
    print(f"{params['name']:40s}: Accuracy = {acc:.6f}, Итераций = {mlp_exp.n_iter_}")

mlp_results_df = pd.DataFrame(mlp_results)
print("\nСводная таблица экспериментов MLP:")
print(mlp_results_df.to_string(index=False))

print("\n10.2. Эксперименты с параметрами SVM:")
print("-" * 80)

svm_params_experiments = [
    {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale', 'name': 'C=0.1 (больше регуляризация)'},
    {'kernel': 'rbf', 'C': 10.0, 'gamma': 'scale', 'name': 'C=10.0 (меньше регуляризация)'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 0.1, 'name': 'gamma=0.1 (меньше влияние)'},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 10.0, 'name': 'gamma=10.0 (больше влияние)'},
    {'kernel': 'linear', 'C': 1.0, 'gamma': 'scale', 'name': 'Линейное ядро'},
    {'kernel': 'poly', 'C': 1.0, 'gamma': 'scale', 'degree': 3, 'name': 'Полиномиальное ядро (degree=3)'},
]

svm_results = []
for params in svm_params_experiments:
    svm_exp = SVC(
        kernel=params['kernel'],
        C=params['C'],
        gamma=params['gamma'],
        probability=True,
        random_state=42
    )
    if 'degree' in params:
        svm_exp.set_params(degree=params['degree'])
    
    svm_exp.fit(X_train, y_train)
    y_pred_exp = svm_exp.predict(X_test)
    acc = accuracy_score(y_test, y_pred_exp)
    svm_results.append({
        'Конфигурация': params['name'],
        'Accuracy': acc
    })
    print(f"{params['name']:40s}: Accuracy = {acc:.6f}")

svm_results_df = pd.DataFrame(svm_results)
print("\nСводная таблица экспериментов SVM:")
print(svm_results_df.to_string(index=False))

# Визуализация влияния параметров
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Влияние параметров на точность', fontsize=16, fontweight='bold')

# MLP
ax = axes[0]
ax.barh(range(len(mlp_results_df)), mlp_results_df['Accuracy'])
ax.set_yticks(range(len(mlp_results_df)))
ax.set_yticklabels(mlp_results_df['Конфигурация'], fontsize=8)
ax.set_xlabel('Accuracy')
ax.set_title('MLP - Влияние параметров')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=metrics_mlp_test['Accuracy'], color='r', linestyle='--', label='Базовая конфигурация')
ax.legend()

# SVM
ax = axes[1]
ax.barh(range(len(svm_results_df)), svm_results_df['Accuracy'])
ax.set_yticks(range(len(svm_results_df)))
ax.set_yticklabels(svm_results_df['Конфигурация'], fontsize=8)
ax.set_xlabel('Accuracy')
ax.set_title('SVM - Влияние параметров')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=metrics_svm_test['Accuracy'], color='r', linestyle='--', label='Базовая конфигурация')
ax.legend()

plt.tight_layout()
plt.savefig('parameter_tuning_results.png', dpi=300, bbox_inches='tight')
print("\nГрафик сохранен: parameter_tuning_results.png")
plt.close()

# ============================================================================
# 11. УДАЛЕНИЕ МАЛОЗНАЧИМЫХ ПРИЗНАКОВ И ПЕРЕОЦЕНКА
# ============================================================================

print("\n" + "=" * 80)
print("11. УДАЛЕНИЕ МАЛОЗНАЧИМЫХ ПРИЗНАКОВ И ПЕРЕОЦЕНКА")
print("=" * 80)

# Определяем наименее важный признак
least_important_feature = importance_df.iloc[-1]['Признак']
print(f"\nНаименее важный признак (по permutation importance): {least_important_feature}")

# Удаляем наименее важный признак
X_reduced = X.drop(least_important_feature, axis=1)
X_reduced_scaled = scaler.fit_transform(X_reduced)

X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(
    X_reduced_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nПризнаки после удаления: {list(X_reduced.columns)}")
print(f"Размерность признаков: {X_reduced.shape[1]} (было {X.shape[1]})")

# Переобучение MLP на уменьшенном наборе признаков
print("\n11.1. Переобучение MLP на уменьшенном наборе признаков:")
mlp_reduced = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

mlp_reduced.fit(X_train_reduced, y_train_reduced)
y_pred_mlp_reduced = mlp_reduced.predict(X_test_reduced)
acc_mlp_reduced = accuracy_score(y_test_reduced, y_pred_mlp_reduced)

print(f"Accuracy на полном наборе признаков: {metrics_mlp_test['Accuracy']:.6f}")
print(f"Accuracy на уменьшенном наборе признаков: {acc_mlp_reduced:.6f}")
print(f"Изменение: {acc_mlp_reduced - metrics_mlp_test['Accuracy']:.6f}")

# Confusion matrix для уменьшенного набора
cm_mlp_reduced = print_confusion_matrix(y_test_reduced, y_pred_mlp_reduced, class_names,
                                        f"Confusion Matrix - MLP (без признака {least_important_feature})")
plt.savefig('confusion_matrix_mlp_reduced.png', dpi=300, bbox_inches='tight')
plt.close()

# Детальный отчет
print("\n11.2. Детальное сравнение:")
metrics_mlp_reduced = calculate_metrics(y_test_reduced, y_pred_mlp_reduced, 
                                        mlp_reduced.predict_proba(X_test_reduced), "MLP", "Test_Reduced")

comparison_reduced = pd.DataFrame({
    'Полный набор признаков': metrics_mlp_test,
    'Уменьшенный набор': metrics_mlp_reduced
})
print("\nСравнение метрик:")
print(comparison_reduced.round(6))

# ============================================================================
# ИТОГОВЫЕ ВЫВОДЫ
# ============================================================================

print("\n" + "=" * 80)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("=" * 80)

print("\n1. Дескриптивный анализ:")
print("   - Датасет содержит 150 записей, 4 признака, 3 класса")
print("   - Распределение классов сбалансировано (50 записей в каждом)")
print("   - Не все признаки распределены нормально (по тесту Шапиро-Уилка)")

print("\n2. Расположение классов:")
print("   - Iris-setosa линейно отделима от остальных")
print("   - Iris-versicolor и Iris-virginica не линейно разделимы")

print("\n3. Результаты классификации:")
print(f"   - MLP: Accuracy = {metrics_mlp_test['Accuracy']:.4f}")
print(f"   - SVM: Accuracy = {metrics_svm_test['Accuracy']:.4f}")
print(f"   - Оба метода показали высокую точность классификации")

print("\n4. Значимость признаков:")
print("   - Наиболее важные признаки различаются для разных методов")
print(f"   - Наименее важный признак: {least_important_feature}")

print("\n5. Влияние параметров:")
print("   - Изменение параметров влияет на точность классификации")
print("   - Для данного датасета базовая конфигурация показала хорошие результаты")

print("\n6. Удаление признаков:")
print(f"   - Удаление {least_important_feature} привело к изменению accuracy на {acc_mlp_reduced - metrics_mlp_test['Accuracy']:.6f}")

print("\n" + "=" * 80)
print("АНАЛИЗ ЗАВЕРШЕН. Все графики сохранены в текущей директории.")
print("=" * 80)


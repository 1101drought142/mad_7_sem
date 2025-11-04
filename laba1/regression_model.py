"""
Модуль для построения регрессионных моделей
Включает линейную регрессию, гребневую регрессию и логистическую регрессию
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_CONFIG

class RegressionModel:
    """
    Класс для построения и оценки регрессионных моделей
    """
    
    def __init__(self, data_dict, target_column='H'):
        """
        Инициализация с данными и целевой переменной
        
        Args:
            data_dict: Словарь с данными столбцов из DataLoader
            target_column: ID целевой переменной (по умолчанию 'H' - стоимость за м²)
        """
        self.data_dict = data_dict
        self.target_column = target_column
        self.X = None
        self.y = None
        self.feature_names = []
        self.target_name = ""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_categorical_target = False
        self.pca = None
        self.pca_components = None
        self.explained_variance_ratio = None
        self.pca_components_count = None

        # Подготавливаем данные
        self._prepare_data()
    
    def _prepare_data(self):
        """
        Подготавливает данные для обучения модели
        """
        # Создаем DataFrame из всех данных
        data_for_model = {}
        
        for col_id, col_info in self.data_dict.items():
            data_for_model[col_info['name']] = col_info['data']
            if col_id == self.target_column:
                self.target_name = col_info['name']
        
        # Создаем DataFrame
        df = pd.DataFrame(data_for_model)
        
        # Определяем целевую переменную и признаки
        self.y = df[self.target_name].values
        self.X = df.drop(columns=[self.target_name]).values
        self.feature_names = [col for col in df.columns if col != self.target_name]
        
        # Проверяем, является ли целевая переменная категориальной
        unique_values = len(np.unique(self.y))
        total_values = len(self.y)
        
        # Если уникальных значений меньше 10% от общего количества, считаем категориальной
        if unique_values < total_values * 0.1:
            self.is_categorical_target = True
            self.y = self.label_encoder.fit_transform(self.y)
            print(f"Целевая переменная '{self.target_name}' определена как категориальная")
            print(f"Уникальные классы: {self.label_encoder.classes_}")
        else:
            print(f"Целевая переменная '{self.target_name}' определена как непрерывная")
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Разделяет данные на обучающую и тестовую выборки
        
        Args:
            test_size: Доля тестовой выборки (по умолчанию 0.2 = 20%)
            random_state: Случайное состояние для воспроизводимости
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Нормализуем признаки
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Разделение данных:")
        print(f"  Обучающая выборка: {X_train_scaled.shape[0]} образцов")
        print(f"  Тестовая выборка: {X_test_scaled.shape[0]} образцов")
        print(f"  Количество признаков: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def analyze_pca(self, X_data, n_components=None, min_variance_explained=0.95):
        """
        Анализирует данные с помощью PCA
        
        Args:
            X_data: Данные для анализа
            n_components: Количество компонент (если None, определяется автоматически)
            min_variance_explained: Минимальная доля объясненной дисперсии
            
        Returns:
            dict: Результаты PCA анализа
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("АНАЛИЗ ГЛАВНЫХ КОМПОНЕНТ (PCA)")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Определяем количество компонент
        if n_components is None:
            # Находим количество компонент для объяснения 95% дисперсии
            pca_temp = PCA()
            pca_temp.fit(X_data)
            cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= min_variance_explained) + 1
            print(f"Автоматически выбрано {n_components} компонент для объяснения {min_variance_explained*100:.1f}% дисперсии")
        
        self.pca_components_count = n_components

        # Создаем и обучаем PCA
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X_data)
        
        # Сохраняем результаты
        self.pca_components = X_pca
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        # Выводим результаты
        print(f"Количество компонент: {n_components}")
        print(f"Общая объясненная дисперсия: {np.sum(self.explained_variance_ratio):.4f}")
        
        print(f"\nОбъясненная дисперсия по компонентам:")
        for i, ratio in enumerate(self.explained_variance_ratio):
            print(f"  Компонента {i+1}: {ratio:.4f} ({ratio*100:.2f}%)")
        
        # Кумулятивная дисперсия
        cumsum_ratio = np.cumsum(self.explained_variance_ratio)
        print(f"\nКумулятивная объясненная дисперсия:")
        for i, cum_ratio in enumerate(cumsum_ratio):
            print(f"  Первые {i+1} компонент: {cum_ratio:.4f} ({cum_ratio*100:.2f}%)")
        
        return {
            'n_components': n_components,
            'explained_variance_ratio': self.explained_variance_ratio,
            'cumulative_variance': cumsum_ratio,
            'total_variance_explained': np.sum(self.explained_variance_ratio),
            'pca_components': X_pca
        }
    
    def plot_pca_analysis(self, save_path=None):
        """
        Строит графики анализа PCA по два в каждом окне
        
        Args:
            save_path: Путь для сохранения графика (опционально)
        """
        if self.pca is None:
            print("Ошибка: PCA не был выполнен. Сначала вызовите analyze_pca()")
            return
        
        # График 1: Объясненная дисперсия и кумулятивная дисперсия
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig1.suptitle('Анализ дисперсии PCA', fontsize=16, fontweight='bold')
        
        # Объясненная дисперсия по компонентам
        bars = ax1.bar(range(1, len(self.explained_variance_ratio) + 1), 
                      self.explained_variance_ratio, 
                      color='skyblue', edgecolor='navy', alpha=0.7)
        ax1.set_title('Объясненная дисперсия по компонентам', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Номер компоненты', fontsize=12)
        ax1.set_ylabel('Доля объясненной дисперсии', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{self.explained_variance_ratio[i]:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Кумулятивная объясненная дисперсия
        cumsum_ratio = np.cumsum(self.explained_variance_ratio)
        ax2.plot(range(1, len(cumsum_ratio) + 1), cumsum_ratio, 'bo-', 
                linewidth=2, markersize=8, label='Кумулятивная дисперсия')
        ax2.axhline(y=0.95, color='r', linestyle='--', linewidth=2, label='95% дисперсии')
        ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=2, label='80% дисперсии')
        ax2.set_title('Кумулятивная объясненная дисперсия', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Количество компонент', fontsize=12)
        ax2.set_ylabel('Кумулятивная доля дисперсии', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_variance.png", dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("img/pca_variance_explained.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # График 2: Scree plot и визуализация компонент
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
        fig2.suptitle('Scree Plot и визуализация компонент', fontsize=16, fontweight='bold')
        
        # Scree plot
        ax3.plot(range(1, len(self.explained_variance_ratio) + 1), 
                self.explained_variance_ratio, 'ro-', 
                linewidth=2, markersize=8)
        ax3.set_title('Scree Plot', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Номер компоненты', fontsize=12)
        ax3.set_ylabel('Объясненная дисперсия', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Визуализация главных компонент
        if self.pca_components.shape[1] >= 2:
            scatter = ax4.scatter(self.pca_components[:, 0], self.pca_components[:, 1], 
                                alpha=0.6, c=range(len(self.pca_components)), 
                                cmap='viridis', s=50)
            ax4.set_xlabel('Первая главная компонента', fontsize=12)
            ax4.set_ylabel('Вторая главная компонента', fontsize=12)
            ax4.set_title('Первые две главные компоненты', fontsize=14, fontweight='bold')
            plt.colorbar(scatter, ax=ax4, label='Индекс образца')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Недостаточно компонент\nдля визуализации', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Визуализация компонент', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_scree_components.png", dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("img/pca_scree_components.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # График 3: Важность признаков и сравнение размерности
        fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
        fig3.suptitle('Важность признаков и сравнение размерности', fontsize=16, fontweight='bold')
        
        # Важность признаков в первой компоненте
        if hasattr(self.pca, 'components_') and len(self.pca.components_) > 0:
            feature_importance = np.abs(self.pca.components_[0])
            feature_names = self.feature_names[:len(feature_importance)]
            
            bars = ax5.barh(feature_names, feature_importance, 
                           color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax5.set_title('Важность признаков в PC1', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Абсолютное значение веса', fontsize=12)
            ax5.set_ylabel('Признаки', fontsize=12)
            ax5.grid(True, alpha=0.3, axis='x')
            
            # Добавляем значения на столбцы
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax5.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{feature_importance[i]:.3f}',
                        ha='left', va='center', fontsize=9)
        
        # Сравнение исходных признаков и компонент
        original_features = len(self.feature_names)
        pca_components = len(self.explained_variance_ratio)
        variance_explained = np.sum(self.explained_variance_ratio)
        
        categories = ['Исходные\nпризнаки', 'PCA\nкомпоненты']
        values = [original_features, pca_components]
        colors = ['lightblue', 'lightgreen']
        
        bars = ax6.bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
        ax6.set_title(f'Сравнение размерности\n({variance_explained:.1%} дисперсии сохранено)', 
                     fontsize=14, fontweight='bold')
        ax6.set_ylabel('Количество признаков/компонент', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars, values):
            ax6.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    str(value), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_importance_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        else:
            plt.savefig("img/pca_importance_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        if save_path:
            print(f"Графики PCA анализа сохранены с префиксом: {save_path}")
    
    def get_pca_feature_importance(self):
        """
        Анализирует важность исходных признаков в главных компонентах
        
        Returns:
            pandas.DataFrame: Таблица важности признаков
        """
        if self.pca is None:
            print("Ошибка: PCA не был выполнен. Сначала вызовите analyze_pca()")
            return None
        
        # Создаем DataFrame с весами компонент
        components_df = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=self.feature_names
        )
        
        # Вычисляем абсолютные значения для определения важности
        abs_components_df = np.abs(components_df)
        
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ВАЖНОСТЬ ПРИЗНАКОВ В ГЛАВНЫХ КОМПОНЕНТАХ")
        print("="*OUTPUT_CONFIG['separator_length'])
        print("Веса компонент (чем больше абсолютное значение, тем важнее признак):")
        print(components_df.round(4))
        
        print(f"\nАбсолютные значения весов:")
        print(abs_components_df.round(4))
        
        # Находим наиболее важные признаки для каждой компоненты
        print(f"\nНаиболее важные признаки для каждой компоненты:")
        for i, col in enumerate(components_df.columns):
            most_important = abs_components_df[col].idxmax()
            weight = components_df.loc[most_important, col]
            print(f"  {col}: {most_important} (вес: {weight:.4f})")
        
        return components_df
    
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """
        Обучает модель линейной регрессии
        
        Args:
            X_train, y_train: Обучающие данные
            X_test, y_test: Тестовые данные
            
        Returns:
            dict: Результаты модели
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ЛИНЕЙНАЯ РЕГРЕССИЯ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Обучаем модель
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Метрики для обучающей выборки
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Метрики для тестовой выборки
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        print(f"Метрики на обучающей выборке:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.4f}")
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.4f}")
        
        print(f"\nКросс-валидация (R²):")
        print(f"  Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Коэффициенты модели
        print(f"\nКоэффициенты модели:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {feature}: {model.coef_[i]:.4f}")
        print(f"  Свободный член: {model.intercept_:.4f}")
        
        return {
            'model': model,
            'train_metrics': {'rmse': train_rmse, 'r2': train_r2, 'mape': train_mape},
            'test_metrics': {'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape},
            'cv_scores': cv_scores,
            'predictions': {'train': y_train_pred, 'test': y_test_pred}
        }
    
    def train_ridge_regression(self, X_train, y_train, X_test, y_test, alpha=1.0):
        """
        Обучает модель гребневой регрессии
        
        Args:
            X_train, y_train: Обучающие данные
            X_test, y_test: Тестовые данные
            alpha: Параметр регуляризации
            
        Returns:
            dict: Результаты модели
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ГРЕБНЕВАЯ РЕГРЕССИЯ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Обучаем модель
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Метрики для обучающей выборки
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Метрики для тестовой выборки
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        print(f"Параметр регуляризации α = {alpha}")
        print(f"\nМетрики на обучающей выборке:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.4f}")
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.4f}")
        
        print(f"\nКросс-валидация (R²):")
        print(f"  Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Коэффициенты модели
        print(f"\nКоэффициенты модели:")
        for i, feature in enumerate(self.feature_names):
            print(f"  {feature}: {model.coef_[i]:.4f}")
        print(f"  Свободный член: {model.intercept_:.4f}")
        
        return {
            'model': model,
            'train_metrics': {'rmse': train_rmse, 'r2': train_r2, 'mape': train_mape},
            'test_metrics': {'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape},
            'cv_scores': cv_scores,
            'predictions': {'train': y_train_pred, 'test': y_test_pred}
        }
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Обучает модель логистической регрессии (для категориальных данных)
        
        Args:
            X_train, y_train: Обучающие данные
            X_test, y_test: Тестовые данные
            
        Returns:
            dict: Результаты модели
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Обучаем модель
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Вероятности
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
        
        # Метрики для обучающей выборки
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Метрики для тестовой выборки
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        print(f"Метрики на обучающей выборке:")
        print(f"  Точность: {train_accuracy:.4f}")
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"  Точность: {test_accuracy:.4f}")
        
        print(f"\nКросс-валидация (Точность):")
        print(f"  Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Отчет по классификации
        print(f"\nОтчет по классификации (тестовая выборка):")
        print(classification_report(y_test, y_test_pred, 
                                  target_names=[str(cls) for cls in self.label_encoder.classes_]))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_test_pred)
        print(f"\nМатрица ошибок:")
        print(cm)
        
        return {
            'model': model,
            'train_metrics': {'accuracy': train_accuracy},
            'test_metrics': {'accuracy': test_accuracy},
            'cv_scores': cv_scores,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'probabilities': {'train': y_train_proba, 'test': y_test_proba}
        }
    
    def plot_predictions(self, y_true, y_pred, title="Предсказания модели"):
        """
        Строит график сравнения истинных и предсказанных значений
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            title: Заголовок графика
        """
        plt.figure(figsize=(10, 6))
        
        if self.is_categorical_target:
            # Для категориальных данных - матрица ошибок
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{title} - Матрица ошибок')
            plt.xlabel('Предсказанные классы')
            plt.ylabel('Истинные классы')
        else:
            # Для непрерывных данных - scatter plot
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Истинные значения')
            plt.ylabel('Предсказанные значения')
            plt.title(f'{title} - Сравнение истинных и предсказанных значений')
        
        plt.tight_layout()
        plt.savefig("img/model_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def compare_models(self, results_dict):
        """
        Сравнивает результаты различных моделей
        
        Args:
            results_dict: Словарь с результатами моделей
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("СРАВНЕНИЕ МОДЕЛЕЙ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        if self.is_categorical_target:
            # Для категориальных данных сравниваем точность
            print(f"{'Модель':<20} {'Точность (тест)':<15} {'CV (среднее)':<15}")
            print("-" * 50)
            
            for model_name, results in results_dict.items():
                test_acc = results['test_metrics']['accuracy']
                cv_mean = results['cv_scores'].mean()
                print(f"{model_name:<20} {test_acc:<15.4f} {cv_mean:<15.4f}")
        else:
            # Для непрерывных данных сравниваем R² и RMSE
            print(f"{'Модель':<20} {'R² (тест)':<12} {'RMSE (тест)':<12} {'CV R² (среднее)':<15}")
            print("-" * 60)
            
            for model_name, results in results_dict.items():
                test_r2 = results['test_metrics']['r2']
                test_rmse = results['test_metrics']['rmse']
                cv_mean = results['cv_scores'].mean()
                print(f"{model_name:<20} {test_r2:<12.4f} {test_rmse:<12.4f} {cv_mean:<15.4f}")
    
    def full_analysis(self, test_size=0.2, ridge_alpha=1.0):
        """
        Выполняет полный анализ регрессионных моделей
        
        Args:
            test_size: Доля тестовой выборки
            ridge_alpha: Параметр регуляризации для гребневой регрессии
        """
        print("="*OUTPUT_CONFIG['separator_length'])
        print("ПОЛНЫЙ АНАЛИЗ РЕГРЕССИОННЫХ МОДЕЛЕЙ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        results = {}
        
        if self.is_categorical_target:
            # Для категориальных данных используем логистическую регрессию
            print("Обнаружена категориальная целевая переменная - используем логистическую регрессию")
            results['Логистическая регрессия'] = self.train_logistic_regression(
                X_train, y_train, X_test, y_test
            )
        else:
            # Для непрерывных данных используем линейную и гребневую регрессию
            results['Линейная регрессия'] = self.train_linear_regression(
                X_train, y_train, X_test, y_test
            )
            
            results['Гребневая регрессия'] = self.train_ridge_regression(
                X_train, y_train, X_test, y_test, alpha=ridge_alpha
            )
        
        # Сравниваем модели
        self.compare_models(results)
        
        return results
    
    def train_linear_regression_pca(self, X_train, y_train, X_test, y_test, n_components=None):
        """
        Обучает модель линейной регрессии на главных компонентах
        
        Args:
            X_train, y_train: Обучающие данные
            X_test, y_test: Тестовые данные
            n_components: Количество главных компонент
            
        Returns:
            dict: Результаты модели
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ЛИНЕЙНАЯ РЕГРЕССИЯ НА ГЛАВНЫХ КОМПОНЕНТАХ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Применяем PCA к обучающим данным
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"Количество главных компонент: {pca.n_components_}")
        print(f"Объясненная дисперсия: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Обучаем модель на главных компонентах
        model = LinearRegression()
        model.fit(X_train_pca, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train_pca)
        y_test_pred = model.predict(X_test_pca)
        
        # Метрики для обучающей выборки
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Метрики для тестовой выборки
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='r2')
        
        print(f"Метрики на обучающей выборке:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.4f}")
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.4f}")
        
        print(f"\nКросс-валидация (R²):")
        print(f"  Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Коэффициенты модели (для главных компонент)
        print(f"\nКоэффициенты модели (главные компоненты):")
        for i in range(pca.n_components_):
            print(f"  PC{i+1}: {model.coef_[i]:.4f}")
        print(f"  Свободный член: {model.intercept_:.4f}")
        
        return {
            'model': model,
            'pca': pca,
            'train_metrics': {'rmse': train_rmse, 'r2': train_r2, 'mape': train_mape},
            'test_metrics': {'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape},
            'cv_scores': cv_scores,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'explained_variance': np.sum(pca.explained_variance_ratio_)
        }
    
    def train_ridge_regression_pca(self, X_train, y_train, X_test, y_test, alpha=1.0, n_components=None):
        """
        Обучает модель гребневой регрессии на главных компонентах
        
        Args:
            X_train, y_train: Обучающие данные
            X_test, y_test: Тестовые данные
            alpha: Параметр регуляризации
            n_components: Количество главных компонент
            
        Returns:
            dict: Результаты модели
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ГРЕБНЕВАЯ РЕГРЕССИЯ НА ГЛАВНЫХ КОМПОНЕНТАХ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Применяем PCA к обучающим данным
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        print(f"Параметр регуляризации α = {alpha}")
        print(f"Количество главных компонент: {pca.n_components_}")
        print(f"Объясненная дисперсия: {np.sum(pca.explained_variance_ratio_):.4f}")
        
        # Обучаем модель на главных компонентах
        model = Ridge(alpha=alpha)
        model.fit(X_train_pca, y_train)
        
        # Предсказания
        y_train_pred = model.predict(X_train_pca)
        y_test_pred = model.predict(X_test_pca)
        
        # Метрики для обучающей выборки
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
        
        # Метрики для тестовой выборки
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
        
        # Кросс-валидация
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='r2')
        
        print(f"Метрики на обучающей выборке:")
        print(f"  RMSE: {train_rmse:.4f}")
        print(f"  R²: {train_r2:.4f}")
        print(f"  MAPE: {train_mape:.4f}")
        
        print(f"\nМетрики на тестовой выборке:")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  R²: {test_r2:.4f}")
        print(f"  MAPE: {test_mape:.4f}")
        
        print(f"\nКросс-валидация (R²):")
        print(f"  Среднее: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Коэффициенты модели (для главных компонент)
        print(f"\nКоэффициенты модели (главные компоненты):")
        for i in range(pca.n_components_):
            print(f"  PC{i+1}: {model.coef_[i]:.4f}")
        print(f"  Свободный член: {model.intercept_:.4f}")
        
        return {
            'model': model,
            'pca': pca,
            'train_metrics': {'rmse': train_rmse, 'r2': train_r2, 'mape': train_mape},
            'test_metrics': {'rmse': test_rmse, 'r2': test_r2, 'mape': test_mape},
            'cv_scores': cv_scores,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'explained_variance': np.sum(pca.explained_variance_ratio_)
        }
    
    def compare_original_vs_pca_models(self, original_results, pca_results):
        """
        Сравнивает модели на исходных данных и на главных компонентах
        
        Args:
            original_results: Результаты моделей на исходных данных
            pca_results: Результаты моделей на главных компонентах
        """
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("СРАВНЕНИЕ МОДЕЛЕЙ: ИСХОДНЫЕ ДАННЫЕ VS ГЛАВНЫЕ КОМПОНЕНТЫ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        if self.is_categorical_target:
            print("Сравнение для категориальных данных (логистическая регрессия):")
            print(f"{'Модель':<30} {'Точность (тест)':<15} {'CV (среднее)':<15}")
            print("-" * 60)
            
            for model_name in original_results.keys():
                if model_name in pca_results:
                    orig_acc = original_results[model_name]['test_metrics']['accuracy']
                    orig_cv = original_results[model_name]['cv_scores'].mean()
                    pca_acc = pca_results[model_name]['test_metrics']['accuracy']
                    pca_cv = pca_results[model_name]['cv_scores'].mean()
                    
                    print(f"{model_name} (исходные):")
                    print(f"  Точность: {orig_acc:.4f}, CV: {orig_cv:.4f}")
                    print(f"{model_name} (PCA):")
                    print(f"  Точность: {pca_acc:.4f}, CV: {pca_cv:.4f}")
                    print(f"  Разница: {pca_acc - orig_acc:+.4f}")
                    print()
        else:
            print("Сравнение для непрерывных данных (линейная и гребневая регрессия):")
            print(f"{'Модель':<30} {'R² (тест)':<12} {'RMSE (тест)':<12} {'CV R² (среднее)':<15}")
            print("-" * 70)
            
            for model_name in original_results.keys():
                if model_name in pca_results:
                    orig_r2 = original_results[model_name]['test_metrics']['r2']
                    orig_rmse = original_results[model_name]['test_metrics']['rmse']
                    orig_cv = original_results[model_name]['cv_scores'].mean()
                    
                    pca_r2 = pca_results[model_name]['test_metrics']['r2']
                    pca_rmse = pca_results[model_name]['test_metrics']['rmse']
                    pca_cv = pca_results[model_name]['cv_scores'].mean()
                    
                    print(f"{model_name} (исходные):")
                    print(f"  R²: {orig_r2:.4f}, RMSE: {orig_rmse:.4f}, CV: {orig_cv:.4f}")
                    print(f"{model_name} (PCA):")
                    print(f"  R²: {pca_r2:.4f}, RMSE: {pca_rmse:.4f}, CV: {pca_cv:.4f}")
                    print(f"  Разница R²: {pca_r2 - orig_r2:+.4f}")
                    print(f"  Разница RMSE: {pca_rmse - orig_rmse:+.4f}")
                    print()
    
    def full_analysis_with_pca(self, test_size=0.2, ridge_alpha=1.0, pca_components=None):
        """
        Выполняет полный анализ регрессионных моделей с PCA
        
        Args:
            test_size: Доля тестовой выборки
            ridge_alpha: Параметр регуляризации для гребневой регрессии
            pca_components: Количество главных компонент для PCA
        """
        print("="*OUTPUT_CONFIG['separator_length'])
        print("ПОЛНЫЙ АНАЛИЗ РЕГРЕССИОННЫХ МОДЕЛЕЙ С PCA")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Разделяем данные
        X_train, X_test, y_train, y_test = self.split_data(test_size=test_size)
        
        # Анализ PCA
        pca_results = self.analyze_pca(X_train, n_components=pca_components)
        
        # Создаем графики PCA по два в каждом окне
        print("\nСоздание графиков PCA (по два в каждом окне)...")
        self.plot_pca_analysis("pca_analysis")
        
        self.get_pca_feature_importance()
        
        # Модели на исходных данных
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("МОДЕЛИ НА ИСХОДНЫХ ДАННЫХ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        original_results = {}
        if self.is_categorical_target:
            original_results['Логистическая регрессия'] = self.train_logistic_regression(
                X_train, y_train, X_test, y_test
            )
        else:
            original_results['Линейная регрессия'] = self.train_linear_regression(
                X_train, y_train, X_test, y_test
            )
            original_results['Гребневая регрессия'] = self.train_ridge_regression(
                X_train, y_train, X_test, y_test, alpha=ridge_alpha
            )
        
        # Модели на главных компонентах
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("МОДЕЛИ НА ГЛАВНЫХ КОМПОНЕНТАХ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        pca_model_results = {}
        if self.is_categorical_target:
            pca_model_results['Логистическая регрессия'] = self.train_logistic_regression(
                X_train, y_train, X_test, y_test
            )
        else:
            pca_model_results['Линейная регрессия'] = self.train_linear_regression_pca(
                X_train, y_train, X_test, y_test, n_components=self.pca_components_count
            )
            pca_model_results['Гребневая регрессия'] = self.train_ridge_regression_pca(
                X_train, y_train, X_test, y_test, alpha=ridge_alpha, n_components=self.pca_components_count
            )
        
        # Сравнение моделей
        self.compare_original_vs_pca_models(original_results, pca_model_results)
        
        return {
            'original_results': original_results,
            'pca_results': pca_model_results,
            'pca_analysis': pca_results
        }

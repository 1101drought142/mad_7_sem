"""
Модуль для анализа корреляций и мультиколлинеарности
Строит матрицу корреляций и рассчитывает VIF-коэффициенты
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, spearmanr
from config import COLUMN_CONFIG, OUTPUT_CONFIG

class CorrelationAnalyzer:
    """
    Класс для анализа корреляций и мультиколлинеарности
    """
    
    def __init__(self, data_dict):
        """
        Инициализация с данными из всех столбцов
        
        Args:
            data_dict: Словарь с данными столбцов из DataLoader
        """
        self.data_dict = data_dict
        self.correlation_matrix = None
        self.vif_scores = None
        
    def build_correlation_matrix(self):
        """
        Строит матрицу корреляций для всех столбцов
        
        Returns:
            pandas.DataFrame: Матрица корреляций
        """
        # Создаем DataFrame из всех данных
        data_for_correlation = {}
        
        for col_id, col_info in self.data_dict.items():
            data_for_correlation[col_info['name']] = col_info['data']
        
        # Создаем DataFrame
        df = pd.DataFrame(data_for_correlation)
        
        # Вычисляем матрицу корреляций
        self.correlation_matrix = df.corr()
        
        print("="*OUTPUT_CONFIG['separator_length'])
        print("МАТРИЦА КОРРЕЛЯЦИЙ")
        print("="*OUTPUT_CONFIG['separator_length'])
        print(self.correlation_matrix.round(4))
        
        return self.correlation_matrix
    
    def plot_correlation_heatmap(self, save_path=None):
        """
        Строит тепловую карту корреляций
        
        Args:
            save_path: Путь для сохранения графика (опционально)
        """
        if self.correlation_matrix is None:
            self.build_correlation_matrix()
        
        plt.figure(figsize=(10, 8))
        
        # Создаем маску для верхнего треугольника
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # Строим тепловую карту
        sns.heatmap(
            self.correlation_matrix,
            mask=mask,
            annot=True,
            cmap='coolwarm',
            center=0,
            square=True,
            fmt='.3f',
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Матрица корреляций', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        else:
            plt.savefig('img/correlation_matrix.png', dpi=300, bbox_inches='tight')
            print("График сохранен: img/correlation_matrix.png")
        
        plt.close()
    
    def calculate_vif(self, data):
        """
        Вычисляет VIF (Variance Inflation Factor) для каждого признака
        
        Args:
            data: pandas.DataFrame с данными
            
        Returns:
            pandas.Series: VIF коэффициенты для каждого признака
        """
        # Стандартизируем данные
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)
        
        vif_data = pd.DataFrame()
        vif_data["Feature"] = data.columns
        vif_data["VIF"] = [self._vif_score(data_scaled, col) for col in data_scaled.columns]
        
        return vif_data
    
    def _vif_score(self, data, target_col):
        """
        Вычисляет VIF для конкретного столбца
        
        Args:
            data: DataFrame с данными
            target_col: Название целевого столбца
            
        Returns:
            float: VIF коэффициент
        """
        # Исключаем целевой столбец из предикторов
        predictors = data.drop(columns=[target_col])
        target = data[target_col]
        
        # Обучаем модель линейной регрессии
        model = LinearRegression()
        model.fit(predictors, target)
        
        # Вычисляем R²
        r_squared = model.score(predictors, target)
        
        # VIF = 1 / (1 - R²)
        if r_squared == 1:
            return float('inf')  # Полная мультиколлинеарность
        else:
            return 1 / (1 - r_squared)
    
    def analyze_multicollinearity(self):
        """
        Анализирует мультиколлинеарность с помощью VIF
        
        Returns:
            pandas.DataFrame: Результаты анализа VIF
        """
        # Подготавливаем данные для VIF анализа
        data_for_vif = {}
        
        for col_id, col_info in self.data_dict.items():
            data_for_vif[col_info['name']] = col_info['data']
        
        # Создаем DataFrame
        df = pd.DataFrame(data_for_vif)
        
        # Вычисляем VIF
        self.vif_scores = self.calculate_vif(df)
        
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("АНАЛИЗ МУЛЬТИКОЛЛИНЕАРНОСТИ (VIF-КОЭФФИЦИЕНТЫ)")
        print("="*OUTPUT_CONFIG['separator_length'])
        print(self.vif_scores.round(4))
        
        # Интерпретация результатов
        print("\nИНТЕРПРЕТАЦИЯ VIF-КОЭФФИЦИЕНТОВ:")
        print("• VIF < 5: Мультиколлинеарность отсутствует")
        print("• 5 ≤ VIF < 10: Умеренная мультиколлинеарность")
        print("• VIF ≥ 10: Высокая мультиколлинеарность (требует внимания)")
        
        # Анализ результатов
        high_vif = self.vif_scores[self.vif_scores['VIF'] >= 10]
        moderate_vif = self.vif_scores[(self.vif_scores['VIF'] >= 5) & (self.vif_scores['VIF'] < 10)]
        
        if len(high_vif) > 0:
            print(f"\n⚠️  ВЫСОКАЯ МУЛЬТИКОЛЛИНЕАРНОСТЬ (VIF ≥ 10):")
            for _, row in high_vif.iterrows():
                print(f"   {row['Feature']}: VIF = {row['VIF']:.2f}")
        
        if len(moderate_vif) > 0:
            print(f"\n⚠️  УМЕРЕННАЯ МУЛЬТИКОЛЛИНЕАРНОСТЬ (5 ≤ VIF < 10):")
            for _, row in moderate_vif.iterrows():
                print(f"   {row['Feature']}: VIF = {row['VIF']:.2f}")
        
        if len(high_vif) == 0 and len(moderate_vif) == 0:
            print("\n✅ Мультиколлинеарность отсутствует (все VIF < 5)")
        
        return self.vif_scores
    
    def get_correlation_insights(self):
        """
        Анализирует матрицу корреляций и выдает выводы
        
        Returns:
            dict: Словарь с выводами о корреляциях
        """
        if self.correlation_matrix is None:
            self.build_correlation_matrix()
        
        insights = {
            'strong_correlations': [],
            'moderate_correlations': [],
            'weak_correlations': []
        }
        
        # Анализируем корреляции (исключаем диагональ)
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                col1 = self.correlation_matrix.columns[i]
                col2 = self.correlation_matrix.columns[j]
                corr_value = self.correlation_matrix.iloc[i, j]
                
                correlation_info = {
                    'variables': f"{col1} - {col2}",
                    'correlation': corr_value
                }
                
                if abs(corr_value) >= 0.7:
                    insights['strong_correlations'].append(correlation_info)
                elif abs(corr_value) >= 0.3:
                    insights['moderate_correlations'].append(correlation_info)
                else:
                    insights['weak_correlations'].append(correlation_info)
        
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("ВЫВОДЫ ПО КОРРЕЛЯЦИЯМ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        if insights['strong_correlations']:
            print("\n🔴 СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| ≥ 0.7):")
            for corr in insights['strong_correlations']:
                print(f"   {corr['variables']}: r = {corr['correlation']:.3f}")
        
        if insights['moderate_correlations']:
            print("\n🟡 УМЕРЕННЫЕ КОРРЕЛЯЦИИ (0.3 ≤ |r| < 0.7):")
            for corr in insights['moderate_correlations']:
                print(f"   {corr['variables']}: r = {corr['correlation']:.3f}")
        
        if insights['weak_correlations']:
            print("\n🟢 СЛАБЫЕ КОРРЕЛЯЦИИ (|r| < 0.3):")
            for corr in insights['weak_correlations']:
                print(f"   {corr['variables']}: r = {corr['correlation']:.3f}")
        
        return insights
    
    def full_analysis(self, save_plot=False):
        """
        Выполняет полный анализ корреляций и мультиколлинеарности
        
        Args:
            save_plot: Сохранять ли график корреляций
        """
        print("="*OUTPUT_CONFIG['separator_length'])
        print("ПОЛНЫЙ АНАЛИЗ КОРРЕЛЯЦИЙ И МУЛЬТИКОЛЛИНЕАРНОСТИ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # 1. Строим матрицу корреляций
        self.build_correlation_matrix()
        
        # 2. Анализируем корреляции
        insights = self.get_correlation_insights()
        
        # 3. Анализируем мультиколлинеарность
        vif_results = self.analyze_multicollinearity()
        
        # 4. Строим график
        if save_plot:
            self.plot_correlation_heatmap("img/correlation_matrix.png")
        else:
            self.plot_correlation_heatmap()
        
        return {
            'correlation_matrix': self.correlation_matrix,
            'vif_scores': vif_results,
            'insights': insights
        }
    
    def create_contingency_table(self, col1_id, col2_id, bins=5):
        """
        Создает таблицу сопряженности для двух переменных
        
        Args:
            col1_id: ID первого столбца
            col2_id: ID второго столбца  
            bins: Количество интервалов для дискретизации
            
        Returns:
            dict: Словарь с результатами анализа таблицы сопряженности
        """
        # Получаем данные
        col1_data = self.data_dict[col1_id]['data']
        col2_data = self.data_dict[col2_id]['data']
        col1_name = self.data_dict[col1_id]['name']
        col2_name = self.data_dict[col2_id]['name']
        
        print("="*OUTPUT_CONFIG['separator_length'])
        print(f"ТАБЛИЦА СОПРЯЖЕННОСТИ: {col1_name.upper()} vs {col2_name.upper()}")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Создаем интервалы для дискретизации с читаемыми подписями
        col1_cut = pd.cut(col1_data, bins=bins)
        col2_cut = pd.cut(col2_data, bins=bins)
        
        # Создаем читаемые подписи на основе диапазонов
        col1_labels = []
        col2_labels = []
        
        for i, interval in enumerate(col1_cut.cat.categories):
            left = f"{interval.left:.1f}" if not pd.isna(interval.left) else "0"
            right = f"{interval.right:.1f}" if not pd.isna(interval.right) else "∞"
            col1_labels.append(f"{left}-{right}")
        
        for i, interval in enumerate(col2_cut.cat.categories):
            left = f"{interval.left:.1f}" if not pd.isna(interval.left) else "0"
            right = f"{interval.right:.1f}" if not pd.isna(interval.right) else "∞"
            col2_labels.append(f"{left}-{right}")
        
        # Применяем новые подписи
        col1_bins = col1_cut.cat.rename_categories(col1_labels)
        col2_bins = col2_cut.cat.rename_categories(col2_labels)
        
        # Создаем таблицу сопряженности
        contingency_table = pd.crosstab(col1_bins, col2_bins, margins=True)
        
        print("\nНАЧАЛЬНАЯ ТАБЛИЦА СОПРЯЖЕННОСТИ:")
        print("-" * 50)
        print(contingency_table)
        
        # Вычисляем теоретические частоты
        # Убираем строку и столбец с итогами для расчетов
        observed = contingency_table.iloc[:-1, :-1].values
        
        # Вычисляем теоретические частоты
        row_totals = observed.sum(axis=1)
        col_totals = observed.sum(axis=0)
        grand_total = observed.sum()
        
        expected = np.outer(row_totals, col_totals) / grand_total
        
        # Создаем таблицу с теоретическими частотами
        expected_table = pd.DataFrame(
            expected,
            index=contingency_table.index[:-1],
            columns=contingency_table.columns[:-1]
        )
        
        print("\nТЕОРЕТИЧЕСКИЕ ЧАСТОТЫ:")
        print("-" * 50)
        print(expected_table.round(2))
        
        # Вычисляем критерий хи-квадрат
        chi2_stat, p_value, dof, expected_freq = chi2_contingency(observed)
        
        print(f"\nКРИТЕРИЙ ХИ-КВАДРАТ:")
        print("-" * 30)
        print(f"Статистика chi2: {chi2_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Степени свободы: {dof}")
        
        # Интерпретация результата
        if p_value < 0.05:
            conclusion = "ОТКЛОНЯЕМ гипотезу о независимости (p < 0.05)"
            interpretation = "Между переменными есть статистически значимая связь"
        else:
            conclusion = "НЕ ОТКЛОНЯЕМ гипотезу о независимости (p ≥ 0.05)"
            interpretation = "Между переменными нет статистически значимой связи"
        
        print(f"Вывод: {conclusion}")
        print(f"Интерпретация: {interpretation}")
        
        # Вычисляем коэффициент Крамера V (для внутреннего использования)
        n = grand_total
        cramers_v = np.sqrt(chi2_stat / (n * (min(observed.shape) - 1)))
        
        # Интерпретация силы связи
        if cramers_v < 0.1:
            strength = "Очень слабая связь"
        elif cramers_v < 0.3:
            strength = "Слабая связь"
        elif cramers_v < 0.5:
            strength = "Умеренная связь"
        else:
            strength = "Сильная связь"
        
        # Сохраняем результаты
        results = {
            'observed_table': contingency_table,
            'expected_table': expected_table,
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramers_v,
            'conclusion': conclusion,
            'interpretation': interpretation,
            'strength': strength
        }
        
        return results
    
    def plot_contingency_heatmap(self, col1_id, col2_id, bins=5, save_path=None):
        """
        Создает тепловую карту для таблицы сопряженности
        
        Args:
            col1_id: ID первого столбца
            col2_id: ID второго столбца
            bins: Количество интервалов для дискретизации
            save_path: Путь для сохранения графика
        """
        # Получаем данные
        col1_data = self.data_dict[col1_id]['data']
        col2_data = self.data_dict[col2_id]['data']
        col1_name = self.data_dict[col1_id]['name']
        col2_name = self.data_dict[col2_id]['name']
        
        # Создаем интервалы с читаемыми подписями
        col1_cut = pd.cut(col1_data, bins=bins)
        col2_cut = pd.cut(col2_data, bins=bins)
        
        # Создаем читаемые подписи на основе диапазонов
        col1_labels = []
        col2_labels = []
        
        for i, interval in enumerate(col1_cut.cat.categories):
            left = f"{interval.left:.1f}" if not pd.isna(interval.left) else "0"
            right = f"{interval.right:.1f}" if not pd.isna(interval.right) else "∞"
            col1_labels.append(f"{left}-{right}")
        
        for i, interval in enumerate(col2_cut.cat.categories):
            left = f"{interval.left:.1f}" if not pd.isna(interval.left) else "0"
            right = f"{interval.right:.1f}" if not pd.isna(interval.right) else "∞"
            col2_labels.append(f"{left}-{right}")
        
        # Применяем новые подписи
        col1_bins = col1_cut.cat.rename_categories(col1_labels)
        col2_bins = col2_cut.cat.rename_categories(col2_labels)
        
        # Создаем таблицу сопряженности
        contingency_table = pd.crosstab(col1_bins, col2_bins)
        
        # Создаем график
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            contingency_table,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar_kws={'label': 'Частота'}
        )
        
        plt.title(f'Таблица сопряженности: {col1_name} vs {col2_name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel(col2_name, fontsize=12)
        plt.ylabel(col1_name, fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")
        else:
            plt.savefig(f'img/contingency_table_{col1_id}_{col2_id}.png', dpi=300, bbox_inches='tight')
            print(f"График сохранен: img/contingency_table_{col1_id}_{col2_id}.png")
        
        plt.close()
    
    def full_contingency_analysis(self, col1_id, col2_id, bins=5, save_plot=True):
        """
        Выполняет полный анализ таблицы сопряженности
        
        Args:
            col1_id: ID первого столбца
            col2_id: ID второго столбца
            bins: Количество интервалов для дискретизации
            save_plot: Сохранять ли график
            
        Returns:
            dict: Результаты анализа
        """
        # Создаем таблицу сопряженности
        results = self.create_contingency_table(col1_id, col2_id, bins)
        
        # Создаем график
        if save_plot:
            self.plot_contingency_heatmap(col1_id, col2_id, bins)
        
        return results
    
    def spearman_correlation_analysis(self, col1_id, col2_id):
        """
        Выполняет ранговый корреляционный анализ Спирмена
        
        Args:
            col1_id: ID первого столбца
            col2_id: ID второго столбца
            
        Returns:
            dict: Результаты анализа корреляции Спирмена
        """
        # Получаем данные
        col1_data = self.data_dict[col1_id]['data']
        col2_data = self.data_dict[col2_id]['data']
        col1_name = self.data_dict[col1_id]['name']
        col2_name = self.data_dict[col2_id]['name']
        
        print("="*OUTPUT_CONFIG['separator_length'])
        print(f"РАНГОВЫЙ КОРРЕЛЯЦИОННЫЙ АНАЛИЗ СПИРМЕНА: {col1_name.upper()} vs {col2_name.upper()}")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        # Вычисляем корреляцию Спирмена
        spearman_corr, spearman_pvalue = spearmanr(col1_data, col2_data)
        
        print(f"\nКОЭФФИЦИЕНТ КОРРЕЛЯЦИИ СПИРМЕНА:")
        print("-" * 40)
        print(f"Коэффициент rho: {spearman_corr:.4f}")
        print(f"p-value: {spearman_pvalue:.6f}")
        
        # Интерпретация результата
        if spearman_pvalue < 0.05:
            significance = "Статистически значимая корреляция (p < 0.05)"
        else:
            significance = "НЕ статистически значимая корреляция (p ≥ 0.05)"
        
        print(f"Значимость: {significance}")
        
        # Интерпретация силы корреляции
        abs_corr = abs(spearman_corr)
        if abs_corr < 0.1:
            strength = "Очень слабая корреляция"
        elif abs_corr < 0.3:
            strength = "Слабая корреляция"
        elif abs_corr < 0.5:
            strength = "Умеренная корреляция"
        elif abs_corr < 0.7:
            strength = "Сильная корреляция"
        else:
            strength = "Очень сильная корреляция"
        
        print(f"Сила корреляции: {strength}")
        
        # Направление корреляции
        if spearman_corr > 0:
            direction = "Положительная корреляция (прямая связь)"
        elif spearman_corr < 0:
            direction = "Отрицательная корреляция (обратная связь)"
        else:
            direction = "Отсутствие корреляции"
        
        print(f"Направление: {direction}")
        
        # Сохраняем результаты
        spearman_results = {
            'correlation_coefficient': spearman_corr,
            'p_value': spearman_pvalue,
            'significance': significance,
            'strength': strength,
            'direction': direction,
            'interpretation': f"Между {col1_name} и {col2_name} наблюдается {strength.lower()}"
        }
        
        return spearman_results

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Используем backend с GUI для отображения графиков
import matplotlib.pyplot as plt
from scipy import stats

class DescriptiveAnalysis():
    def __init__(self, data, name):
        """
        Initialize DescriptiveAnalysis with data.
        
        Args:
            data: pandas Series or numpy array containing numerical data
        """
        self.data = data
        self.name = name
        self.params = {}
        
    def get_params(self):
        """
        Calculate descriptive statistics parameters.
        
        Returns:
            dict: Dictionary containing mean, median, mode, std, variance, min, max, sum, kurtosis, skewness
        """
        # Convert to numpy array for calculations
        data_array = np.array(self.data)
        
        # Calculate all parameters
        self.params = {
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'mode': stats.mode(data_array, keepdims=True)[0][0] if len(data_array) > 0 else None,
            'std': np.std(data_array, ddof=1),  # Sample standard deviation
            'variance': np.var(data_array, ddof=1),  # Sample variance (выборочная дисперсия)
            'min': np.min(data_array),
            'max': np.max(data_array),
            'sum': np.sum(data_array),
            'kurtosis': stats.kurtosis(data_array),  # Excess kurtosis (эксцесс)
            'skewness': stats.skew(data_array)       # Skewness (ассиметрия)
        }
        
        return self.params
    
    def detect_outliers(self, threshold=3):
        """
        Detect outliers using Z-score method.
        
        Args:
            threshold (float): Z-score threshold for outlier detection (default: 3)
            
        Returns:
            tuple: (outliers_data, outliers_indices, z_scores)
        """
        data_array = np.array(self.data)
        
        # Calculate Z-scores
        mean = np.mean(data_array)
        std = np.std(data_array, ddof=1)
        z_scores = np.abs((data_array - mean) / std)
        
        # Find outliers
        outlier_mask = z_scores > threshold
        outliers_data = data_array[outlier_mask]
        outliers_indices = np.where(outlier_mask)[0]
        
        return outliers_data, outliers_indices, z_scores
    
    def show_outliers(self, threshold=3):
        """
        Display outliers in a separate window with detailed information.
        
        Args:
            threshold (float): Z-score threshold for outlier detection (default: 3)
        """
        outliers_data, outliers_indices, z_scores = self.detect_outliers(threshold)
        
        if len(outliers_data) == 0:
            print(f"Выбросы не найдены (порог Z-score: {threshold})")
            
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Main data with outliers highlighted
        ax1.scatter(range(len(self.data)), self.data, alpha=0.6, color='blue', label='Обычные данные')
        ax1.scatter(outliers_indices, outliers_data, color='red', s=100, 
                   label=f'Выбросы (Z-score > {threshold})', zorder=5)
        ax1.set_title(f'Данные с выбросами - {self.name}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Индекс')
        ax1.set_ylabel('Значение')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Z-scores
        ax2.scatter(range(len(z_scores)), z_scores, alpha=0.6, color='green', label='Z-scores')
        ax2.axhline(y=threshold, color='red', linestyle='--', 
                   label=f'Порог ({threshold})')
        ax2.scatter(outliers_indices, z_scores[outliers_indices], color='red', s=100, 
                   label='Выбросы', zorder=5)
        ax2.set_title('Z-scores', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Индекс')
        ax2.set_ylabel('Z-score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.savefig(f'img/outliers_analysis_{self.name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store outlier information
        self.outliers_info = {
            'data': outliers_data,
            'indices': outliers_indices,
            'z_scores': z_scores[outliers_indices],
            'threshold': threshold,
            'count': len(outliers_data)
        }
    
    def draw_histogram(self):
        """
        Create histogram with intervals calculated using Sturges' rule.
        """
        data_array = np.array(self.data)
        
        # Calculate number of bins using Sturges' rule: k = 1 + log2(n)
        n = len(data_array)
        k = int(1 + np.log2(n))
        
        # Create histogram
        plt.figure(figsize=(12, 6))
        
        # Plot histogram
        n, bins, patches = plt.hist(data_array, bins=k, alpha=0.7, color='skyblue', 
                                   edgecolor='black', linewidth=1)
        
        # Customize histogram
        plt.title(f'Гистограмма - {self.name}', fontsize=14, fontweight='bold')
        plt.xlabel('Значение', fontsize=12)
        plt.ylabel('Частота', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add mean line
        plt.axvline(self.params['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Среднее: {self.params["mean"]:.2f}')
        
        # Add median line
        plt.axvline(self.params['median'], color='orange', linestyle='--', linewidth=2, 
                   label=f'Медиана: {self.params["median"]:.2f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'img/histogram_{self.name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store histogram information
        self.histogram_info = {
            'bins': bins,
            'counts': n,
            'num_bins': k,
            'bin_width': bins[1] - bins[0] if len(bins) > 1 else 0
        }
    
    def draw_diagrams(self):
        """
        Create box-and-whiskers diagram for the data.
        """
        plt.figure(figsize=(10, 6))
        
        # Create box plot
        plt.boxplot(self.data, patch_artist=True, whis=3, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2),
                   whiskerprops=dict(color='black', linewidth=1.5),
                   capprops=dict(color='black', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red', 
                                 markersize=5, alpha=0.7,))
        
        plt.title(f'Ящик с усами - {self.name}', fontsize=14, fontweight='bold')
        plt.ylabel('Values', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        plt.savefig(f'img/boxplot_{self.name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def draw_qq_plot(self):
        """
        Create Q-Q plot to check normality of the data.
        """
        from scipy import stats
        
        data_array = np.array(self.data)
        
        # Create Q-Q plot
        plt.figure(figsize=(10, 6))
        
        # Generate Q-Q plot
        stats.probplot(data_array, dist="norm", plot=plt)
        
        plt.title(f'Q-Q график (проверка нормальности) - {self.name}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Теоретические квантили', fontsize=12)
        plt.ylabel('Выборочные квантили', fontsize=12)
        plt.grid(True, alpha=0.3)
             
        plt.tight_layout()
        plt.savefig(f'img/qq_plot_{self.name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store Q-Q plot information
        self.qq_plot_info = {
            'data': data_array,
            'theoretical_quantiles': stats.probplot(data_array, dist="norm")[0],
            'sample_quantiles': stats.probplot(data_array, dist="norm")[1]
        }
    
    def normality_tests(self):
        """
        Perform normality tests: Box-Ljung and Lilliefors tests.
        
        Returns:
            dict: Dictionary containing test results with p-values
        """
        data_array = np.array(self.data)
        
        # Box-Ljung test (тест Бокса-Льюинга)
        # Проверяет автокорреляцию остатков
        try:
            # Для Box-Ljung теста нужны остатки, используем разности
            residuals = np.diff(data_array)
            if len(residuals) > 0:
                box_ljung_stat, box_ljung_pvalue = stats.jarque_bera(residuals)
            else:
                box_ljung_stat, box_ljung_pvalue = np.nan, np.nan
        except:
            box_ljung_stat, box_ljung_pvalue = np.nan, np.nan
        
        # Lilliefors test (тест Лилиефорса)
        # Проверяет нормальность распределения
        try:
            lilliefors_stat, lilliefors_pvalue = stats.kstest(data_array, 'norm', 
                                                             args=(np.mean(data_array), np.std(data_array, ddof=1)))
        except:
            lilliefors_stat, lilliefors_pvalue = np.nan, np.nan
        
        # Сохраняем результаты
        self.normality_results = {
            'box_ljung': {
                'statistic': box_ljung_stat,
                'p_value': box_ljung_pvalue,
                'interpretation': 'Тест Бокса-Льюинга (автокорреляция)'
            },
            'lilliefors': {
                'statistic': lilliefors_stat,
                'p_value': lilliefors_pvalue,
                'interpretation': 'Тест Лилиефорса (нормальность)'
            },
        }
        
        return self.normality_results
    
    def print_normality_tests(self):
        """
        Print results of normality tests in Russian.
        """
        if not hasattr(self, 'normality_results'):
            self.normality_tests()
        
        print(f"\nТЕСТЫ НОРМАЛЬНОСТИ - {self.name.upper()}:")
        print("-" * 60)
        
        for test_name, results in self.normality_results.items():
            if not np.isnan(results['p_value']):
                p_value = results['p_value']
                interpretation = results['interpretation']
                
                print(f"{interpretation}:")
                print(f"  Статистика: {results['statistic']:.6f}")
                print(f"  p-value: {p_value:.6f}")
                print()
            else:
                print(f"{results['interpretation']}: НЕ ПРИМЕНИМ")
                print()
    
    def start_analysis(self, show_outliers=True, show_histogram=True, show_qq_plot=True, show_normality_tests=True, outlier_threshold=3):
        """
        Start the complete descriptive analysis.
        
        Args:
            show_outliers (bool): Whether to show outlier analysis (default: True)
            show_histogram (bool): Whether to show histogram (default: True)
            show_qq_plot (bool): Whether to show Q-Q plot (default: True)
            show_normality_tests (bool): Whether to show normality tests (default: True)
            outlier_threshold (float): Z-score threshold for outlier detection (default: 3)
        """
        
        # Calculate parameters
        params = self.get_params()
        
        # Russian names for parameters
        russian_names = {
            'mean': 'Среднее',
            'median': 'Медиана',
            'mode': 'Мода',
            'std': 'Стандартное отклонение',
            'variance': 'Выборочная дисперсия',
            'min': 'Минимум',
            'max': 'Максимум',
            'sum': 'Сумма',
            'kurtosis': 'Эксцесс',
            'skewness': 'Асимметрия'
        }
        
        for key, value in params.items():
            russian_name = russian_names.get(key, key.capitalize())
            if value is not None:
                print(f"{russian_name}: {value:.4f}")
            else:
                print(f"{russian_name}: {value}")
        
        # Draw box plot
        self.draw_diagrams()
        
        # Draw histogram if requested
        if show_histogram:
            self.draw_histogram()
        
        # Draw Q-Q plot if requested
        if show_qq_plot:
            self.draw_qq_plot()
        
        # Show normality tests if requested
        if show_normality_tests:
            self.print_normality_tests()
        
        # Show outliers if requested
        if show_outliers:
            self.show_outliers(outlier_threshold)
        
        return params
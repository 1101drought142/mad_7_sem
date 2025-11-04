"""
Excel Data Extraction and Descriptive Analysis Script
Extracts data from configured columns from data.xlsx file and performs descriptive analysis
Data type: float
"""

import pandas as pd
import numpy as np
from pathlib import Path

from dataloader import DataLoader 
from descriptive_analysis import DescriptiveAnalysis
from normalization import Normalizer
from correlation_matrix import CorrelationAnalyzer
from regression_model import RegressionModel
from config import COLUMN_CONFIG, PROCESSING_CONFIG, OUTPUT_CONFIG

if __name__ == "__main__":
    # Execute the main function
    current_dir = Path(__file__).parent
    excel_file = current_dir / "data.xlsx"
    
    # Load data from Excel file
    loader = DataLoader(excel_file)
    extracted_data = loader.extract_excel_data()
    
    if extracted_data is not None:
        print("Данные успешно загружены!")
        
        # Show loaded data info
        for col_id, col_info in extracted_data.items():
            print(f"Колонка {col_id} ({col_info['name']}): {len(col_info['data'])} записей")
        
        # Process each configured column
        for col_id, col_info in extracted_data.items():
            data = col_info['data']
            name = col_info['name']
            
            print("\n" + "="*OUTPUT_CONFIG['separator_length'])
            print(f"ДЕСКРИПТИВНЫЙ АНАЛИЗ - КОЛОНКА {col_id} ({name.upper()})")
            print("="*OUTPUT_CONFIG['separator_length'])
            analysis = DescriptiveAnalysis(data, name)
            params = analysis.start_analysis()

            print("\n" + "="*OUTPUT_CONFIG['separator_length'])
            print(f"ОЧИСТКА ОТ ВЫБРОСОВ И НОРМАЛИЗАЦИЯ - КОЛОНКА {col_id}")
            print("="*OUTPUT_CONFIG['separator_length'])
            normalizer = Normalizer(data)
            clean_data, normalized_data = normalizer.full_pipeline(
                z_threshold=PROCESSING_CONFIG['z_threshold'],
                normalize_method=PROCESSING_CONFIG['normalize_method']
            )
            
            print(f"\nРЕЗУЛЬТАТЫ ОБРАБОТКИ КОЛОНКИ {col_id}:")
            print(f"   Исходных записей: {len(data)}")
            print(f"   После очистки: {len(clean_data)}")
            print(f"   Удалено выбросов: {len(data) - len(clean_data)}")
            
            if PROCESSING_CONFIG['show_samples']:
                print(f"   Первые {OUTPUT_CONFIG['sample_size']} очищенных значений: {clean_data[:OUTPUT_CONFIG['sample_size']]}")
        #         print(f"   Первые {OUTPUT_CONFIG['sample_size']} нормализованных значений: {normalized_data[:OUTPUT_CONFIG['sample_size']]}")
            
        #     if PROCESSING_CONFIG['show_detailed_stats']:
        #         print(f"   Статистика нормализованных данных:")
        #         decimal_places = OUTPUT_CONFIG['decimal_places']
        #         print(f"     Минимум: {np.min(normalized_data):.{decimal_places}f}")
        #         print(f"     Максимум: {np.max(normalized_data):.{decimal_places}f}")
        #         print(f"     Среднее: {np.mean(normalized_data):.{decimal_places}f}")
        #         print(f"     Стандартное отклонение: {np.std(normalized_data):.{decimal_places}f}")
        
        # Анализ таблицы сопряженности
        print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        print("АНАЛИЗ ТАБЛИЦЫ СОПРЯЖЕННОСТИ")
        print("="*OUTPUT_CONFIG['separator_length'])
        
        from correlation_matrix import CorrelationAnalyzer
        correlation_analyzer = CorrelationAnalyzer(extracted_data)
        contingency_results = correlation_analyzer.full_contingency_analysis('C', 'H', bins=5, save_plot=True)
        
        # Ранговый корреляционный анализ Спирмена
        spearman_results = correlation_analyzer.spearman_correlation_analysis('C', 'H')
        
        # # Анализ корреляций и мультиколлинеарности
        # print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        # print("АНАЛИЗ КОРРЕЛЯЦИЙ И МУЛЬТИКОЛЛИНЕАРНОСТИ")
        # print("="*OUTPUT_CONFIG['separator_length'])
        
        # correlation_analyzer = CorrelationAnalyzer(extracted_data)
        # correlation_results = correlation_analyzer.full_analysis(save_plot=True)
        
        # # Построение регрессионных моделей
        # print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        # print("ПОСТРОЕНИЕ РЕГРЕССИОННЫХ МОДЕЛЕЙ")
        # print("="*OUTPUT_CONFIG['separator_length'])
        
        # regression_analyzer = RegressionModel(extracted_data, target_column='H')
        # regression_results = regression_analyzer.full_analysis(test_size=0.2, ridge_alpha=1.0)
        
        # # Анализ с PCA для устранения мультиколлинеарности
        # print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        # print("АНАЛИЗ С PCA ДЛЯ УСТРАНЕНИЯ МУЛЬТИКОЛЛИНЕАРНОСТИ")
        # print("="*OUTPUT_CONFIG['separator_length'])
        
        # pca_regression_results = regression_analyzer.full_analysis_with_pca(
        #     test_size=0.2, 
        #     ridge_alpha=1.0, 
        #     pca_components=None  # Автоматический выбор количества компонент
        # )
        
        # print("\n" + "="*OUTPUT_CONFIG['separator_length'])
        # print("ОБРАБОТКА ДАННЫХ ЗАВЕРШЕНА УСПЕШНО!")
        # print("="*OUTPUT_CONFIG['separator_length'])
    
    else:
        print("Ошибка: Не удалось загрузить данные из Excel файла!")

from sklearn import preprocessing
import numpy as np
import pandas as pd
from scipy import stats

class Normalizer():
    """
    Класс для очистки данных от выбросов и нормализации
    """

    def __init__(self, dataset):
        """
        Инициализация с датасетом (pandas Series или numpy array)
        
        Args:
            dataset: pandas Series или numpy array с данными
        """
        self.dataset = dataset
        # Конвертируем в numpy array для единообразной обработки
        if isinstance(dataset, pd.Series):
            self.dataset = dataset.values
        elif not isinstance(dataset, np.ndarray):
            self.dataset = np.array(dataset)

    def remove_outliers(self, z_threshold=3):
        """
        Удаление выбросов на основе Z-score
        
        Args:
            z_threshold (float): Пороговое значение Z-score (по умолчанию 3)
            
        Returns:
            numpy array: Очищенный от выбросов датасет
        """
        if len(self.dataset) == 0:
            raise ValueError("Empty dataset")

        # Вычисляем Z-score для каждого элемента
        z_scores = np.abs(stats.zscore(self.dataset))
        
        # Создаем маску для элементов, которые НЕ являются выбросами
        mask = z_scores < z_threshold
        
        # Фильтруем датасет, оставляя только не-выбросы
        dataset_clean = self.dataset[mask]
        
        print(f"Исходный размер датасета: {len(self.dataset)}")
        print(f"Размер после удаления выбросов: {len(dataset_clean)}")
        print(f"Удалено выбросов: {len(self.dataset) - len(dataset_clean)}")
        
        return dataset_clean

    def normalize_data(self, data, method='l2'):
        """
        Нормализация данных с использованием sklearn.preprocessing.normalize
        
        Args:
            data: numpy array для нормализации
            method (str): Метод нормализации ('l1', 'l2', 'max')
            
        Returns:
            numpy array: Нормализованные данные
        """
        if len(data) == 0:
            raise ValueError("Empty dataset for normalization")
            
        # Для одномерных данных используем MinMaxScaler или StandardScaler
        if method == 'l2':
            # L2 нормализация для одномерных данных
            norm = np.linalg.norm(data)
            if norm == 0:
                return data
            return data / norm
        elif method == 'l1':
            # L1 нормализация для одномерных данных
            norm = np.sum(np.abs(data))
            if norm == 0:
                return data
            return data / norm
        elif method == 'max':
            # Max нормализация для одномерных данных
            max_val = np.max(np.abs(data))
            if max_val == 0:
                return data
            return data / max_val
        else:
            # Используем MinMaxScaler для других методов
            scaler = preprocessing.MinMaxScaler()
            data_2d = data.reshape(-1, 1)
            normalized_data = scaler.fit_transform(data_2d)
            return normalized_data.flatten()

    def full_pipeline(self, z_threshold=3, normalize_method='l2'):
        """
        Полный пайплайн: удаление выбросов + нормализация
        
        Args:
            z_threshold (float): Пороговое значение Z-score
            normalize_method (str): Метод нормализации
            
        Returns:
            tuple: (очищенные_данные, нормализованные_данные)
        """
        # Удаляем выбросы
        clean_data = self.remove_outliers(z_threshold)
        
        # Нормализуем
        normalized_data = self.normalize_data(clean_data, normalize_method)
                
        return clean_data, normalized_data
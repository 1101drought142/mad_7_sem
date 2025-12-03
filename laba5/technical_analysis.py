"""
Лабораторная работа 5. Технический анализ. Паттерны
Алгоритм для определения тренда, линий поддержки/сопротивления,
паттернов и индикаторов технического анализа.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Предупреждение: statsmodels не установлен. Некоторые функции анализа временных рядов будут недоступны.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Предупреждение: seaborn не установлен. Календарная тепловая карта будет упрощенной.")


class TechnicalAnalyzer:
    """Класс для технического анализа финансовых данных"""
    
    def __init__(self, data_path='data.csv'):
        """
        Инициализация анализатора
        
        Args:
            data_path: путь к файлу с данными
        """
        self.data = self.load_data(data_path)
        self.prepare_data()
        
    def load_data(self, path):
        """Загрузка и предобработка данных"""
        # Чтение CSV с учетом пробелов в заголовках
        df = pd.read_csv(path, skipinitialspace=True)
        
        # Очистка названий колонок от пробелов
        df.columns = df.columns.str.strip()
        
        # Очистка данных от символов $ и преобразование в числовой формат
        for col in ['Close/Last', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '').astype(float)
        
        # Преобразование даты
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Переименование колонок для удобства
        df.columns = ['Date', 'Close', 'Volume', 'Open', 'High', 'Low']
        
        return df
    
    def prepare_data(self):
        """Подготовка данных для анализа"""
        # Вычисление дополнительных метрик
        self.data['HL_PCT'] = (self.data['High'] - self.data['Low']) / self.data['Close'] * 100
        self.data['PCT_change'] = (self.data['Close'] - self.data['Open']) / self.data['Open'] * 100
        
    def calculate_ma(self, period=20):
        """
        Вычисление скользящей средней (Moving Average)
        
        Args:
            period: период для расчета MA
        """
        self.data[f'MA_{period}'] = self.data['Close'].rolling(window=period).mean()
        return self.data[f'MA_{period}']
    
    def calculate_rsi(self, period=14):
        """
        Вычисление индикатора RSI (Relative Strength Index)
        
        Args:
            period: период для расчета RSI
        """
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        self.data['RSI'] = rsi
        return rsi
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """
        Вычисление индикатора MACD (Moving Average Convergence Divergence)
        
        Args:
            fast: период быстрой EMA
            slow: период медленной EMA
            signal: период сигнальной линии
        """
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        self.data['MACD'] = macd_line
        self.data['MACD_signal'] = signal_line
        self.data['MACD_hist'] = histogram
        
        return macd_line, signal_line, histogram
    
    def find_support_resistance(self, window=20, threshold=0.02):
        """
        Поиск линий поддержки и сопротивления
        
        Args:
            window: размер окна для поиска локальных экстремумов
            threshold: порог для определения значимости уровня
        """
        support_levels = []
        resistance_levels = []
        
        # Поиск локальных минимумов (поддержка)
        for i in range(window, len(self.data) - window):
            if self.data['Low'].iloc[i] == self.data['Low'].iloc[i-window:i+window+1].min():
                support_levels.append((i, self.data['Low'].iloc[i]))
        
        # Поиск локальных максимумов (сопротивление)
        for i in range(window, len(self.data) - window):
            if self.data['High'].iloc[i] == self.data['High'].iloc[i-window:i+window+1].max():
                resistance_levels.append((i, self.data['High'].iloc[i]))
        
        # Группировка близких уровней
        support = self._group_levels(support_levels, threshold)
        resistance = self._group_levels(resistance_levels, threshold)
        
        # Выбор наиболее значимых уровней
        support = sorted(support, key=lambda x: x[1])[:3]  # Топ-3 поддержки
        resistance = sorted(resistance, key=lambda x: x[1], reverse=True)[:3]  # Топ-3 сопротивления
        
        return support, resistance
    
    def _group_levels(self, levels, threshold):
        """Группировка близких уровней"""
        if not levels:
            return []
        
        grouped = []
        levels_sorted = sorted(levels, key=lambda x: x[1])
        
        current_group = [levels_sorted[0]]
        for level in levels_sorted[1:]:
            if abs(level[1] - current_group[-1][1]) / current_group[-1][1] < threshold:
                current_group.append(level)
            else:
                # Сохраняем средний уровень группы
                avg_price = np.mean([l[1] for l in current_group])
                avg_idx = int(np.mean([l[0] for l in current_group]))
                grouped.append((avg_idx, avg_price))
                current_group = [level]
        
        if current_group:
            avg_price = np.mean([l[1] for l in current_group])
            avg_idx = int(np.mean([l[0] for l in current_group]))
            grouped.append((avg_idx, avg_price))
        
        return grouped
    
    def detect_double_bottom(self, lookback=50, min_distance=20, tolerance=0.05):
        """
        Обнаружение паттерна "Двойное дно"
        
        Args:
            lookback: период поиска назад
            min_distance: минимальное расстояние между двумя днами
            tolerance: допустимое отклонение цен двух доньев
        """
        patterns = []
        
        for i in range(lookback, len(self.data) - 10):
            # Поиск локального минимума
            window_data = self.data.iloc[i-lookback:i]
            min_idx = window_data['Low'].idxmin()
            min_price = self.data.loc[min_idx, 'Low']
            min_pos = self.data.index.get_loc(min_idx)
            
            # Поиск второго минимума после первого
            if min_pos + min_distance < len(self.data):
                future_window = self.data.iloc[min_pos+min_distance:min_pos+lookback]
                if len(future_window) > 0:
                    min_idx2 = future_window['Low'].idxmin()
                    min_price2 = self.data.loc[min_idx2, 'Low']
                    min_pos2 = self.data.index.get_loc(min_idx2)
                    
                    # Проверка условий двойного дна
                    price_diff = abs(min_price - min_price2) / min_price
                    if price_diff <= tolerance and min_pos2 - min_pos >= min_distance:
                        # Проверка наличия промежуточного пика
                        between_data = self.data.iloc[min_pos:min_pos2]
                        if len(between_data) > 0:
                            peak_price = between_data['High'].max()
                            peak_above = (peak_price - min_price) / min_price > 0.03  # Пик должен быть выше на 3%
                            
                            if peak_above:
                                patterns.append({
                                    'type': 'double_bottom',
                                    'first_bottom_idx': min_pos,
                                    'second_bottom_idx': min_pos2,
                                    'first_bottom_price': min_price,
                                    'second_bottom_price': min_price2,
                                    'peak_price': peak_price,
                                    'peak_idx': between_data['High'].idxmax()
                                })
        
        return patterns
    
    def determine_trend(self, short_period=20, long_period=50):
        """
        Определение тренда на основе скользящих средних
        
        Returns:
            'uptrend', 'downtrend' или 'sideways'
        """
        ma_short = self.calculate_ma(short_period)
        ma_long = self.calculate_ma(long_period)
        
        # Определение тренда для последних значений
        recent_short = ma_short.iloc[-10:].mean()
        recent_long = ma_long.iloc[-10:].mean()
        
        if recent_short > recent_long * 1.02:
            return 'uptrend'
        elif recent_short < recent_long * 0.98:
            return 'downtrend'
        else:
            return 'sideways'
    
    def plot_candlestick_chart(self):
        """Построение графика японских свечей с анализом"""
        # Подготовка данных для mplfinance
        df_plot = self.data.set_index('Date')
        df_plot = df_plot[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Вычисление индикаторов
        ma_20 = self.calculate_ma(20)
        ma_50 = self.calculate_ma(50)
        rsi = self.calculate_rsi(14)
        macd, macd_signal, macd_hist = self.calculate_macd()
        
        # Поиск уровней поддержки и сопротивления
        support, resistance = self.find_support_resistance()
        
        # Поиск паттернов
        patterns = self.detect_double_bottom()
        
        # Определение тренда
        trend = self.determine_trend()
        
        # Создание дополнительных графиков
        apds = []
        
        # Добавление скользящих средних
        apds.append(mpf.make_addplot(ma_20.values, color='blue', width=1, alpha=0.7, label='MA20'))
        apds.append(mpf.make_addplot(ma_50.values, color='red', width=1, alpha=0.7, label='MA50'))
        
        # Построение основного графика свечей
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style='yahoo',
            volume=True,
            addplot=apds,
            returnfig=True,
            figsize=(16, 10),
            title=f'Технический анализ. Тренд: {trend}'
        )
        
        ax_main = axes[0]
        ax_volume = axes[2] if len(axes) > 2 else None
        
        # Добавление линий поддержки и сопротивления
        for idx, price in support:
            if idx < len(self.data):
                date = self.data.iloc[idx]['Date']
                ax_main.axhline(y=price, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label='Поддержка' if idx == support[0][0] else '')
        
        for idx, price in resistance:
            if idx < len(self.data):
                date = self.data.iloc[idx]['Date']
                ax_main.axhline(y=price, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Сопротивление' if idx == resistance[0][0] else '')
        
        # Выделение паттернов
        for pattern in patterns[:3]:  # Показываем первые 3 паттерна
            first_idx = pattern['first_bottom_idx']
            second_idx = pattern['second_bottom_idx']
            peak_idx = pattern['peak_idx']
            
            if first_idx < len(self.data) and second_idx < len(self.data):
                first_date = self.data.iloc[first_idx]['Date']
                second_date = self.data.iloc[second_idx]['Date']
                
                # Выделение точек двойного дна
                ax_main.scatter([first_date, second_date], 
                              [pattern['first_bottom_price'], pattern['second_bottom_price']],
                              color='blue', s=100, marker='v', zorder=5, label='Двойное дно' if pattern == patterns[0] else '')
        
        ax_main.legend(loc='upper left')
        ax_main.set_ylabel('Цена ($)')
        
        plt.tight_layout()
        plt.savefig('photos/candlestick_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_indicators(self):
        """Построение графиков индикаторов"""
        rsi = self.calculate_rsi(14)
        macd, macd_signal, macd_hist = self.calculate_macd()
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # График цены с MA
        ax1 = axes[0]
        ax1.plot(self.data['Date'], self.data['Close'], label='Цена закрытия', linewidth=1.5)
        ma_20 = self.calculate_ma(20)
        ma_50 = self.calculate_ma(50)
        ax1.plot(self.data['Date'], ma_20, label='MA(20)', linewidth=1.5, alpha=0.7)
        ax1.plot(self.data['Date'], ma_50, label='MA(50)', linewidth=1.5, alpha=0.7)
        ax1.set_ylabel('Цена ($)')
        ax1.set_title('Цена и скользящие средние')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График RSI
        ax2 = axes[1]
        ax2.plot(self.data['Date'], rsi, label='RSI(14)', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Перекупленность (70)')
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Перепроданность (30)')
        ax2.fill_between(self.data['Date'], 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI')
        ax2.set_title('Индикатор RSI (Relative Strength Index)')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График MACD
        ax3 = axes[2]
        ax3.plot(self.data['Date'], macd, label='MACD', color='blue', linewidth=1.5)
        ax3.plot(self.data['Date'], macd_signal, label='Сигнальная линия', color='red', linewidth=1.5)
        ax3.bar(self.data['Date'], macd_hist, label='Гистограмма', alpha=0.3, color='gray')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_ylabel('MACD')
        ax3.set_xlabel('Дата')
        ax3.set_title('Индикатор MACD (Moving Average Convergence Divergence)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Форматирование дат для более компактного отображения
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/indicators.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_support_resistance(self):
        """Детальный график с линиями поддержки и сопротивления"""
        support, resistance = self.find_support_resistance()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # График цены
        ax.plot(self.data['Date'], self.data['Close'], label='Цена закрытия', linewidth=2, color='black')
        ax.fill_between(self.data['Date'], self.data['Low'], self.data['High'], 
                       alpha=0.2, color='gray', label='Диапазон High-Low')
        
        # Линии поддержки
        for idx, price in support:
            if idx < len(self.data):
                ax.axhline(y=price, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                          label=f'Поддержка ${price:.2f}' if idx == support[0][0] else '')
                # Аннотация
                date = self.data.iloc[idx]['Date']
                ax.annotate(f'${price:.2f}', xy=(date, price), xytext=(10, 10),
                           textcoords='offset points', fontsize=9, color='green',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Линии сопротивления
        for idx, price in resistance:
            if idx < len(self.data):
                ax.axhline(y=price, color='red', linestyle='--', linewidth=2, alpha=0.7,
                          label=f'Сопротивление ${price:.2f}' if idx == resistance[0][0] else '')
                # Аннотация
                date = self.data.iloc[idx]['Date']
                ax.annotate(f'${price:.2f}', xy=(date, price), xytext=(10, -20),
                           textcoords='offset points', fontsize=9, color='red',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена ($)')
        ax.set_title('Линии поддержки и сопротивления')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат для более компактного отображения
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/support_resistance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_patterns(self):
        """График с обнаруженными паттернами"""
        patterns = self.detect_double_bottom()
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # График цены
        ax.plot(self.data['Date'], self.data['Close'], label='Цена закрытия', linewidth=2, color='black')
        
        # Выделение паттернов
        for i, pattern in enumerate(patterns[:5]):  # Показываем первые 5 паттернов
            first_idx = pattern['first_bottom_idx']
            second_idx = pattern['second_bottom_idx']
            peak_idx = pattern['peak_idx']
            
            if first_idx < len(self.data) and second_idx < len(self.data):
                first_date = self.data.iloc[first_idx]['Date']
                second_date = self.data.iloc[second_idx]['Date']
                peak_date = self.data.iloc[peak_idx]['Date'] if peak_idx < len(self.data) else None
                
                # Выделение точек двойного дна
                ax.scatter([first_date, second_date], 
                          [pattern['first_bottom_price'], pattern['second_bottom_price']],
                          color='blue', s=150, marker='v', zorder=5, 
                          label='Двойное дно' if i == 0 else '')
                
                # Соединение точек
                ax.plot([first_date, second_date], 
                       [pattern['first_bottom_price'], pattern['second_bottom_price']],
                       color='blue', linestyle=':', linewidth=2, alpha=0.5)
                
                # Выделение пика
                if peak_date:
                    ax.scatter([peak_date], [pattern['peak_price']],
                              color='orange', s=150, marker='^', zorder=5)
                
                # Аннотация
                mid_date = self.data.iloc[(first_idx + second_idx) // 2]['Date']
                ax.annotate(f'Двойное дно #{i+1}', 
                           xy=(mid_date, (pattern['first_bottom_price'] + pattern['second_bottom_price']) / 2),
                           xytext=(0, -30), textcoords='offset points',
                           fontsize=10, color='blue',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7))
        
        ax.set_xlabel('Дата')
        ax.set_ylabel('Цена ($)')
        ax.set_title('Обнаруженные паттерны: Двойное дно')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат для более компактного отображения
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def generate_report(self):
        """Генерация текстового отчета с результатами анализа"""
        trend = self.determine_trend()
        support, resistance = self.find_support_resistance()
        patterns = self.detect_double_bottom()
        rsi = self.calculate_rsi(14)
        macd, macd_signal, macd_hist = self.calculate_macd()
        
        report = []
        report.append("=" * 80)
        report.append("ОТЧЕТ ПО ТЕХНИЧЕСКОМУ АНАЛИЗУ")
        report.append("=" * 80)
        report.append("")
        
        report.append("1. ОПИСАНИЕ ВРЕМЕННОГО РЯДА")
        report.append("-" * 80)
        report.append(f"Период данных: {self.data['Date'].min().strftime('%Y-%m-%d')} - {self.data['Date'].max().strftime('%Y-%m-%d')}")
        report.append(f"Количество точек: {len(self.data)}")
        report.append(f"Минимальная цена: ${self.data['Low'].min():.2f}")
        report.append(f"Максимальная цена: ${self.data['High'].max():.2f}")
        report.append(f"Средняя цена закрытия: ${self.data['Close'].mean():.2f}")
        report.append(f"Волатильность (std): ${self.data['Close'].std():.2f}")
        report.append("")
        
        report.append("2. ОПРЕДЕЛЕНИЕ ТРЕНДА")
        report.append("-" * 80)
        report.append(f"Текущий тренд: {trend.upper()}")
        ma_20 = self.calculate_ma(20)
        ma_50 = self.calculate_ma(50)
        report.append(f"MA(20): ${ma_20.iloc[-1]:.2f}")
        report.append(f"MA(50): ${ma_50.iloc[-1]:.2f}")
        if ma_20.iloc[-1] > ma_50.iloc[-1]:
            report.append("Интерпретация: Бычий тренд (MA20 > MA50)")
        else:
            report.append("Интерпретация: Медвежий тренд (MA20 < MA50)")
        report.append("")
        
        report.append("3. ЛИНИИ ПОДДЕРЖКИ И СОПРОТИВЛЕНИЯ")
        report.append("-" * 80)
        report.append("Уровни поддержки:")
        for idx, price in support:
            report.append(f"  - ${price:.2f} (дата: {self.data.iloc[idx]['Date'].strftime('%Y-%m-%d')})")
        report.append("")
        report.append("Уровни сопротивления:")
        for idx, price in resistance:
            report.append(f"  - ${price:.2f} (дата: {self.data.iloc[idx]['Date'].strftime('%Y-%m-%d')})")
        report.append("")
        
        report.append("4. ОБНАРУЖЕННЫЕ ПАТТЕРНЫ")
        report.append("-" * 80)
        report.append(f"Найдено паттернов 'Двойное дно': {len(patterns)}")
        for i, pattern in enumerate(patterns[:5], 1):
            report.append(f"\nПаттерн #{i}:")
            report.append(f"  Первое дно: ${pattern['first_bottom_price']:.2f} ({self.data.iloc[pattern['first_bottom_idx']]['Date'].strftime('%Y-%m-%d')})")
            report.append(f"  Второе дно: ${pattern['second_bottom_price']:.2f} ({self.data.iloc[pattern['second_bottom_idx']]['Date'].strftime('%Y-%m-%d')})")
            report.append(f"  Промежуточный пик: ${pattern['peak_price']:.2f}")
        report.append("")
        
        report.append("5. ИНДИКАТОРЫ")
        report.append("-" * 80)
        report.append(f"RSI(14): {rsi.iloc[-1]:.2f}")
        if rsi.iloc[-1] > 70:
            report.append("  Интерпретация: Перекупленность (сигнал на продажу)")
        elif rsi.iloc[-1] < 30:
            report.append("  Интерпретация: Перепроданность (сигнал на покупку)")
        else:
            report.append("  Интерпретация: Нейтральная зона")
        report.append("")
        
        report.append(f"MACD: {macd.iloc[-1]:.2f}")
        report.append(f"Сигнальная линия: {macd_signal.iloc[-1]:.2f}")
        report.append(f"Гистограмма: {macd_hist.iloc[-1]:.2f}")
        if macd.iloc[-1] > macd_signal.iloc[-1]:
            report.append("  Интерпретация: Бычий сигнал (MACD > Signal)")
        else:
            report.append("  Интерпретация: Медвежий сигнал (MACD < Signal)")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_time_series_line(self):
        """Линейный график временного ряда"""
        fig, ax = plt.subplots(figsize=(16, 6))
        
        ax.plot(self.data['Date'], self.data['Close'], linewidth=1.5, color='#2E86AB', label='Цена закрытия')
        ax.fill_between(self.data['Date'], self.data['Low'], self.data['High'], 
                       alpha=0.2, color='gray', label='Диапазон High-Low')
        
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Цена ($)', fontsize=12)
        ax.set_title('Линейный график временного ряда', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат для более компактного отображения
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/time_series_line.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_seasonal_decomposition(self, period=252):
        """
        График сезонной декомпозиции временного ряда
        
        Args:
            period: период сезонности (252 для годовой сезонности в торговых днях)
        """
        if not STATSMODELS_AVAILABLE:
            print("Ошибка: statsmodels не установлен. Установите: pip install statsmodels")
            return None
        
        # Подготовка данных для декомпозиции
        ts = self.data.set_index('Date')['Close']
        
        # Если данных недостаточно для заданного периода, уменьшаем период
        if len(ts) < period * 2:
            period = len(ts) // 4 if len(ts) > 20 else None
        
        try:
            if period:
                decomposition = seasonal_decompose(ts, model='additive', period=period, extrapolate_trend='freq')
            else:
                # Если период не задан, используем мультипликативную модель без сезонности
                decomposition = seasonal_decompose(ts, model='additive', period=None)
        except Exception as e:
            print(f"Ошибка при декомпозиции: {e}. Используется упрощенная декомпозиция.")
            # Упрощенная декомпозиция
            trend = ts.rolling(window=min(50, len(ts)//10)).mean()
            seasonal = ts - trend
            residual = ts - trend - seasonal
            decomposition = type('obj', (object,), {
                'observed': ts,
                'trend': trend,
                'seasonal': seasonal,
                'resid': residual
            })()
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('Сезонная декомпозиция временного ряда', fontsize=16, fontweight='bold')
        
        # Исходный ряд
        axes[0].plot(decomposition.observed, color='#2E86AB', linewidth=1.5)
        axes[0].set_ylabel('Исходный ряд', fontsize=11)
        axes[0].set_title('Исходный временной ряд', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Тренд
        axes[1].plot(decomposition.trend, color='#A23B72', linewidth=1.5)
        axes[1].set_ylabel('Тренд', fontsize=11)
        axes[1].set_title('Тренд', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Сезонность
        axes[2].plot(decomposition.seasonal, color='#F18F01', linewidth=1.5)
        axes[2].set_ylabel('Сезонность', fontsize=11)
        axes[2].set_title('Сезонная составляющая', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Остаток
        axes[3].plot(decomposition.resid, color='#C73E1D', linewidth=1.5)
        axes[3].set_ylabel('Остаток', fontsize=11)
        axes[3].set_xlabel('Дата', fontsize=11)
        axes[3].set_title('Остаточная составляющая', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        axes[3].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Форматирование дат для более компактного отображения
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/seasonal_decomposition.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_acf_pacf(self, lags=40):
        """
        Графики автокорреляционной (ACF) и частичной автокорреляционной (PACF) функций
        
        Args:
            lags: количество лагов для анализа
        """
        if not STATSMODELS_AVAILABLE:
            print("Ошибка: statsmodels не установлен. Установите: pip install statsmodels")
            return None
        
        ts = self.data['Close'].values
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('Автокорреляционный анализ', fontsize=16, fontweight='bold')
        
        # ACF
        plot_acf(ts, lags=lags, ax=axes[0], alpha=0.05, title='Автокорреляционная функция (ACF)')
        axes[0].set_xlabel('Лаг', fontsize=11)
        axes[0].set_ylabel('Автокорреляция', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # PACF
        plot_pacf(ts, lags=lags, ax=axes[1], alpha=0.05, title='Частичная автокорреляционная функция (PACF)')
        axes[1].set_xlabel('Лаг', fontsize=11)
        axes[1].set_ylabel('Частичная автокорреляция', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('photos/acf_pacf.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_calendar_heatmap(self):
        """Календарная тепловая карта для выявления внутринедельных и внутримесячных паттернов"""
        # Подготовка данных
        df_heatmap = self.data.copy()
        df_heatmap['Year'] = df_heatmap['Date'].dt.year
        df_heatmap['Month'] = df_heatmap['Date'].dt.month
        df_heatmap['Day'] = df_heatmap['Date'].dt.day
        df_heatmap['Weekday'] = df_heatmap['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
        df_heatmap['Week'] = df_heatmap['Date'].dt.isocalendar().week
        
        # Создание матрицы для тепловой карты (месяц x день недели)
        pivot_month_weekday = df_heatmap.pivot_table(
            values='Close', 
            index='Month', 
            columns='Weekday', 
            aggfunc='mean'
        )
        
        # Создание матрицы для тепловой карты (год x месяц)
        pivot_year_month = df_heatmap.pivot_table(
            values='Close', 
            index='Year', 
            columns='Month', 
            aggfunc='mean'
        )
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle('Календарные паттерны во временном ряде', fontsize=16, fontweight='bold')
        
        # Тепловая карта: месяц x день недели
        if SEABORN_AVAILABLE:
            sns.heatmap(pivot_month_weekday, annot=False, fmt='.1f', cmap='YlOrRd', 
                       ax=axes[0], cbar_kws={'label': 'Средняя цена закрытия ($)'})
        else:
            im1 = axes[0].imshow(pivot_month_weekday.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            axes[0].set_xticks(range(len(pivot_month_weekday.columns)))
            axes[0].set_xticklabels(['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'])
            axes[0].set_yticks(range(len(pivot_month_weekday.index)))
            axes[0].set_yticklabels(pivot_month_weekday.index)
            plt.colorbar(im1, ax=axes[0], label='Средняя цена закрытия ($)')
        
        axes[0].set_xlabel('День недели', fontsize=11)
        axes[0].set_ylabel('Месяц', fontsize=11)
        axes[0].set_title('Средняя цена по месяцам и дням недели', fontsize=12, fontweight='bold')
        
        # Тепловая карта: год x месяц
        if SEABORN_AVAILABLE:
            sns.heatmap(pivot_year_month, annot=True, fmt='.0f', cmap='viridis', 
                       ax=axes[1], cbar_kws={'label': 'Средняя цена закрытия ($)'})
        else:
            im2 = axes[1].imshow(pivot_year_month.values, aspect='auto', cmap='viridis', interpolation='nearest')
            axes[1].set_xticks(range(len(pivot_year_month.columns)))
            axes[1].set_xticklabels(pivot_year_month.columns)
            axes[1].set_yticks(range(len(pivot_year_month.index)))
            axes[1].set_yticklabels(pivot_year_month.index)
            plt.colorbar(im2, ax=axes[1], label='Средняя цена закрытия ($)')
        
        axes[1].set_xlabel('Месяц', fontsize=11)
        axes[1].set_ylabel('Год', fontsize=11)
        axes[1].set_title('Средняя цена по годам и месяцам', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('photos/calendar_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig
    
    def plot_forecast(self, forecast_periods=30, confidence_level=0.95):
        """
        График с прогнозом и доверительным интервалом
        
        Args:
            forecast_periods: количество периодов для прогноза
            confidence_level: уровень доверительного интервала (0.95 = 95%)
        """
        if not STATSMODELS_AVAILABLE:
            print("Ошибка: statsmodels не установлен. Установите: pip install statsmodels")
            return None
        
        ts = self.data.set_index('Date')['Close']
        
        # Установка частоты данных (рабочие дни)
        try:
            ts = ts.asfreq('B')  # 'B' = Business day frequency
        except:
            pass  # Если не удалось установить частоту, продолжаем без нее
        
        # Разделение на обучающую и тестовую выборки
        train_size = int(len(ts) * 0.9)
        train = ts[:train_size]
        test = ts[train_size:] if train_size < len(ts) else pd.Series(dtype=float)
        
        # Построение модели ARIMA
        try:
            # Автоматический подбор параметров (упрощенный)
            # Пробуем разные порядки модели
            best_model = None
            best_aic = np.inf
            
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(train, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                        except:
                            continue
            
            if best_model is None:
                # Если не удалось подобрать модель, используем простую
                model = ARIMA(train, order=(1, 1, 1))
                fitted_model = model.fit()
            else:
                fitted_model = best_model
            
            # Прогноз
            forecast_result = fitted_model.get_forecast(steps=forecast_periods)
            forecast = forecast_result.predicted_mean
            
            # Доверительный интервал
            forecast_ci = forecast_result.conf_int(alpha=1-confidence_level)
            
            # Создание индекса для прогноза
            last_date = ts.index[-1]
            # Пытаемся определить частоту
            if len(ts) > 1:
                freq = pd.infer_freq(ts.index[-10:])
                if freq is None:
                    freq = 'B'  # По умолчанию рабочие дни
            else:
                freq = 'B'
            
            forecast_index = pd.date_range(start=last_date, periods=forecast_periods+1, freq=freq)[1:]
            forecast.index = forecast_index
            forecast_ci.index = forecast_index
            
        except Exception as e:
            print(f"Ошибка при построении модели ARIMA: {e}. Используется простое экспоненциальное сглаживание.")
            # Простое экспоненциальное сглаживание как запасной вариант
            alpha = 0.3
            forecast = []
            last_value = train.iloc[-1]
            trend = (train.iloc[-1] - train.iloc[-min(10, len(train))]) / min(10, len(train))
            
            for i in range(forecast_periods):
                forecast.append(last_value + trend * (i + 1))
            
            forecast = pd.Series(forecast, index=pd.date_range(start=ts.index[-1], periods=forecast_periods+1, freq='D')[1:])
            
            # Простой доверительный интервал на основе стандартного отклонения
            std_dev = train.diff().std()
            forecast_ci = pd.DataFrame({
                'lower': forecast - 1.96 * std_dev,
                'upper': forecast + 1.96 * std_dev
            }, index=forecast.index)
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Исторические данные
        ax.plot(ts.index, ts.values, label='Исторические данные', color='#2E86AB', linewidth=2)
        
        # Тестовая выборка (если есть)
        if len(test) > 0:
            ax.plot(test.index, test.values, label='Фактические значения (тест)', 
                   color='green', linewidth=2, linestyle='--')
        
        # Прогноз
        ax.plot(forecast.index, forecast.values, label='Прогноз', 
               color='#F18F01', linewidth=2, linestyle='--')
        
        # Доверительный интервал
        ax.fill_between(forecast.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1],
                       alpha=0.3, color='orange', label=f'Доверительный интервал ({confidence_level*100:.0f}%)')
        
        # Вертикальная линия разделения
        ax.axvline(x=ts.index[-1], color='red', linestyle=':', linewidth=2, 
                  label='Начало прогноза', alpha=0.7)
        
        ax.set_xlabel('Дата', fontsize=12)
        ax.set_ylabel('Цена закрытия ($)', fontsize=12)
        ax.set_title('Прогноз временного ряда с доверительным интервалом', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Форматирование дат для более компактного отображения
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('photos/forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return fig


def main():
    """Основная функция для выполнения анализа"""
    print("Загрузка данных...")
    analyzer = TechnicalAnalyzer('data.csv')
    
    print("Выполнение технического анализа...")
    
    # Генерация всех графиков
    print("Построение графика японских свечей...")
    analyzer.plot_candlestick_chart()
    
    print("Построение графиков индикаторов...")
    analyzer.plot_indicators()
    
    print("Построение графика поддержки/сопротивления...")
    analyzer.plot_support_resistance()
    
    print("Построение графика паттернов...")
    analyzer.plot_patterns()
    
    # Анализ временных рядов
    print("\nВыполнение анализа временных рядов...")
    
    print("Построение линейного графика временного ряда...")
    analyzer.plot_time_series_line()
    
    print("Построение графика сезонной декомпозиции...")
    analyzer.plot_seasonal_decomposition()
    
    print("Построение графиков ACF и PACF...")
    analyzer.plot_acf_pacf()
    
    print("Построение календарной тепловой карты...")
    analyzer.plot_calendar_heatmap()
    
    print("Построение графика прогноза...")
    analyzer.plot_forecast()
    
    # Генерация отчета
    print("\nГенерация отчета...")
    report = analyzer.generate_report()
    print(report)
    
    # Сохранение отчета
    with open('photos/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\nАнализ завершен!")
    print("Результаты сохранены в папку 'photos':")
    print("\nТехнический анализ:")
    print("  - candlestick_analysis.png - График японских свечей с анализом")
    print("  - indicators.png - Графики индикаторов (RSI, MACD, MA)")
    print("  - support_resistance.png - Линии поддержки и сопротивления")
    print("  - patterns.png - Обнаруженные паттерны")
    print("\nАнализ временных рядов:")
    print("  - time_series_line.png - Линейный график временного ряда")
    print("  - seasonal_decomposition.png - Сезонная декомпозиция")
    print("  - acf_pacf.png - Автокорреляционные функции (ACF и PACF)")
    print("  - calendar_heatmap.png - Календарная тепловая карта")
    print("  - forecast.png - Прогноз с доверительным интервалом")
    print("\nОтчет:")
    print("  - analysis_report.txt - Текстовый отчет")


if __name__ == '__main__':
    main()


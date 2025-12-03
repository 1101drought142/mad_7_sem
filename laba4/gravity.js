const Dataset = require('libs/dataset/v2');
const loadedData = Editor.getLoadedData();
const data = Dataset.processData(loadedData, 'data', Editor);

const params = Editor.getParams();

// Функция для нормализации значений (приводит к диапазону 0-1)
function normalizeValue(value, min, max) {
    if (max === min) return 0.5; // Если все значения одинаковые, возвращаем середину
    return (value - min) / (max - min);
}

// Функция для вычисления min и max для каждого поля
function calculateFieldStats(data, fields) {
    const stats = {};
    
    fields.forEach(field => {
        const values = data
            .map(row => parseFloat(row[field]))
            .filter(val => !isNaN(val) && isFinite(val));
        
        if (values.length > 0) {
            stats[field] = {
                min: Math.min(...values),
                max: Math.max(...values)
            };
        }
    });
    
    return stats;
}

// Функция для группировки данных по полю класса
function groupByClass(data, classField) {
    const grouped = {};
    
    data.forEach(row => {
        // Безопасно извлекаем название класса и преобразуем в строку
        let className = row[classField];
        if (className === null || className === undefined || className === '') {
            className = 'Unknown';
        } else {
            className = String(className).trim();
            if (className === '') {
                className = 'Unknown';
            }
        }
        
        if (!grouped[className]) {
            grouped[className] = [];
        }
        grouped[className].push(row);
    });
    
    return grouped;
}

// Палитра цветов для классов
function getClassColor(className, index, totalClasses) {
    // Используем палитру из 10 различных цветов
    const colors = [
        '#1f77b4', // синий
        '#ff7f0e', // оранжевый
        '#2ca02c', // зеленый
        '#d62728', // красный
        '#9467bd', // фиолетовый
        '#8c564b', // коричневый
        '#e377c2', // розовый
        '#7f7f7f', // серый
        '#bcbd22', // желто-зеленый
        '#17becf'  // голубой
    ];
    
    // Используем индекс для выбора цвета из палитры
    return colors[index % colors.length];
}

// Проверяем наличие данных
if (!data || data.length === 0) {
    module.exports = {
        error: 'No data available'
    };
} else {
    // Получаем числовые поля для анализа
    let allFields = [];
    
    if (params.parallelFields && params.parallelFields.length > 0) {
        // Проверяем, что указанные поля существуют в данных
        const availableKeys = Object.keys(data[0] || {});
        allFields = params.parallelFields.filter(field => availableKeys.includes(field));
        
        // Если указанные поля не найдены, используем все числовые поля
        if (allFields.length === 0) {
            allFields = availableKeys.filter(key => {
                const sampleValue = data[0][key];
                return !isNaN(parseFloat(sampleValue)) && isFinite(sampleValue);
            });
        }
    } else if (params.heatmapFields && params.heatmapFields.length > 0) {
        // Используем heatmapFields как fallback
        const availableKeys = Object.keys(data[0] || {});
        allFields = params.heatmapFields.filter(field => availableKeys.includes(field));
        
        if (allFields.length === 0) {
            allFields = availableKeys.filter(key => {
                const sampleValue = data[0][key];
                return !isNaN(parseFloat(sampleValue)) && isFinite(sampleValue);
            });
        }
    } else {
        // Если поля не указаны, используем все числовые поля
        allFields = Object.keys(data[0] || {})
            .filter(key => {
                const sampleValue = data[0][key];
                return !isNaN(parseFloat(sampleValue)) && isFinite(sampleValue);
            });
    }
    
    if (allFields.length === 0) {
        module.exports = {
            error: `No numeric fields found. Available fields: ${Object.keys(data[0] || {}).join(', ')}`
        };
    } else {
        // Ограничиваем количество полей для читаемости (максимум 8)
        const fields = allFields.slice(0, 8);
        
        // Получаем поле класса из параметров
        // Убеждаемся, что это валидная строка
        let classField = params.Species || 'Species';
        if (typeof classField !== 'string' || classField.trim() === '') {
            classField = 'Species';
        } else {
            classField = classField.trim();
        }
        
        // Проверяем наличие поля класса в данных
        const availableKeys = Object.keys(data[0] || {});
        let actualClassField = availableKeys.includes(classField) ? classField : null;
        
        // Если поле класса не найдено, пробуем найти по другим вариантам
        if (!actualClassField) {
            const possibleClassFields = ['Species', 'species', 'Class', 'class', 'label', 'Label'];
            for (const possibleField of possibleClassFields) {
                if (availableKeys.includes(possibleField)) {
                    actualClassField = possibleField;
                    break;
                }
            }
        }
        
        // Исключаем поле класса из списка полей для графика
        const fieldsForChart = fields.filter(field => field !== actualClassField);
        
        // Вычисляем статистику для нормализации (используем fieldsForChart)
        const fieldStats = calculateFieldStats(data, fieldsForChart);
        
        // Ограничиваем количество строк данных для производительности (максимум 100)
        const displayData = data.slice(0, 100);
        
        // Создаем категории для оси X (названия полей) - БЕЗ поля класса
        const categories = fieldsForChart;
        
        // Группируем данные по классам, если поле класса доступно
        let seriesData = [];
        
        if (actualClassField) {
            const groupedData = groupByClass(displayData, actualClassField);
            const classNames = Object.keys(groupedData);
            
            // Создаем серии для каждого класса
            classNames.forEach((className, classIndex) => {
                const classData = groupedData[className];
                const classColor = getClassColor(className, classIndex, classNames.length);
                
                // Для каждого объекта в классе создаем отдельную серию (линию)
                classData.forEach((row, rowIndex) => {
                    const seriesPoints = fieldsForChart.map((field, fieldIndex) => {
                        const rawValue = parseFloat(row[field]);
                        
                        // Проверяем валидность значения
                        if (isNaN(rawValue) || !isFinite(rawValue)) {
                            return null; // Пропускаем невалидные значения
                        }
                        
                        // Нормализуем значение
                        const stats = fieldStats[field];
                        const normalizedValue = stats ? normalizeValue(rawValue, stats.min, stats.max) : 0.5;
                        
                        // Создаем actionParams для интерактивности
                        const actionParams = {};
                        fieldsForChart.forEach(f => {
                            if (row[f] !== undefined) {
                                actionParams[f] = row[f];
                            }
                        });
                        // Добавляем класс в actionParams
                        if (row[actualClassField] !== undefined) {
                            actionParams[actualClassField] = row[actualClassField];
                        }
                        
                        return {
                            x: fieldIndex, // Позиция на оси X (индекс поля)
                            y: normalizedValue, // Нормализованное значение (0-1)
                            custom: {
                                rawValue: rawValue, // Исходное значение для tooltip
                                field: field,
                                actionParams: actionParams
                            }
                        };
                    }).filter(point => point !== null); // Убираем null значения
                    
                    // Используем название класса как имя серии
                    // Убеждаемся, что имя всегда валидная непустая строка
                    let name = String(className || 'Unknown').trim();
                    if (name === '' || name === 'null' || name === 'undefined') {
                        name = 'Unknown';
                    }
                    
                    // Пропускаем серии без данных
                    if (seriesPoints.length === 0) {
                        return;
                    }
                    
                    seriesData.push({
                        type: 'line',
                        name: name,
                        data: seriesPoints,
                        color: classColor, // Назначаем цвет классу
                        lineWidth: 1.5,
                        marker: {
                            enabled: true,
                            radius: 3,
                            fillColor: classColor, // Цвет маркеров тоже должен соответствовать классу
                            lineColor: classColor,
                            lineWidth: 1
                        },
                        opacity: 0.7 // Полупрозрачность для лучшей визуализации множества линий
                    });
                });
            });
        } else {
            // Если поле класса не найдено, используем старую логику с номерами строк
            seriesData = displayData.map((row, rowIndex) => {
                const seriesPoints = fieldsForChart.map((field, fieldIndex) => {
                    const rawValue = parseFloat(row[field]);
                    
                    // Проверяем валидность значения
                    if (isNaN(rawValue) || !isFinite(rawValue)) {
                        return null; // Пропускаем невалидные значения
                    }
                    
                    // Нормализуем значение
                    const stats = fieldStats[field];
                    const normalizedValue = stats ? normalizeValue(rawValue, stats.min, stats.max) : 0.5;
                    
                    // Создаем actionParams для интерактивности
                    const actionParams = {};
                    fieldsForChart.forEach(f => {
                        if (row[f] !== undefined) {
                            actionParams[f] = row[f];
                        }
                    });
                    
                    return {
                        x: fieldIndex, // Позиция на оси X (индекс поля)
                        y: normalizedValue, // Нормализованное значение (0-1)
                        custom: {
                            rawValue: rawValue, // Исходное значение для tooltip
                            field: field,
                            actionParams: actionParams
                        }
                    };
                }).filter(point => point !== null); // Убираем null значения
                
                // Создаем имя серии (можно использовать индекс или первые значения)
                const name = `Row ${rowIndex + 1}`;
                
                return {
                    type: 'line',
                    name: name,
                    data: seriesPoints,
                    lineWidth: 1,
                    marker: {
                        enabled: true,
                        radius: 3
                    },
                    opacity: 0.6 // Полупрозрачность для лучшей визуализации множества линий
                };
            });
        }
        
        // Экспортируем структуру графика для DataLens
        module.exports = {
            chart: {
                margin: {
                    left: 60,
                    right: 20,
                    top: 20,
                    bottom: 60,
                }
            },
            series: {
                data: seriesData
            },
            xAxis: {
                type: 'category',
                categories: categories,
                title: {
                    text: 'Features'
                }
            },
            yAxis: [{
                type: 'linear',
                title: {
                    text: 'Normalized Value (0-1)'
                },
                min: 0,
                max: 1
            }]
        };
    }
}

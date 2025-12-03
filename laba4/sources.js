const {buildSource} = require('libs/dataset/v2');

const params = Editor.getParams();
const heatmapFields = params.heatmapFields || [];
const classField = params.Species || 'Species';

// Создаем источник данных
// Если heatmapFields указаны, используем их, иначе получаем все данные
const sourceConfig = {
    id: Editor.getId('dataset'),
};

// Добавляем колонки только если они указаны
// Если массив пустой, DataLens вернет все доступные колонки
if (heatmapFields.length > 0) {
    // Включаем поле класса в запрос, если его еще нет
    const columns = [...heatmapFields];
    // Добавляем поле класса только если оно указано и является валидной строкой
    if (classField && typeof classField === 'string' && classField.trim() !== '' && !columns.includes(classField)) {
        columns.push(classField);
    }
    sourceConfig.columns = columns;
}

module.exports = {
    data: buildSource(sourceConfig),
};
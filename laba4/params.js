module.exports = {
    // Поля датасета
    SepalLength: 'SepalLengthCm',
    SepalWidth: 'SepalWidthCm',
    PetalLength: 'PetalLengthCm',
    PetalWidth: 'PetalWidthCm',
    Species: 'Species',
    
    // Настройки для параллельных координат
    parallelFields: ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    
    // Настройки heatmap (для совместимости, используется как fallback)
    heatmapFields: ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    correlationMethod: 'pearson', // 'pearson', 'spearman', 'kendall'
    
    // Настройки отображения
    showValues: true,
    colorScale: 'RdYlBu',
    reverseColorScale: false
};
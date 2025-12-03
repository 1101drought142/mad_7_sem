# === 0. Очистка окружения ===
rm(list = ls())
if (exists(".Random.seed")) rm(.Random.seed)

# === 1. Загрузка данных ===
# Устанавливаем пакет readxl если нужно
if(!require(readxl)) install.packages("readxl", repos = "https://cran.r-project.org")
library(readxl)

# Устанавливаем пакет ggplot2 для визуализации если нужно
if(!require(ggplot2)) install.packages("ggplot2", repos = "https://cran.r-project.org")
library(ggplot2)

# Загружаем данные из Excel файла с правильными типами колонок
# Порядок: Номер (skip), Дата сделка (skip), Возраст, Дистанция до метро, Количество магазинов рядом, Широта, Долгота, Стоимость за ед площади
file_path <- "houses/data.xlsx"
data <- read_excel(file_path, col_types = c("numeric", "skip", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"))

# Переименовываем колонки для удобства
colnames(data) <- c("Номер", "Возраст", "Дистанция_до_метро", "Количество_магазинов_рядом", "Широта", "Долгота", "Стоимость_за_ед_площади")

# === 2. Подготовка данных ===
df <- data.frame(
  metro_distance = as.numeric(data$Дистанция_до_метро),      # Независимая переменная: Расстояние до метро
  price_per_unit = as.numeric(data$Стоимость_за_ед_площади), # Зависимая переменная: Стоимость за м2
  number_of_stores = as.numeric(data$Количество_магазинов_рядом)  # Количество магазинов рядом
)

# Убедимся, что переменные numeric
df$metro_distance <- as.numeric(df$metro_distance)
df$price_per_unit <- as.numeric(df$price_per_unit)

cat("Исходное количество наблюдений:", nrow(df), "\n")

# === 2.1. Удаление выбросов методом IQR ===
cat("\n=== Обнаружение и удаление выбросов ===\n")

# Функция для обнаружения выбросов по методу IQR
detect_outliers_iqr <- function(var, var_name) {
  var_clean <- var[!is.na(var)]
  Q1 <- quantile(var_clean, 0.25, na.rm = TRUE)
  Q3 <- quantile(var_clean, 0.75, na.rm = TRUE)
  IQR_val <- Q3 - Q1
  lower_bound <- Q1 - 1.5 * IQR_val
  upper_bound <- Q3 + 1.5 * IQR_val
  
  outliers <- which(var < lower_bound | var > upper_bound)
  
  cat("\n", var_name, ":\n")
  cat("  Q1 =", round(Q1, 2), ", Q3 =", round(Q3, 2), ", IQR =", round(IQR_val, 2), "\n")
  cat("  Границы: [", round(lower_bound, 2), ", ", round(upper_bound, 2), "]\n", sep = "")
  cat("  Количество выбросов:", length(outliers), "\n")
  
  return(outliers)
}

# Обнаруживаем выбросы для каждой переменной
outliers_metro <- detect_outliers_iqr(df$metro_distance, "Расстояние до метро")
outliers_price <- detect_outliers_iqr(df$price_per_unit, "Стоимость за м2")
outliers_stores <- detect_outliers_iqr(df$number_of_stores, "Количество магазинов рядом")

# Объединяем все индексы выбросов
all_outliers <- unique(c(outliers_metro, outliers_price, outliers_stores))

cat("\nОбщее количество строк с выбросами (по любой переменной):", length(all_outliers), "\n")
cat("Процент выбросов:", round(100 * length(all_outliers) / nrow(df), 2), "%\n")

# Удаляем выбросы
if (length(all_outliers) > 0) {
  df <- df[-all_outliers, ]
  cat("Удалено строк:", length(all_outliers), "\n")
  cat("Осталось наблюдений:", nrow(df), "\n")
} else {
  cat("Выбросы не обнаружены.\n")
}

# === 2.2. Убираем NA ===
df <- na.omit(df)

cat("\nКоличество наблюдений после удаления NA:", nrow(df), "\n")
cat("Переменные:\n")
cat("  metro_distance - Расстояние до метро\n")
cat("  price_per_unit - Стоимость за м2 (зависимая переменная)\n")
cat("  number_of_stores - Количество магазинов рядом\n")

# === 3. Формирование фактора (metro_distance в 3 равные группы) ===
# Разбиваем на три равные группы по квантилям
df$metro_group <- cut(df$metro_distance,
                   breaks = quantile(df$metro_distance, probs = seq(0, 1, length = 4), na.rm = TRUE),
                   include.lowest = TRUE,
                   labels = c("Близко", "Средне", "Далеко"))

# Проверка распределения по группам
cat("\nРаспределение по группам расстояния до метро:\n")
print(table(df$metro_group))

# === 4. Однофакторный дисперсионный анализ (ANOVA) ===
# Зависимая переменная: price_per_unit (Стоимость за м2)
# Фактор: metro_group (группы расстояния до метро)
anova_model <- aov(price_per_unit ~ metro_group, data = df)
summary(anova_model)

# === 5. Интерпретация гипотезы ===
# H0: средняя стоимость за м2 одинакова во всех 3 группах расстояния до метро
# H1: хотя бы в одной группе средняя стоимость отличается

# === 6. График для наглядности ===
boxplot(price_per_unit ~ metro_group, data = df,
        col = c("lightblue", "lightgreen", "lightpink"),
        main = "Стоимость за м2 по группам расстояния до метро",
        xlab = "Группа расстояния до метро", ylab = "Стоимость за м2")

# === 6.1. Многослойная гистограмма для сравнения распределений по группам ===
# Создаем папку для графиков если её нет
if (!dir.exists("img")) dir.create("img")

# Многослойная гистограмма с прозрачностью
histogram_plot <- ggplot(df, aes(x = price_per_unit, fill = metro_group)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 30, color = "black", size = 0.3) +
  scale_fill_manual(values = c("Близко" = "#87CEEB", "Средне" = "#90EE90", "Далеко" = "#FFB6C1"),
                    name = "Группа расстояния\nдо метро") +
  labs(title = "Многослойная гистограмма: Стоимость за м2 по группам расстояния до метро",
       x = "Стоимость за м2",
       y = "Частота") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right")

# Сохраняем график
ggsave("img/histogram_density_by_groups.png", plot = histogram_plot, 
       width = 10, height = 6, dpi = 300)
cat("\nГрафик многослойной гистограммы сохранен: img/histogram_density_by_groups.png\n")

# === 6.2. График плотности для сравнения формы распределений ===
density_plot <- ggplot(df, aes(x = price_per_unit, fill = metro_group, color = metro_group)) +
  geom_density(alpha = 0.5, size = 1) +
  scale_fill_manual(values = c("Близко" = "#87CEEB", "Средне" = "#90EE90", "Далеко" = "#FFB6C1"),
                    name = "Группа расстояния\nдо метро") +
  scale_color_manual(values = c("Близко" = "#4682B4", "Средне" = "#32CD32", "Далеко" = "#FF69B4"),
                     name = "Группа расстояния\nдо метро") +
  labs(title = "График плотности: Сравнение распределений стоимости за м2 по группам",
       x = "Стоимость за м2",
       y = "Плотность") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right")

# Сохраняем график
ggsave("img/density_plot_by_groups.png", plot = density_plot, 
       width = 10, height = 6, dpi = 300)
cat("График плотности сохранен: img/density_plot_by_groups.png\n")

# === 7. Формирование второго фактора (количество магазинов рядом в группы) ===
# Разбиваем на три равные группы по квантилям
df$stores_group <- cut(df$number_of_stores,
                   breaks = quantile(df$number_of_stores, probs = seq(0, 1, length = 4), na.rm = TRUE),
                   include.lowest = TRUE,
                   labels = c("Мало", "Средне", "Много"))

cat("\nРаспределение по группам количества магазинов рядом:\n")
print(table(df$stores_group))

# === 8. Двухфакторный дисперсионный анализ (metro_group и stores_group) ===
# Проверяем влияние группы расстояния до метро и количества магазинов рядом на стоимость
anova_model2 <- aov(price_per_unit ~ metro_group * stores_group, data = df)
summary(anova_model2)

# === 9. Интерпретация гипотезы ===
# H0: стоимость за м2 не зависит от metro_group, stores_group и их взаимодействия
# H1: хотя бы один фактор (или их взаимодействие) влияет на стоимость за м2

# === 9.1. Визуализация для двухфакторного ANOVA ===
# Многослойная гистограмма с двумя факторами
histogram_2factor <- ggplot(df, aes(x = price_per_unit, fill = metro_group)) +
  geom_histogram(alpha = 0.6, position = "identity", bins = 25, color = "black", size = 0.3) +
  facet_wrap(~ stores_group, ncol = 3, labeller = labeller(stores_group = c("Мало" = "Мало магазинов", 
                                                                             "Средне" = "Средне магазинов", 
                                                                             "Много" = "Много магазинов"))) +
  scale_fill_manual(values = c("Близко" = "#87CEEB", "Средне" = "#90EE90", "Далеко" = "#FFB6C1"),
                    name = "Группа расстояния\nдо метро") +
  labs(title = "Многослойная гистограмма: Стоимость за м2 по группам (двухфакторный анализ)",
       x = "Стоимость за м2",
       y = "Частота") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right",
        strip.text = element_text(size = 10, face = "bold"))

ggsave("img/histogram_2factor_anova.png", plot = histogram_2factor, 
       width = 12, height = 6, dpi = 300)
cat("\nГрафик многослойной гистограммы для двухфакторного ANOVA сохранен: img/histogram_2factor_anova.png\n")

# График плотности с двумя факторами
density_2factor <- ggplot(df, aes(x = price_per_unit, fill = metro_group, color = metro_group)) +
  geom_density(alpha = 0.5, size = 1) +
  facet_wrap(~ stores_group, ncol = 3, labeller = labeller(stores_group = c("Мало" = "Мало магазинов", 
                                                                             "Средне" = "Средне магазинов", 
                                                                             "Много" = "Много магазинов"))) +
  scale_fill_manual(values = c("Близко" = "#87CEEB", "Средне" = "#90EE90", "Далеко" = "#FFB6C1"),
                    name = "Группа расстояния\nдо метро") +
  scale_color_manual(values = c("Близко" = "#4682B4", "Средне" = "#32CD32", "Далеко" = "#FF69B4"),
                     name = "Группа расстояния\nдо метро") +
  labs(title = "График плотности: Сравнение распределений (двухфакторный анализ)",
       x = "Стоимость за м2",
       y = "Плотность") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
        legend.position = "right",
        strip.text = element_text(size = 10, face = "bold"))

ggsave("img/density_2factor_anova.png", plot = density_2factor, 
       width = 12, height = 6, dpi = 300)
cat("График плотности для двухфакторного ANOVA сохранен: img/density_2factor_anova.png\n")

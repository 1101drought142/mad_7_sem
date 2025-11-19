# === 0. Очистка окружения ===
rm(list = ls())
if (exists(".Random.seed")) rm(.Random.seed)

# === 1. Загрузка данных ===
# Устанавливаем пакет readxl если нужно
if(!require(readxl)) install.packages("readxl", repos = "https://cran.r-project.org")
library(readxl)

# Загружаем данные из Excel файла с правильными типами колонок
# Порядок: Номер (skip), Дата сделка (skip), Возраст, Дистанция до метро, Количество магазинов рядом, Широта, Долгота, Стоимость за ед площади
file_path <- "houses/data.xlsx"
data <- read_excel(file_path, col_types = c("numeric", "skip", "numeric", "numeric", "numeric", "numeric", "numeric", "numeric"))

# Переименовываем колонки для удобства
colnames(data) <- c("Номер", "Возраст", "Дистанция_до_метро", "Количество_магазинов_рядом", "Широта", "Долгота", "Стоимость_за_ед_площади")

# === 2. Подготовка данных ===
df <- data.frame(
  metro_distance = as.numeric(data$Дистанция_до_метро),      # Независимая переменная: Расстояние до метро
  price_per_unit = as.numeric(data$Стоимость_за_ед_площади)  # Зависимая переменная: Стоимость за м2
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

# Объединяем все индексы выбросов
all_outliers <- unique(c(outliers_metro, outliers_price))

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
cat("  metro_distance - Расстояние до метро (независимая)\n")
cat("  price_per_unit - Стоимость за м2 (зависимая)\n")

# === 3. Линейная регрессия price_per_unit ~ metro_distance ===
lm_model <- lm(price_per_unit ~ metro_distance, data = df)

# Вывод результатов модели
summary(lm_model)

# === 4. Остатки ===
residuals_lm <- residuals(lm_model)

std_resid <- rstandard(lm_model)
hist(std_resid, breaks = 20, col = "lightgray", border = "black",
     main = "Гистограмма стандартизированных остатков", xlab = "Стандартизированные остатки")

# === 5. График модели ===
plot(df$metro_distance, df$price_per_unit,
     pch = 19, col = "blue",
     xlab = "Расстояние до метро", ylab = "Стоимость за м2",
     main = "Линейная регрессия: Стоимость за м2 ~ Расстояние до метро")
abline(lm_model, col = "red", lwd = 2)

# === 6. График остатков ===
plot(df$metro_distance, residuals_lm,
     pch = 19, col = "darkgreen",
     xlab = "Расстояние до метро", ylab = "Остатки",
     main = "График остатков")
abline(h = 0, col = "red", lwd = 2)

# === 7. Q-Q график остатков ===
qqnorm(residuals_lm, main = "Q-Q Plot остатков")
qqline(residuals_lm, col = "red", lwd = 2)

# === 8. Гистограмма остатков ===
hist(residuals_lm,
     breaks = 9, col = "lightgray", border = "black",
     main = "Гистограмма остатков", xlab = "Остатки")

# === 9. Уравнение регрессии ===
coeffs <- coef(lm_model)
eq <- paste0("Стоимость_за_м2 = ",
             round(coeffs[1], 2), " + ",
             round(coeffs[2], 2), " * Расстояние_до_метро")
cat("\nУравнение регрессии:\n", eq, "\n")

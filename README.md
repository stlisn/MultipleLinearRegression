
# Отчёт по лабораторной работе: Построение и оценка линейной многофакторной модели (ЛМФМ)

## Цель работы и постановка задачи

**Цель работы**: Разработка программы для построения и анализа линейной многофакторной модели (ЛМФМ), позволяющей:
- оценивать статистическую значимость факторов и их взаимосвязь с откликом;
- проверять модель на мультиколлинеарность;
- оценивать адекватность модели на основе коэффициента детерминации и ошибок;
- выполнять предсказания отклика для новых данных.

**Постановка задачи**:
1. Считать данные из CSV-файла, содержащего факторы и отклик.
2. Построить ЛМФМ, отобрав значимые факторы на основе уровня значимости, указанного пользователем.
3. Оценить модель на предмет мультиколлинеарности факторов.
4. Проверить модель на адекватность (статистическую значимость).
5. Сохранить или вывести результаты предсказания для новых данных.

---

## Краткое описание процедуры построения и оценки ЛМФМ

1. **Считывание данных**:
   - Данные факторов и отклика считываются из CSV-файла с разделителем `;`.

2. **Анализ факторов**:
   - Для каждого фактора рассчитываются:
     - p-значение для проверки статистической значимости;
     - коэффициент корреляции с откликом.
   - Пользователь выбирает значимые факторы для включения в модель.

3. **Проверка на мультиколлинеарность**:
   - Рассчитывается матрица корреляции между всеми выбранными факторами.

4. **Оценка адекватности модели**:
   - Рассчитываются:
     - Коэффициент детерминации R^2;
     - F-статистика и её p-значение.
   - Проверяется статистическая значимость модели.
   - Вычисляются ошибки предсказания (средняя относительная ошибка и RMSE).

5. **Предсказания для новых данных**:
   - Новые данные загружаются из указанного файла.
   - Предсказания выводятся на экран или сохраняются в файл.

---

## Демонстрация работы программы

### Входные данные

- **Данные для обучения**: `learning_data.csv`
- **Данные для прогноза**: `data_for_prediction.csv`
- **Уровень значимости**: 0.05


### Результаты выполнения

1. **Выбранные значимые факторы**: `X1, X3, X4` (пример выбора).
2. **Анализ связи факторов с откликом и их статистическую значимость**:
   ```
   Фактор: X1, p-значение: 0.2189, корреляция с y: 0.4889
   Фактор: X2, p-значение: 0.0678, корреляция с y: 0.6722
   Фактор: X3, p-значение: 0.0866, корреляция с y: 0.6413
   Фактор: X4, p-значение: 0.0207, корреляция с y: 0.7862
   Фактор: X5, p-значение: 0.0573, корреляция с y: 0.6919
   ```
2. **Проверка на мультиколлинеарность**:
   ```
   Корреляция между X1 и X3: 0.3131
   Корреляция между X1 и X4: 0.0215
   Корреляция между X3 и X4: 0.5501
   ```
3. **Модель**:
   - R^2: 0.85
   - F-статистика: 7.46
   - p-значение F-теста: 0.04
   - RMSE:  33.73
   - Модель адекватна

### Выходные данные

- Файл с прогнозами: `predictions.csv`.

---

## Выводы

Разработанная программа позволяет:
1. Анализировать статистическую значимость факторов и их связь с откликом.
2. Построить адекватную ЛМФМ, выявляя значимые факторы.
3. Оценить адекватность модели и её ошибки.
4. Выполнять точные предсказания для новых данных.

Программа доказала свою функциональность на предоставленных данных. 

import pandas as pd
import numpy as np
import statsmodels.api as sm

def read_data(file_path):
    """
    Считывает данные из текстового файла.
    """
    data = pd.read_csv(file_path, sep=';')
    return data

def analyze_factors(X, y):
    """
    Анализирует связь факторов с откликом и их статистическую значимость.
    """
    factor_analysis = {}
    for col in X.columns:
        X_with_intercept = sm.add_constant(X[col], prepend=False)
        model = sm.OLS(y, X_with_intercept).fit()
        p_value = model.pvalues[col]
        corr_y = np.corrcoef(X[col], y)[0, 1]
        factor_analysis[col] = {
            "p_value": p_value,
            "corr_with_y": corr_y
        }
        print(f"Фактор: {col}, p-значение: {p_value:.4f}, корреляция с y: {corr_y:.4f}")
    
    print("\nВыберите факторы, которые хотите оставить:")
    global selected_factors
    selected_factors = input("Введите имена факторов через запятую: ").strip().split(',')
    selected_factors = [factor.strip() for factor in selected_factors]
    
    return X[selected_factors]

def check_multicollinearity(X):
    """
    Проверяет мультиколлинеарность факторов.
    """
    corr_matrix = X.corr()
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            print(f"Корреляция между {corr_matrix.index[i]} и {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.4f}")

def evaluate_model(X, y, alpha):
    """
    Оценивает модель.
    """
    X_with_intercept = sm.add_constant(X, prepend=False)
    model = sm.OLS(y, X_with_intercept).fit()
    print(model.summary())
    
    # Коэффициент детерминации
    r2 = model.rsquared
    f_stat = model.fvalue
    f_p_value = model.f_pvalue
    
    print(f"R^2: {r2:.2f}, F-статистика: {f_stat:.2f}, p-значение F-теста: {f_p_value:.2f}")
    if f_p_value < alpha:
        print("Модель адекватна (статистически значима).")
    else:
        print("Модель неадекватна.")
    
    # Ошибки
    y_pred = model.predict(X_with_intercept)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    print(f"RMSE: {rmse:.2f}")
    
    return model

def predict_new_values(model, new_data_file):
    """
    Предсказание для новых данных.
    """
    new_data = pd.read_csv(new_data_file, sep=';')
    new_data = new_data[selected_factors]
    new_data_with_intercept = sm.add_constant(new_data, prepend=False)
    predictions = model.predict(new_data_with_intercept).round(2)
    print("Предсказания для новых данных:")
    print(predictions)
    return predictions

selected_factors = ''

# Считывание данных
input_file = 'learning_data.csv'
data = read_data(input_file)

# Разделение на факторы и отклик
y = data['y']
X = data.drop(columns=['y'])

# Ввод уровня значимости
alpha = float(input("Введите уровень значимости (например, 0.05): "))

# Анализ факторов
X_selected = analyze_factors(X, y)

# Проверка на мультиколлинеарность
check_multicollinearity(X_selected)

# Оценка модели
model = evaluate_model(X_selected, y, alpha)

# Предсказания:
new_data_file = 'data_for_prediction.csv'
predictions = predict_new_values(model, new_data_file)
output_file = 'predictions.csv'
predictions.to_csv(output_file, index=True, header=False)
    

from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Загрузка модели
loaded_model = joblib.load('water_pred_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Получение данных из веб-формы
        ph = float(request.form['ph'])
        Hardness = float(request.form['Hardness'])
        Solids = float(request.form['Solids'])
        Chloramines = float(request.form['Chloramines'])
        Sulfate = float(request.form['Sulfate'])
        Conductivity = float(request.form['Conductivity'])
        Organic_carbon = float(request.form['Organic_carbon'])
        Trihalomethanes = float(request.form['Trihalomethanes'])
        Turbidity = float(request.form['Turbidity'])

        # Создание DataFrame с полученными данными
        data = pd.DataFrame([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]],
                            columns=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate','Conductivity' , 'Organic_carbon', 'Trihalomethanes', 'Turbidity'])

        # Получение предсказания от модели
        prediction = loaded_model.predict(data)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)

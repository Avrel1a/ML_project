import flask
from flask import Flask, render_template, request
import pickle
import pandas as pd
import sklearn

app = flask.Flask(__name__, template_folder='Templates')

@app.rout('/', methods=['POST', 'GET'])
@app.rout('/index', methods=['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    if flask.request.method == 'POST':
        with open('RF_model1.pkl', 'rb') as f1:
            RF_model1_loaded = pickle.load(f1)
        with open('RF_model2.pkl', 'rb') as f2:
            RF_model2_loaded = pickle.load(f2)
        with open('preprocessor1.pkl', 'rb') as p1:
            preprocessor1_loaded = pickle.load(p1)
        with open('preprocessor2.pkl', 'rb') as p2:
            preprocessor2_loaded = pickle.load(p2)

        input_features = ['Соотношение матрица-наполнитель', 'Плотность, кг/м3', 'Модуль упругости, ГПа',
                    'Количество отвердителя, м.%', 'Содержание эпоксидных групп,%_2', 'Температура вспышки, С_2',
                    'Поверхностная плотность, г/м2', 'Потребление смолы, г/м2', 'Угол нашивки, град', 'Шаг нашивки',
                    'Плотность нашивки']

        df = pd.DataFrame(
            [
                [request.form.get(k) for k in input_features]
            ], columns=input_features
        )

        x1 = preprocessor1_loaded.transform(df)
        x2 = preprocessor2_loaded.transform(df)

        y1_pred = RF_model1_loaded.predict(x1)
        y2_pred = RF_model2_loaded.predict(x2)

        return render_template('main.html', output_1=y1_pred, output_2=y2_pred)

if __name__ == '__main__':
    app.run()



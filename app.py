import flask
from flask import Flask, render_template, request
import pandas as pd
import pickle
import sklearn
from sklearn.ensemble import RandomForestRegressor

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['POST', 'GET'])
@app.route('/index', methods=['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    if flask.request.method == 'POST':
        with open('models/RF_model1.pkl', 'rb') as f1:
            RF_model1_loaded = pickle.load(f1)
        with open('models/RF_model2.pkl', 'rb') as f2:
            RF_model2_loaded = pickle.load(f2)
        with open('models/preprocessor1.pkl', 'rb') as p1:
            preprocessor1_loaded = pickle.load(p1)
        with open('models/preprocessor2.pkl', 'rb') as p2:
            preprocessor2_loaded = pickle.load(p2)

        var1 = float(flask.request.form['Соотношение матрица-наполнитель'])
        var2 = float(flask.request.form['Плотность, кг/м3'])
        var3 = float(flask.request.form['Модуль упругости, ГПа'])
        var4 = float(flask.request.form['Количество отвердителя, м.%'])
        var5 = float(flask.request.form['Содержание эпоксидных групп,%_2'])
        var6 = float(flask.request.form['Температура вспышки, С_2'])
        var7 = float(flask.request.form['Поверхностная плотность, г/м2'])
        var8 = float(flask.request.form['Потребление смолы, г/м2'])
        var9 = float(flask.request.form['Угол нашивки, град'])
        var10 = float(flask.request.form['Шаг нашивки'])
        var11 = float(flask.request.form['Плотность нашивки'])

        list_of_features = [var1, var2, var3, var4, var5, var6, var7, var8, var9, var10, var11]

        sample_x = pd.DataFrame(list_of_features, index=[0])

        x1 = preprocessor1_loaded.transform(sample_x)
        x2 = preprocessor2_loaded.transform(sample_x)

        y1_pred = RF_model1_loaded.predict(x1)
        y2_pred = RF_model2_loaded.predict(x2)

        return render_template('main.html', y1_pred=y1_pred, y2_pred=y2_pred)

if __name__ == '__main__':
    app.run()



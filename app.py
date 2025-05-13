from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    file = request.files['file']
    date_col = request.form['date_col']
    value_col = request.form['value_col']
    periods = int(request.form['periods'])
    model_choice = request.form['model']

    df = pd.read_csv(file)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, value_col]].dropna()
    df = df.set_index(date_col).asfreq('MS')
    df = df.fillna(method='ffill')

    last_date = df.index[-1]
    freq = df.index.inferred_freq or 'MS'
    forecast_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)

    # Model accuracy storage
    model_accuracies = {}

    # Holt-Winters
    hw_model = ExponentialSmoothing(df[value_col], trend='add', seasonal='add', seasonal_periods=12)
    hw_fit = hw_model.fit()
    hw_forecast = hw_fit.forecast(periods)
    hw_accuracy = mean_absolute_percentage_error(df[value_col][-periods:], hw_fit.fittedvalues[-periods:]) if len(df) > periods else None
    model_accuracies['holtwinters'] = f"{(1 - hw_accuracy) * 100:.2f}%" if hw_accuracy is not None else "N/A"

    # ARIMA
    arima_model = ARIMA(df[value_col], order=(5, 1, 0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=periods)
    arima_accuracy = mean_absolute_percentage_error(df[value_col][-periods:], arima_fit.predict(start=len(df)-periods, end=len(df)-1)) if len(df) > periods else None
    model_accuracies['arima'] = f"{(1 - arima_accuracy) * 100:.2f}%" if arima_accuracy is not None else "N/A"

    # Naive
    naive_forecast = pd.Series([df[value_col].iloc[-1]] * periods, index=forecast_index)
    naive_accuracy = mean_absolute_percentage_error(df[value_col][-periods:], [df[value_col].iloc[-1]] * periods) if len(df) > periods else None
    model_accuracies['naive'] = f"{(1 - naive_accuracy) * 100:.2f}%" if naive_accuracy is not None else "N/A"

    # Plot Comparison Chart
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df[value_col], label='Historical', color='black')
    plt.plot(forecast_index, hw_forecast, label='Holt-Winters Forecast', color='red')
    plt.plot(forecast_index, arima_forecast, label='ARIMA Forecast', color='green')
    plt.plot(forecast_index, naive_forecast, label='Naive Forecast', color='blue')
    plt.title('Model Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/comparison.png')
    plt.close()

    # Selected Model Plot
    if model_choice == 'holtwinters':
        forecast = hw_forecast
        accuracy = model_accuracies['holtwinters']
        label = 'Holt-Winters Forecast'
    elif model_choice == 'arima':
        forecast = arima_forecast
        accuracy = model_accuracies['arima']
        label = 'ARIMA Forecast'
    elif model_choice == 'naive':
        forecast = naive_forecast
        accuracy = model_accuracies['naive']
        label = 'Naive Forecast'
    else:
        return "Invalid model selected."

    # Selected model graph
    plt.figure(figsize=(8, 4))
    plt.plot(df.index, df[value_col], label='Historical', color='black')
    plt.plot(forecast_index, forecast, label=label, color='orange')
    plt.title(f'{label}')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/selected_model.png')
    plt.close()

    # Forecast Results
    forecast_df = pd.DataFrame({'ds': forecast_index, 'yhat': forecast})
    forecast_df['ds'] = forecast_df['ds'].dt.strftime('%Y-%m-%d')
    forecast_list = forecast_df.to_dict(orient='records')
    forecast_df.to_csv("forecast.csv", index=False)

    return render_template(
        "index.html",
        predictions=forecast_list,
        model=model_choice.upper(),
        accuracy=accuracy,
        model_accuracies=model_accuracies,
        show_downloads=True,
        show_plot=True
    )

@app.route('/download_csv')
def download_csv():
    return send_file("forecast.csv", as_attachment=True)

@app.route('/download_chart')
def download_chart():
    return send_file("static/selected_model.png", as_attachment=True)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)

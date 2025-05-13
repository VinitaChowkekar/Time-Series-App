# 📈 Time Series Forecasting App

A simple web-based Time Series Forecasting application built using **Flask**. Upload your time series data and generate forecasts using models like **Holt-Winters**, **ARIMA**, and **Naive Forecasting**.

---

## 🚀 Features

- Upload CSV with time series data
- Choose forecasting model: Holt-Winters, ARIMA, Naive
- Visualize forecasts with interactive plots
- Download forecast results as CSV and image
- View model-wise accuracy (MAPE)

---

## 🧠 Models Used

- **Holt-Winters** (Triple Exponential Smoothing)
- **ARIMA**
- **Naive Forecast**

---

## 🗂️ Project Structure

.

├── app.py # Main Flask app

├── templates/

│ └── index.html # UI for uploading and viewing results

├── static/

│ ├── timepic.jpeg # Background image

│ ├── comparison.png # Forecast comparison plot

│ └── selected_model.png# Selected model's forecast plot

├── requirements.txt # Python dependencies

└── README.md # Project info


---

## ⚙️ How to Run

### 1. Clone the repository

~~~bash
git clone https://github.com/VinitaChowkekar/Time-Series-App.git
cd your-repo-name
~~~

## 2. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

## 3. Run the app

~~~bash
python app.py
~~~

👩‍💻 Author
-Vinita Chowkekar

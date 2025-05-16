from flask import Flask, render_template, request
import pandas as pd
import joblib
import os
from datetime import datetime, timedelta
import numpy as np
from reorder_point_calculation import ReorderPointCalculator  # ✅ Import reorder module

app = Flask(__name__)

# Load Forecasting and Stockout Models
forecast_model = joblib.load('xgb_model.pkl')
stockout_model = joblib.load('stockout_model.pkl')
season_encoder = joblib.load('season_encoder_stockout.pkl')

# Load Supplier Model and Encoder
supplier_model = joblib.load('supplier_model.pkl')
supplier_encoder = joblib.load('supplier_encoder.pkl')

# Load Supplier Data
df = pd.read_csv('inventory_data_without_supplier_performance.csv')
dealer_ids = df['Dealer_ID'].unique()[:10]

# ---------------------- Home ----------------------
@app.route('/')
def index():
    return render_template('index.html')

# ---------------------- Forecasting ----------------------
@app.route('/forecast')
def forecast():
    return render_template('forecast.html')

@app.route('/forecast_result', methods=['POST'])
def forecast_result():
    # Get form data
    season = int(request.form['season'])
    consumption_rate = float(request.form['consumption_rate'])
    rolling_avg = float(request.form['rolling_avg'])
    days = int(request.form['days'])

    # Generate forecast with basic trend and noise
    base = rolling_avg
    trend = 0.2  # Add a small trend upwards
    noise_std_dev = 0.5  # Add some randomness

    forecast = []
    start_date = datetime.today()

    for i in range(days):
        date = start_date + timedelta(days=i)
        predicted = base + (i * trend) + np.random.normal(0, noise_std_dev)
        predicted = max(0, round(predicted))  # No negative predictions
        forecast.append({'Date': date.strftime('%Y-%m-%d'), 'Predicted_Quantity': predicted})

    return render_template('forecast_result.html', tables=forecast)

# ---------------------- Stockout Risk ----------------------
@app.route('/stockout')
def stockout():
    return render_template('stockout.html')

@app.route('/stockout_result', methods=['POST'])
def stockout_result():
    try:
        input_data = pd.DataFrame([{
            'Consumption_Rate': float(request.form['consumption_rate']),
            'Lag_Quantity': float(request.form['lag_quantity']),
            'Lag_Consumption': float(request.form['lag_consumption']),
            'RollingAvg_Consumption_7': float(request.form['rolling_7']),
            'RollingAvg_Consumption_30': float(request.form['rolling_30']),
            'IsWeekend': int(request.form['is_weekend']),
            'Season': request.form['season']
        }])

        input_data['Season'] = season_encoder.transform(input_data[['Season']])
        prediction = stockout_model.predict(input_data)[0]

        result = "LOW" if prediction == 0 else "HIGH"
        return render_template('stockout_result.html', result=result)

    except Exception as e:
        return f"Error in Stockout Prediction: {str(e)}"

# ---------------------- Supplier Performance ----------------------
@app.route('/supplier')
def supplier():
    return render_template('supplier.html', dealer_ids=dealer_ids)

@app.route('/supplier_result', methods=['POST'])
def supplier_result():
    try:
        dealer_id = request.form.get('dealer_id', '')
        on_time_rate = round(float(request.form.get('on_time_rate', 0)), 2)
        defect_rate = round(float(request.form.get('defect_rate', 0)), 2)
        lead_time = round(float(request.form.get('lead_time', 0)), 2)

        input_data = pd.DataFrame([{
            'OnTimeRate': on_time_rate,
            'DefectRate': defect_rate,
            'LeadTime': lead_time
        }])

        prediction = supplier_model.predict(input_data)[0]
        performance = supplier_encoder.inverse_transform([prediction])[0]

        return render_template('supplier_result.html', dealer_id=dealer_id, result=performance)

    except Exception as e:
        return f"Error in Supplier Evaluation: {str(e)}"

# ---------------------- Reorder Point Module ----------------------
@app.route("/reorder")
def reorder_home():
    return render_template("reorder_point_form.html")

@app.route("/reorder_result", methods=["POST"])
def reorder_result():
    try:
        avg_daily_demand = float(request.form["avg_daily_demand"])
        lead_time = float(request.form["lead_time"])
        demand_std_dev = request.form.get("demand_std_dev")
        std_dev = float(demand_std_dev) if demand_std_dev else None

        calculator = ReorderPointCalculator()
        result = calculator.calculate(avg_daily_demand, lead_time, std_dev)

        return render_template("reorder_point_result.html", **result)

    except Exception as e:
        return f"Error in Reorder Point Calculation: {str(e)}"

# ---------------------- Run Flask App ----------------------
if __name__ == '__main__':
    app.run(debug=True)

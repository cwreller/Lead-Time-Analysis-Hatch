from flask import Flask, request, render_template
import pandas as pd
from datetime import timedelta
import joblib
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load the trained pipeline and test data
pipeline = joblib.load('pipeline.pkl')
X_test, y_test = joblib.load('test_data.pkl')

# Recalculate the MAE
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

default_values = {
    'Product Names': 'Rest PLUS 2nd gen',
    'Departure Port': 'default_port',
    'Destination Name': 'default_destination',
    'Origin Actual Cargo Ready Date': pd.to_datetime('2023-01-01')
}

def predict_arrival_date(order_details, pipeline, default_values, confidence_interval=0.85):
    # Extract input details with default values
    product_name = order_details.get('Product Names', default_values['Product Names'])
    departure_port = order_details.get('Departure Port', default_values['Departure Port'])
    destination_name = order_details.get('Destination Name', default_values['Destination Name'])
    cargo_ready_date = pd.to_datetime(order_details.get('Origin Actual Cargo Ready Date', default_values['Origin Actual Cargo Ready Date']))

    # Create a dataframe for the new input
    new_data = pd.DataFrame({
        'Product Names': [product_name],
        'Departure Port': [departure_port],
        'Destination Name': [destination_name],
        'Cargo Ready Month': [cargo_ready_date.month],
        'Cargo Ready Day': [cargo_ready_date.day],
        'Cargo Ready DayOfWeek': [cargo_ready_date.dayofweek]
    })

    # Predict the lead time
    predicted_lead_time = int(round(pipeline.predict(new_data)[0]))
    z_score = norm.ppf((1 + confidence_interval) / 2)
    error_margin = z_score * mae

    estimated_arrival_date = cargo_ready_date + timedelta(days=predicted_lead_time)
    lower_bound = cargo_ready_date + timedelta(days=predicted_lead_time - error_margin)
    upper_bound = cargo_ready_date + timedelta(days=predicted_lead_time + error_margin)

    # Format the dates
    estimated_arrival_date_str = estimated_arrival_date.strftime('%B %d, %Y')
    lower_bound_str = lower_bound.strftime('%B %d, %Y')
    upper_bound_str = upper_bound.strftime('%B %d, %Y')

    return predicted_lead_time, estimated_arrival_date_str, lower_bound_str, upper_bound_str, confidence_interval * 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    order_details = {
        'Product Names': request.form['product_name'],
        'Departure Port': request.form['departure_port'],
        'Destination Name': request.form['destination_name'],
        'Origin Actual Cargo Ready Date': request.form['cargo_ready_date']
    }
    predicted_lead_time, estimated_arrival_date, lower_bound, upper_bound, confidence_percentage = predict_arrival_date(order_details, pipeline, default_values)
    return render_template('index.html', predicted_lead_time=predicted_lead_time, estimated_arrival_date=estimated_arrival_date, lower_bound=lower_bound, upper_bound=upper_bound, confidence_percentage=confidence_percentage, product_name=order_details['Product Names'], departure_port=order_details['Departure Port'], destination_name=order_details['Destination Name'], cargo_ready_date=order_details['Origin Actual Cargo Ready Date'])

@app.route('/map')
def map():
    return render_template('map.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
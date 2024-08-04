from flask import Flask, request, jsonify
from waitress import serve
import util
import logging
import pickle

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    locations = util.get_location_names()
    logging.debug(f"Locations: {locations}")  # Locations değerini logla
    response = jsonify({
        'locations': locations
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    total_sqft=float(request.form['total_sqft'])
    location=request.form['location']
    bhk=int(request.form['bhk'])
    bath=int(request.form['bath'])
    response=jsonify({
        'estimated_price':util.get_estimated_price(location,total_sqft,bhk,bath)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    print("Starting Python Flask Server for Home Price Prediction")
    logging.debug("Loading artifacts...")
    util.load_saved_artifacts()  # Artifacts yükle
    #logging.debug(f"Data columns: {util.__data_columns}")
    logging.debug(f"Locations: {util.get_location_names()}")
    serve(app, host='0.0.0.0', port=5000)

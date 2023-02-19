# Importing Required Modules
from flask import Flask, render_template, request, Markup,jsonify
import numpy as np
import pandas as pd
import requests
import pickle
import io
from pprint import pprint

app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))

# Importing the trained picle model

crop_recommendation_model_path = 'rf_model.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

# Function to fetch latitude and longitude of given city

def fetch_coordinates(city_name,state_name,country_name):
    """
    Fetch and returns the latitude,longitude of a city
    :params: city_name,state_name,country_name
    :return: latitude,longitude
    """
    api_key = open('geo_api_key.txt','r').read()

    city = city_name
    state = state_name
    country = country_name    


    urlx = "http://api.openweathermap.org/geo/1.0/direct?q={city}+,{state},{country}&limit=1&appid={api_key}".format(city=city,state=state,country=country,api_key=api_key)
    geo_api_response = requests.get(urlx).json()
    latitude = geo_api_response[0]['lat']
    longitude =geo_api_response[0]['lon']
    return latitude,longitude

# Function to fetch weather given latitude and longitude

def fetch_weather(latitude,longitude):
    """
    Fetch and returns the temperature and humidity of cordinates
    params : latitude,longitude
    :return: temperature, humidity
    """
    
    api_key = open('weather_api_key.txt','r').read()

    lat =latitude
    lon = longitude

    urlx = "https://api.agromonitoring.com/agro/1.0/weather/forecast?lat={lat}&lon={lon}&appid={api_key}".format(api_key=api_key,lat=lat,lon=lon)
    weather_api_response = requests.get(urlx).json()

    humidity    = weather_api_response[1]['main']['humidity']
    temperature_kelvin = weather_api_response[1]['main']['temp']
    temperature_celsius = temperature_kelvin - 273.15

    return humidity,temperature_celsius


@app.route('/')
def home():
    return render_template('crop.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    
    '''
    For rendering results on HTML GUI
    '''
    if request.method == 'POST':
        N = int(request.form['Nitrogen'])
        P = int(request.form['Phosphorous'])
        K = int(request.form['Pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        
        # Calling the fetch_coordinates and fetch_weather functions

        lat,lon = fetch_coordinates(city,state,country)
        humidity,temperature=fetch_weather(lat,lon)

        data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        my_prediction = model.predict(data)
        final_prediction = my_prediction[0]
        
        return render_template('crop.html', prediction_text='The Best suitable crop is {}'.format(final_prediction), title='Harvestify - Home')


if __name__ == "__main__":
    app.run(debug=True)

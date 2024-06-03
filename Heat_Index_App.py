import datetime
import math
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model


@st.cache_resource
def load_lstm_model():
    return load_model('lstm_model_st.keras')

saved_lstm_model = load_lstm_model()

def classify_value(value, text):
    if 27 < value < 33:
        return f':yellow[{text}]'
    elif 33 < value < 42:
        return f':orange[{text}]'
    elif 42 < value < 52:
        return f':red[{text}]'
    elif value > 52:
        return f':violet[{text}]'
    return text 


st.title("Heat Index Forecast")

# date_str = '2023-04-13'
# date_today = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()

# date_str = '2024-05-20' # user input date today
# date_today = datetime.datetime.strptime(date_str, '%Y-%m-%d').date()


date_today = st.date_input("Date Today", datetime.date.today())

def generate_epoch_times(given_date):
    given_date = datetime.datetime.combine(given_date, datetime.datetime.min.time())
    return [(given_date + datetime.timedelta(days=i) - datetime.datetime(1970, 1, 1)).total_seconds() for i in range(-7, 7)]

epoch_times_list = generate_epoch_times(date_today)

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def heat_index_celsius(temperature_celsius, relative_humidity):
    temperature = celsius_to_fahrenheit(temperature_celsius)
    hi_simple = 0.5 * (temperature + 61.0 + ((temperature - 68.0) * 1.2) + (relative_humidity * 0.094))
    if hi_simple >= 80:

            hi = (-42.379 +
                2.04901523 * temperature +
                10.14333127 * relative_humidity -
                0.22475541 * temperature * relative_humidity -
                0.00683783 * temperature**2 -
                0.05481717 * relative_humidity**2 +
                0.00122874 * temperature**2 * relative_humidity +
                0.00085282 * temperature * relative_humidity**2 -
                0.00000199 * temperature**2 * relative_humidity**2)

     
            if relative_humidity < 13 and temperature >= 80 and temperature <= 112:
                adjustment = ((13 - relative_humidity) / 4) * math.sqrt((17 - abs(temperature - 95.0)) / 17)
                hi -= adjustment

            
            elif relative_humidity > 85 and temperature >= 80 and temperature <= 87:
                adjustment = ((relative_humidity - 85) / 10) * ((87 - temperature) / 5)
                hi += adjustment

            return (hi - 32) * 5/9

    else:
            return (hi_simple - 32) * 5/9
        
        
def postprocess_temp(temp):
    return temp * apparent_temp_training_std + apparent_temp_training_mean

def postprocess_rh(rh):
    return rh * rel_humidity_std + rel_humidity_training_mean

apparent_temp_training_mean = 26.208763
apparent_temp_training_std = 1.451434
rel_humidity_training_mean = 79.419437
rel_humidity_std = 7.419842

# Use provided inputs
# AT_inputs = [32, 32, 31, 32, 32, 30, 29]
# RH_inputs = [61, 61, 62, 62, 63, 81, 90]
# AT_inputs = [37.7, 35.8, 37.8, 35.7, 34.4, 37, 33.9]  # try other inputs
# RH_inputs = [33, 44, 37, 44, 52, 41, 56]  # try other inputs
st.write("### Input Apparent Temperatures (AT)")

with st.form("AT_form"):
    AT_inputs = [st.number_input(f"{i} day/s ago (AT)", key=f"AT_{i}") for i in range(1, 8)]
    at_submit_button = st.form_submit_button(label='Submit')


st.write("### Input Relative Humidities (RH)")

with st.form("RH_form"):
    RH_inputs = [st.number_input(f"{i} day/s ago (RH)", min_value=0, max_value=100, key=f"RH_{i}") for i in range(1, 8)]
    rh_submit_button = st.form_submit_button(label='Submit')
    

if 'heat_indices' not in st.session_state:
    st.session_state.heat_indices = [0] * 7

if st.button("Forecast"):
    with st.spinner('Forecasting...'):
        apparent_temperatures = AT_inputs
        relative_humidities = RH_inputs
        year_sin = np.sin(2 * np.pi * np.array(epoch_times_list) / (365.2425 * 24 * 3600))
        year_cos = np.cos(2 * np.pi * np.array(epoch_times_list) / (365.2425 * 24 * 3600))

        user_input_past_temp = np.array([
            [apparent_temperatures[6], relative_humidities[6], year_sin[0], year_cos[0]],
            [apparent_temperatures[5], relative_humidities[5], year_sin[1], year_cos[1]],
            [apparent_temperatures[4], relative_humidities[4], year_sin[2], year_cos[2]],
            [apparent_temperatures[3], relative_humidities[3], year_sin[3], year_cos[3]],
            [apparent_temperatures[2], relative_humidities[2], year_sin[4], year_cos[4]],
            [apparent_temperatures[1], relative_humidities[1], year_sin[5], year_cos[5]],
            [apparent_temperatures[0], relative_humidities[0], year_sin[6], year_cos[6]]
        ]).reshape(1, 7, 4)

        forecast_app_temp = []
        forecast_rel_hum = []
        heat_indices = []

        for i in range(7):
            predictions = saved_lstm_model(user_input_past_temp)
            app_temp_and_rel_hum = predictions[0].numpy()
            apparent_temperature = postprocess_temp(app_temp_and_rel_hum[0])
            relative_humidity = postprocess_rh(app_temp_and_rel_hum[1])
            forecast_app_temp.append(apparent_temperature)
            forecast_rel_hum.append(relative_humidity)
            prediction = np.array([app_temp_and_rel_hum[0], app_temp_and_rel_hum[1], year_sin[i + 7], year_cos[i + 7]])
            heat_indices.append(heat_index_celsius(apparent_temperature, relative_humidity))
            user_input_past_temp = np.delete(user_input_past_temp, 0, axis=1)
            user_input_past_temp = np.append(user_input_past_temp, prediction).reshape(1, 7, 4)

        st.session_state.heat_indices = heat_indices

   
        if len(st.session_state.heat_indices) >= 7:
            st.write("### Results")
            rcol1, rcol2, rcol3, rcol4 = st.columns(4)
            rcol5, rcol6, rcol7, _ = st.columns(4)

            rcol1.metric(label=classify_value(st.session_state.heat_indices[0], "Today"), value=f"{st.session_state.heat_indices[0]:,.2f} °C")
            rcol2.metric(label=classify_value(st.session_state.heat_indices[1], "Tomorrow"), value=f"{st.session_state.heat_indices[1]:,.2f} °C")
            rcol3.metric(label=classify_value(st.session_state.heat_indices[2], "2 days from today"), value=f"{st.session_state.heat_indices[2]:,.2f} °C")
            rcol4.metric(label=classify_value(st.session_state.heat_indices[3], "3 days from today"), value=f"{st.session_state.heat_indices[3]:,.2f} °C")
            rcol5.metric(label=classify_value(st.session_state.heat_indices[4], "4 days from today"), value=f"{st.session_state.heat_indices[4]:,.2f} °C")
            rcol6.metric(label=classify_value(st.session_state.heat_indices[5], "5 days from today"), value=f"{st.session_state.heat_indices[5]:,.2f} °C")
            rcol7.metric(label=classify_value(st.session_state.heat_indices[6], "6 days from today"), value=f"{st.session_state.heat_indices[6]:,.2f} °C")
        else:
            st.error("Forecasting did not produce enough heat index values. Please try again.")

        st.write(f"Heat Index Forecast: {heat_indices}")
        st.write(f"Apparent Temperature Forecast: {forecast_app_temp}")
        st.write(f"Relative Humidity Forecast: {forecast_rel_hum}")

if st.button("Reset"):
    st.session_state.heat_indices = [0] * 7


hi_classes = {
    'Class': ['Caution', 'Extreme Caution', 'Danger', 'Extreme Danger'],
    'Description': [
        'Heat index values between 27°C and 32°C. Fatigue is possible with prolonged exposure and/or physical activity.',
        'Heat index values between 33°C and 41°C. Heat cramps and heat exhaustion are possible with prolonged exposure and/or physical activity.',
        'Heat index values between 42°C and 51°C. Heatstroke, heat cramps, and heat exhaustion are likely with prolonged exposure and/or physical activity.',
        'Heat index values above 52°C. Heatstroke is highly likely with continued exposure.'
    ]
}


hi_df = pd.DataFrame(hi_classes)


def color_negative_red(val):
    if val == 'Caution':
        return 'color: yellow'
    elif val == 'Extreme Caution':
        return 'color: #E6CC00'
    elif val == 'Danger':
        return 'color: orange'
    elif val == 'Extreme Danger':
        return 'color: red'


styled_df = hi_df.style.map(color_negative_red)


st.table(styled_df)

import joblib
import streamlit as st
import pandas as pd

try: 
    model = joblib.load('xgb_model2.pkl')
    st.write('Model Loaded Successfully')
except Exception as e:
    st.write(f"Error Loading Model: {e}")
    st.stop()

claim_mapping = {
    1: "Yes", 
    0: "No"
}

def main():
    st.title('Claim Predictor')
    #['policy_tenure', 'age_of_car', 'age_of_policyholder','area_cluster', 'population_density', 'make', 'segment', 'model','fuel_type', 'max_torque', 'max_power', 'engine_type', 'airbags','is_esc', 'is_adjustable_steering', 'is_tpms', 'is_parking_sensors','is_parking_camera', 'rear_brakes_type', 'displacement', 'cylinder','transmission_type', 'gear_box', 'steering_type', 'turning_radius','length', 'width', 'height', 'gross_weight', 'is_front_fog_lights','is_rear_window_wiper', 'is_rear_window_washer','is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks','is_central_locking', 'is_power_steering','is_driver_seat_height_adjustable', 'is_day_night_rear_view_mirror','is_ecw', 'is_speed_alert', 'ncap_rating']
    p_tenure = st.number_input('Policy Tenure', min_value=0, max_value=1)
    age_car = st.number_input('Age of Car', min_value=0, max_value=1)
    age_holder = st.number_input('Age of Policy Holder', min_value=0, max_value=1)
    pop_density = st.number_input('Population Density', min_value=200, max_value=100000)
    make  = st.number_input('Make', min_value=1, max_value=5)
    torque = st.number_input('Max Torque', min_value=0, max_value=500)
    torque_rpm = st.number_input('Torque RPM', min_value=0, max_value=5000)
    power = st.number_input('Power', min_value=0, max_value=200)
    power_rpm = st.number_input('Power RPM', min_value=0, max_value=10000)
    airbags = st.number_input('Airbags', min_value=1, max_value=10)     
    ncap = st.number_input('NCAP Rating', min_value=0, max_value=5)
    disp = st.number_input('Displacement', min_value=100, max_value=2000)
    cyl = st.number_input('Cylinders', min_value=3, max_value=4)
    gear_box = st.number_input('Gear Box', min_value=5, max_value=6)
    turning_rad = st.number_input('Turning Radius', min_value=0, max_value=10)
    length = st.number_input('Length', min_value=2000, max_value=5000)
    width = st.number_input('Width', min_value=1000, max_value=2000)
    height = st.number_input('Height', min_value=1000, max_value=2000)
    weight = st.number_input('Gross Weight', min_value=1000, max_value=2000)

    area_cluster = st.selectbox('Area Cluster', ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21', 'C22'])
    segment = st.selectbox('Segment', ['A', 'B1', 'B2', 'C1', 'C2', 'Utility'])
    mod = st.selectbox('Model', ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11'])
    eng_type = st.selectbox('Engine Type', ['1.0 SCe', '1.2 L K Series Engine', '1.2 L K12N Dualjet', '1.5 L U2 CRDi', '1.5 Turbocharged Revotorq', '1.5 Turbocharged Revotron', 'F8D Petrol Engine', 'G12B', 'K Series Dual jet', 'K10C', 'i-DTEC'])
    esc = st.selectbox('Escapable?', [True, False])
    adj_str = st.selectbox('Adjustable Steering?', [True, False])
    tpms = st.selectbox('TPMS?', [True, False])
    par_sen = st.selectbox('Parking Sensors?', [True, False])
    par_cam = st.selectbox('Parking Camera?', [True, False])
    front_fog = st.selectbox('Front Fog Lights?', [True, False])
    rear_wiper = st.selectbox('Rear Window Wiper?', [True, False])
    rear_washer = st.selectbox('Rear Window Washer?', [True, False])
    rear_defogger = st.selectbox('Rear Window Defogger?', [True, False])
    brake_ass = st.selectbox('Brake Assist?', [True, False])
    power_lock = st.selectbox('Power Door Lock?', [True, False])
    cen_lock = st.selectbox('Central Locking?', [True, False])
    pow_steer = st.selectbox('Power Steering?', [True, False])
    driver_adj = st.selectbox('Driver Seat Height Adjustable?', [True, False])
    day_night = st.selectbox('Day Night Rear View Mirror?', [True, False])
    ecw = st.selectbox('ECW?', [True, False])
    speed_al = st.selectbox('Speed Alert?', [True, False])

    fuels = ['CNG', 'Diesel', 'Petrol']
    fuel_type = st.selectbox('Fuel Type', fuels)
    one_hot_encoded_fuel = [0] * len(fuels)
    fuel_index = fuels.index(fuel_type)
    one_hot_encoded_fuel[fuel_index] = 1

    transmissions = ['Automatic', 'Manual']
    trans_type = st.selectbox('Transmission Type', transmissions)
    one_hot_encoded_trans = [0] * len(transmissions)
    trans_index = transmissions.index(trans_type)
    one_hot_encoded_trans[trans_index] = 1

    steerings = ['Electric', 'Manual', 'Power']
    steer_type = st.selectbox('Steering Type', steerings)
    one_hot_encoded_steer = [0] * len(steerings)
    steer_index = steerings.index(steer_type)
    one_hot_encoded_steer[steer_index] = 1

    rear_brakes = ['Disc', 'Drum']
    rear_type = st.selectbox('Rear Brakes Type', rear_brakes)
    one_hot_encoded_rear = [0] * len(rear_brakes)
    rear_index = rear_brakes.index(rear_type)
    one_hot_encoded_rear[rear_index] = 1

    data = pd.DataFrame({
        'policy_tenure': [p_tenure], 
        'age_of_car': [age_car], 
        'age_of_policyholder': [age_holder],
        'area_cluster': [{'C1':1, 'C2':2, 'C3':3, 'C4':4, 'C5':5, 'C6':6, 'C7':7, 'C8':8, 'C9':9, 'C10':10, 'C11':11, 'C12':12, 'C13':13, 'C14':14, 'C15':15, 'C16':16, 'C17':17, 'C18':18, 'C19':19, 'C20':20, 'C21':21, 'C22':22}[area_cluster]],
        'population_density': [pop_density], 
        'make': [make], 
        'segment': [{'A': 0, 'B1': 1, 'B2': 2, 'C1': 3, 'C2': 4, 'Utility': 5}[segment]], 
        'model': [{'M1':1, 'M2':2, 'M3':3, 'M4':4, 'M5':5, 'M6':6, 'M7':7, 'M8':8, 'M9':9, 'M10':10, 'M11':11}[mod]],
        'engine_type': [{'1.0 SCe': 0, '1.2 L K Series Engine': 1, '1.2 L K12N Dualjet': 2, '1.5 L U2 CRDi': 3, '1.5 Turbocharged Revotorq': 4, '1.5 Turbocharged Revotron': 5, 'F8D Petrol Engine': 6, 'G12B': 7, 'K Series Dual jet': 8, 'K10C': 9, 'i-DTEC': 10}[eng_type]],
        'airbags': [airbags],
        'is_esc': [esc],
        'is_adjustable_steering': [adj_str], 
        'is_tpms': [tpms],
        'is_parking_sensors': [par_sen], 
        'is_parking_camera': [par_cam], 
        'displacement': [disp], 
        'cylinder': [cyl],
        'gear_box': [gear_box], 
        'turning_radius': [turning_rad], 
        'length': [length], 
        'width': [width], 
        'height': [height],
        'gross_weight': [weight], 
        'is_front_fog_lights': [front_fog], 
        'is_rear_window_wiper': [rear_wiper],
        'is_rear_window_washer': [rear_washer],
        'is_rear_window_defogger': [rear_defogger], 
        'is_brake_assist': [brake_ass],
        'is_power_door_locks': [power_lock], 
        'is_central_locking': [cen_lock], 
        'is_power_steering': [pow_steer],
        'is_driver_seat_height_adjustable': [driver_adj], 
        'is_day_night_rear_view_mirror': [day_night],
        'is_ecw': [ecw], 
        'is_speed_alert': [speed_al], 
        'ncap_rating': [ncap],
        'fuel_type_CNG': [one_hot_encoded_fuel[0]],
        'fuel_type_Diesel': [one_hot_encoded_fuel[1]],
        'fuel_type_Petrol': [one_hot_encoded_fuel[2]], 
        'steering_type_Electric': [one_hot_encoded_steer[0]],
        'steering_type_Manual': [one_hot_encoded_steer[1]], 
        'steering_type_Power': [one_hot_encoded_steer[2]],
        'transmission_type_Automatic': [one_hot_encoded_trans[0]], 
        'transmission_type_Manual': [one_hot_encoded_trans[1]],
        'rear_brakes_type_Disc': [one_hot_encoded_rear[0]], 
        'rear_brakes_type_Drum': [one_hot_encoded_rear[1]], 
        'torque': [torque],
        'torque_rpm': [torque_rpm], 
        'power': [power], 
        'power_rpm': [power_rpm]
    })

    st.write('Input Data for Predictions: ')
    st.write(data)

    if st.button('Predict'):
        try:
            prediction = model.predict(data)
            claim = claim_mapping.get(prediction[0], 'Unknown Claim Status')
            st.write(f'Predicted Claim: {claim}')
        except Exception as e:
            st.write(f"Error during prediction: {e}")

if __name__ == '__main__':
    main()
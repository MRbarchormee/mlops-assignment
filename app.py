import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model

# -----------------------------
# 🔧 CONFIG
# -----------------------------
MODEL_PATH = 'car_price_pipeline'  # match your save_model() name
PAGE_TITLE = 'Car Price Prediction (PyCaret)'

# -----------------------------
# 🧠 LOAD PIPELINE (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

# -----------------------------
# 🖥️ UI
# -----------------------------
st.set_page_config(page_title=PAGE_TITLE, page_icon='🚗', layout='centered')
st.title(PAGE_TITLE)
st.caption('Enter car details below and click **Predict** to get an estimated price (in 10k INR units).')

# -----------------------------
# 📝 INPUTS
# -----------------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    kilometers = st.number_input('Kilometers_Driven', min_value=0, step=100, value=50000)
    mileage = st.number_input('Mileage', min_value=0.0, step=0.1, value=18.0)
    engine = st.number_input('Engine', min_value=0, step=100, value=1200)

with col2:
    power = st.number_input('Power', min_value=0.0, step=0.01, value=80.0)
    year = st.number_input('Year', min_value=1990, max_value=2025, value=2015)
    seats = st.number_input('Seats', min_value=2, max_value=10, value=5)

with col3:
    fuel_type = st.selectbox('Fuel_Type', ['Petrol', 'Diesel', 'CNG', 'LPG', 'Electric'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic'])
    location = st.selectbox('Location', ['Mumbai', 'Chennai', 'Hyderabad', 'Jaipur', 'Pune', 'Kolkata', 'Delhi', 'Kochi', 'Bangalore', 'Coimbatore', 'Ahmedabad'])
    brand = st.selectbox('Brand', 
                        ['Ambassador',
                        'Audi',
                        'BMW',
                        'Chevrolet',
                        'Datsun',
                        'Fiat',
                        'Force',
                        'Ford',
                        'Honda',
                        'Hyundai',
                        'ISUZU',
                        'Jeep',
                        'Land',
                        'Mahindra',
                        'Maruti',
                        'Mercedes-Benz',
                        'Mini',
                        'Mitsubishi',
                        'Nissan',
                        'Renault',
                        'Skoda',
                        'Smart',
                        'Tata',
                        'Toyota',
                        'Volkswagen',
                        'Volvo'])

with col4:
    owner_type = st.selectbox('Owner_Type', ['First', 'Second', 'Third', 'Fourth & Above'])

# -----------------------------
# 🧮 PREDICT
# -----------------------------
if st.button('Predict Price'):
    input_df = pd.DataFrame([{
        'Kilometers_Driven': kilometers,
        'Mileage': mileage,
        'Engine': engine,
        'Power': power,
        'Year': int(year),
        'Seats': int(seats),
        'Fuel_Type': fuel_type,
        'Transmission': transmission,
        'Location': location,
        'Owner_Type': owner_type,
        'Brand': brand
    }])

    pred_df = predict_model(model, data=input_df)

    # PyCaret regression adds 'Label' for predicted values
    predicted_price = pred_df.loc[0, 'prediction_label']
    # actual_price = predicted_price * 1000

    st.subheader('✅ Prediction')
    st.metric(label='Estimated Price in INR', value=f"{predicted_price:,.2f}")

    with st.expander('See full transformed prediction row'):
        st.dataframe(pred_df, use_container_width=True)

# -----------------------------
# 🧪 BATCH PREDICTION
# -----------------------------
st.markdown('---')
st.subheader('Batch Prediction (CSV)')
st.caption('Upload a CSV with the same feature columns used during training.')

csv_file = st.file_uploader('Upload CSV', type=['csv'])
if csv_file is not None:
    batch_df = pd.read_csv(csv_file)

    # Ensure Brand column exists if CSV has Brand_Model
    if 'Brand_Model' in batch_df.columns:
        batch_df['Brand'] = batch_df['Brand_Model'].str.split(' ', 1).str[0]

    if st.button('Predict on CSV'):
        batch_preds = predict_model(model, data=batch_df)
        st.success('Predictions generated!')
        st.dataframe(batch_preds, use_container_width=True)

        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label='Download Predictions CSV',
            data=convert_df(batch_preds),
            file_name='predictions.csv',
            mime='text/csv'
        )

# -----------------------------
# ❓ HELP / NOTES
# -----------------------------
# st.markdown("""
# **Notes**
# - Features must match the PyCaret experiment exactly (Brand instead of Brand_Model, numerical/categorical preprocessing handled in pipeline).
# - Price is in INR units.
# """)

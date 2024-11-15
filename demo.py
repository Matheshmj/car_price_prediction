import streamlit as st
import pandas as pd
import pickle

# Load the previously saved model pipeline
with open('xgb_model.pkl', 'rb') as model_file:
    pipeline = pickle.load(model_file)

# Load the encoder and scaler used during training
with open('encoder.pkl', 'rb') as enc_file:
    encoder = pickle.load(enc_file)

with open('scaler.pkl', 'rb') as scale_file:
    scaler = pickle.load(scale_file)

# App Title and Header
st.set_page_config(page_title="Car Price Prediction", layout="wide", page_icon="🚗")
st.title("🚗 Car Price Prediction App")
st.markdown(
    """
    Welcome to the **Car Price Prediction App**!  
    Fill in the details below to get an estimate of your car's price.  
    """)

# Sidebar for additional instructions
with st.sidebar:
    st.header("💡 Instructions")
    st.markdown(
        """
        - Select the relevant features of your car.
        - Click on **Predict Price** to see the result.
        - All fields are required for accurate prediction.
        """
    )
    st.write("")

# Layout: Two columns for input fields
col1, col2 = st.columns(2)

with col1:
    body_type = st.selectbox("Body Type", ['SUV', 'Sedan', 'Hatchback', 'Coupe', 'Convertible', 'Wagon'])
    brand = st.selectbox("Brand", ['Jeep', 'Toyota', 'Honda', 'BMW', 'Ford', 'Mercedes'])
    model = st.text_input("Model", "Jeep Compass")  # Text input for model
    model_year = st.number_input("Model Year", min_value=1900, max_value=2024, value=2020)
    kms_driven = st.number_input("Kms Driven", min_value=0, value=20000)

with col2:
    fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Petrol', 'Electric', 'CNG', 'Hybrid'])
    transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])
    max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=170.0)
    torque = st.number_input("Torque (Nm)", min_value=0.0, value=350.0)
    mileage = st.number_input("Mileage (km/l)", min_value=0.0, value=17.1)
    engine = st.number_input("Engine (cc)", min_value=0, value=1956)

# Additional inputs below
st.subheader("Additional Information")
col3, col4 = st.columns(2)

with col3:
    owner_no = st.number_input("Number of Owners", min_value=1, value=1)
    insurance_validity = st.selectbox("Insurance Validity", ['Third Party insurance', 'Comprehensive insurance', 'None'])

with col4:
    seats = st.number_input("Seats", min_value=1, value=5)
    city = st.selectbox("City", ['Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata'])

# Button to predict price
st.markdown("---")
if st.button("💰 Predict Price"):
    # Create the DataFrame for prediction
    new_data = {
        'body type': [body_type],
        'Brand': [brand],
        'model': [model],
        'modelYear': [model_year],
        'Kms Driven': [kms_driven],
        'Fuel Type': [fuel_type],
        'Transmission': [transmission],
        'Max Power': [max_power],
        'Torque': [torque],
        'Mileage': [mileage],
        'Engine': [engine],
        'ownerNo': [owner_no],
        'Insurance Validity': [insurance_validity],
        'Seats': [seats],
        'City': [city]
    }

    # Convert new_data to a DataFrame
    new_df = pd.DataFrame(new_data)

    # Match training column names
    categorical_cols = ['body type', 'Brand', 'model', 'Fuel Type', 'Transmission', 'Insurance Validity', 'City']
    numerical_cols = [col for col in new_df.columns if col not in categorical_cols]

    # Convert categorical columns to 'category' dtype
    for col in categorical_cols:
        if col in new_df.columns:
            new_df[col] = new_df[col].astype('category')

    # Apply encoder to categorical columns and ensure correct shape
    encoded_data = encoder.transform(new_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # Replace original categorical columns with the encoded ones
    new_df = pd.concat([new_df[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # Apply scaler to numerical columns
    scaled_data = scaler.transform(new_df)
    scaled_df = pd.DataFrame(scaled_data, columns=new_df.columns)

    # Predict the price using the loaded pipeline
    try:
        predicted_price = pipeline.predict(scaled_df)

        # Show the predicted price with visualization
        st.success(f"✨ Predicted Price of the car(in lakhs): ₹{predicted_price[0]:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")

# Footer
st.markdown("---")

# import streamlit as st
# import joblib
# import pandas as pd

# model = joblib.load('Ensemble (Multi-Layer Perceptrons, XGBoost, Random Forest).pkl')  

# st.title('Prediksi pH Tanah')

# st.write("""
#     Masukkan nilai-nilai fitur tanah untuk memprediksi nilai pH tanah.
# """)

# P = st.number_input("P", min_value=0.0, max_value=100.0, value=10.0)
# SAND = st.number_input("SAND", min_value=0.0, max_value=100.0, value=30.0)
# CLAY = st.number_input("CLAY", min_value=0.0, max_value=100.0, value=30.0)
# N = st.number_input("N", min_value=0.0, max_value=100.0, value=5.0)
# K = st.number_input("K", min_value=0.0, max_value=100.0, value=2.0)
# Ca = st.number_input("Ca", min_value=0.0, max_value=100.0, value=10.0)
# Mg = st.number_input("Mg", min_value=0.0, max_value=100.0, value=5.0)
# Na = st.number_input("Na", min_value=0.0, max_value=100.0, value=1.0)
# CEC = st.number_input("CEC", min_value=0.0, max_value=100.0, value=25.0)
# SAR = st.number_input("SAR", min_value=0.0, max_value=100.0, value=1.0)
# ESP = st.number_input("ESP", min_value=0.0, max_value=100.0, value=5.0)

# total_elements = Ca + Mg + K
# if total_elements > 0:
#     perc_Ca = (Ca / total_elements) * 100
#     perc_Mg = (Mg / total_elements) * 100
#     perc_K = (K / total_elements) * 100
# else:
#     perc_Ca = perc_Mg = perc_K = 0.0

# st.write(f"% Ca: {perc_Ca:.2f}")
# st.write(f"% Mg: {perc_Mg:.2f}")
# st.write(f"% K: {perc_K:.2f}")

# input_data = pd.DataFrame([[P, SAND, CLAY, N, K, Ca, Mg, Na, CEC, SAR, ESP, perc_Ca, perc_Mg, perc_K]],
#                           columns=["P", "SAND", "CLAY", "N", "K", "Ca", "Mg", "Na", "CEC", "SAR", "ESP", "% Ca", "% Mg", "% K"])

# input_data = input_data[model.feature_names_in_]  # Menyesuaikan dengan nama fitur yang digunakan saat pelatihan

# if st.button('Prediksi pH'):
#     pH_pred = model.predict(input_data)
    
#     pH_pred_rounded = round(pH_pred[0])  # Membulatkan angka prediksi

#     ph_labels = {
#         0: 'Strongly acidic',
#         1: 'Moderately acidic',
#         2: 'Neutral',
#         3: 'Moderately alkaline',
#         4: 'Strongly alkaline'
#     }

#     pH_class = ph_labels.get(pH_pred_rounded, 'Unknown')  # Default ke 'Unknown' jika prediksi tidak ada dalam mapping

#     st.write(f"Prediksi pH tanah: {pH_class}")

import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load('Ensemble (Multi-Layer Perceptrons, XGBoost, Random Forest).pkl')

# Streamlit app title and custom styling
st.title('🌱 Prediksi pH Tanah')
st.markdown("""
    <style>
    .title {
        font-size: 36px;
        color: #2e8b57;
        font-family: 'Arial';
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Masukkan nilai-nilai fitur tanah untuk memprediksi nilai pH tanah.</p>', unsafe_allow_html=True)

# Input form with columns for better layout
col1, col2 = st.columns(2)

with col1:
    P = st.number_input("P (Phosphorus)", min_value=0.0, max_value=100.0, value=10.0)
    SAND = st.number_input("SAND (%)", min_value=0.0, max_value=100.0, value=30.0)
    CLAY = st.number_input("CLAY (%)", min_value=0.0, max_value=100.0, value=30.0)
    N = st.number_input("N (Nitrogen)", min_value=0.0, max_value=100.0, value=5.0)
    K = st.number_input("K (Potassium)", min_value=0.0, max_value=100.0, value=2.0)
    Ca = st.number_input("Ca (Calcium)", min_value=0.0, max_value=100.0, value=10.0)

with col2:
    Mg = st.number_input("Mg (Magnesium)", min_value=0.0, max_value=100.0, value=5.0)
    Na = st.number_input("Na (Sodium)", min_value=0.0, max_value=100.0, value=1.0)
    CEC = st.number_input("CEC (Cation Exchange Capacity)", min_value=0.0, max_value=100.0, value=25.0)
    SAR = st.number_input("SAR (Sodium Adsorption Ratio)", min_value=0.0, max_value=100.0, value=1.0)
    ESP = st.number_input("ESP (Exchangeable Sodium Percentage)", min_value=0.0, max_value=100.0, value=5.0)

# Calculating the percentage of elements
total_elements = Ca + Mg + K
if total_elements > 0:
    perc_Ca = (Ca / total_elements) * 100
    perc_Mg = (Mg / total_elements) * 100
    perc_K = (K / total_elements) * 100
else:
    perc_Ca = perc_Mg = perc_K = 0.0

# Display calculated percentages
st.write(f"% Ca: {perc_Ca:.2f}%")
st.write(f"% Mg: {perc_Mg:.2f}%")
st.write(f"% K: {perc_K:.2f}%")

# Prepare the input data for prediction
input_data = pd.DataFrame([[P, SAND, CLAY, N, K, Ca, Mg, Na, CEC, SAR, ESP, perc_Ca, perc_Mg, perc_K]],
                          columns=["P", "SAND", "CLAY", "N", "K", "Ca", "Mg", "Na", "CEC", "SAR", "ESP", "% Ca", "% Mg", "% K"])

input_data = input_data[model.feature_names_in_]  # Align features with model

# Prediction button
if st.button('Prediksi pH'):
    pH_pred = model.predict(input_data)
    pH_pred_rounded = round(pH_pred[0])  # Round the predicted value
    
    # Define pH class labels
    ph_labels = {
        0: 'Strongly acidic',
        1: 'Moderately acidic',
        2: 'Neutral',
        3: 'Moderately alkaline',
        4: 'Strongly alkaline'
    }

    # Get corresponding pH class
    pH_class = ph_labels.get(pH_pred_rounded, 'Unknown')

    # Display result
    st.write(f"Prediksi pH tanah: **{pH_class}**")

    # Plot a bar chart for visualizing pH prediction
    fig, ax = plt.subplots()
    ax.bar(ph_labels.values(), [1 if x == pH_class else 0 for x in ph_labels.values()])
    ax.set_title("Prediksi pH Tanah")
    ax.set_xlabel("Kategori pH")
    ax.set_ylabel("Probabilitas")
    st.pyplot(fig)

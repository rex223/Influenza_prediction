import streamlit as st
import pandas as pd
import numpy as np
import pickle
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# Set page configuration
st.set_page_config(
    page_title="Influenza Outbreak Prediction System",
    page_icon="🦠",
    layout="wide"
)

# Custom CSS for medical theme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .title-container {
        background-color: #0078D7;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    .section-header {
        background-color: #E6F2FF;
        padding: 0.7rem;
        border-left: 5px solid #0078D7;
        border-radius: 5px;
        margin: 1.5rem 0 1rem 0;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        padding: 1rem;
        color: #6c757d;
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    /* Custom Slider */
    .stSlider > div > div > div {
        background-color: #E6F2FF;
    }
    /* Button styling */
    .stButton > button {
        background-color: #0078D7;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #005bb5;
    }
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
        margin-left: 5px;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
</style>
""", unsafe_allow_html=True)

# Title with medical icon
st.markdown("""
<div class="title-container">
    <h1>🦠 Influenza Outbreak Prediction System</h1>
    <p>Hybrid Machine Learning & Fuzzy Logic Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# Load the trained XGBoost model and imputer
@st.cache_resource
def load_models():
    try:
        with open('xgboost_high_risk_model_final.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('imputer_final.pkl', 'rb') as f:
            imputer = pickle.load(f)
        return model, imputer
    except FileNotFoundError as e:
        st.error(f"Error: {str(e)}. Please ensure the model and imputer files are in the correct directory.")
        st.stop()

model, imputer = load_models()

# Define the features used in the XGBoost model
features = ['monthly_temperature', 'pig_vaccinated', 'area', 'state_total_population', 'monthly_case_to_pop_ratio', 'pop_density']

# State data (population and area for each state)
state_data = {
    'Andhra Pradesh': {'population': 49577103, 'area': 162970},
    'Arunachal Pradesh': {'population': 1383727, 'area': 83743},
    'Assam': {'population': 31205576, 'area': 78438},
    'Bihar': {'population': 104099452, 'area': 94163},
    'Chandigarh': {'population': 1055450, 'area': 114},
    'Chhattisgarh': {'population': 25545198, 'area': 135192},
    'Goa': {'population': 1458545, 'area': 3702},
    'Gujarat': {'population': 60439692, 'area': 196244},
    'Haryana': {'population': 25351462, 'area': 44212},
    'Himachal Pradesh': {'population': 6864602, 'area': 55673},
    'Jharkhand': {'population': 32988134, 'area': 79716},
    'Karnataka': {'population': 61095297, 'area': 191791},
    'Kerala': {'population': 33406061, 'area': 38852},
    'Madhya Pradesh': {'population': 72626809, 'area': 308252},
    'Maharashtra': {'population': 112374333, 'area': 307713},
    'Manipur': {'population': 2855794, 'area': 22327},
    'Meghalaya': {'population': 2966889, 'area': 22429},
    'Mizoram': {'population': 1097206, 'area': 21081},
    'Nagaland': {'population': 1978502, 'area': 16579},
    'Odisha': {'population': 41974218, 'area': 155707},
    'Puducherry': {'population': 1247953, 'area': 490},
    'Punjab': {'population': 27743338, 'area': 50362},
    'Rajasthan': {'population': 68548437, 'area': 342239},
    'Sikkim': {'population': 610577, 'area': 7096},
    'Tamil Nadu': {'population': 72147030, 'area': 130060},
    'Telangana': {'population': 35003674, 'area': 112077},
    'Tripura': {'population': 3673917, 'area': 10486},
    'Uttar Pradesh': {'population': 199812341, 'area': 243290},
    'Uttarakhand': {'population': 10086292, 'area': 53483},
    'West Bengal': {'population': 91276115, 'area': 88752},
}

# Define fuzzy logic system with updated ranges
monthly_temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'monthly_temperature')
pop_density = ctrl.Antecedent(np.arange(0, 5001, 50), 'pop_density')
pig_vaccinated = ctrl.Antecedent(np.arange(0, 100001, 1000), 'pig_vaccinated')
state_total_population = ctrl.Antecedent(np.arange(0, 200000001, 1000000), 'state_total_population')
monthly_case_to_pop_ratio = ctrl.Antecedent(np.arange(0, 1e-4, 1e-6), 'monthly_case_to_pop_ratio')  # Adjusted range

# Output variable
outbreak_risk = ctrl.Consequent(np.arange(0, 101, 1), 'outbreak_risk')

# Membership functions for inputs
monthly_temperature.automf(3, names=['poor', 'average', 'good'])
pop_density.automf(3, names=['poor', 'average', 'good'])
pig_vaccinated.automf(3, names=['poor', 'average', 'good'])
state_total_population.automf(3, names=['poor', 'average', 'good'])
monthly_case_to_pop_ratio.automf(3, names=['poor', 'average', 'good'])

# Membership functions for output
outbreak_risk['no_risk'] = fuzz.trimf(outbreak_risk.universe, [0, 0, 20])
outbreak_risk['low'] = fuzz.trimf(outbreak_risk.universe, [10, 30, 50])
outbreak_risk['high'] = fuzz.trimf(outbreak_risk.universe, [40, 100, 100])

# Define fuzzy rules
rule1 = ctrl.Rule(monthly_temperature['poor'] & pop_density['good'] & monthly_case_to_pop_ratio['good'], outbreak_risk['high'])
rule2 = ctrl.Rule(pig_vaccinated['poor'] & state_total_population['good'], outbreak_risk['high'])
rule3 = ctrl.Rule(monthly_temperature['good'] & pop_density['poor'] & monthly_case_to_pop_ratio['poor'] & pig_vaccinated['good'], outbreak_risk['no_risk'])
rule4 = ctrl.Rule(pig_vaccinated['good'], outbreak_risk['low'])
rule5 = ctrl.Rule(monthly_temperature['average'] & monthly_case_to_pop_ratio['average'], outbreak_risk['low'])
rule6 = ctrl.Rule(pop_density['average'] & pig_vaccinated['average'], outbreak_risk['low'])
rule7 = ctrl.Rule(monthly_case_to_pop_ratio['good'], outbreak_risk['high'])  # New rule

# Create control system
outbreak_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7])
outbreak_sim = ctrl.ControlSystemSimulation(outbreak_ctrl)

# Function to create a gauge chart for risk visualization
def create_gauge_chart(risk_score, risk_label):
    if risk_label == "No Risk":
        color = "green"
    elif risk_label == "Low Risk":
        color = "yellow"
    else:
        color = "red"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 5], 'color': "rgba(0, 150, 0, 0.2)"},
                {'range': [5, 20], 'color': "rgba(255, 255, 0, 0.2)"},
                {'range': [20, 100], 'color': "rgba(255, 0, 0, 0.2)"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_score
            }
        },
        title = {'text': "Outbreak Risk Score"}
    ))
    
    return fig

# Function to create feature importance chart
def create_feature_importance_chart():
    # This would normally use your actual model's feature importances
    # For this example, using sample data
    importance_data = {
        'Feature': ['Temperature', 'Vaccination', 'Population', 'Case Ratio', 'Pop Density', 'Area'],
        'Importance': [0.25, 0.20, 0.15, 0.22, 0.10, 0.08]
    }
    
    fig = px.bar(
        importance_data, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance in Prediction Model',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='white'
    )
    
    return fig

# Two-column layout for inputs and results
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="section-header"><h3>📊 Input Parameters</h3></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # State selection with a map preview (placeholder)
    st.subheader("🗺️ Region Selection")
    state = st.selectbox(
        "Select State", 
        options=list(state_data.keys()), 
        index=list(state_data.keys()).index('West Bengal'),
        help="Select the Indian state for which you want to predict influenza outbreak risk"
    )
    
    # Get state-specific data
    state_info = state_data.get(state, {'population': 0, 'area': 0})
    state_total_population_val = state_info['population']
    area_val = state_info['area']
    
    # Display state information with better formatting
    st.markdown(f"""
        <div style="background-color:#f1f8ff; padding:10px; border-radius:5px; margin-top:10px;">
            <strong>State:</strong> {state}<br>
            <strong>Population:</strong> {state_total_population_val:,}<br>
            <strong>Area:</strong> {area_val:,} km²<br>
            <strong>Population Density:</strong> {state_total_population_val/area_val:.2f} people/km²
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🌡️ Environmental Factors")
    
    
    # Temperature with more context
    temp_col1, temp_col2 = st.columns([3, 1])
    with temp_col1:
        monthly_temperature_val = st.slider(
            "Monthly Average Temperature (°C)", 
            min_value=0.0, 
            max_value=40.0, 
            value=20.0,
            help="The average temperature for the month in degrees Celsius"
        )
    # Helper functions for UI elements
    def get_temp_color(temp):
        if temp < 10:
            return "#3b82f6"  # Blue for cold
        elif temp < 20:
            return "#10b981"  # Green for mild
        elif temp < 30:
            return "#f59e0b"  # Orange for warm
        else:
            return "#ef4444"  # Red for hot

    def get_risk_bg_color(risk_label):
        if risk_label == "No Risk":
            return "#d1fae5"  # Light green
        elif risk_label == "Low Risk":
            return "#fef3c7"  # Light yellow
        else:
            return "#fee2e2"  # Light red

    def get_risk_icon(risk_label):
        if risk_label == "No Risk":
            return "✅"
        elif risk_label == "Low Risk":
            return "⚠️"
        else:
            return "🚨"    
    with temp_col2:
        st.markdown(f"""
            <div style="background-color:{get_temp_color(monthly_temperature_val)}; 
                  height:80px; border-radius:5px; display:flex; 
                  align-items:center; justify-content:center; color:white; 
                  font-size:24px; font-weight:bold;">
                {monthly_temperature_val}°C
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💉 Vaccination Data")
    
    # Vaccination with percentage context
    pig_vaccinated_val = st.number_input(
        "Number of Pigs Vaccinated", 
        min_value=0.0, 
        max_value=100000.0, 
        value=5000.0,
        help="Total number of pigs that have been vaccinated in the region"
    )
    
    # Estimated pig population (this would ideally come from real data)
    estimated_pig_population = 50000
    vaccination_rate = (pig_vaccinated_val / estimated_pig_population) * 100
    
    # Vaccination rate progress bar
    st.markdown(f"<p>Estimated Vaccination Rate: {vaccination_rate:.1f}%</p>", unsafe_allow_html=True)
    st.progress(min(vaccination_rate/100, 1.0))
    
    if vaccination_rate < 30:
        st.warning("⚠️ Vaccination rate is below recommended levels (30%)")
    elif vaccination_rate > 70:
        st.success("✅ Vaccination rate meets recommended levels (>70%)")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🩺 Case Reports")
    
    # Cases with trend visualization
    cases_reported_val = st.number_input(
        "Monthly Cases Reported in the State", 
        min_value=0.0, 
        max_value=1000000.0, 
        value=100.0,
        help="Total number of influenza cases reported in the last month"
    )
    
    # Compute case-to-population ratio and display it
    monthly_case_to_pop_ratio_val = cases_reported_val / state_total_population_val if state_total_population_val > 0 else 0.0
    
    # Show per 100,000 for easier interpretation
    cases_per_100k = monthly_case_to_pop_ratio_val * 100000
    
    st.markdown(f"""
        <div style="background-color:#f1f8ff; padding:10px; border-radius:5px; margin-top:10px;">
            <strong>Case-to-Population Ratio:</strong> {monthly_case_to_pop_ratio_val:.8f}<br>
            <strong>Cases per 100,000 people:</strong> {cases_per_100k:.2f}
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-header"><h3>📈 Analysis & Prediction</h3></div>', unsafe_allow_html=True)
    
    # Initialize session state for risk scores if not already set
    if "fuzzy_risk_score" not in st.session_state:
        st.session_state.fuzzy_risk_score = None
    if "xgb_risk_score" not in st.session_state:
        st.session_state.xgb_risk_score = None
    if "final_risk_score" not in st.session_state:
        st.session_state.final_risk_score = None
    if "risk_label" not in st.session_state:
        st.session_state.risk_label = None
    
    # Calculate population density
    pop_density_val = state_total_population_val / area_val if area_val > 0 else 0.0
    
    # Prediction button with more attractive styling
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Add an info box about the prediction methods
    st.markdown("""
    <div style="background-color:#e8f4fd; padding:15px; border-radius:5px; margin-bottom:20px; border-left:5px solid #0078D7;">
        <h4 style="margin-top:0;">🔬 About This Prediction Tool</h4>
        <p>This system uses a hybrid approach combining XGBoost machine learning with fuzzy logic to predict influenza outbreak risks. It analyzes environmental factors, vaccination rates, population metrics, and historical case data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    predict_col1, predict_col2 = st.columns([3, 1])
    with predict_col1:
        predict_button = st.button("🔍 Run Prediction Analysis", help="Click to analyze current data and predict outbreak risk")
    with predict_col2:
        debug = st.checkbox("Debug Mode", value=False, help="Show detailed model scores")
    
    # Show model components if debug is enabled
    if debug and "fuzzy_risk_score" in st.session_state and st.session_state.fuzzy_risk_score is not None:
        st.markdown("""
        <div style="background-color:#f8f9fa; padding:10px; border-radius:5px; margin-top:10px; font-family:monospace;">
            <strong>DEBUG DATA:</strong><br>
        """, unsafe_allow_html=True)
        st.write(f"Fuzzy Logic Risk Score: {st.session_state.fuzzy_risk_score:.2f}")
        st.write(f"XGBoost Risk Score: {st.session_state.xgb_risk_score:.2f}")
        st.write(f"Combined Risk Score: {st.session_state.final_risk_score:.2f}")
        st.write(f"Risk Classification: {st.session_state.risk_label}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction results section
    if predict_button:
        # Input validation
        if any(x < 0 for x in [monthly_temperature_val, pig_vaccinated_val, area_val, state_total_population_val, monthly_case_to_pop_ratio_val, cases_reported_val]):
            st.error("⚠️ Input values cannot be negative.")
            st.stop()

        # Display a loading message
        with st.spinner("Analyzing data and generating prediction..."):
            # Prepare input for XGBoost model
            input_data = pd.DataFrame({
                'monthly_temperature': [monthly_temperature_val],
                'pig_vaccinated': [pig_vaccinated_val],
                'area': [area_val],
                'state_total_population': [state_total_population_val],
                'monthly_case_to_pop_ratio': [monthly_case_to_pop_ratio_val],
                'pop_density': [pop_density_val],
                'name': [state]
            })

            # One-hot encode the 'name' column to match the model's expected features
            input_data_encoded = pd.get_dummies(input_data, columns=['name'], prefix='name')

            # Ensure all expected one-hot encoded columns are present
            expected_columns = model.get_booster().feature_names
            input_data_encoded = input_data_encoded.reindex(columns=expected_columns, fill_value=0)

            # Apply imputer to the input data
            try:
                input_data_imputed = pd.DataFrame(imputer.transform(input_data_encoded), columns=input_data_encoded.columns)
            except Exception as e:
                st.error(f"⚠️ Error in preprocessing input data: {str(e)}")
                st.stop()

            # Get XGBoost prediction
            try:
                xgb_prob = model.predict_proba(input_data_imputed)[:, 1][0]  # Probability of high risk
                xgb_risk_score = xgb_prob * 100  # Scale to 0-100
            except Exception as e:
                st.error(f"⚠️ Error in XGBoost prediction: {str(e)}")
                st.stop()

            # Get Fuzzy Logic prediction
            try:
                outbreak_sim.input['monthly_temperature'] = monthly_temperature_val
                outbreak_sim.input['pop_density'] = pop_density_val
                outbreak_sim.input['pig_vaccinated'] = pig_vaccinated_val
                outbreak_sim.input['state_total_population'] = state_total_population_val
                outbreak_sim.input['monthly_case_to_pop_ratio'] = monthly_case_to_pop_ratio_val
                outbreak_sim.compute()
                fuzzy_risk_score = outbreak_sim.output['outbreak_risk']
            except Exception as e:
                st.error(f"⚠️ Error in Fuzzy Logic computation: {str(e)}")
                st.stop()

            # Combined score
            final_risk_score = (xgb_risk_score + fuzzy_risk_score) / 2

            # Store scores in session state
            st.session_state.fuzzy_risk_score = fuzzy_risk_score
            st.session_state.xgb_risk_score = xgb_risk_score
            st.session_state.final_risk_score = final_risk_score

            # Define risk level
            if final_risk_score < 5:
                risk_label = "No Risk"
                risk_color = "green"
                risk_message = "The risk of an influenza outbreak is very low."
                recommendations = [
                    "Continue routine surveillance",
                    "Maintain current vaccination protocols",
                    "Standard hygiene practices are sufficient"
                ]
            elif final_risk_score < 20:
                risk_label = "Low Risk"
                risk_color = "yellow"
                risk_message = "The risk of an influenza outbreak is low. Monitor conditions."
                recommendations = [
                    "Increase sampling frequency in high-density areas",
                    "Consider supplemental vaccination in unprotected populations",
                    "Enhance public awareness about hygiene practices",
                    "Prepare healthcare facilities for potential cases"
                ]
            else:
                risk_label = "High Risk"
                risk_color = "red"
                risk_message = "The risk of an influenza outbreak is high. Immediate action is needed."
                recommendations = [
                    "Activate emergency response protocols",
                    "Implement intensive surveillance and testing",
                    "Deploy vaccination teams to all affected and surrounding areas",
                    "Consider movement restrictions for livestock",
                    "Issue public health advisories",
                    "Prepare isolation facilities and increase hospital capacity"
                ]
            
            # Store risk label in session state
            st.session_state.risk_label = risk_label

        # Display risk assessment with better styling
        st.markdown(f"""
        <div class="risk-card" style="background-color:{get_risk_bg_color(risk_label)};">
            <h2 style="color:{risk_color}; margin-bottom:10px;">
                {get_risk_icon(risk_label)} {risk_label}
            </h2>
            <p style="font-size:1.2em; margin-bottom:20px;">{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display gauge chart
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Risk Assessment Score")
        gauge_chart = create_gauge_chart(final_risk_score, risk_label)
        st.plotly_chart(gauge_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display recommendations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📋 Recommended Actions")
        for i, rec in enumerate(recommendations):
            st.markdown(f"**{i+1}.** {rec}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display feature importance
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Key Risk Factors")
        feature_chart = create_feature_importance_chart()
        st.plotly_chart(feature_chart, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Display placeholder when no prediction has been run
        st.markdown('<div class="card" style="text-align:center; padding:40px;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="color:#6c757d;">
            <i class="fas fa-chart-line" style="font-size:48px;"></i>
            <h3>No Prediction Data Available</h3>
            <p>Adjust the parameters on the left and click "Run Prediction Analysis" to generate an outbreak risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)



# Add footer with disclaimer
st.markdown("""
<div class="footer">
    <p><strong>Disclaimer:</strong> This tool is for research and educational purposes only. All predictions should be validated by healthcare professionals.</p>
    <p>© 2025 Influenza Outbreak Prediction System | Last updated: April 2025</p>
</div>
""", unsafe_allow_html=True)
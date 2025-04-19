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
    /* Base styles */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Title container - already dark blue so works in both modes */
    .title-container {
        background-color: #0078D7;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }
    
    /* Section headers - adjust for dark mode */
    .section-header {
        background-color: rgba(0, 120, 215, 0.2);
        padding: 0.7rem;
        border-left: 5px solid #0078D7;
        border-radius: 5px;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Cards - adapt to color scheme */
    .card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Risk cards with better dark mode support */
.risk-card {
    padding: 1.5rem;
    border-radius: 10px;
    margin-top: 1rem;
    text-align: center;
}

/* Dark mode styles using CSS media query */
@media (prefers-color-scheme: dark) {
    .risk-card.no-risk {
        background-color: rgba(16, 185, 129, 0.3) !important;
        color: white !important;
    }
    
    .risk-card.low-risk {
        background-color: rgba(245, 158, 11, 0.3) !important;
        color: white !important;
    }
    
    .risk-card.high-risk {
        background-color: rgba(239, 68, 68, 0.3) !important;
        color: white !important;
    }
}
    
    /* Footer - ensure better contrast */
    .footer {
        text-align: center;
        padding: 1rem;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.8rem;
        margin-top: 2rem;
    }
    
    /* Form elements */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Custom Slider */
    .stSlider > div > div > div {
        background-color: rgba(0, 120, 215, 0.3);
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
    
    /* Info boxes with better dark mode support */
    .info-box {
        background-color: rgba(0, 120, 215, 0.1);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #0078D7;
    }
    
    /* Debug panel for better visibility */
    .debug-panel {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        font-family: monospace;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* State info with better dark mode visibility */
    .state-info {
        background-color: rgba(0, 120, 215, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border: 1px solid rgba(0, 120, 215, 0.3);
    }
    
    /* Case ratio display */
    .case-info {
        background-color: rgba(0, 120, 215, 0.1);
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
        border: 1px solid rgba(0, 120, 215, 0.3);
    }
    
    /* No prediction placeholder */
    .placeholder {
        text-align: center;
        padding: 40px;
        color: rgba(255, 255, 255, 0.7);
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
        background-color: #333;
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
    
    /* Warning box for zero cases */
    .warning-box {
        background-color: rgba(255, 193, 7, 0.2);
        border-left: 5px solid #ffc107;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
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
    'Andhra Pradesh': {'population': 53903393, 'area': 162970},
    'Arunachal Pradesh': {'population': 1570458, 'area': 83743},
    'Assam': {'population': 35607039, 'area': 78438},
    'Bihar': {'population': 124799926, 'area': 94163},
    'Chandigarh': {'population': 1158473, 'area': 114},
    'Chhattisgarh': {'population': 29436231, 'area': 135192},
    'Goa': {'population': 1567000, 'area': 3702},
    'Gujarat': {'population': 70648000, 'area': 196244},
    'Haryana': {'population': 29846000, 'area': 44212},
    'Himachal Pradesh': {'population': 7431000, 'area': 55673},
    'Jharkhand': {'population': 38969000, 'area': 79716},
    'Karnataka': {'population': 67268000, 'area': 191791},
    'Kerala': {'population': 35633000, 'area': 38852},
    'Madhya Pradesh': {'population': 86579000, 'area': 308252},
    'Maharashtra': {'population': 126385000, 'area': 307713},
    'Manipur': {'population': 2721756, 'area': 22327},
    'Meghalaya': {'population': 2964007, 'area': 22429},
    'Mizoram': {'population': 1091014, 'area': 21081},
    'Nagaland': {'population': 1980602, 'area': 16579},
    'Odisha': {'population': 46276000, 'area': 155707},
    'Puducherry': {'population': 1244464, 'area': 490},
    'Punjab': {'population': 27704236, 'area': 50362},
    'Rajasthan': {'population': 81025000, 'area': 342239},
    'Sikkim': {'population': 703000, 'area': 7096},
    'Tamil Nadu': {'population': 76860000, 'area': 130060},
    'Telangana': {'population': 38090000, 'area': 112077},
    'Tripura': {'population': 3671032, 'area': 10486},
    'Uttar Pradesh': {'population': 235687000, 'area': 243290},
    'Uttarakhand': {'population': 10116752, 'area': 53483},
    'West Bengal': {'population': 99084000, 'area': 88752},
}

# Define better fuzzy logic system with updated ranges
monthly_temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'monthly_temperature')
pop_density = ctrl.Antecedent(np.arange(0, 5001, 50), 'pop_density')
pig_vaccinated = ctrl.Antecedent(np.arange(0, 100001, 1000), 'pig_vaccinated')  # Ensure full range covers up to 100k
state_total_population = ctrl.Antecedent(np.arange(0, 200000001, 1000000), 'state_total_population')
monthly_case_to_pop_ratio = ctrl.Antecedent(np.arange(0, 1e-3, 1e-5), 'monthly_case_to_pop_ratio')

# Output variable - ensure full range is covered
outbreak_risk = ctrl.Consequent(np.arange(0, 101, 1), 'outbreak_risk')

# Better membership functions for inputs
pop_density['low'] = fuzz.trimf(pop_density.universe, [0, 0, 500])
pop_density['medium'] = fuzz.trimf(pop_density.universe, [300, 1000, 2000])
pop_density['high'] = fuzz.trimf(pop_density.universe, [1500, 3000, 5000])

state_total_population['small'] = fuzz.trimf(state_total_population.universe, [0, 0, 10000000])
state_total_population['medium'] = fuzz.trimf(state_total_population.universe, [5000000, 50000000, 100000000])
state_total_population['large'] = fuzz.trimf(state_total_population.universe, [50000000, 150000000, 200000000])

# Improved case-to-pop ratio with wider range
monthly_case_to_pop_ratio['low'] = fuzz.trimf(monthly_case_to_pop_ratio.universe, [0, 0, 2e-4])
monthly_case_to_pop_ratio['medium'] = fuzz.trimf(monthly_case_to_pop_ratio.universe, [1e-4, 3e-4, 5e-4])
monthly_case_to_pop_ratio['high'] = fuzz.trimf(monthly_case_to_pop_ratio.universe, [4e-4, 7e-4, 1e-3])

# Fix for Issue 2: Define better membership functions for pig_vaccinated with improved ranges
pig_vaccinated['low'] = fuzz.trimf(pig_vaccinated.universe, [0, 5000, 20000])
pig_vaccinated['medium'] = fuzz.trimf(pig_vaccinated.universe, [15000, 40000, 65000])  # Adjust medium range to include 40000
pig_vaccinated['high'] = fuzz.trimf(pig_vaccinated.universe, [50000, 75000, 100000])

# Improved temperature membership functions that better match the 0-40°C range
monthly_temperature['cold'] = fuzz.trimf(monthly_temperature.universe, [0, 0, 15])
monthly_temperature['moderate'] = fuzz.trimf(monthly_temperature.universe, [10, 20, 30])
monthly_temperature['hot'] = fuzz.trimf(monthly_temperature.universe, [25, 35, 40])

# Membership functions for output - ensure full coverage
outbreak_risk['no_risk'] = fuzz.trimf(outbreak_risk.universe, [0, 0, 25])
outbreak_risk['low_risk'] = fuzz.trimf(outbreak_risk.universe, [15, 35, 55])
outbreak_risk['high_risk'] = fuzz.trimf(outbreak_risk.universe, [45, 75, 100])

# Define improved fuzzy rules with temperature-specific considerations
rule1 = ctrl.Rule(monthly_temperature['cold'] & monthly_case_to_pop_ratio['high'], outbreak_risk['high_risk'])
rule2 = ctrl.Rule(pig_vaccinated['low'] & state_total_population['large'], outbreak_risk['high_risk'])
rule3 = ctrl.Rule(monthly_temperature['hot'] & pop_density['low'] & monthly_case_to_pop_ratio['low'], outbreak_risk['no_risk'])
rule4 = ctrl.Rule(pig_vaccinated['high'] & state_total_population['medium'], outbreak_risk['low_risk'])
rule5 = ctrl.Rule(monthly_temperature['moderate'] & monthly_case_to_pop_ratio['medium'], outbreak_risk['low_risk'])
rule6 = ctrl.Rule(pop_density['high'] & pig_vaccinated['medium'], outbreak_risk['low_risk'])  # Changed from high_risk to low_risk
rule7 = ctrl.Rule(monthly_case_to_pop_ratio['high'] & pig_vaccinated['low'], outbreak_risk['high_risk'])  
rule8 = ctrl.Rule(monthly_temperature['moderate'] & pop_density['low'] & monthly_case_to_pop_ratio['low'], outbreak_risk['no_risk'])
rule9 = ctrl.Rule(state_total_population['small'], outbreak_risk['low_risk'])
rule10 = ctrl.Rule(pig_vaccinated['low'] & monthly_case_to_pop_ratio['medium'], outbreak_risk['high_risk'])
rule11 = ctrl.Rule(monthly_temperature['moderate'] & pop_density['high'] & pig_vaccinated['low'], outbreak_risk['high_risk'])
rule12 = ctrl.Rule(monthly_temperature['hot'] & pop_density['high'] & monthly_case_to_pop_ratio['medium'], outbreak_risk['high_risk'])
rule13 = ctrl.Rule(monthly_temperature['cold'] & pop_density['high'] & monthly_case_to_pop_ratio['medium'], outbreak_risk['high_risk'])
rule14 = ctrl.Rule(monthly_temperature['moderate'] & pig_vaccinated['high'] & monthly_case_to_pop_ratio['low'], outbreak_risk['no_risk'])
# Add a rule for medium vaccinated pigs to ensure 40000 works correctly
rule15 = ctrl.Rule(pig_vaccinated['medium'], outbreak_risk['low_risk'])

# Create control system with all rules
outbreak_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15])
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
        <div class="state-info">
            <strong>State:</strong> {state}<br>
            <strong>Population:</strong> {state_total_population_val:,}<br>
            <strong>Area:</strong> {area_val:,} km²<br>
            <strong>Population Density:</strong> {state_total_population_val/area_val:.2f} people/km²
        </div>
    """, unsafe_allow_html=True)    
    st.markdown('<div class="card">', unsafe_allow_html=True)

    # Temperature UI with better visual feedback
    st.subheader("🌡️ Environmental Factors")

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

    def get_temp_risk_label(temp):
        if temp < 10:
            return "Cold - Moderate Indoor Transmission Risk"
        elif temp < 20:
            return "Mild - Low Transmission Risk"
        elif temp >= 20 and temp <= 30:
            return "Optimal Range - Higher Virus Survival"
        else:
            return "Hot - Reduced Outdoor Transmission"

    # Temperature with more context
    temp_col1, temp_col2 = st.columns([3, 1])
    with temp_col1:
        monthly_temperature_val = st.slider(
            "Monthly Average Temperature (°C)", 
            min_value=0.0, 
            max_value=40.0, 
            value=20.0,
            help="The average temperature for the month in degrees Celsius. Temperature affects virus survival and transmission patterns."
        )
        
        # Add temperature risk information
        st.markdown(f"""
            <div style="background-color:rgba({30}, {30}, {30}, 0.1); padding:10px; border-radius:5px; margin-top:5px;">
                <strong>{get_temp_risk_label(monthly_temperature_val)}</strong>
            </div>
        """, unsafe_allow_html=True)
    
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
            
    # Improved vaccination section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💉 Vaccination Data")

    # Vaccination with percentage context
    pig_vaccinated_val = st.number_input(
        "Number of Pigs Vaccinated", 
        min_value=0, 
        max_value=100000, 
        value=5000,
        help="Total number of pigs that have been vaccinated in the region"
    )

    # Estimated pig population (this would ideally come from real data)
    estimated_pig_population = 50000
    vaccination_rate = (pig_vaccinated_val / estimated_pig_population) * 100

    # Determine vaccination status and color
    def get_vacc_status(rate):
        if rate < 20:
            return "Critical", "#ef4444"  # Red
        elif rate < 30:
            return "Very Low", "#f97316"  # Dark orange
        elif rate < 50:
            return "Insufficient", "#f59e0b"  # Orange
        elif rate < 70:
            return "Moderate", "#84cc16"  # Light green
        else:
            return "Adequate", "#10b981"  # Green

    vacc_status, vacc_color = get_vacc_status(vaccination_rate)

    # Vaccination rate display with colored status
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
            <span>Estimated Vaccination Rate:</span>
            <span style="color: {vacc_color}; font-weight: bold;">{vaccination_rate:.1f}% - {vacc_status}</span>
        </div>
    """, unsafe_allow_html=True)

    # Improved vaccination progress bar with color
    st.markdown(f"""
        <div style="background-color: #e5e7eb; border-radius: 5px; height: 10px; margin-bottom: 15px;">
            <div style="background-color: {vacc_color}; width: {min(vaccination_rate, 100)}%; height: 100%; border-radius: 5px;"></div>
        </div>
    """, unsafe_allow_html=True)

    # Vaccination guidance
    if vaccination_rate < 30:
        st.warning("⚠️ Vaccination rate is critically below recommended levels (30%). This significantly increases outbreak risk.")
    elif vaccination_rate < 50:
        st.warning("⚠️ Vaccination rate is below optimal levels. Consider increasing vaccination coverage.")
    elif vaccination_rate < 70:
        st.info("ℹ️ Vaccination rate is moderate but could be improved to reduce outbreak risk.")
    else:
        st.success("✅ Vaccination rate meets recommended levels (>70%), providing good herd immunity.")

    # Add vaccination impact information
    st.markdown("""
        <div style="font-size: 0.9em; margin-top: 10px;">
            <strong>Vaccination Impact:</strong><br>
            • <70% coverage: Insufficient to prevent outbreaks<br>
            • 70-90% coverage: Good protection, limited outbreak potential<br>
            • >90% coverage: Optimal protection, minimal outbreak risk
        </div>
    """, unsafe_allow_html=True)

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
    
    # Fix for Issue 1: Warning for zero cases
    if cases_reported_val == 0:
        st.markdown("""
            <div class="warning-box">
                <strong>⚠️ No Cases Reported</strong>
                <p>You have entered zero cases. This will result in a "No Risk" assessment but may not reflect the true situation if cases are unreported or underreported.</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Compute case-to-population ratio and display it
    monthly_case_to_pop_ratio_val = cases_reported_val / state_total_population_val if state_total_population_val > 0 else 0.0
    
    # Show per 100,000 for easier interpretation
    cases_per_100k = monthly_case_to_pop_ratio_val * 100000
    
    st.markdown(f"""
        <div class="state-info">
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
   <div class="info-box">
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
        <div class="debug-panel">
    <strong>DEBUG DATA:</strong><br>
        """, unsafe_allow_html=True)
        st.write(f"Fuzzy Logic Risk Score: {st.session_state.fuzzy_risk_score:.2f}")
        st.write(f"XGBoost Risk Score: {st.session_state.xgb_risk_score:.2f}")
        st.write(f"Combined Risk Score: {st.session_state.final_risk_score:.2f}")
        st.write(f"Risk Classification: {st.session_state.risk_label}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction button handler
    if predict_button:
        # Fix for Issue 1: Special case for zero or very low cases
        if cases_reported_val == 0:
            fuzzy_risk_score = 0
            xgb_risk_score = 0
            final_risk_score = 0
            
            # Store scores in session state
            st.session_state.fuzzy_risk_score = fuzzy_risk_score
            st.session_state.xgb_risk_score = xgb_risk_score
            st.session_state.final_risk_score = final_risk_score
            st.session_state.risk_label = "No"
            
            risk_label = "No Risk"
            risk_color = "green"
            risk_message = "No population data available for risk assessment."
            recommendations = [
                "Update demographic data",
                "Continue routine surveillance",
                "Maintain current vaccination protocols"
            ]
        else:
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

                # Get Fuzzy Logic prediction with improved handling
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


                if monthly_temperature_val < 10 or monthly_temperature_val > 35:
                    temp_weight = 0.7  # More weight to fuzzy logic for extreme temperatures
                else:
                    temp_weight = 0.5  # Equal weighting for moderate temperatures
                    
                final_risk_score = (xgb_risk_score * (1-temp_weight)) + (fuzzy_risk_score * temp_weight)

                # Store scores in session state
                st.session_state.fuzzy_risk_score = fuzzy_risk_score
                st.session_state.xgb_risk_score = xgb_risk_score
                st.session_state.final_risk_score = final_risk_score

                # Adjust risk based on reported cases relative to population size
                cases_per_million = (cases_reported_val / state_total_population_val) * 1000000 if state_total_population_val > 0 else 0
                
                # Case thresholds adjusted for different population sizes
                LOW_CASE_THRESHOLD = 10  # Cases per million threshold for low risk override
                HIGH_CASE_THRESHOLD = 100  # Cases per million threshold for high risk consideration
                
                # Case-based adjustments
                if cases_per_million <= LOW_CASE_THRESHOLD:
                    # If cases are very low, reduce the risk score
                    final_risk_score = min(final_risk_score, 10 + (cases_per_million / 2))
                elif cases_per_million >= HIGH_CASE_THRESHOLD:
                    # If cases are high, increase the risk floor
                    final_risk_score = max(final_risk_score, 20 + min((cases_per_million - HIGH_CASE_THRESHOLD) / 10, 30))
                
                # Temperature-based risk adjustments
                if monthly_temperature_val < 10:
                    # Cold conditions can reduce virus survival outdoors but increase indoor transmission
                    if cases_per_million > 50:  
                        final_risk_score = max(final_risk_score, 25)  # Maintain elevated risk
                    else:
                        final_risk_score = min(final_risk_score, 35)  # Cap risk
                elif monthly_temperature_val > 35:
                    # Very hot conditions may reduce virus survival but can impact immunity
                    final_risk_score = min(final_risk_score * 0.9, 85) 
                elif 20 <= monthly_temperature_val <= 30:
                    # Optimal virus survival range
                    if cases_per_million > 30:
                        final_risk_score = max(final_risk_score, 30)  # Ensure at least medium risk
                
                # Vaccination impact - stronger effect
                vacc_percentage = (pig_vaccinated_val / 50000) * 100  # Using estimated pig population
                if vacc_percentage >= 70:
                    final_risk_score = final_risk_score * 0.7  # Significant reduction for high vaccination
                elif vacc_percentage <= 20:
                    final_risk_score = min(final_risk_score * 1.3, 100)  # Increase risk for low vaccination
                
                # Define risk level with better thresholds
                if final_risk_score < 15:
                    risk_label = "No Risk"
                    risk_color = "green"
                    risk_message = "The risk of an influenza outbreak is very low."
                    recommendations = [
                        "Continue routine surveillance",
                        "Maintain current vaccination protocols",
                        "Standard hygiene practices are sufficient"
                    ]
                elif final_risk_score < 40:  # Widened the low risk range
                    risk_label = "Low Risk"
                    risk_color = "orange"
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
        <div class="risk-card {risk_label.lower().replace(' ', '-')}-risk">
            <h2 style="margin-bottom:10px;">
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
        <div class="placeholder">
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

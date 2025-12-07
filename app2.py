"""
DASHBOARD DHT11 REAL-TIME dengan Download CSV
Fixed version - tanpa statsmodels dependency
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
import os
import time
import json
import warnings

# ==================== CONFIGURATION ====================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="DHT11 Real-Time Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== HIVEMQ CONFIGURATION ====================
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"

# ==================== PATH CONFIGURATION ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAININGDHT_DIR = os.path.join(BASE_DIR, "Trainingdht")
MODELS_DIR = os.path.join(TRAININGDHT_DIR, "models")
CSV_PKL_DIR = os.path.join(TRAININGDHT_DIR, "csv_pkl")
CSV_FILE = os.path.join(TRAININGDHT_DIR, "sensor_data.csv")

# Buat folder jika belum ada
os.makedirs(TRAININGDHT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CSV_PKL_DIR, exist_ok=True)

# ==================== DATA MANAGEMENT FUNCTIONS ====================
def load_csv_data():
    """Load data dari CSV file"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, delimiter=';')
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return pd.DataFrame()

def save_to_csv(data_dict):
    """Save data ke CSV file"""
    try:
        timestamp = datetime.now().strftime('%H;%M;%S')
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        new_row = {
            'timestamp': timestamp,
            'temperature': data_dict.get('temperature', 0),
            'humidity': data_dict.get('humidity', 0),
            'label': data_dict.get('label', ''),
            'date': date_str
        }
        
        df_new = pd.DataFrame([new_row])
        
        if os.path.exists(CSV_FILE):
            df_existing = pd.read_csv(CSV_FILE, delimiter=';')
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined.to_csv(CSV_FILE, sep=';', index=False)
        else:
            df_new.to_csv(CSV_FILE, sep=';', index=False)
            
        return True
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")
        return False

# ==================== LOAD MODELS dari CSV.PKL ====================
@st.cache_resource
def load_models_from_csv_pkl():
    """Load models dari folder CSV.PKL"""
    models = {}
    scaler = None
    
    try:
        # 1. Load scaler dari CSV.PKL
        scaler_csv_pkl = os.path.join(CSV_PKL_DIR, 'scaler.csv.pkl')
        if os.path.exists(scaler_csv_pkl):
            with open(scaler_csv_pkl, 'rb') as f:
                scaler_info = pickle.load(f)
                
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                if isinstance(scaler_info, pd.DataFrame):
                    scaler.mean_ = scaler_info['mean'].values
                    scaler.scale_ = scaler_info['scale'].values
                elif isinstance(scaler_info, StandardScaler):
                    scaler = scaler_info
                    
            st.success("✅ Loaded scaler from CSV.PKL")
        
        # 2. Load 3 ML models dari CSV.PKL
        model_files = [
            ('Decision Tree', 'decision_tree.csv.pkl'),
            ('K-Nearest Neighbors', 'k_nearest_neighbors.csv.pkl'),
            ('Logistic Regression', 'logistic_regression.csv.pkl')
        ]
        
        for model_name, filename in model_files:
            csv_pkl_path = os.path.join(CSV_PKL_DIR, filename)
            
            if os.path.exists(csv_pkl_path):
                with open(csv_pkl_path, 'rb') as f:
                    model_data = pickle.load(f)
                    
                    if isinstance(model_data, dict) and 'model' in model_data:
                        models[model_name] = model_data['model']
                    elif hasattr(model_data, 'predict'):
                        models[model_name] = model_data
        
        # 3. Fallback: Load dari folder models
        if not models:
            model_files_pkl = {
                'Decision Tree': 'decision_tree.pkl',
                'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
                'Logistic Regression': 'logistic_regression.pkl'
            }
            
            for model_name, filename in model_files_pkl.items():
                model_path = os.path.join(MODELS_DIR, filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        models[model_name] = pickle.load(f)
        
        if models:
            st.success(f"✅ Loaded {len(models)} ML models")
        else:
            st.error("❌ No models found!")
            
        return models, scaler
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None

# ==================== HIVEMQ SIMPLE CONNECTION ====================
def connect_to_hivemq():
    """Simple connection to HiveMQ untuk real-time data"""
    try:
        import paho.mqtt.client as mqtt
        
        client = mqtt.Client()
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        client.tls_set()
        
        received_messages = []
        latest_data = None
        
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("✅ Connected to HiveMQ")
                client.subscribe(DHT_TOPIC)
                print(f"📡 Subscribed to: {DHT_TOPIC}")
        
        def on_message(client, userdata, msg):
            nonlocal latest_data, received_messages
            try:
                data = json.loads(msg.payload.decode())
                data['received_time'] = datetime.now()
                data['timestamp_str'] = datetime.now().strftime('%H:%M:%S')
                
                latest_data = data
                received_messages.append(data)
                
                # Save to CSV
                save_to_csv(data)
                
                # Keep only last 100 messages in memory
                if len(received_messages) > 100:
                    received_messages = received_messages[-100:]
                    
            except Exception as e:
                print(f"Error processing message: {e}")
        
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        
        time.sleep(2)
        
        return client, latest_data, received_messages
        
    except Exception as e:
        st.error(f"⚠️ HiveMQ Connection Error: {e}")
        return None, None, []

# ==================== PREDICTION FUNCTIONS ====================
def get_label_color(label):
    colors = {
        'DINGIN': '#3498db',
        'NORMAL': '#2ecc71',
        'PANAS': '#e74c3c',
        'UNKNOWN': '#95a5a6',
        'ERROR': '#f39c12'
    }
    return colors.get(label, '#95a5a6')

def predict_with_models(models, scaler, temperature, humidity, hour=None, minute=None):
    """Predict dengan semua model dari CSV.PKL"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            if scaler is not None and hasattr(scaler, 'transform'):
                try:
                    features_scaled = scaler.transform(features)
                except:
                    features_scaled = features
            else:
                features_scaled = features
            
            pred_code = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                try:
                    probs = model.predict_proba(features_scaled)[0]
                    confidence = probs[pred_code] if len(probs) > pred_code else 1.0
                except:
                    confidence = 1.0
                    probs = [0, 0, 0]
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(pred_code, 'UNKNOWN')
            
            predictions[model_name] = {
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                    'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                    'PANAS': float(probs[2]) if len(probs) > 2 else 0
                },
                'color': get_label_color(label)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'color': '#f39c12'
            }
    
    return predictions

# ==================== MAIN DASHBOARD ====================
def main():
    st.title("🌡️ DHT11 Real-Time Dashboard")
    st.markdown("**Live sensor data • ML Predictions • CSV Download**")
    
    # Initialize session state
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = load_csv_data()
    
    # ===== SIDEBAR =====
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # Load ML Models
        st.subheader("🤖 ML Models")
        if st.button("🔄 Load Models", use_container_width=True):
            st.cache_resource.clear()
            st.rerun()
        
        # Model Selection
        show_dt = st.checkbox("Decision Tree", value=True, key="show_dt")
        show_knn = st.checkbox("K-Nearest Neighbors", value=True, key="show_knn")
        show_lr = st.checkbox("Logistic Regression", value=True, key="show_lr")
        
        # HiveMQ Connection
        st.subheader("📡 HiveMQ")
        use_hivemq = st.checkbox("Enable HiveMQ", value=True, key="use_hivemq")
        
        # Display Settings
        st.subheader("📊 Display")
        auto_refresh = st.checkbox("Auto Refresh", value=True, key="auto_refresh")
        refresh_rate = st.slider("Refresh (seconds)", 1, 10, 3)
        
        # CSV Data Management
        st.subheader("📁 CSV Data")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Load CSV", use_container_width=True):
                st.session_state.csv_data = load_csv_data()
                st.success(f"Loaded {len(st.session_state.csv_data)} records")
        
        with col2:
            if st.button("🗑️ Clear CSV", use_container_width=True):
                if os.path.exists(CSV_FILE):
                    os.remove(CSV_FILE)
                    st.session_state.csv_data = pd.DataFrame()
                    st.success("CSV cleared")
        
        # System Info
        st.subheader("📈 System Info")
        st.write(f"CSV Records: {len(st.session_state.csv_data)}")
        st.write(f"Last: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
    
    # Load ML Models
    all_models, scaler = load_models_from_csv_pkl()
    
    if not all_models:
        st.error("❌ No ML models found. Please run training first.")
        st.stop()
    
    # Filter active models
    selected_models = {
        'Decision Tree': show_dt,
        'K-Nearest Neighbors': show_knn,
        'Logistic Regression': show_lr
    }
    
    active_models = {name: model for name, model in all_models.items() 
                    if selected_models.get(name, True)}
    
    # ===== REAL-TIME DATA SECTION =====
    st.markdown("---")
    st.subheader("📡 REAL-TIME SENSOR DATA")
    
    # Initialize HiveMQ
    hivemq_client = None
    latest_data = None
    messages_history = []
    
    if use_hivemq:
        try:
            hivemq_client, latest_data, messages_history = connect_to_hivemq()
            
            if hivemq_client:
                st.success("✅ Connected to HiveMQ")
            else:
                st.warning("⚠️ Could not connect to HiveMQ")
                use_hivemq = False
        except Exception as e:
            st.warning(f"⚠️ HiveMQ Error: {e}")
            use_hivemq = False
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if latest_data and 'temperature' in latest_data:
            temp = latest_data['temperature']
            st.metric("🌡️ Temperature", f"{temp}°C", delta="LIVE")
        else:
            temp = 24.0
            st.metric("🌡️ Temperature", f"{temp}°C", delta="MANUAL")
    
    with col2:
        if latest_data and 'humidity' in latest_data:
            hum = latest_data['humidity']
            st.metric("💧 Humidity", f"{hum}%", delta="LIVE")
        else:
            hum = 65.0
            st.metric("💧 Humidity", f"{hum}%", delta="MANUAL")
    
    with col3:
        data_count = len(messages_history) if messages_history else 0
        st.metric("📊 Live Points", data_count)
    
    with col4:
        csv_count = len(st.session_state.csv_data)
        st.metric("📁 CSV Records", csv_count)
    
    with col5:
        status = "🟢 LIVE" if hivemq_client else "🔴 OFFLINE"
        st.metric("📡 Status", status)
    
    # ===== ML PREDICTIONS SECTION =====
    st.markdown("---")
    st.subheader("🤖 ML PREDICTIONS")
    
    # Use latest data or manual
    if latest_data:
        temp = latest_data.get('temperature', 24.0)
        hum = latest_data.get('humidity', 65.0)
        data_source = "HiveMQ LIVE"
    else:
        # Manual input fallback
        col1, col2 = st.columns(2)
        with col1:
            temp = st.number_input("Temperature (°C)", 15.0, 35.0, 24.0, 0.5, key="manual_temp")
        with col2:
            hum = st.number_input("Humidity (%)", 30.0, 90.0, 65.0, 1.0, key="manual_hum")
        data_source = "MANUAL INPUT"
    
    hour = datetime.now().hour
    minute = datetime.now().minute
    
    # Make predictions
    predictions = predict_with_models(active_models, scaler, temp, hum, hour, minute)
    
    # Display predictions
    if predictions:
        cols = st.columns(min(3, len(predictions)))
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            if idx >= len(cols):
                break
                
            with cols[idx]:
                color = pred['color']
                
                # Prediction card
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, {color}15, white);
                    border-radius: 12px;
                    padding: 20px;
                    border: 2px solid {color};
                    margin-bottom: 15px;
                    text-align: center;
                ">
                    <h3 style="color: #333; margin: 0 0 10px 0;">
                        {model_name}
                    </h3>
                    <h1 style="color: {color}; margin: 15px 0; font-size: 28px;">
                        {pred['label']}
                    </h1>
                    <div style="background: {color}20; padding: 8px 15px; border-radius: 20px;">
                        <span style="color: #333; font-weight: bold;">
                            {pred['confidence']:.1%} confidence
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Probability bars
                if 'probabilities' in pred:
                    prob_df = pd.DataFrame({
                        'Label': ['DINGIN', 'NORMAL', 'PANAS'],
                        'Probability': [
                            pred['probabilities']['DINGIN'],
                            pred['probabilities']['NORMAL'],
                            pred['probabilities']['PANAS']
                        ]
                    })
                    
                    fig = px.bar(
                        prob_df,
                        x='Label',
                        y='Probability',
                        color='Label',
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        range_y=[0, 1]
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=180,
                        margin=dict(t=0, b=0, l=0, r=0),
                        yaxis=dict(tickformat=".0%")
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Model agreement
        labels = [p['label'] for p in predictions.values()]
        if len(set(labels)) == 1:
            st.success(f"✅ **ALL MODELS AGREE:** {labels[0]}")
        else:
            st.warning(f"⚠️ **MODELS DISAGREE:** {', '.join(set(labels))}")
    
    # ===== CSV DATA DOWNLOAD SECTION =====
    st.markdown("---")
    st.subheader("📁 CSV DATA MANAGEMENT")
    
    tab1, tab2, tab3 = st.tabs(["📥 Download", "📊 Preview", "📈 Visualization"])
    
    with tab1:
        st.markdown("### Download CSV Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download current CSV
            if not st.session_state.csv_data.empty:
                csv = st.session_state.csv_data.to_csv(index=False, sep=';')
                st.download_button(
                    label="📥 Download Current CSV",
                    data=csv,
                    file_name="dht11_current_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No CSV data available")
        
        with col2:
            # Download with timestamp
            if not st.session_state.csv_data.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv = st.session_state.csv_data.to_csv(index=False, sep=';')
                st.download_button(
                    label="📅 Download with Timestamp",
                    data=csv,
                    file_name=f"dht11_data_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Download live data
            if messages_history:
                live_df = pd.DataFrame(messages_history)
                csv = live_df.to_csv(index=False)
                st.download_button(
                    label="📡 Download Live Data",
                    data=csv,
                    file_name="dht11_live_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("No live data available")
    
    with tab2:
        st.markdown("### CSV Data Preview")
        
        if not st.session_state.csv_data.empty:
            st.dataframe(
                st.session_state.csv_data,
                use_container_width=True,
                height=300
            )
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", len(st.session_state.csv_data))
            with col2:
                if 'temperature' in st.session_state.csv_data.columns:
                    avg_temp = st.session_state.csv_data['temperature'].mean()
                    st.metric("Avg Temp", f"{avg_temp:.1f}°C")
            with col3:
                if 'humidity' in st.session_state.csv_data.columns:
                    avg_hum = st.session_state.csv_data['humidity'].mean()
                    st.metric("Avg Hum", f"{avg_hum:.1f}%")
        else:
            st.info("No CSV data available")
    
    with tab3:
        st.markdown("### Data Visualization")
        
        if not st.session_state.csv_data.empty and len(st.session_state.csv_data) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'temperature' in st.session_state.csv_data.columns:
                    fig = px.histogram(
                        st.session_state.csv_data,
                        x='temperature',
                        title='Temperature Distribution',
                        color_discrete_sequence=['#e74c3c']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'humidity' in st.session_state.csv_data.columns:
                    fig = px.histogram(
                        st.session_state.csv_data,
                        x='humidity',
                        title='Humidity Distribution',
                        color_discrete_sequence=['#3498db']
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # Scatter plot WITHOUT trendline (remove statsmodels dependency)
            if 'temperature' in st.session_state.csv_data.columns and 'humidity' in st.session_state.csv_data.columns:
                fig = px.scatter(
                    st.session_state.csv_data,
                    x='temperature',
                    y='humidity',
                    title='Temperature vs Humidity'
                    # REMOVED: trendline='ols' - cause of error
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need more data for visualization")
    
    # ===== REAL-TIME CHARTS =====
    if use_hivemq and messages_history and len(messages_history) > 1:
        st.markdown("---")
        st.subheader("📈 REAL-TIME CHARTS")
        
        df_history = pd.DataFrame(messages_history)
        
        if 'received_time' in df_history.columns and 'temperature' in df_history.columns:
            # Temperature trend
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_history['received_time'],
                y=df_history['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#e74c3c', width=2)
            ))
            
            fig.update_layout(
                title='Live Temperature Trend',
                height=300,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== FOOTER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 System Summary**")
        st.write(f"- Active Models: {len(active_models)}")
        st.write(f"- CSV Records: {len(st.session_state.csv_data)}")
        st.write(f"- Live Data: {len(messages_history)} points")
    
    with col2:
        st.markdown("**⚡ Quick Actions**")
        if st.button("🔄 Refresh All", use_container_width=True):
            st.session_state.last_refresh = datetime.now()
            st.session_state.csv_data = load_csv_data()
            st.rerun()
    
    with col3:
        st.markdown("**📁 Export Options**")
        if st.button("📊 Export Report", use_container_width=True):
            st.info("Report generation feature coming soon!")
    
    # Auto-refresh
    if auto_refresh:
        current_time = datetime.now()
        if (current_time - st.session_state.last_refresh).seconds >= refresh_rate:
            st.session_state.last_refresh = current_time
            time.sleep(0.1)
            st.rerun()

if __name__ == "__main__":
    main()

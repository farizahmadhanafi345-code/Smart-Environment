# dashboard_iot_complete.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import pickle
import os
import time
import paho.mqtt.client as mqtt
import ssl
import threading
import warnings
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "IoT_Dataset")
CSV_FILE = os.path.join(DATA_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(DATA_DIR, "ml_models")

# Konfigurasi HiveMQ REAL
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1760514170127",
    "password": "0r8ULyh9&duT1,BHg%.M",
    "use_ssl": True,
    "keepalive": 20
}

# Topics REAL
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="IoT Smart Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected { background-color: #10B981; }
    .status-disconnected { background-color: #EF4444; }
    .status-waiting { background-color: #F59E0B; }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border: 2px solid;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }
    
    .data-table {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }
    
    .log-container {
        background: #1F2937;
        color: #E5E7EB;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.9rem;
    }
    
    /* Color coding for labels */
    .label-dingin { color: #3B82F6; font-weight: bold; }
    .label-normal { color: #10B981; font-weight: bold; }
    .label-panas { color: #EF4444; font-weight: bold; }
    
    /* Animation for real-time updates */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==================== INISIALISASI STATE ====================
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        "temperature": 25.0,
        "humidity": 65.0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "label": "NORMAL",
        "label_encoded": 1,
        "last_update": datetime.now()
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None
    st.session_state.mqtt_connected = False

if 'data_history' not in st.session_state:
    st.session_state.data_history = []

if 'system_logs' not in st.session_state:
    st.session_state.system_logs = []

# ==================== FUNGSI MQTT ====================
def add_log(message):
    """Tambahkan log ke sistem"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    st.session_state.system_logs.append(log_entry)
    
    # Batasi jumlah log
    if len(st.session_state.system_logs) > 50:
        st.session_state.system_logs = st.session_state.system_logs[-50:]

def on_mqtt_connect(client, userdata, flags, rc):
    """Callback ketika terhubung ke MQTT"""
    if rc == 0:
        st.session_state.mqtt_connected = True
        client.subscribe(DHT_TOPIC)
        add_log("✅ Connected to HiveMQ Cloud")
    else:
        st.session_state.mqtt_connected = False
        add_log(f"❌ Connection failed with code: {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    """Callback ketika terputus dari MQTT"""
    st.session_state.mqtt_connected = False
    add_log("⚠️ Disconnected from HiveMQ")

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima data sensor"""
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        # Tentukan label berdasarkan suhu
        if temperature < 25:
            label = "DINGIN"
            label_encoded = 0
        elif temperature > 28:
            label = "PANAS"
            label_encoded = 2
        else:
            label = "NORMAL"
            label_encoded = 1
        
        # Update sensor data
        st.session_state.sensor_data.update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "label": label,
            "label_encoded": label_encoded,
            "last_update": datetime.now()
        })
        
        # Tambah ke history
        st.session_state.data_history.append({
            "timestamp": datetime.now(),
            "temperature": temperature,
            "humidity": humidity,
            "label": label
        })
        
        # Batasi history
        if len(st.session_state.data_history) > 100:
            st.session_state.data_history = st.session_state.data_history[-100:]
        
        add_log(f"📡 Received: {temperature}°C, {humidity}% → {label}")
        
    except Exception as e:
        add_log(f"❌ Error processing MQTT message: {e}")

def connect_mqtt():
    """Connect ke HiveMQ Cloud"""
    try:
        if st.session_state.mqtt_client and st.session_state.mqtt_connected:
            return True
        
        client = mqtt.Client()
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
        
        client.on_connect = on_mqtt_connect
        client.on_disconnect = on_mqtt_disconnect
        client.on_message = on_mqtt_message
        
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=20)
        client.loop_start()
        
        st.session_state.mqtt_client = client
        
        # Tunggu koneksi
        time.sleep(2)
        return True
        
    except Exception as e:
        add_log(f"❌ MQTT Connection failed: {e}")
        return False

def send_led_command(command):
    """Kirim perintah ke LED"""
    if st.session_state.mqtt_client and st.session_state.mqtt_connected:
        try:
            st.session_state.mqtt_client.publish(LED_TOPIC, command)
            add_log(f"💡 LED command sent: {command}")
            return True
        except:
            return False
    return False

# ==================== LOAD MODELS & DATA ====================
@st.cache_resource
def load_ml_models():
    """Load ML models dari file"""
    models = {}
    
    try:
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            scaler = None
            add_log("⚠️ Scaler file not found")
        
        # Load models
        model_files = [
            ('Decision Tree', 'decision_tree.pkl'),
            ('Random Forest', 'random_forest.pkl'),
            ('K-Nearest Neighbors', 'k_nearest_neighbors.pkl'),
            ('Logistic Regression', 'logistic_regression.pkl')
        ]
        
        for name, filename in model_files:
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                add_log(f"✅ Loaded model: {name}")
            else:
                add_log(f"⚠️ Model not found: {filename}")
        
        return models, scaler
    
    except Exception as e:
        add_log(f"❌ Error loading models: {e}")
        return None, None

def load_dataset():
    """Load dataset dari CSV file"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, delimiter=';')
            
            # Konversi timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['time'] = df['timestamp'].dt.strftime('%H:%M')
                df['date'] = df['timestamp'].dt.date
            
            add_log(f"📊 Dataset loaded: {len(df)} records")
            return df
        else:
            add_log("📝 Dataset file not found, starting empty")
            return pd.DataFrame()
            
    except Exception as e:
        add_log(f"❌ Error loading dataset: {e}")
        return pd.DataFrame()

def predict_with_models(models, scaler, temperature, humidity, hour=None, minute=None):
    """Prediksi dengan semua model ML"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    if not models or scaler is None:
        return {}
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Prepare features
            features = np.array([[temperature, humidity, hour, minute]])
            features_scaled = scaler.transform(features)
            
            # Predict
            pred_code = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code]
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            # Map to label
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
                'label_encoded': int(pred_code)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e)
            }
    
    return predictions

# ==================== SIDEBAR ====================
def sidebar_controls():
    """Sidebar controls untuk dashboard"""
    with st.sidebar:
        st.markdown('<div class="main-header"><h3>⚙️ CONTROL PANEL</h3></div>', unsafe_allow_html=True)
        
        # Status koneksi
        st.subheader("🔗 Connection Status")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.session_state.mqtt_connected:
                st.markdown('<span class="status-indicator status-connected"></span> CONNECTED', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-indicator status-disconnected"></span> DISCONNECTED', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.sensor_data['timestamp']:
                last_update = datetime.now() - st.session_state.sensor_data['last_update']
                seconds = last_update.total_seconds()
                if seconds < 10:
                    st.success("LIVE")
                elif seconds < 60:
                    st.warning(f"{int(seconds)}s ago")
                else:
                    st.error(f"{int(seconds/60)}m ago")
        
        # Kontrol koneksi
        st.subheader("🌐 MQTT Control")
        col_conn1, col_conn2 = st.columns(2)
        
        with col_conn1:
            if st.button("🔗 Connect", use_container_width=True, type="primary"):
                with st.spinner("Connecting..."):
                    if connect_mqtt():
                        st.success("Connected!")
                        time.sleep(1)
                        st.rerun()
        
        with col_conn2:
            if st.button("🔌 Disconnect", use_container_width=True):
                if st.session_state.mqtt_client:
                    st.session_state.mqtt_client.loop_stop()
                    st.session_state.mqtt_connected = False
                    add_log("Disconnected from HiveMQ")
                    st.warning("Disconnected")
                    st.rerun()
        
        st.markdown("---")
        
        # Manual input untuk testing
        st.subheader("🧪 Manual Testing")
        
        manual_temp = st.slider("Temperature (°C)", 15.0, 35.0, 
                               st.session_state.sensor_data['temperature'], 0.1)
        manual_hum = st.slider("Humidity (%)", 30.0, 90.0,
                              st.session_state.sensor_data['humidity'], 0.1)
        
        if st.button("🔮 Predict with ML", use_container_width=True):
            # Update sensor data dengan manual input
            st.session_state.sensor_data.update({
                "temperature": manual_temp,
                "humidity": manual_hum,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "last_update": datetime.now()
            })
            add_log(f"Manual input: {manual_temp}°C, {manual_hum}%")
            st.rerun()
        
        st.markdown("---")
        
        # Kontrol LED
        st.subheader("💡 LED Control")
        
        col_led1, col_led2 = st.columns(2)
        with col_led1:
            if st.button("🔴 RED", use_container_width=True):
                send_led_command("merah")
            
            if st.button("🟢 GREEN", use_container_width=True):
                send_led_command("hijau")
        
        with col_led2:
            if st.button("🟡 YELLOW", use_container_width=True):
                send_led_command("kuning")
            
            if st.button("⚫ OFF", use_container_width=True):
                send_led_command("off")
        
        st.markdown("---")
        
        # Dataset control
        st.subheader("📊 Dataset")
        
        dataset_df = load_dataset()
        if not dataset_df.empty:
            st.info(f"📁 {len(dataset_df)} records loaded")
            
            # Download button
            csv_data = dataset_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"iot_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.warning("No dataset available")
        
        # Refresh button
        if st.button("🔄 Refresh Dashboard", use_container_width=True, type="secondary"):
            st.rerun()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", value=False)
        
        return auto_refresh

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌡️ IoT SMART DASHBOARD</h1>
        <h4>Real-time Sensor Monitoring & ML Prediction System</h4>
        <p>Connected to ESP32 DHT11 via HiveMQ Cloud</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    auto_refresh = sidebar_controls()
    
    # Load models dan data
    models, scaler = load_ml_models()
    dataset_df = load_dataset()
    
    # ============ ROW 1: REAL-TIME SENSOR DATA ============
    st.subheader("📡 LIVE SENSOR DATA")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.sensor_data['temperature']
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #EF4444;">
            <h3 style="color: #EF4444; margin-top: 0;">🌡️ TEMPERATURE</h3>
            <h1 style="color: #EF4444; font-size: 2.8rem;">{temp:.1f} °C</h1>
            <p>Real-time from DHT11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.sensor_data['humidity']
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #3B82F6;">
            <h3 style="color: #3B82F6; margin-top: 0;">💧 HUMIDITY</h3>
            <h1 style="color: #3B82F6; font-size: 2.8rem;">{hum:.1f} %</h1>
            <p>Real-time from DHT11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        label = st.session_state.sensor_data['label']
        label_class = f"label-{label.lower()}"
        
        # Tentukan warna berdasarkan label
        if label == "DINGIN":
            color = "#3B82F6"
            icon = "🥶"
        elif label == "PANAS":
            color = "#EF4444"
            icon = "🔥"
        else:
            color = "#10B981"
            icon = "✅"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {color};">
            <h3 style="color: {color}; margin-top: 0;">🏷️ STATUS</h3>
            <h1 style="color: {color}; font-size: 2.8rem;">{icon} {label}</h1>
            <p>Based on temperature threshold</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        last_update = st.session_state.sensor_data['last_update']
        time_diff = datetime.now() - last_update
        seconds = time_diff.total_seconds()
        
        if seconds < 5:
            status_color = "#10B981"
            status_text = "LIVE"
        elif seconds < 30:
            status_color = "#F59E0B"
            status_text = f"{int(seconds)}s ago"
        else:
            status_color = "#EF4444"
            status_text = f"{int(seconds)}s ago"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {status_color};">
            <h3 style="color: {status_color}; margin-top: 0;">🕐 LAST UPDATE</h3>
            <h1 style="color: {status_color}; font-size: 2.8rem;">{status_text}</h1>
            <p>{st.session_state.sensor_data['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ============ ROW 2: ML PREDICTIONS ============
    st.subheader("🤖 MACHINE LEARNING PREDICTIONS")
    
    if models and scaler:
        # Get predictions
        predictions = predict_with_models(
            models, scaler,
            st.session_state.sensor_data['temperature'],
            st.session_state.sensor_data['humidity'],
            datetime.now().hour,
            datetime.now().minute
        )
        
        if predictions:
            # Display predictions in columns
            n_models = len(predictions)
            pred_cols = st.columns(n_models)
            
            for idx, (model_name, pred) in enumerate(predictions.items()):
                with pred_cols[idx]:
                    # Tentukan warna berdasarkan label
                    label_color = {
                        'DINGIN': '#3B82F6',
                        'NORMAL': '#10B981',
                        'PANAS': '#EF4444',
                        'ERROR': '#6B7280'
                    }.get(pred['label'], '#6B7280')
                    
                    st.markdown(f"""
                    <div class="prediction-card" style="border-color: {label_color};">
                        <h3 style="color: {label_color};">{model_name}</h3>
                        <h1 style="color: {label_color}; font-size: 2.5rem; margin: 1rem 0;">
                            {pred['label']}
                        </h1>
                        <p style="font-size: 1.2rem;">
                            Confidence: <strong>{pred.get('confidence', 0):.1%}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Model agreement
            st.subheader("📊 Model Agreement Analysis")
            
            labels = [pred['label'] for pred in predictions.values() if pred['label'] != 'ERROR']
            if labels:
                unique_labels = set(labels)
                
                if len(unique_labels) == 1:
                    st.success(f"✅ All models agree: **{list(unique_labels)[0]}**")
                    st.balloons()
                else:
                    st.warning(f"⚠️ Models disagree: {', '.join(unique_labels)}")
                    
                    # Show detailed probabilities
                    with st.expander("View Detailed Probabilities"):
                        prob_data = []
                        for model_name, pred in predictions.items():
                            if 'probabilities' in pred:
                                row = {'Model': model_name}
                                row.update(pred['probabilities'])
                                prob_data.append(row)
                        
                        if prob_data:
                            prob_df = pd.DataFrame(prob_data)
                            st.dataframe(prob_df, use_container_width=True)
            
            # Prediction comparison chart
            if any('probabilities' in pred for pred in predictions.values()):
                st.subheader("📈 Prediction Probability Comparison")
                
                # Prepare data for chart
                chart_data = []
                for model_name, pred in predictions.items():
                    if 'probabilities' in pred:
                        for label, prob in pred['probabilities'].items():
                            chart_data.append({
                                'Model': model_name,
                                'Label': label,
                                'Probability': prob
                            })
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data)
                    
                    fig = px.bar(
                        chart_df,
                        x='Model',
                        y='Probability',
                        color='Label',
                        barmode='group',
                        color_discrete_map={
                            'DINGIN': '#3B82F6',
                            'NORMAL': '#10B981',
                            'PANAS': '#EF4444'
                        },
                        title="Model Prediction Probabilities"
                    )
                    
                    fig.update_layout(
                        height=400,
                        showlegend=True,
                        yaxis_range=[0, 1]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No predictions available. Check if models are loaded correctly.")
    else:
        st.warning("ML models not loaded. Run training first or check model files.")
    
    st.markdown("---")
    
    # ============ ROW 3: DATA VISUALIZATION ============
    st.subheader("📊 DATA VISUALIZATION & HISTORY")
    
    # Tabs untuk berbagai visualisasi
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Trends", "🎯 Temperature Analysis", "📋 Historical Data", "📝 System Logs"])
    
    with tab1:
        # Real-time trends dari history
        if len(st.session_state.data_history) > 1:
            history_df = pd.DataFrame(st.session_state.data_history)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Temperature Trend', 'Humidity Trend'),
                vertical_spacing=0.1
            )
            
            # Temperature plot
            fig.add_trace(
                go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['temperature'],
                    mode='lines+markers',
                    name='Temperature',
                    line=dict(color='#EF4444', width=2),
                    marker=dict(size=6)
                ),
                row=1, col=1
            )
            
            # Add threshold lines
            fig.add_hline(y=25, line_dash="dash", line_color="#3B82F6", 
                         annotation_text="Cold (<25°C)", row=1, col=1)
            fig.add_hline(y=28, line_dash="dash", line_color="#EF4444",
                         annotation_text="Hot (>28°C)", row=1, col=1)
            
            # Humidity plot
            fig.add_trace(
                go.Scatter(
                    x=history_df['timestamp'],
                    y=history_df['humidity'],
                    mode='lines+markers',
                    name='Humidity',
                    line=dict(color='#3B82F6', width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=True,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Collecting real-time data... Connect to HiveMQ to see live trends.")
    
    with tab2:
        # Temperature analysis
        if not dataset_df.empty:
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                # Temperature distribution
                fig1 = px.histogram(
                    dataset_df,
                    x='temperature',
                    nbins=20,
                    title='Temperature Distribution',
                    color_discrete_sequence=['#EF4444']
                )
                fig1.update_layout(height=300)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_analysis2:
                # Temperature vs Humidity scatter
                fig2 = px.scatter(
                    dataset_df,
                    x='temperature',
                    y='humidity',
                    color='label',
                    title='Temperature vs Humidity',
                    color_discrete_map={
                        'DINGIN': '#3B82F6',
                        'NORMAL': '#10B981',
                        'PANAS': '#EF4444'
                    },
                    hover_data=['timestamp']
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Label distribution pie chart
            if 'label' in dataset_df.columns:
                label_counts = dataset_df['label'].value_counts()
                fig3 = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title='Label Distribution in Dataset',
                    color=label_counts.index,
                    color_discrete_map={
                        'DINGIN': '#3B82F6',
                        'NORMAL': '#10B981',
                        'PANAS': '#EF4444'
                    }
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No historical data available. Collect data first.")
    
    with tab3:
        # Historical data table
        if not dataset_df.empty:
            st.subheader("📋 Dataset Preview")
            
            # Filter controls
            col_filter1, col_filter2 = st.columns(2)
            with col_filter1:
                date_filter = st.selectbox("Filter by date", 
                                         ['All dates'] + sorted(dataset_df['date'].unique().astype(str).tolist()))
            
            with col_filter2:
                label_filter = st.selectbox("Filter by label", 
                                          ['All labels'] + sorted(dataset_df['label'].unique().tolist()))
            
            # Apply filters
            filtered_df = dataset_df.copy()
            if date_filter != 'All dates':
                filtered_df = filtered_df[filtered_df['date'].astype(str) == date_filter]
            if label_filter != 'All labels':
                filtered_df = filtered_df[filtered_df['label'] == label_filter]
            
            # Show statistics
            st.write(f"**Showing {len(filtered_df)} of {len(dataset_df)} records**")
            
            # Display table
            display_cols = ['timestamp', 'temperature', 'humidity', 'label']
            if all(col in filtered_df.columns for col in display_cols):
                st.dataframe(
                    filtered_df[display_cols].sort_values('timestamp', ascending=False).head(50),
                    use_container_width=True,
                    height=400
                )
            else:
                st.dataframe(filtered_df.head(50), use_container_width=True, height=400)
            
            # Download filtered data
            csv_filtered = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv_filtered,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No historical data available.")
    
    with tab4:
        # System logs
        st.subheader("📝 System Logs")
        
        # Log container
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        
        # Display logs (newest first)
        logs = st.session_state.system_logs[::-1]
        for log in logs[:30]:  # Show last 30 logs
            # Color code log messages
            if "✅" in log or "Connected" in log:
                st.markdown(f'<span style="color: #10B981;">{log}</span>', unsafe_allow_html=True)
            elif "❌" in log or "Error" in log or "failed" in log.lower():
                st.markdown(f'<span style="color: #EF4444;">{log}</span>', unsafe_allow_html=True)
            elif "⚠️" in log or "Warning" in log or "Disconnected" in log:
                st.markdown(f'<span style="color: #F59E0B;">{log}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color: #E5E7EB;">{log}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            st.session_state.system_logs = []
            st.rerun()
    
    st.markdown("---")
    
    # ============ ROW 4: SYSTEM STATUS ============
    st.subheader("⚙️ SYSTEM STATUS")
    
    col_status1, col_status2, col_status3 = st.columns(3)
    
    with col_status1:
        st.markdown("""
        <div class="data-table">
            <h4>🔗 Connection Status</h4>
            <p><strong>MQTT:</strong> {'✅ Connected' if st.session_state.mqtt_connected else '❌ Disconnected'}</p>
            <p><strong>Broker:</strong> HiveMQ Cloud</p>
            <p><strong>Topic:</strong> {DHT_TOPIC}</p>
        </div>
        """.format(DHT_TOPIC=DHT_TOPIC), unsafe_allow_html=True)
    
    with col_status2:
        st.markdown(f"""
        <div class="data-table">
            <h4>📊 Data Status</h4>
            <p><strong>Live Data:</strong> {'✅ Receiving' if len(st.session_state.data_history) > 0 else '⏳ Waiting'}</p>
            <p><strong>History Size:</strong> {len(st.session_state.data_history)} records</p>
            <p><strong>Dataset Size:</strong> {len(dataset_df)} records</p>
            <p><strong>Last Data:</strong> {st.session_state.sensor_data['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_status3:
        ml_status = "✅ Loaded" if models and len(models) > 0 else "⚠️ Not Loaded"
        model_count = len(models) if models else 0
        
        st.markdown(f"""
        <div class="data-table">
            <h4>🤖 ML System Status</h4>
            <p><strong>ML Models:</strong> {ml_status}</p>
            <p><strong>Models Loaded:</strong> {model_count}</p>
            <p><strong>Scaler:</strong> {'✅ Loaded' if scaler else '⚠️ Not Loaded'}</p>
            <p><strong>Predictions:</strong> Active</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 1rem;">
        <p><strong>🌐 IoT Smart Dashboard v2.0</strong> | Real-time DHT11 Monitoring & ML Prediction</p>
        <p>Connected to ESP32 via HiveMQ Cloud | Powered by Streamlit & Scikit-learn</p>
        <p>© 2024 IoT Smart Monitoring System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh jika diaktifkan
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ==================== RUN DASHBOARD ====================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Dashboard Error: {str(e)}")
        st.info("Please refresh the page or check your connection.")

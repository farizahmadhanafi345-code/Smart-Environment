# dashboard_final_complete.py
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
import threading
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
BASE_DIR = "Data_Collector"
MODELS_DIR = os.path.join(BASE_DIR, "models")
CSV_FILE = os.path.join(BASE_DIR, "sensor_data.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# MQTT Configuration - SAMA dengan ESP32
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"

# Topics - SAMA dengan ESP32
TOPIC_SENSOR = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
TOPIC_LED = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"
TOPIC_PREDICTION = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_prediction"

# Buffer untuk data real-time
BUFFER_SIZE = 100

# ==================== GLOBAL VARIABLES ====================
# Inisialisasi session state
if 'realtime_data' not in st.session_state:
    st.session_state.realtime_data = deque(maxlen=BUFFER_SIZE)
    
if 'realtime_predictions' not in st.session_state:
    st.session_state.realtime_predictions = deque(maxlen=BUFFER_SIZE)

if 'ml_models' not in st.session_state:
    st.session_state.ml_models = None

if 'ml_scaler' not in st.session_state:
    st.session_state.ml_scaler = None

if 'ml_metadata' not in st.session_state:
    st.session_state.ml_metadata = None

if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

if 'mqtt_connected' not in st.session_state:
    st.session_state.mqtt_connected = False

# ==================== LOAD ML MODELS (DARI TRAINING) ====================
@st.cache_resource
def load_ml_models():
    """Load ML models dari hasil training sebelumnya"""
    try:
        print("📦 Loading ML models from training...")
        
        models = {}
        metadata = {}
        scaler = None
        
        # 1. Load metadata
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✅ Loaded metadata: {list(metadata.get('models_trained', []))}")
        
        # 2. Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Loaded scaler")
        
        # 3. Load individual models
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
                print(f"✅ Loaded {name}")
        
        # 4. Load historical data
        historical_data = None
        if os.path.exists(CSV_FILE):
            historical_data = pd.read_csv(CSV_FILE, delimiter=';')
            print(f"✅ Loaded historical data: {len(historical_data)} records")
        
        return models, scaler, metadata, historical_data
        
    except Exception as e:
        st.error(f"❌ Error loading ML models: {e}")
        return None, None, {}, None

# ==================== ML PREDICTION FUNCTIONS ====================
def make_ml_prediction(temperature, humidity, hour=None, minute=None):
    """Buat prediksi menggunakan semua model ML yang sudah di-training"""
    if st.session_state.ml_models is None or st.session_state.ml_scaler is None:
        return None
    
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    # Prepare features (SAMA dengan training)
    features = np.array([[temperature, humidity, hour, minute]])
    features_scaled = st.session_state.ml_scaler.transform(features)
    
    predictions = {}
    
    for model_name, model in st.session_state.ml_models.items():
        try:
            # Predict
            pred_code = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code] * 100
            else:
                confidence = 100.0
                probs = [0, 0, 0]
            
            # Map to label (SAMA dengan training)
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(pred_code, 'UNKNOWN')
            
            predictions[model_name] = {
                'label': label,
                'label_encoded': int(pred_code),
                'confidence': float(confidence),
                'probabilities': {
                    'DINGIN': float(probs[0]) * 100 if len(probs) > 0 else 0,
                    'NORMAL': float(probs[1]) * 100 if len(probs) > 1 else 0,
                    'PANAS': float(probs[2]) * 100 if len(probs) > 2 else 0
                },
                'color': get_label_color(label),
                'icon': get_label_icon(label),
                'description': get_label_description(label)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'error': str(e),
                'color': '#f39c12',
                'icon': '❌'
            }
    
    return predictions

def get_label_color(label):
    colors = {
        'DINGIN': '#3498db',    # Blue
        'NORMAL': '#2ecc71',    # Green
        'PANAS': '#e74c3c',     # Red
        'UNKNOWN': '#95a5a6',
        'ERROR': '#f39c12'
    }
    return colors.get(label, '#95a5a6')

def get_label_icon(label):
    icons = {
        'DINGIN': '❄️',
        'NORMAL': '✅',
        'PANAS': '🔥',
        'UNKNOWN': '❓',
        'ERROR': '❌'
    }
    return icons.get(label, '❓')

def get_label_description(label):
    descriptions = {
        'DINGIN': 'Suhu < 25°C',
        'NORMAL': 'Suhu 25-28°C',
        'PANAS': 'Suhu > 28°C'
    }
    return descriptions.get(label, '')

# ==================== MQTT CLIENT ====================
class IntegratedMQTTClient:
    """MQTT Client yang TERHUBUNG dengan sistem ML"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        
    def connect(self):
        """Connect ke MQTT broker"""
        try:
            self.client = mqtt.Client()
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set()
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            
            # Connect
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            time.sleep(2)  # Wait for connection
            return True
            
        except Exception as e:
            st.error(f"❌ MQTT Connection failed: {str(e)}")
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            st.session_state.mqtt_connected = True
            
            # Subscribe ke topics
            client.subscribe([
                (TOPIC_SENSOR, 1),
                (TOPIC_PREDICTION, 1)
            ])
            
            # Kirim status connected
            client.publish(TOPIC_LED, "dashboard_online", qos=1)
            
            print(f"✅ MQTT Connected & Subscribed")
        else:
            print(f"❌ MQTT Connection failed: {rc}")
    
    def _on_message(self, client, userdata, msg):
        """Process incoming MQTT messages"""
        try:
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            timestamp = datetime.now()
            
            if msg.topic == TOPIC_SENSOR:
                # Data dari ESP32
                sensor_record = {
                    'timestamp': timestamp,
                    'temperature': float(data.get('temperature', 0)),
                    'humidity': float(data.get('humidity', 0)),
                    'label': data.get('label', 'UNKNOWN'),
                    'label_encoded': data.get('label_encoded', 1),
                    'date': data.get('date', timestamp.strftime('%Y-%m-%d')),
                    'time_str': data.get('timestamp', '00:00:00'),
                    'source': 'ESP32',
                    'record': data.get('record', 0)
                }
                
                # Simpan ke buffer
                st.session_state.realtime_data.append(sensor_record)
                
                # Buat prediksi ML OTOMATIS dari data sensor
                if (st.session_state.ml_models is not None and 
                    st.session_state.ml_scaler is not None):
                    
                    hour = int(sensor_record['time_str'].split(';')[0]) if ';' in sensor_record['time_str'] else timestamp.hour
                    minute = int(sensor_record['time_str'].split(';')[1]) if ';' in sensor_record['time_str'] else timestamp.minute
                    
                    ml_predictions = make_ml_prediction(
                        sensor_record['temperature'],
                        sensor_record['humidity'],
                        hour,
                        minute
                    )
                    
                    # Simpan prediksi ML
                    for model_name, prediction in ml_predictions.items():
                        pred_record = {
                            'timestamp': timestamp,
                            'model': model_name,
                            'prediction': prediction['label'],
                            'confidence': prediction['confidence'],
                            'temperature': sensor_record['temperature'],
                            'humidity': sensor_record['humidity'],
                            'source': 'ML_Model',
                            'label_encoded': prediction['label_encoded']
                        }
                        st.session_state.realtime_predictions.append(pred_record)
                
                print(f"📥 Sensor: {sensor_record['temperature']}°C, {sensor_record['humidity']}%")
                
            elif msg.topic == TOPIC_PREDICTION:
                # Data prediksi dari training system (jika ada)
                pred_record = {
                    'timestamp': timestamp,
                    'model': data.get('model', 'Unknown'),
                    'prediction': data.get('label', 'UNKNOWN'),
                    'confidence': float(data.get('confidence', 0)),
                    'temperature': float(data.get('temperature', 0)),
                    'humidity': float(data.get('humidity', 0)),
                    'source': 'Training_System',
                    'publish_time': data.get('publish_time', '')
                }
                st.session_state.realtime_predictions.append(pred_record)
                
                print(f"📥 ML Prediction: {pred_record['model']} -> {pred_record['prediction']}")
                
        except Exception as e:
            print(f"❌ Error processing MQTT: {str(e)}")
    
    def send_led_command(self, command):
        """Kirim perintah LED ke ESP32"""
        if self.connected and self.client:
            try:
                self.client.publish(TOPIC_LED, command, qos=1)
                print(f"📤 Sent LED: {command}")
                return True
            except:
                return False
        return False
    
    def disconnect(self):
        """Disconnect dari MQTT"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            st.session_state.mqtt_connected = False

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="ESP32 DHT Complete System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .system-header {
        background: linear-gradient(90deg, #1E88E5, #0D47A1);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .component-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid;
        margin-bottom: 1rem;
    }
    .ml-card { border-left-color: #9b59b6; }
    .sensor-card { border-left-color: #3498db; }
    .dashboard-card { border-left-color: #2ecc71; }
    .prediction-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
def create_sidebar():
    """Create sidebar dengan semua kontrol"""
    st.sidebar.title("🤖 Integrated System Controls")
    st.sidebar.markdown("---")
    
    # System Status
    st.sidebar.subheader("📊 System Status")
    
    ml_status = "✅ Loaded" if st.session_state.ml_models else "❌ Not Loaded"
    mqtt_status = "🟢 Connected" if st.session_state.mqtt_connected else "🔴 Disconnected"
    data_status = f"📈 {len(st.session_state.realtime_data)} points"
    
    st.sidebar.markdown(f"""
    - ML Models: {ml_status}
    - MQTT: {mqtt_status}
    - Real-time Data: {data_status}
    """)
    
    st.sidebar.markdown("---")
    
    # 1. ML Models Section
    st.sidebar.subheader("🧠 ML Models")
    
    if st.session_state.ml_models:
        for model_name in st.session_state.ml_models.keys():
            st.sidebar.markdown(f"✅ {model_name}")
        
        # Manual Prediction
        st.sidebar.markdown("**🔮 Manual Prediction:**")
        manual_temp = st.sidebar.slider("Temp (°C)", 15.0, 35.0, 24.0, 0.5)
        manual_hum = st.sidebar.slider("Hum (%)", 30.0, 90.0, 65.0, 1.0)
        
        if st.sidebar.button("Predict with ML", use_container_width=True):
            predictions = make_ml_prediction(manual_temp, manual_hum)
            if predictions:
                st.session_state.manual_predictions = predictions
                st.sidebar.success("Prediction made!")
    else:
        st.sidebar.warning("ML models not loaded")
        if st.sidebar.button("Load ML Models", use_container_width=True):
            models, scaler, metadata, historical = load_ml_models()
            st.session_state.ml_models = models
            st.session_state.ml_scaler = scaler
            st.session_state.ml_metadata = metadata
            st.session_state.historical_data = historical
            st.rerun()
    
    st.sidebar.markdown("---")
    
    # 2. MQTT Connection
    st.sidebar.subheader("📡 MQTT Connection")
    
    if not st.session_state.mqtt_connected:
        if st.sidebar.button("Connect to ESP32", type="primary", use_container_width=True):
            with st.spinner("Connecting..."):
                mqtt_client = IntegratedMQTTClient()
                if mqtt_client.connect():
                    st.session_state.mqtt_client = mqtt_client
                    st.rerun()
    else:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("Disconnect", use_container_width=True):
                if st.session_state.mqtt_client:
                    st.session_state.mqtt_client.disconnect()
                st.session_state.mqtt_connected = False
                st.rerun()
    
    st.sidebar.markdown("---")
    
    # 3. ESP32 Control
    st.sidebar.subheader("💡 ESP32 Control")
    
    if st.session_state.mqtt_connected:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔴 Merah", use_container_width=True):
                st.session_state.mqtt_client.send_led_command("merah")
        with col2:
            if st.button("🟢 Hijau", use_container_width=True):
                st.session_state.mqtt_client.send_led_command("hijau")
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            if st.button("🟡 Kuning", use_container_width=True):
                st.session_state.mqtt_client.send_led_command("kuning")
        with col4:
            if st.button("⚫ Off", use_container_width=True):
                st.session_state.mqtt_client.send_led_command("off")
    else:
        st.sidebar.info("Connect to ESP32 first")
    
    st.sidebar.markdown("---")
    
    # 4. Display Settings
    st.sidebar.subheader("⚙️ Display Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    show_ml_charts = st.sidebar.checkbox("Show ML Charts", value=True)
    show_raw_data = st.sidebar.checkbox("Show Raw Data", value=False)
    
    buffer_size = st.sidebar.slider("Buffer Size", 10, 200, BUFFER_SIZE, 10)
    
    if buffer_size != BUFFER_SIZE:
        st.session_state.realtime_data = deque(st.session_state.realtime_data, maxlen=buffer_size)
        st.session_state.realtime_predictions = deque(st.session_state.realtime_predictions, maxlen=buffer_size)
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Components:**
    1. ESP32 Data Collector
    2. ML Training Models  
    3. Real-time Dashboard
    4. MQTT Communication
    """)
    
    return {
        'auto_refresh': auto_refresh,
        'show_ml_charts': show_ml_charts,
        'show_raw_data': show_raw_data,
        'buffer_size': buffer_size
    }

# ==================== ML PERFORMANCE DASHBOARD ====================
def show_ml_performance():
    """Tampilkan performance ML models dari training"""
    if st.session_state.ml_metadata is None:
        return
    
    st.subheader("📊 ML Models Performance (From Training)")
    
    if 'performance' in st.session_state.ml_metadata:
        perf_data = st.session_state.ml_metadata['performance']
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame([
            {
                'Model': model,
                'Accuracy': perf['accuracy'],
                'Precision': perf['precision'],
                'Recall': perf['recall'],
                'F1-Score': perf['f1_score'],
                'CV Score': perf['cv_mean']
            }
            for model, perf in perf_data.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart comparison
            fig = px.bar(
                metrics_df.melt(id_vars=['Model'], value_vars=['Accuracy', 'F1-Score']),
                x='Model',
                y='value',
                color='variable',
                barmode='group',
                title="Model Accuracy & F1-Score",
                labels={'value': 'Score', 'variable': 'Metric'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart
            fig = go.Figure()
            
            for model in metrics_df['Model']:
                model_data = metrics_df[metrics_df['Model'] == model].iloc[0]
                fig.add_trace(go.Scatterpolar(
                    r=[
                        model_data['Accuracy'],
                        model_data['Precision'], 
                        model_data['Recall'],
                        model_data['F1-Score']
                    ],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=model
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Model Performance Comparison",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Show training info
        with st.expander("Training Information"):
            st.json(st.session_state.ml_metadata)

# ==================== REAL-TIME DASHBOARD ====================
def show_realtime_dashboard():
    """Tampilkan dashboard real-time"""
    st.subheader("📡 Real-time Data Stream")
    
    # Current Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.session_state.realtime_data:
            latest = list(st.session_state.realtime_data)[-1]
            st.metric("🌡️ Current Temp", f"{latest['temperature']:.1f}°C")
        else:
            st.metric("🌡️ Current Temp", "N/A")
    
    with col2:
        if st.session_state.realtime_data:
            latest = list(st.session_state.realtime_data)[-1]
            st.metric("💧 Current Hum", f"{latest['humidity']:.1f}%")
        else:
            st.metric("💧 Current Hum", "N/A")
    
    with col3:
        if st.session_state.realtime_predictions:
            latest_pred = list(st.session_state.realtime_predictions)[-1]
            st.metric("🤖 Latest Prediction", latest_pred['prediction'])
        else:
            st.metric("🤖 Latest Prediction", "N/A")
    
    with col4:
        st.metric("📊 Data Points", len(st.session_state.realtime_data))
    
    # Charts
    if st.session_state.realtime_data:
        data_list = list(st.session_state.realtime_data)
        df = pd.DataFrame(data_list)
        
        # Temperature chart
        fig1 = px.line(
            df,
            x='timestamp',
            y='temperature',
            title="🌡️ Temperature Trend",
            labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'}
        )
        fig1.add_hline(y=25, line_dash="dash", line_color="blue", annotation_text="DINGIN")
        fig1.add_hline(y=28, line_dash="dash", line_color="red", annotation_text="PANAS")
        st.plotly_chart(fig1, use_container_width=True)
        
        # ML Predictions chart
        if st.session_state.realtime_predictions:
            pred_list = list(st.session_state.realtime_predictions)
            pred_df = pd.DataFrame(pred_list)
            
            fig2 = px.scatter(
                pred_df,
                x='timestamp',
                y='confidence',
                color='model',
                size='confidence',
                hover_data=['prediction', 'temperature'],
                title="🤖 ML Predictions Confidence",
                labels={'confidence': 'Confidence (%)', 'timestamp': 'Time'}
            )
            st.plotly_chart(fig2, use_container_width=True)

# ==================== PREDICTION COMPARISON ====================
def show_prediction_comparison():
    """Bandingkan prediksi dari semua model"""
    if (not st.session_state.realtime_data or 
        not st.session_state.realtime_predictions):
        return
    
    st.subheader("🔍 Prediction Comparison")
    
    # Get latest data
    latest_data = list(st.session_state.realtime_data)[-1]
    latest_predictions = list(st.session_state.realtime_predictions)[-3:]  # Last 3 predictions
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Latest Sensor Data:**")
        st.json({
            'Temperature': f"{latest_data['temperature']}°C",
            'Humidity': f"{latest_data['humidity']}%",
            'Label': latest_data['label'],
            'Time': latest_data['time_str']
        })
    
    with col2:
        st.markdown("**ML Predictions:**")
        if latest_predictions:
            for pred in latest_predictions:
                color = get_label_color(pred['prediction'])
                st.markdown(f"""
                <div style="background-color: {color}20; padding: 10px; border-radius: 5px; margin: 5px 0;">
                    <strong>{pred['model']}</strong>: {pred['prediction']} 
                    <span style="float: right;">{pred['confidence']:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Check agreement
        if len(latest_predictions) > 1:
            predictions = [p['prediction'] for p in latest_predictions]
            if len(set(predictions)) == 1:
                st.success(f"✅ All models agree: {predictions[0]}")
            else:
                st.warning(f"⚠️ Models disagree: {', '.join(set(predictions))}")

# ==================== TRAINING DATA ANALYSIS ====================
def show_training_analysis():
    """Analisis data training yang digunakan"""
    if st.session_state.historical_data is None:
        return
    
    st.subheader("📚 Training Data Analysis")
    
    df = st.session_state.historical_data
    
    tab1, tab2, tab3 = st.tabs(["Statistics", "Distribution", "Raw Data"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            avg_temp = df['temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
        with col3:
            avg_hum = df['humidity'].mean()
            st.metric("Avg Humidity", f"{avg_hum:.1f}%")
        
        # Label distribution
        label_counts = df['label'].value_counts()
        fig = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="Label Distribution in Training Data",
            color=label_counts.index,
            color_discrete_map={
                'DINGIN': '#3498db',
                'NORMAL': '#2ecc71',
                'PANAS': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Scatter plot
        fig = px.scatter(
            df,
            x='temperature',
            y='humidity',
            color='label',
            title="Temperature vs Humidity (Training Data)",
            color_discrete_map={
                'DINGIN': '#3498db',
                'NORMAL': '#2ecc71',
                'PANAS': '#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.dataframe(df.head(20), use_container_width=True)

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.markdown("""
    <div class="system-header">
        <h1 style="margin: 0; text-align: center;">🤖 ESP32 DHT Complete ML System</h1>
        <p style="margin: 0; text-align: center; opacity: 0.9;">
            Real-time Monitoring + ML Predictions + Training Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load ML models on startup
    if st.session_state.ml_models is None:
        with st.spinner("Loading ML models and training data..."):
            models, scaler, metadata, historical = load_ml_models()
            st.session_state.ml_models = models
            st.session_state.ml_scaler = scaler
            st.session_state.ml_metadata = metadata
            st.session_state.historical_data = historical
    
    # Sidebar
    controls = create_sidebar()
    
    # Auto-refresh
    if controls['auto_refresh'] and st.session_state.mqtt_connected:
        time.sleep(2)
        st.rerun()
    
    # System Overview
    st.subheader("🏗️ System Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="component-card ml-card">
            <h3>🧠 ML Training System</h3>
            <p><strong>Models:</strong> Decision Tree, KNN, Logistic Regression</p>
            <p><strong>Features:</strong> Temperature, Humidity, Time</p>
            <p><strong>Labels:</strong> DINGIN, NORMAL, PANAS</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="component-card sensor-card">
            <h3>📡 ESP32 Data Collector</h3>
            <p><strong>Sensor:</strong> DHT11 (GPIO 4)</p>
            <p><strong>Output:</strong> Temperature, Humidity</p>
            <p><strong>Storage:</strong> SPIFFS + MQTT</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="component-card dashboard-card">
            <h3>📊 Real-time Dashboard</h3>
            <p><strong>Features:</strong> Live monitoring, ML predictions</p>
            <p><strong>Control:</strong> ESP32 LED control</p>
            <p><strong>Analysis:</strong> Training data visualization</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Real-time Dashboard", 
        "🧠 ML Performance", 
        "📚 Training Data",
        "🔧 System Controls"
    ])
    
    with tab1:
        show_realtime_dashboard()
        show_prediction_comparison()
    
    with tab2:
        show_ml_performance()
    
    with tab3:
        show_training_analysis()
    
    with tab4:
        st.subheader("System Controls & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**MQTT Configuration:**")
            st.code(f"""
Broker: {MQTT_BROKER}
Port: {MQTT_PORT}
Username: {MQTT_USERNAME}
Topics:
  - Sensor: {TOPIC_SENSOR}
  - LED: {TOPIC_LED}
  - Prediction: {TOPIC_PREDICTION}
            """)
        
        with col2:
            st.markdown("**ML Configuration:**")
            if st.session_state.ml_metadata:
                st.json(st.session_state.ml_metadata)
            else:
                st.info("ML metadata not loaded")
        
        # Debug info
        with st.expander("Debug Information"):
            st.json({
                "realtime_data_points": len(st.session_state.realtime_data),
                "prediction_points": len(st.session_state.realtime_predictions),
                "ml_models_loaded": list(st.session_state.ml_models.keys()) if st.session_state.ml_models else [],
                "mqtt_connected": st.session_state.mqtt_connected,
                "historical_records": len(st.session_state.historical_data) if st.session_state.historical_data is not None else 0
            })
    
    # Footer
    st.markdown("---")
    update_time = datetime.now().strftime('%H:%M:%S')
    
    if controls['auto_refresh']:
        st.caption(f"🔄 Auto-refresh enabled | Last update: {update_time}")
    else:
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()

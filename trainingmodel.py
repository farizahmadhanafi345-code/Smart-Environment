"""
DHT11 REAL-TIME DASHBOARD WITH HIVEMQ
Streamlit Dashboard untuk monitoring real-time sensor DHT11 dengan ML predictions
"""

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
import queue
import warnings
from pathlib import Path

# ==================== CONFIGURATION ====================
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DHT11 Real-Time Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PATH SETUP ====================
# Gunakan path yang sama dengan training
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_OPTIONS = [
    r"C:\Users\USER\OneDrive\Documents\broker\dashboardstreamlit",
    SCRIPT_DIR
]

BASE_DIR = None
for base_dir in BASE_DIR_OPTIONS:
    if os.path.exists(base_dir):
        BASE_DIR = base_dir
        break

if BASE_DIR is None:
    BASE_DIR = SCRIPT_DIR

# Path untuk Trainingdht (sama dengan training)
TRAININGDHT_DIR = os.path.join(BASE_DIR, "Trainingdht")
CSV_FILE = os.path.join(TRAININGDHT_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(TRAININGDHT_DIR, "models")
REPORTS_DIR = os.path.join(TRAININGDHT_DIR, "reports")
CSV_PKL_DIR = os.path.join(TRAININGDHT_DIR, "csv_pkl")

# Buat folder jika belum ada
os.makedirs(TRAININGDHT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CSV_PKL_DIR, exist_ok=True)

# ==================== HIVEMQ CONFIGURATION ====================
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
PREDICTION_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_prediction"

# ==================== HIVEMQ MANAGER ====================
class HiveMQManager:
    """Manage real-time HiveMQ connection"""
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.message_queue = queue.Queue(maxsize=100)
        self.received_data = []
        self.max_history = 500
        self.status = "Disconnected"
        self.last_message_time = None
        
    def connect(self):
        """Connect to HiveMQ broker"""
        try:
            self.client = mqtt.Client()
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set()
            
            # Set callbacks
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message
            self.client.on_disconnect = self._on_disconnect
            
            # Connect
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            
            time.sleep(2)
            return True
            
        except Exception as e:
            self.status = f"Connection error: {str(e)[:50]}"
            return False
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback for successful connection"""
        if rc == 0:
            self.connected = True
            self.status = "Connected"
            client.subscribe(DHT_TOPIC)
            client.subscribe(PREDICTION_TOPIC)
            print(f"✅ Connected to HiveMQ and subscribed to topics")
        else:
            self.connected = False
            self.status = f"Connection failed (code: {rc})"
    
    def _on_message(self, client, userdata, msg):
        """Callback for received messages"""
        try:
            data = json.loads(msg.payload.decode())
            data['topic'] = msg.topic
            data['received_time'] = datetime.now().strftime('%H:%M:%S')
            data['timestamp'] = datetime.now().isoformat()
            data['message_type'] = 'sensor' if msg.topic == DHT_TOPIC else 'prediction'
            
            # Add to queue
            self.message_queue.put(data)
            
            # Store in history
            self.received_data.append(data)
            if len(self.received_data) > self.max_history:
                self.received_data = self.received_data[-self.max_history:]
            
            self.last_message_time = datetime.now()
            
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback for disconnection"""
        self.connected = False
        self.status = "Disconnected"
    
    def get_latest_message(self):
        """Get latest message from queue"""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_all_messages(self):
        """Get all received messages"""
        return self.received_data.copy()
    
    def get_sensor_messages(self):
        """Get only sensor messages"""
        return [m for m in self.received_data if m.get('topic') == DHT_TOPIC]
    
    def get_prediction_messages(self):
        """Get only prediction messages"""
        return [m for m in self.received_data if m.get('topic') == PREDICTION_TOPIC]
    
    def publish_prediction(self, prediction_data):
        """Publish prediction to HiveMQ"""
        if self.connected and self.client:
            try:
                payload = json.dumps(prediction_data)
                result = self.client.publish(PREDICTION_TOPIC, payload, qos=1)
                return result.rc == mqtt.MQTT_ERR_SUCCESS
            except Exception as e:
                print(f"Publish error: {e}")
                return False
        return False
    
    def disconnect(self):
        """Disconnect from HiveMQ"""
        if self.client and self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            self.status = "Manually disconnected"

# ==================== LOAD MODELS FROM TRAINING ====================
@st.cache_resource
def load_trained_models():
    """Load trained models and scaler from training directory"""
    try:
        models = {}
        
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # Fallback to CSV_PKL scaler
            scaler_csv_pkl = os.path.join(CSV_PKL_DIR, "scaler.csv.pkl")
            if os.path.exists(scaler_csv_pkl):
                with open(scaler_csv_pkl, 'rb') as f:
                    scaler_info = pickle.load(f)
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    if isinstance(scaler_info, pd.DataFrame):
                        scaler.mean_ = scaler_info['mean'].values
                        scaler.scale_ = scaler_info['scale'].values
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
        
        # Load models from models directory
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
            'Logistic Regression': 'logistic_regression.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[model_name] = pickle.load(f)
            else:
                # Try CSV_PKL format
                csv_pkl_path = os.path.join(CSV_PKL_DIR, filename.replace('.pkl', '.csv.pkl'))
                if os.path.exists(csv_pkl_path):
                    with open(csv_pkl_path, 'rb') as f:
                        model_data = pickle.load(f)
                        if isinstance(model_data, dict) and 'model' in model_data:
                            models[model_name] = model_data['model']
        
        if not models:
            st.warning("No trained models found. Please run model_training.py first")
            return None, None
        
        return models, scaler
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

@st.cache_data
def load_dataset():
    """Load historical dataset"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, delimiter=';')
            
            if 'timestamp' in df.columns:
                # Process timestamp
                df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
            
            return df
        else:
            return pd.DataFrame()
    except:
        return pd.DataFrame()

def get_label_color(label):
    """Get color based on label"""
    colors = {
        'DINGIN': '#3498db',    # Blue
        'NORMAL': '#2ecc71',    # Green
        'PANAS': '#e74c3c',     # Red
        'UNKNOWN': '#95a5a6',   # Gray
        'ERROR': '#f39c12'      # Orange
    }
    return colors.get(label, '#95a5a6')

def predict_with_models(models, scaler, temperature, humidity, hour=None, minute=None):
    """Make predictions using all trained models"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    
    try:
        if hasattr(scaler, 'transform'):
            features_scaled = scaler.transform(features)
        else:
            features_scaled = features
    except:
        features_scaled = features
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            pred_code = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code] if len(probs) > pred_code else 1.0
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
                'color': get_label_color(label),
                'features': [temperature, humidity, hour, minute]
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'color': '#f39c12'
            }
    
    return predictions

# ==================== SIDEBAR CONTROLS ====================
def render_sidebar(mqtt_manager, models):
    """Render sidebar controls"""
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # HiveMQ Connection Status
    st.sidebar.subheader("📡 HiveMQ Connection")
    
    if mqtt_manager.connected:
        st.sidebar.success("✅ Connected")
        if st.sidebar.button("Disconnect", key="disconnect_btn"):
            mqtt_manager.disconnect()
            st.rerun()
    else:
        st.sidebar.error("❌ Disconnected")
        if st.sidebar.button("Connect", key="connect_btn"):
            if mqtt_manager.connect():
                st.rerun()
    
    # Model Selection
    st.sidebar.subheader("🤖 Active Models")
    
    selected_models = {}
    if models:
        for model_name in models.keys():
            selected_models[model_name] = st.sidebar.checkbox(
                model_name, 
                value=True, 
                key=f"model_{model_name}"
            )
    
    # Real-time Settings
    st.sidebar.subheader("🌐 Real-time Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True, key="auto_refresh")
    if auto_refresh:
        refresh_interval = st.sidebar.slider("Refresh (seconds)", 1, 10, 2, key="refresh_interval")
    
    # Manual Prediction
    st.sidebar.subheader("🔮 Manual Prediction")
    
    manual_temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 24.0, 0.5, key="manual_temp")
    manual_hum = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0, key="manual_hum")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        manual_hour = st.sidebar.number_input("Hour", 0, 23, datetime.now().hour, key="manual_hour")
    with col2:
        manual_minute = st.sidebar.number_input("Minute", 0, 59, datetime.now().minute, key="manual_minute")
    
    # Data Management
    st.sidebar.subheader("📊 Data")
    
    if st.sidebar.button("Clear Messages", key="clear_msg"):
        mqtt_manager.received_data = []
        st.rerun()
    
    if st.sidebar.button("Save to CSV", key="save_csv"):
        messages = mqtt_manager.get_sensor_messages()
        if messages:
            new_data = pd.DataFrame(messages)
            if os.path.exists(CSV_FILE):
                existing_data = pd.read_csv(CSV_FILE, delimiter=';')
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data.to_csv(CSV_FILE, sep=';', index=False)
            else:
                new_data.to_csv(CSV_FILE, sep=';', index=False)
            st.sidebar.success(f"Saved {len(messages)} messages")
    
    # System Info
    st.sidebar.subheader("📈 System Info")
    
    df = load_dataset()
    if not df.empty:
        st.sidebar.metric("Dataset Records", len(df))
    
    if models:
        st.sidebar.metric("Loaded Models", len(models))
    
    return {
        'selected_models': selected_models,
        'manual_input': (manual_temp, manual_hum, manual_hour, manual_minute),
        'auto_refresh': auto_refresh,
        'refresh_interval': refresh_interval if 'refresh_interval' in locals() else 2
    }

# ==================== MAIN DASHBOARD ====================
def main():
    st.title("🌡️ DHT11 Real-Time Dashboard")
    st.markdown("Live monitoring of temperature/humidity with ML predictions via HiveMQ")
    
    # Initialize components
    if 'mqtt_manager' not in st.session_state:
        st.session_state.mqtt_manager = HiveMQManager()
    
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    mqtt_manager = st.session_state.mqtt_manager
    
    # Load trained models
    models, scaler = load_trained_models()
    
    if models is None or scaler is None:
        st.error("❌ Failed to load ML models. Please run model_training.py first.")
        return
    
    # Render sidebar and get controls
    controls = render_sidebar(mqtt_manager, models)
    selected_models = controls['selected_models']
    manual_input = controls['manual_input']
    auto_refresh = controls['auto_refresh']
    
    # Filter active models
    active_models = {name: model for name, model in models.items() 
                     if selected_models.get(name, True)}
    
    # ===== ROW 1: REAL-TIME METRICS =====
    st.markdown("---")
    
    # Get latest sensor data
    messages = mqtt_manager.get_sensor_messages()
    latest_sensor = messages[-1] if messages else None
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if latest_sensor:
            temp = latest_sensor.get('temperature', 0)
            st.metric("🌡️ Live Temperature", f"{temp}°C", 
                     delta="Live" if mqtt_manager.connected else "Offline")
        else:
            st.metric("🌡️ Temperature", f"{manual_input[0]}°C", "Manual")
    
    with col2:
        if latest_sensor:
            hum = latest_sensor.get('humidity', 0)
            st.metric("💧 Live Humidity", f"{hum}%", 
                     delta="Live" if mqtt_manager.connected else "Offline")
        else:
            st.metric("💧 Humidity", f"{manual_input[1]}%", "Manual")
    
    with col3:
        msg_count = len(messages)
        st.metric("📥 Messages", msg_count, 
                 delta=f"+{len(mqtt_manager.get_all_messages()) - msg_count}" if mqtt_manager.get_prediction_messages() else None)
    
    with col4:
        status_color = "🟢" if mqtt_manager.connected else "🔴"
        status_text = "Connected" if mqtt_manager.connected else "Disconnected"
        st.metric("📡 HiveMQ", status_text, status_color)
    
    # ===== ROW 2: REAL-TIME PREDICTIONS =====
    st.markdown("---")
    st.subheader("🔮 Real-time Predictions")
    
    # Use live data if available, otherwise use manual input
    if latest_sensor and mqtt_manager.connected:
        temp = latest_sensor.get('temperature', manual_input[0])
        hum = latest_sensor.get('humidity', manual_input[1])
        hour = datetime.now().hour
        minute = datetime.now().minute
        source = "Live Sensor"
    else:
        temp, hum, hour, minute = manual_input
        source = "Manual Input"
    
    # Make predictions
    predictions = predict_with_models(active_models, scaler, temp, hum, hour, minute)
    
    # Display predictions
    cols = st.columns(min(3, len(predictions)))
    
    for idx, (model_name, pred) in enumerate(predictions.items()):
        if idx >= len(cols):
            break
            
        with cols[idx]:
            color = pred['color']
            
            # Prediction card
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, {color}15, {color}05);
                border-radius: 10px;
                padding: 15px;
                border-left: 5px solid {color};
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            ">
                <h4 style="margin: 0; color: #333; font-size: 14px;">
                    {model_name} <span style="font-size: 10px; color: #666;">({source})</span>
                </h4>
                <h2 style="margin: 10px 0; color: {color}; font-size: 24px;">
                    {pred['label']}
                </h2>
                <p style="margin: 5px 0; font-size: 12px;">
                    Confidence: <strong>{pred['confidence']:.1%}</strong>
                </p>
                <p style="margin: 5px 0; font-size: 11px; color: #666;">
                    Temp: {temp}°C | Hum: {hum}%
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability chart
            if 'probabilities' in pred:
                prob_data = pd.DataFrame({
                    'Class': ['DINGIN', 'NORMAL', 'PANAS'],
                    'Probability': [
                        pred['probabilities']['DINGIN'],
                        pred['probabilities']['NORMAL'], 
                        pred['probabilities']['PANAS']
                    ]
                })
                
                fig = px.bar(
                    prob_data,
                    x='Class',
                    y='Probability',
                    color='Class',
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
                    margin=dict(t=10, b=10, l=10, r=10),
                    yaxis=dict(tickformat=".0%")
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Publish button
            if mqtt_manager.connected:
                if st.button(f"📤 Publish", key=f"pub_{model_name}", use_container_width=True):
                    prediction_data = {
                        'model': model_name,
                        'label': pred['label'],
                        'confidence': pred['confidence'],
                        'temperature': temp,
                        'humidity': hum,
                        'hour': hour,
                        'minute': minute,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'dashboard'
                    }
                    
                    if mqtt_manager.publish_prediction(prediction_data):
                        st.success(f"✅ Published {model_name} prediction!")
                        time.sleep(0.5)
                        st.rerun()
    
    # ===== ROW 3: REAL-TIME DATA STREAM =====
    st.markdown("---")
    st.subheader("📡 Live Data Stream")
    
    if mqtt_manager.connected:
        tab1, tab2, tab3 = st.tabs(["📊 Sensor Data", "🤖 Predictions", "📈 Charts"])
        
        with tab1:
            # Display recent sensor data
            sensor_messages = mqtt_manager.get_sensor_messages()[-20:]  # Last 20 messages
            
            if sensor_messages:
                sensor_df = pd.DataFrame(sensor_messages)
                st.dataframe(
                    sensor_df[['received_time', 'temperature', 'humidity']].tail(10),
                    use_container_width=True,
                    height=300
                )
                
                # Line chart for temperature
                if len(sensor_df) > 1:
                    sensor_df['time'] = pd.to_datetime(sensor_df['received_time'], format='%H:%M:%S', errors='coerce')
                    sensor_df = sensor_df.sort_values('time')
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=sensor_df['time'],
                        y=sensor_df['temperature'],
                        mode='lines+markers',
                        name='Temperature',
                        line=dict(color='#e74c3c', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=sensor_df['time'],
                        y=sensor_df['humidity'],
                        name='Humidity',
                        line=dict(color='#3498db', width=2),
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title='Real-time Sensor Data',
                        yaxis=dict(title='Temperature (°C)', side='left'),
                        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
                        height=300,
                        showlegend=True,
                        xaxis_title='Time'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Waiting for sensor data...")
        
        with tab2:
            # Display prediction history
            pred_messages = mqtt_manager.get_prediction_messages()[-10:]
            
            if pred_messages:
                pred_df = pd.DataFrame(pred_messages)
                st.dataframe(
                    pred_df[['received_time', 'model', 'label', 'confidence']].tail(10),
                    use_container_width=True,
                    height=300
                )
                
                # Prediction accuracy over time
                if len(pred_df) > 1:
                    pred_counts = pred_df['label'].value_counts()
                    
                    fig = px.pie(
                        values=pred_counts.values,
                        names=pred_counts.index,
                        color=pred_counts.index,
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        title='Prediction Distribution'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions published yet")
        
        with tab3:
            # Combined charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Temperature distribution
                if messages:
                    temp_values = [m.get('temperature', 0) for m in messages if 'temperature' in m]
                    if temp_values:
                        fig = px.histogram(
                            x=temp_values,
                            title='Temperature Distribution',
                            labels={'x': 'Temperature (°C)'},
                            color_discrete_sequence=['#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Humidity distribution
                if messages:
                    hum_values = [m.get('humidity', 0) for m in messages if 'humidity' in m]
                    if hum_values:
                        fig = px.histogram(
                            x=hum_values,
                            title='Humidity Distribution',
                            labels={'x': 'Humidity (%)'},
                            color_discrete_sequence=['#3498db']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Connect to HiveMQ to see real-time data")
    
    # ===== ROW 4: MODEL PERFORMANCE & HISTORY =====
    st.markdown("---")
    st.subheader("📊 Model Performance & History")
    
    # Load historical data
    df = load_dataset()
    
    if not df.empty:
        tab1, tab2 = st.tabs(["📈 Data Analysis", "🤖 Model Info"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot
                if 'temperature' in df.columns and 'humidity' in df.columns:
                    fig = px.scatter(
                        df,
                        x='temperature',
                        y='humidity',
                        color='label' if 'label' in df.columns else None,
                        title="Historical Temperature vs Humidity",
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        } if 'label' in df.columns else None
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Hourly averages
                if 'hour' in df.columns:
                    hourly_avg = df.groupby('hour').agg({
                        'temperature': 'mean',
                        'humidity': 'mean'
                    }).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hourly_avg['hour'],
                        y=hourly_avg['temperature'],
                        name='Temperature',
                        line=dict(color='#e74c3c')
                    ))
                    fig.add_trace(go.Scatter(
                        x=hourly_avg['hour'],
                        y=hourly_avg['humidity'],
                        name='Humidity',
                        line=dict(color='#3498db'),
                        yaxis='y2'
                    ))
                    
                    fig.update_layout(
                        title='Average by Hour of Day',
                        xaxis_title='Hour',
                        yaxis_title='Temperature (°C)',
                        yaxis2=dict(title='Humidity (%)', overlaying='y', side='right'),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Model information
            st.markdown("**Loaded Models:**")
            for model_name, model in models.items():
                with st.expander(f"{model_name}"):
                    st.write(f"Type: {type(model).__name__}")
                    
                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        st.write(f"Parameters: {len(params)}")
                        
                        # Show important parameters
                        important_params = ['max_depth', 'n_neighbors', 'C', 'solver', 'criterion']
                        for param in important_params:
                            if param in params:
                                st.write(f"- {param}: {params[param]}")
                    
                    # Show if model is active
                    st.write(f"Status: {'✅ Active' if selected_models.get(model_name, True) else '❌ Inactive'}")
    else:
        st.info("No historical data available")
    
    # ===== FOOTER =====
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**System Status**")
        st.write(f"- HiveMQ: {'🟢 Connected' if mqtt_manager.connected else '🔴 Disconnected'}")
        st.write(f"- Models: {len(active_models)} active")
        st.write(f"- Messages: {len(mqtt_manager.get_all_messages())}")
    
    with col2:
        st.markdown("**HiveMQ Info**")
        st.write(f"- Broker: `{MQTT_BROKER}`")
        st.write(f"- Topic: `{DHT_TOPIC}`")
        st.write(f"- Last: {mqtt_manager.last_message_time.strftime('%H:%M:%S') if mqtt_manager.last_message_time else 'Never'}")
    
    with col3:
        st.markdown("**Quick Actions**")
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        if st.button("💾 Export Data", use_container_width=True):
            messages = mqtt_manager.get_all_messages()
            if messages:
                df_export = pd.DataFrame(messages)
                csv = df_export.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"hivemq_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    # Auto-refresh
    if auto_refresh:
        refresh_interval = controls.get('refresh_interval', 2)
        current_time = datetime.now()
        
        if (current_time - st.session_state.last_refresh).seconds >= refresh_interval:
            st.session_state.last_refresh = current_time
            time.sleep(0.1)  # Small delay to allow UI updates
            st.rerun()

if __name__ == "__main__":
    main()

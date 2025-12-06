# dashboard_ultimate_final.py
import streamlit as st
import paho.mqtt.client as mqtt
import json
import csv
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import ssl
import threading
import pickle
import warnings
import socket
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HIVEMQ ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "fariz_device_main",
    "password": "F4riz#Device2025!",
    "use_ssl": True,
    "keepalive": 60
}

# Topics
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# ==================== TEST CREDENTIALS ALTERNATIF ====================
MQTT_TEST_CONFIGS = [
    {
        "name": "Main Credentials",
        "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
        "port": 8883,
        "username": "fariz_device_main",
        "password": "F4riz#Device2025!",
        "use_ssl": True
    },
    {
        "name": "Web Client Credentials",
        "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
        "port": 8883,
        "username": "hivemq.webclient.1764923408610",
        "password": "9y&f74G1*pWSD.tQdXa@",
        "use_ssl": True
    },
    {
        "name": "Port 8884 (WebSockets)",
        "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
        "port": 8884,
        "username": "fariz_device_main",
        "password": "F4riz#Device2025!",
        "use_ssl": True
    }
]

# ==================== FALLBACK MODE ====================
FALLBACK_MODE = False  # Akan diaktifkan jika MQTT gagal

# ==================== INISIALISASI STATE ====================
if 'iot_dashboard' not in st.session_state:
    st.session_state.iot_dashboard = {
        # Connection State
        "mqtt_connected": False,
        "connection_status": "disconnected",
        "connection_test_result": None,
        "current_config": MQTT_CONFIG,
        "fallback_mode": False,
        
        # Sensor Data
        "temperature": 25.0,
        "humidity": 65.0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": "NORMAL",
        "status_code": 1,
        
        # Data Storage
        "sensor_history": [],
        "data_points": 0,
        "last_update": datetime.now(),
        
        # ML System
        "ml_models": {},
        "ml_trained": False,
        "ml_predictions": {},
        "ml_training_data": [],
        
        # Control
        "led_status": "off",
        "auto_refresh": True,
        
        # System
        "logs": [],
        "start_time": datetime.now()
    }

# ==================== CONNECTION TEST FUNCTION ====================
def test_mqtt_connection(config):
    """Test koneksi MQTT dan return hasil"""
    try:
        log_message(f"Testing connection: {config['name']}")
        log_message(f"Broker: {config['broker']}:{config['port']}")
        log_message(f"Username: {config['username']}")
        
        # Test network connectivity
        try:
            socket.create_connection((config['broker'], config['port']), timeout=5)
            log_message("✅ Network connection OK")
        except:
            log_message("❌ Network connection failed")
            return False, "Network connection failed"
        
        # Create MQTT client
        client = mqtt.Client(f"test_client_{int(time.time())}")
        
        # Set credentials
        client.username_pw_set(config['username'], config['password'])
        
        # Set SSL
        if config.get('use_ssl', True):
            client.tls_set(tls_version=ssl.PROTOCOL_TLS)
            client.tls_insecure_set(True)
        
        connection_result = {"success": False, "error": None}
        
        def on_connect_test(client, userdata, flags, rc):
            if rc == 0:
                connection_result["success"] = True
            else:
                connection_result["success"] = False
                error_msgs = {
                    1: "Protocol version mismatch",
                    2: "Invalid client ID",
                    3: "Server unavailable",
                    4: "Bad username/password",
                    5: "Not authorized"
                }
                connection_result["error"] = error_msgs.get(rc, f"Error code: {rc}")
        
        client.on_connect = on_connect_test
        
        # Connect with timeout
        client.connect(config['broker'], config['port'], 10)
        client.loop_start()
        
        # Wait for connection
        time.sleep(3)
        
        client.loop_stop()
        client.disconnect()
        
        if connection_result["success"]:
            log_message(f"✅ Connection test SUCCESS: {config['name']}")
            return True, "Connection successful"
        else:
            log_message(f"❌ Connection test FAILED: {connection_result.get('error', 'Unknown error')}")
            return False, connection_result.get("error", "Unknown error")
            
    except Exception as e:
        error_msg = str(e)
        log_message(f"❌ Connection exception: {error_msg}")
        return False, error_msg

# ==================== LOGGING ====================
def log_message(message, level="INFO"):
    """Log message ke system"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CONNECTION": "🔗",
        "SENSOR": "🌡️",
        "ML": "🤖"
    }
    
    icon = icons.get(level, "📝")
    
    colors = {
        "INFO": "#3B82F6",
        "SUCCESS": "#10B981",
        "WARNING": "#F59E0B",
        "ERROR": "#EF4444",
        "CONNECTION": "#8B5CF6",
        "SENSOR": "#EF4444",
        "ML": "#7C3AED"
    }
    
    color = colors.get(level, "#6B7280")
    
    log_entry = {
        "time": timestamp,
        "message": message,
        "icon": icon,
        "color": color,
        "level": level
    }
    
    st.session_state.iot_dashboard["logs"].insert(0, log_entry)
    
    # Limit logs
    if len(st.session_state.iot_dashboard["logs"]) > 50:
        st.session_state.iot_dashboard["logs"] = st.session_state.iot_dashboard["logs"][:50]
    
    # Print to console
    print(f"[{timestamp}] {icon} {message}")

# ==================== ML SYSTEM ====================
class ML_System:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.trained = False
        
    def train_models(self, data):
        """Train KNN, DT, LR models"""
        try:
            if len(data) < 10:
                return False, "Need at least 10 samples"
            
            df = pd.DataFrame(data)
            X = df[['temperature', 'humidity']].values
            y = df['status_code'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            models_to_train = {
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(max_depth=5),
                "Logistic Regression": LogisticRegression(max_iter=1000)
            }
            
            results = {}
            
            for name, model in models_to_train.items():
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.models[name] = model
                results[name] = {
                    "accuracy": accuracy,
                    "model": model
                }
                
                log_message(f"{name} trained - Accuracy: {accuracy:.2%}", "ML")
            
            self.trained = True
            st.session_state.iot_dashboard["ml_trained"] = True
            st.session_state.iot_dashboard["ml_models"] = self.models
            
            return True, "Models trained successfully"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict(self, temperature, humidity):
        """Make predictions with all models"""
        if not self.trained:
            return {}
        
        try:
            features = np.array([[temperature, humidity]])
            features_scaled = self.scaler.transform(features)
            
            predictions = {}
            label_map = {0: "DINGIN 🥶", 1: "NORMAL ✅", 2: "PANAS 🔥"}
            
            for name, model in self.models.items():
                pred_code = model.predict(features_scaled)[0]
                
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_scaled)[0]
                    confidence = probs[pred_code]
                else:
                    confidence = 1.0
                    probs = [0.33, 0.33, 0.34]
                
                predictions[name] = {
                    "label": label_map.get(pred_code, "UNKNOWN"),
                    "confidence": float(confidence),
                    "probabilities": {
                        "DINGIN": float(probs[0]) if len(probs) > 0 else 0,
                        "NORMAL": float(probs[1]) if len(probs) > 1 else 0,
                        "PANAS": float(probs[2]) if len(probs) > 2 else 0
                    }
                }
            
            return predictions
            
        except Exception as e:
            log_message(f"Prediction error: {e}", "ERROR")
            return {}

# Initialize ML system
if 'ml_system' not in st.session_state:
    st.session_state.ml_system = ML_System()

# ==================== FALLBACK DATA GENERATOR ====================
def generate_fallback_data():
    """Generate fallback data jika MQTT gagal"""
    # Simulate sensor data
    base_temp = 25.0
    base_hum = 65.0
    
    # Add some variation
    temp_variation = np.random.uniform(-2, 2)
    hum_variation = np.random.uniform(-5, 5)
    
    temperature = round(base_temp + temp_variation, 1)
    humidity = round(base_hum + hum_variation, 1)
    
    # Determine status
    if temperature < 25:
        status, color, code = "DINGIN 🥶", "#3B82F6", 0
    elif temperature > 28:
        status, color, code = "PANAS 🔥", "#EF4444", 2
    else:
        status, color, code = "NORMAL ✅", "#10B981", 1
    
    return temperature, humidity, status, color, code

# ==================== MQTT MANAGER ====================
class MQTT_Manager:
    def __init__(self):
        self.client = None
        self.connected = False
        
    def connect(self, config):
        """Connect to MQTT broker"""
        try:
            log_message(f"Connecting to {config['broker']}...", "CONNECTION")
            
            # Test connection first
            success, message = test_mqtt_connection(config)
            if not success:
                return False, message
            
            # Create client
            self.client = mqtt.Client(f"dashboard_{int(time.time())}")
            self.client.username_pw_set(config['username'], config['password'])
            
            if config.get('use_ssl', True):
                self.client.tls_set(tls_version=ssl.PROTOCOL_TLS)
                self.client.tls_insecure_set(True)
            
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            self.client.connect(config['broker'], config['port'], 60)
            self.client.loop_start()
            
            # Wait for connection
            time.sleep(2)
            
            if self.connected:
                st.session_state.iot_dashboard["current_config"] = config
                return True, "Connected successfully"
            else:
                return False, "Connection timeout"
                
        except Exception as e:
            return False, str(e)
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            st.session_state.iot_dashboard["mqtt_connected"] = True
            st.session_state.iot_dashboard["connection_status"] = "connected"
            client.subscribe(DHT_TOPIC)
            log_message("✅ MQTT Connected and subscribed!", "SUCCESS")
        else:
            log_message(f"❌ Connection failed: {rc}", "ERROR")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
        st.session_state.iot_dashboard["mqtt_connected"] = False
        st.session_state.iot_dashboard["connection_status"] = "disconnected"
        log_message("⚠️ Disconnected from MQTT", "WARNING")
    
    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            temperature = float(data.get('temperature', 0))
            humidity = float(data.get('humidity', 0))
            
            # Update data
            update_sensor_data(temperature, humidity, "MQTT")
            
        except Exception as e:
            log_message(f"Message error: {e}", "ERROR")
    
    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            st.session_state.iot_dashboard["mqtt_connected"] = False
            log_message("Disconnected", "INFO")
    
    def send_led_command(self, command):
        if self.connected and self.client:
            try:
                self.client.publish(LED_TOPIC, command)
                st.session_state.iot_dashboard["led_status"] = command
                log_message(f"LED: {command}", "INFO")
                return True
            except:
                return False
        return False

# Initialize MQTT manager
if 'mqtt_manager' not in st.session_state:
    st.session_state.mqtt_manager = MQTT_Manager()

# ==================== DATA UPDATE FUNCTION ====================
def update_sensor_data(temperature, humidity, source="FALLBACK"):
    """Update sensor data (bisa dari MQTT atau fallback)"""
    # Determine status
    if temperature < 25:
        status, color, code = "DINGIN 🥶", "#3B82F6", 0
    elif temperature > 28:
        status, color, code = "PANAS 🔥", "#EF4444", 2
    else:
        status, color, code = "NORMAL ✅", "#10B981", 1
    
    # Update state
    st.session_state.iot_dashboard.update({
        "temperature": temperature,
        "humidity": humidity,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": status,
        "status_code": code,
        "data_points": st.session_state.iot_dashboard["data_points"] + 1,
        "last_update": datetime.now()
    })
    
    # Add to history
    st.session_state.iot_dashboard["sensor_history"].append({
        "timestamp": datetime.now(),
        "temperature": temperature,
        "humidity": humidity,
        "status": status,
        "color": color,
        "source": source
    })
    
    # Keep only last 100 records
    if len(st.session_state.iot_dashboard["sensor_history"]) > 100:
        st.session_state.iot_dashboard["sensor_history"] = st.session_state.iot_dashboard["sensor_history"][-100:]
    
    # Add to ML training data
    st.session_state.iot_dashboard["ml_training_data"].append({
        "temperature": temperature,
        "humidity": humidity,
        "status_code": code
    })
    
    # Make ML predictions
    if st.session_state.iot_dashboard["ml_trained"]:
        predictions = st.session_state.ml_system.predict(temperature, humidity)
        st.session_state.iot_dashboard["ml_predictions"] = predictions
    
    log_message(f"{source}: {temperature:.1f}°C, {humidity:.1f}% → {status}", "SENSOR")

# ==================== AUTO-UPDATE THREAD ====================
def fallback_update_thread():
    """Thread untuk update data fallback"""
    while st.session_state.iot_dashboard.get("fallback_mode", False):
        if not st.session_state.iot_dashboard["mqtt_connected"]:
            # Generate fallback data
            temp, hum, status, color, code = generate_fallback_data()
            update_sensor_data(temp, hum, "FALLBACK")
        
        time.sleep(3)  # Update every 3 seconds

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="IoT Dashboard Ultimate",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .dashboard-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .model-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .model-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 5px 15px rgba(59,130,246,0.1);
    }
    
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .connected { background: #10B981; color: white; }
    .disconnected { background: #EF4444; color: white; }
    .fallback { background: #F59E0B; color: white; }
    
    .log-item {
        padding: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    
    .big-number {
        font-size: 2.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .connection-test-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success { background: #D1FAE5; border: 1px solid #A7F3D0; color: #065F46; }
    .error { background: #FEE2E2; border: 1px solid #FECACA; color: #991B1B; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 2.8rem;">🚀 ULTIMATE IOT DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Real-time • ML Ready • Connection Test • Fallback Mode</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 Connection Manager")
        
        # Status display
        col_status, col_mode = st.columns(2)
        with col_status:
            if st.session_state.iot_dashboard["mqtt_connected"]:
                st.markdown('<div class="status-badge connected">🟢 CONNECTED</div>', unsafe_allow_html=True)
            elif st.session_state.iot_dashboard.get("fallback_mode", False):
                st.markdown('<div class="status-badge fallback">🟡 FALLBACK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-badge disconnected">🔴 DISCONNECTED</div>', unsafe_allow_html=True)
        
        with col_mode:
            st.caption(f"Data: {st.session_state.iot_dashboard['data_points']}")
        
        # Connection Test Section
        st.subheader("🧪 Connection Test")
        
        test_config = st.selectbox(
            "Test Configuration",
            options=MQTT_TEST_CONFIGS,
            format_func=lambda x: x["name"]
        )
        
        if st.button("🔍 Test Connection", use_container_width=True):
            with st.spinner("Testing connection..."):
                success, message = test_mqtt_connection(test_config)
                
                if success:
                    st.markdown(f"""
                    <div class="connection-test-result success">
                        ✅ <strong>SUCCESS</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="connection-test-result error">
                        ❌ <strong>FAILED</strong><br>
                        {message}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Connect Button
        if not st.session_state.iot_dashboard["mqtt_connected"]:
            if st.button("🔗 Connect with Selected Config", type="primary", use_container_width=True):
                with st.spinner("Connecting..."):
                    success, message = st.session_state.mqtt_manager.connect(test_config)
                    if success:
                        st.success("Connected!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Failed: {message}")
                        
                        # Offer fallback mode
                        if st.button("🔄 Enable Fallback Mode", use_container_width=True):
                            st.session_state.iot_dashboard["fallback_mode"] = True
                            # Start fallback thread
                            threading.Thread(target=fallback_update_thread, daemon=True).start()
                            st.success("Fallback mode activated!")
                            st.rerun()
        else:
            if st.button("🔌 Disconnect", use_container_width=True):
                st.session_state.mqtt_manager.disconnect()
                st.warning("Disconnected")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        # ML Control
        st.header("🤖 ML Control")
        
        train_data_count = len(st.session_state.iot_dashboard["ml_training_data"])
        
        if st.button(f"🏋️ Train ML Models ({train_data_count} samples)", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=train_data_count < 10):
            with st.spinner("Training KNN, DT, LR..."):
                success, message = st.session_state.ml_system.train_models(
                    st.session_state.iot_dashboard["ml_training_data"]
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
                time.sleep(2)
                st.rerun()
        
        # Manual Prediction Test
        st.markdown("---")
        st.subheader("🧪 Test Prediction")
        
        col_temp, col_hum = st.columns(2)
        with col_temp:
            test_temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
        with col_hum:
            test_hum = st.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
        
        if st.button("🔮 Predict", use_container_width=True,
                    disabled=not st.session_state.iot_dashboard["ml_trained"]):
            predictions = st.session_state.ml_system.predict(test_temp, test_hum)
            st.session_state.iot_dashboard["ml_predictions"] = predictions
            st.success("Predictions generated!")
            st.rerun()
        
        st.markdown("---")
        
        # LED Control (only if connected)
        st.header("💡 LED Control")
        
        col_led1, col_led2 = st.columns(2)
        with col_led1:
            if st.button("🔴 RED", use_container_width=True,
                        disabled=not st.session_state.iot_dashboard["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("merah")
                st.success("Sent RED")
            
            if st.button("🟢 GREEN", use_container_width=True,
                        disabled=not st.session_state.iot_dashboard["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("hijau")
                st.success("Sent GREEN")
        
        with col_led2:
            if st.button("🟡 YELLOW", use_container_width=True,
                        disabled=not st.session_state.iot_dashboard["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("kuning")
                st.success("Sent YELLOW")
            
            if st.button("⚫ OFF", use_container_width=True,
                        disabled=not st.session_state.iot_dashboard["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("off")
                st.success("Sent OFF")
        
        st.caption(f"LED: {st.session_state.iot_dashboard['led_status'].upper()}")
        
        st.markdown("---")
        
        # System Control
        st.header("⚙️ System")
        
        auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", 
                                  value=st.session_state.iot_dashboard["auto_refresh"])
        st.session_state.iot_dashboard["auto_refresh"] = auto_refresh
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.iot_dashboard["logs"] = []
            st.rerun()
    
    # ============ MAIN DASHBOARD ============
    
    # Row 1: Live Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.iot_dashboard["temperature"]
        temp_color = "#3B82F6" if temp < 25 else "#EF4444" if temp > 28 else "#10B981"
        
        st.markdown(f"""
        <div class="dashboard-card" style="border-left-color: {temp_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🌡️ TEMPERATURE</h3>
            <div class="big-number pulse" style="color: {temp_color};">
                {temp:.1f}°C
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                {st.session_state.iot_dashboard['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.iot_dashboard["humidity"]
        hum_color = "#3B82F6"
        
        st.markdown(f"""
        <div class="dashboard-card" style="border-left-color: {hum_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">💧 HUMIDITY</h3>
            <div class="big-number pulse" style="color: {hum_color};">
                {hum:.1f}%
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                {st.session_state.iot_dashboard['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.iot_dashboard["status"]
        status_color = "#3B82F6" if "DINGIN" in status else "#EF4444" if "PANAS" in status else "#10B981"
        
        st.markdown(f"""
        <div class="dashboard-card" style="border-left-color: {status_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🏷️ ROOM STATUS</h3>
            <div class="big-number" style="color: {status_color}; font-size: 2rem;">
                {status}
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                Based on temperature
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ml_status = "TRAINED" if st.session_state.iot_dashboard["ml_trained"] else "NOT TRAINED"
        ml_color = "#10B981" if st.session_state.iot_dashboard["ml_trained"] else "#F59E0B"
        ml_icon = "✅" if st.session_state.iot_dashboard["ml_trained"] else "⚠️"
        
        st.markdown(f"""
        <div class="dashboard-card" style="border-left-color: {ml_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🤖 ML STATUS</h3>
            <div class="big-number" style="color: {ml_color}; font-size: 1.8rem;">
                {ml_icon} {ml_status}
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                KNN • DT • LR Ready
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: ML Predictions
    st.subheader("🔮 ML Model Predictions")
    
    if st.session_state.iot_dashboard["ml_predictions"]:
        predictions = st.session_state.iot_dashboard["ml_predictions"]
        
        col_knn, col_dt, col_lr = st.columns(3)
        
        # KNN
        with col_knn:
            if "KNN" in predictions:
                pred = predictions["KNN"]
                color = "#3B82F6" if "DINGIN" in pred["label"] else "#EF4444" if "PANAS" in pred["label"] else "#10B981"
                
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: {color};">K-Nearest Neighbors</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px;">
                        <small>🥶: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Decision Tree
        with col_dt:
            if "Decision Tree" in predictions:
                pred = predictions["Decision Tree"]
                color = "#3B82F6" if "DINGIN" in pred["label"] else "#EF4444" if "PANAS" in pred["label"] else "#10B981"
                
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: {color};">Decision Tree</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px;">
                        <small>🥶: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Logistic Regression
        with col_lr:
            if "Logistic Regression" in predictions:
                pred = predictions["Logistic Regression"]
                color = "#3B82F6" if "DINGIN" in pred["label"] else "#EF4444" if "PANAS" in pred["label"] else "#10B981"
                
                st.markdown(f"""
                <div class="model-card">
                    <h4 style="color: {color};">Logistic Regression</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px;">
                        <small>🥶: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        if st.session_state.iot_dashboard["ml_trained"]:
            st.info("⏳ No predictions yet. Waiting for data...")
        else:
            st.warning("🤖 Train ML models first to enable predictions")
    
    st.markdown("---")
    
    # Row 3: Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Charts", "📋 History", "📝 Logs"])
    
    with tab1:
        if len(st.session_state.iot_dashboard["sensor_history"]) > 1:
            history_df = pd.DataFrame(st.session_state.iot_dashboard["sensor_history"])
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_temp = px.line(
                    history_df, 
                    x='timestamp', 
                    y='temperature',
                    title='Temperature Trend',
                    color_discrete_sequence=['#EF4444']
                )
                fig_temp.add_hline(y=25, line_dash="dash", line_color="#3B82F6")
                fig_temp.add_hline(y=28, line_dash="dash", line_color="#EF4444")
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col_chart2:
                fig_hum = px.line(
                    history_df,
                    x='timestamp',
                    y='humidity',
                    title='Humidity Trend',
                    color_discrete_sequence=['#3B82F6']
                )
                st.plotly_chart(fig_hum, use_container_width=True)
        else:
            st.info("📈 Waiting for data...")
    
    with tab2:
        if st.session_state.iot_dashboard["sensor_history"]:
            df = pd.DataFrame(st.session_state.iot_dashboard["sensor_history"][-20:])
            df['timestamp'] = df['timestamp'].dt.strftime('%H:%M:%S')
            st.dataframe(df[['timestamp', 'temperature', 'humidity', 'status', 'source']], 
                        use_container_width=True)
        else:
            st.info("📋 No data history yet")
    
    with tab3:
        log_container = st.container(height=400)
        
        with log_container:
            for log in st.session_state.iot_dashboard["logs"][:30]:
                st.markdown(f"""
                <div style="color: {log['color']}; font-family: 'Courier New'; 
                         font-size: 0.85rem; padding: 0.3rem 0; border-bottom: 1px solid #E5E7EB;">
                    {log['icon']} <strong>[{log['time']}]</strong> {log['message']}
                </div>
                """, unsafe_allow_html=True)
    
    # Auto-refresh
    if st.session_state.iot_dashboard["auto_refresh"]:
        time.sleep(3)
        st.rerun()

# ==================== RUN ====================
if __name__ == "__main__":
    main()

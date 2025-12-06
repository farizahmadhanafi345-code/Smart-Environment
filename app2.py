# dashboard_complete_final.py
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HIVEMQ REAL ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "fariz_device_main",
    "password": "F4riz#Device2025!",
    "use_ssl": True,
    "keepalive": 60,
    "clean_session": True
}

# Topics REAL dari ESP32 Anda
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# Path untuk ML models
ML_MODELS_DIR = "ml_models"
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# ==================== DEBUG MODE ====================
DEBUG_MODE = True

def debug_log(message):
    """Log untuk debugging"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[DEBUG {timestamp}] {message}")

# ==================== INISIALISASI STATE ====================
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        # Connection State
        "mqtt_connected": False,
        "connection_status": "disconnected",  # disconnected, connecting, connected, error
        "last_connection_attempt": None,
        "connection_errors": [],
        "reconnect_count": 0,
        
        # Sensor Data
        "temperature": 25.0,
        "humidity": 65.0,
        "timestamp": "",
        "status": "NORMAL",
        "status_code": 1,
        
        # Data History
        "sensor_history": [],
        "csv_records": [],
        "data_points": 0,
        
        # ML System
        "ml_models": {},  # KNN, DT, LR
        "ml_scaler": None,
        "ml_trained": False,
        "ml_training_data": [],
        "ml_predictions": {},
        "ml_performance": {},
        
        # Control
        "led_status": "off",
        "auto_refresh": True,
        
        # System
        "system_logs": [],
        "start_time": datetime.now()
    }

# ==================== LOGGING SYSTEM ====================
def add_system_log(message, level="INFO", source="SYSTEM"):
    """Enhanced logging system"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    level_config = {
        "DEBUG": {"icon": "🔍", "color": "#6B7280"},
        "INFO": {"icon": "ℹ️", "color": "#3B82F6"},
        "SUCCESS": {"icon": "✅", "color": "#10B981"},
        "WARNING": {"icon": "⚠️", "color": "#F59E0B"},
        "ERROR": {"icon": "❌", "color": "#EF4444"},
        "MQTT": {"icon": "📡", "color": "#8B5CF6"},
        "SENSOR": {"icon": "🌡️", "color": "#EF4444"},
        "ML": {"icon": "🤖", "color": "#7C3AED"}
    }
    
    config = level_config.get(level, level_config["INFO"])
    
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "level": level,
        "icon": config["icon"],
        "color": config["color"],
        "source": source
    }
    
    st.session_state.app_state["system_logs"].insert(0, log_entry)
    
    if len(st.session_state.app_state["system_logs"]) > 50:
        st.session_state.app_state["system_logs"] = st.session_state.app_state["system_logs"][:50]
    
    # Console debug
    debug_log(f"[{level}] {message}")

# ==================== ML SYSTEM (KNN, DT, LR) ====================
class MLSystem:
    def __init__(self):
        self.models = {
            "KNN": None,
            "Decision Tree": None,
            "Logistic Regression": None
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.load_models()
    
    def load_models(self):
        """Load trained models dari file"""
        try:
            loaded_count = 0
            
            for name in self.models.keys():
                model_file = os.path.join(ML_MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
                if os.path.exists(model_file):
                    with open(model_file, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    loaded_count += 1
                    add_system_log(f"Loaded {name} model", "SUCCESS", "ML")
            
            # Load scaler
            scaler_file = os.path.join(ML_MODELS_DIR, "scaler.pkl")
            if os.path.exists(scaler_file):
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if loaded_count > 0:
                self.is_trained = True
                st.session_state.app_state["ml_trained"] = True
                add_system_log(f"{loaded_count} ML models loaded", "SUCCESS", "ML")
                return True
            else:
                add_system_log("No trained models found", "WARNING", "ML")
                return False
                
        except Exception as e:
            add_system_log(f"Error loading models: {e}", "ERROR", "ML")
            return False
    
    def prepare_training_data(self):
        """Siapkan data training dari collected data"""
        if len(st.session_state.app_state["ml_training_data"]) < 10:
            return None, None, "Need at least 10 samples"
        
        try:
            df = pd.DataFrame(st.session_state.app_state["ml_training_data"])
            X = df[['temperature', 'humidity']].values
            y = df['status_code'].values
            
            return X, y, None
            
        except Exception as e:
            return None, None, str(e)
    
    def train_all_models(self):
        """Train KNN, Decision Tree, dan Logistic Regression"""
        try:
            X, y, error = self.prepare_training_data()
            if error:
                return False, error
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models
            models_config = {
                "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            }
            
            performance = {}
            
            for name, model in models_config.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate accuracy
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model
                self.models[name] = model
                performance[name] = {
                    "accuracy": float(accuracy),
                    "model": model,
                    "y_test": y_test,
                    "y_pred": y_pred
                }
                
                # Save model to file
                model_file = os.path.join(ML_MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
                
                add_system_log(f"{name} trained: {accuracy:.2%}", "SUCCESS", "ML")
            
            # Save scaler
            scaler_file = os.path.join(ML_MODELS_DIR, "scaler.pkl")
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.is_trained = True
            st.session_state.app_state["ml_trained"] = True
            st.session_state.app_state["ml_performance"] = performance
            
            # Update ML models in session state
            st.session_state.app_state["ml_models"] = self.models
            
            return True, f"Models trained successfully with {len(X)} samples"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict_all(self, temperature, humidity):
        """Predict dengan semua model"""
        if not self.is_trained:
            return {}
        
        try:
            features = np.array([[temperature, humidity]])
            features_scaled = self.scaler.transform(features)
            
            predictions = {}
            label_map = {0: "DINGIN", 1: "NORMAL", 2: "PANAS"}
            
            for name, model in self.models.items():
                if model is None:
                    continue
                
                # Predict
                pred_code = model.predict(features_scaled)[0]
                
                # Get probabilities
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_scaled)[0]
                    confidence = float(probs[pred_code])
                else:
                    confidence = 1.0
                    probs = [0.33, 0.33, 0.34]
                
                predictions[name] = {
                    "label": label_map.get(pred_code, "UNKNOWN"),
                    "confidence": confidence,
                    "label_code": int(pred_code),
                    "probabilities": {
                        "DINGIN": float(probs[0]) if len(probs) > 0 else 0,
                        "NORMAL": float(probs[1]) if len(probs) > 1 else 0,
                        "PANAS": float(probs[2]) if len(probs) > 2 else 0
                    }
                }
            
            return predictions
            
        except Exception as e:
            add_system_log(f"Prediction error: {e}", "ERROR", "ML")
            return {}

# ==================== MQTT CONNECTION MANAGER ====================
class MQTTConnectionManager:
    def __init__(self):
        self.client = None
        self.connection_in_progress = False
        
    def on_connect(self, client, userdata, flags, rc):
        """Callback ketika terhubung"""
        debug_log(f"on_connect called with rc={rc}")
        
        if rc == 0:
            # Connection successful
            st.session_state.app_state["mqtt_connected"] = True
            st.session_state.app_state["connection_status"] = "connected"
            st.session_state.app_state["reconnect_count"] = 0
            
            # Subscribe to topic
            client.subscribe(DHT_TOPIC)
            
            add_system_log(f"✅ Connected to HiveMQ! Subscribed to {DHT_TOPIC}", "SUCCESS", "MQTT")
            debug_log(f"Subscribed to topic: {DHT_TOPIC}")
            
        else:
            # Connection failed
            st.session_state.app_state["mqtt_connected"] = False
            st.session_state.app_state["connection_status"] = "error"
            
            error_messages = {
                1: "Incorrect protocol version",
                2: "Invalid client identifier",
                3: "Server unavailable",
                4: "Bad username or password",
                5: "Not authorized"
            }
            
            error_msg = error_messages.get(rc, f"Unknown error {rc}")
            add_system_log(f"❌ Connection failed: {error_msg}", "ERROR", "MQTT")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback ketika terputus"""
        debug_log(f"on_disconnect called with rc={rc}")
        
        st.session_state.app_state["mqtt_connected"] = False
        st.session_state.app_state["connection_status"] = "disconnected"
        
        if rc != 0:
            add_system_log("⚠️ Unexpected disconnection", "WARNING", "MQTT")
            # Try to reconnect
            self.schedule_reconnect()
    
    def on_message(self, client, userdata, msg):
        """Callback ketika menerima pesan"""
        try:
            debug_log(f"Message received on topic: {msg.topic}")
            
            data = json.loads(msg.payload.decode('utf-8'))
            temperature = float(data.get('temperature', 0))
            humidity = float(data.get('humidity', 0))
            
            debug_log(f"Parsed data: temp={temperature}, hum={humidity}")
            
            # Determine status
            if temperature < 25:
                status, color, code = "DINGIN 🥶", "#3B82F6", 0
            elif temperature > 28:
                status, color, code = "PANAS 🔥", "#EF4444", 2
            else:
                status, color, code = "NORMAL ✅", "#10B981", 1
            
            # Update sensor data
            st.session_state.app_state.update({
                "temperature": temperature,
                "humidity": humidity,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": status,
                "status_code": code,
                "data_points": st.session_state.app_state["data_points"] + 1
            })
            
            # Add to history
            st.session_state.app_state["sensor_history"].append({
                "timestamp": datetime.now(),
                "temperature": temperature,
                "humidity": humidity,
                "status": status.split()[0],
                "color": color
            })
            
            # Keep only last 100 records
            if len(st.session_state.app_state["sensor_history"]) > 100:
                st.session_state.app_state["sensor_history"] = st.session_state.app_state["sensor_history"][-100:]
            
            # Add to ML training data
            st.session_state.app_state["ml_training_data"].append({
                "temperature": temperature,
                "humidity": humidity,
                "status_code": code
            })
            
            # Make ML predictions if trained
            if 'ml_system' in st.session_state and st.session_state.ml_system.is_trained:
                predictions = st.session_state.ml_system.predict_all(temperature, humidity)
                st.session_state.app_state["ml_predictions"] = predictions
            
            # Save to CSV
            self.save_to_csv(temperature, humidity, status, code)
            
            add_system_log(f"📡 Data: {temperature:.1f}°C, {humidity:.1f}% → {status}", "SENSOR", "SENSOR")
            
        except Exception as e:
            add_system_log(f"Error processing message: {e}", "ERROR", "MQTT")
    
    def save_to_csv(self, temperature, humidity, status, code):
        """Simpan data ke CSV"""
        try:
            csv_file = "iot_data.csv"
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                
                if not file_exists:
                    writer.writerow(['timestamp', 'temperature', 'humidity', 'status', 'status_code'])
                
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                writer.writerow([
                    timestamp,
                    f"{temperature:.2f}",
                    f"{humidity:.2f}",
                    status,
                    code
                ])
            
            # Store in memory for display
            st.session_state.app_state["csv_records"].append({
                "timestamp": timestamp,
                "temperature": temperature,
                "humidity": humidity,
                "status": status.split()[0]
            })
            
            if len(st.session_state.app_state["csv_records"]) > 20:
                st.session_state.app_state["csv_records"] = st.session_state.app_state["csv_records"][-20:]
                
        except Exception as e:
            add_system_log(f"CSV save error: {e}", "ERROR", "SYSTEM")
    
    def schedule_reconnect(self):
        """Jadwalkan reconnect"""
        if st.session_state.app_state["reconnect_count"] < 5:  # Max 5 attempts
            st.session_state.app_state["reconnect_count"] += 1
            delay = min(2 ** st.session_state.app_state["reconnect_count"], 30)
            
            add_system_log(f"Reconnect attempt {st.session_state.app_state['reconnect_count']} in {delay}s", "WARNING", "MQTT")
            
            def reconnect():
                time.sleep(delay)
                if not st.session_state.app_state["mqtt_connected"]:
                    self.connect_mqtt()
            
            threading.Thread(target=reconnect, daemon=True).start()
    
    def connect_mqtt(self):
        """Connect ke MQTT broker"""
        if self.connection_in_progress:
            return False
        
        try:
            self.connection_in_progress = True
            st.session_state.app_state["connection_status"] = "connecting"
            st.session_state.app_state["last_connection_attempt"] = datetime.now()
            
            add_system_log(f"Connecting to {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}...", "INFO", "MQTT")
            
            # Create client
            client_id = f"dashboard_{int(time.time())}"
            self.client = mqtt.Client(client_id=client_id, clean_session=MQTT_CONFIG["clean_session"])
            
            # Set credentials
            self.client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
            
            # Set SSL if enabled
            if MQTT_CONFIG["use_ssl"]:
                self.client.tls_set(tls_version=ssl.PROTOCOL_TLS)
                self.client.tls_insecure_set(True)  # For testing
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # Connect
            self.client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=MQTT_CONFIG["keepalive"])
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection result
            for i in range(20):  # Wait up to 2 seconds
                if st.session_state.app_state["mqtt_connected"]:
                    break
                time.sleep(0.1)
            
            self.connection_in_progress = False
            
            if st.session_state.app_state["mqtt_connected"]:
                return True
            else:
                add_system_log("Connection timeout", "ERROR", "MQTT")
                return False
                
        except Exception as e:
            self.connection_in_progress = False
            st.session_state.app_state["connection_status"] = "error"
            
            error_msg = str(e)
            add_system_log(f"Connection error: {error_msg}", "ERROR", "MQTT")
            
            # Store error
            st.session_state.app_state["connection_errors"].append({
                "time": datetime.now(),
                "error": error_msg
            })
            
            return False
    
    def disconnect_mqtt(self):
        """Disconnect dari MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                st.session_state.app_state["mqtt_connected"] = False
                st.session_state.app_state["connection_status"] = "disconnected"
                add_system_log("Disconnected from MQTT", "INFO", "MQTT")
                return True
        except:
            pass
        return False
    
    def send_led_command(self, command):
        """Kirim perintah LED"""
        if self.client and st.session_state.app_state["mqtt_connected"]:
            try:
                self.client.publish(LED_TOPIC, command)
                st.session_state.app_state["led_status"] = command
                add_system_log(f"LED command sent: {command}", "SUCCESS", "CONTROL")
                return True
            except Exception as e:
                add_system_log(f"Failed to send LED command: {e}", "ERROR", "CONTROL")
                return False
        return False

# ==================== INITIALIZE SYSTEMS ====================
if 'mqtt_manager' not in st.session_state:
    st.session_state.mqtt_manager = MQTTConnectionManager()

if 'ml_system' not in st.session_state:
    st.session_state.ml_system = MLSystem()

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="ESP32 IoT Dashboard Pro",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 15px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    }
    
    .ml-model-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .ml-model-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 5px 15px rgba(59, 130, 246, 0.1);
    }
    
    .connection-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    .connected { background: #D1FAE5; color: #065F46; border: 1px solid #A7F3D0; }
    .connecting { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
    .disconnected { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    .error { background: #FEE2E2; color: #991B1B; border: 1px solid #FECACA; }
    
    .log-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    
    .data-value {
        font-size: 2.5rem;
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
    
    .tab-content {
        padding: 1rem 0;
    }
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
        <h1 style="margin: 0; font-size: 2.8rem;">🤖 ESP32 SMART DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Real-time IoT + KNN • Decision Tree • Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 Connection Control")
        
        # Connection status
        status = st.session_state.app_state["connection_status"]
        status_display = {
            "connected": ("🟢 CONNECTED", "connected"),
            "connecting": ("🟡 CONNECTING", "connecting"),
            "disconnected": ("🔴 DISCONNECTED", "disconnected"),
            "error": ("🔴 ERROR", "error")
        }.get(status, ("⚪ UNKNOWN", "disconnected"))
        
        col_status, col_info = st.columns([2, 1])
        with col_status:
            st.markdown(f'<div class="connection-status {status_display[1]}">{status_display[0]}</div>', unsafe_allow_html=True)
        with col_info:
            st.caption(f"Data: {st.session_state.app_state['data_points']}")
        
        # Connection buttons
        if not st.session_state.app_state["mqtt_connected"]:
            if st.button("🔗 Connect to ESP32", type="primary", use_container_width=True):
                with st.spinner("Connecting to HiveMQ..."):
                    if st.session_state.mqtt_manager.connect_mqtt():
                        st.success("Connected successfully!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Connection failed. Check credentials.")
        else:
            if st.button("🔌 Disconnect", type="secondary", use_container_width=True):
                if st.session_state.mqtt_manager.disconnect_mqtt():
                    st.warning("Disconnected")
                    time.sleep(1)
                    st.rerun()
        
        # Connection details expander
        with st.expander("🔍 Connection Details"):
            st.write(f"**Broker:** `{MQTT_CONFIG['broker']}`")
            st.write(f"**Port:** `{MQTT_CONFIG['port']}`")
            st.write(f"**Username:** `{MQTT_CONFIG['username']}`")
            st.write(f"**Topic:** `{DHT_TOPIC}`")
            st.write(f"**Reconnect attempts:** {st.session_state.app_state['reconnect_count']}")
            
            if st.session_state.app_state["connection_errors"]:
                st.write("**Recent Errors:**")
                for error in st.session_state.app_state["connection_errors"][-3:]:
                    st.error(f"{error['time'].strftime('%H:%M:%S')}: {error['error'][:50]}...")
        
        st.markdown("---")
        
        # ML Control Section
        st.header("🤖 ML System Control")
        
        # ML Status
        if st.session_state.app_state["ml_trained"]:
            st.success(f"✅ {len([m for m in st.session_state.ml_system.models.values() if m is not None])} models trained")
        else:
            st.warning("⚠️ Models not trained")
        
        # Train ML button
        train_samples = len(st.session_state.app_state["ml_training_data"])
        if st.button(f"🏋️ Train ML Models ({train_samples} samples)", 
                    type="secondary", 
                    use_container_width=True,
                    disabled=train_samples < 10):
            with st.spinner("Training KNN, DT, LR..."):
                success, message = st.session_state.ml_system.train_all_models()
                if success:
                    st.success(message)
                else:
                    st.error(message)
                time.sleep(2)
                st.rerun()
        
        # Manual prediction test
        st.markdown("---")
        st.subheader("🧪 Test Prediction")
        
        col_temp, col_hum = st.columns(2)
        with col_temp:
            test_temp = st.number_input("Temp (°C)", 15.0, 35.0, 25.0, 0.5)
        with col_hum:
            test_hum = st.number_input("Hum (%)", 30.0, 90.0, 65.0, 1.0)
        
        if st.button("🔮 Predict with ML", use_container_width=True,
                    disabled=not st.session_state.app_state["ml_trained"]):
            predictions = st.session_state.ml_system.predict_all(test_temp, test_hum)
            if predictions:
                st.session_state.app_state["ml_predictions"] = predictions
                st.success("Predictions generated!")
                st.rerun()
            else:
                st.warning("No predictions available")
        
        st.markdown("---")
        
        # LED Control
        st.header("💡 LED Control")
        
        col_led1, col_led2 = st.columns(2)
        with col_led1:
            if st.button("🔴 RED", use_container_width=True,
                        disabled=not st.session_state.app_state["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("merah")
                st.success("Sent: RED")
            
            if st.button("🟢 GREEN", use_container_width=True,
                        disabled=not st.session_state.app_state["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("hijau")
                st.success("Sent: GREEN")
        
        with col_led2:
            if st.button("🟡 YELLOW", use_container_width=True,
                        disabled=not st.session_state.app_state["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("kuning")
                st.success("Sent: YELLOW")
            
            if st.button("⚫ OFF", use_container_width=True,
                        disabled=not st.session_state.app_state["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("off")
                st.success("Sent: OFF")
        
        st.caption(f"Status: {st.session_state.app_state['led_status'].upper()}")
        
        st.markdown("---")
        
        # System Controls
        st.header("⚙️ System")
        
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", 
                                  value=st.session_state.app_state["auto_refresh"])
        st.session_state.app_state["auto_refresh"] = auto_refresh
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.app_state["system_logs"] = []
            st.rerun()
    
    # ============ MAIN DASHBOARD ============
    
    # Row 1: Live Sensor Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.app_state["temperature"]
        temp_color = "#3B82F6" if temp < 25 else "#EF4444" if temp > 28 else "#10B981"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {temp_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🌡️ TEMPERATURE</h3>
            <div class="data-value pulse" style="color: {temp_color};">
                {temp:.1f}°C
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                {st.session_state.app_state['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.app_state["humidity"]
        hum_color = "#3B82F6"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {hum_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">💧 HUMIDITY</h3>
            <div class="data-value pulse" style="color: {hum_color};">
                {hum:.1f}%
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                {st.session_state.app_state['timestamp']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.app_state["status"]
        status_color = "#3B82F6" if "DINGIN" in status else "#EF4444" if "PANAS" in status else "#10B981"
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {status_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🏷️ ROOM STATUS</h3>
            <div class="data-value" style="color: {status_color}; font-size: 2rem;">
                {status}
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                Auto-classified
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        ml_status = "✅ TRAINED" if st.session_state.app_state["ml_trained"] else "⚠️ NOT TRAINED"
        ml_color = "#10B981" if st.session_state.app_state["ml_trained"] else "#F59E0B"
        trained_count = len([m for m in st.session_state.ml_system.models.values() if m is not None])
        
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: {ml_color};">
            <h3 style="color: #6B7280; margin: 0 0 0.5rem 0;">🤖 ML STATUS</h3>
            <div class="data-value" style="color: {ml_color}; font-size: 1.8rem;">
                {ml_status}
            </div>
            <p style="color: #9CA3AF; margin: 0; font-size: 0.9rem;">
                {trained_count}/3 models ready
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: ML Predictions Display
    st.subheader("🔮 ML Model Predictions")
    
    if st.session_state.app_state["ml_predictions"]:
        col_knn, col_dt, col_lr = st.columns(3)
        
        predictions = st.session_state.app_state["ml_predictions"]
        
        # KNN Prediction
        with col_knn:
            if "KNN" in predictions:
                pred = predictions["KNN"]
                color = "#3B82F6" if pred["label"] == "DINGIN" else "#EF4444" if pred["label"] == "PANAS" else "#10B981"
                
                st.markdown(f"""
                <div class="ml-model-card">
                    <h4 style="color: {color}; margin: 0 0 0.5rem 0;">K-Nearest Neighbors</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p style="font-weight: bold; color: #6B7280;">
                        Confidence: {pred['confidence']:.1%}
                    </p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                        <small>🥶 DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅ NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥 PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Decision Tree Prediction
        with col_dt:
            if "Decision Tree" in predictions:
                pred = predictions["Decision Tree"]
                color = "#3B82F6" if pred["label"] == "DINGIN" else "#EF4444" if pred["label"] == "PANAS" else "#10B981"
                
                st.markdown(f"""
                <div class="ml-model-card">
                    <h4 style="color: {color}; margin: 0 0 0.5rem 0;">Decision Tree</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p style="font-weight: bold; color: #6B7280;">
                        Confidence: {pred['confidence']:.1%}
                    </p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                        <small>🥶 DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅ NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥 PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Logistic Regression Prediction
        with col_lr:
            if "Logistic Regression" in predictions:
                pred = predictions["Logistic Regression"]
                color = "#3B82F6" if pred["label"] == "DINGIN" else "#EF4444" if pred["label"] == "PANAS" else "#10B981"
                
                st.markdown(f"""
                <div class="ml-model-card">
                    <h4 style="color: {color}; margin: 0 0 0.5rem 0;">Logistic Regression</h4>
                    <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                    <p style="font-weight: bold; color: #6B7280;">
                        Confidence: {pred['confidence']:.1%}
                    </p>
                    <div style="background: #F3F4F6; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
                        <small>🥶 DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                        <small>✅ NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                        <small>🔥 PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        if st.session_state.app_state["ml_trained"]:
            st.info("⏳ Waiting for data to make predictions")
        else:
            st.warning("🤖 Train ML models first to enable predictions")
    
    st.markdown("---")
    
    # Row 3: Tabs for Charts, Data, and ML Performance
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts", "📊 ML Performance", "📋 Data History", "📝 System Logs"])
    
    with tab1:
        # Charts Tab
        if len(st.session_state.app_state["sensor_history"]) > 1:
            history_df = pd.DataFrame(st.session_state.app_state["sensor_history"])
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                fig_temp = px.line(
                    history_df, 
                    x='timestamp', 
                    y='temperature',
                    title='Temperature Trend',
                    labels={'temperature': 'Temperature (°C)', 'timestamp': 'Time'}
                )
                fig_temp.update_traces(line_color='#EF4444', line_width=3)
                fig_temp.add_hline(y=25, line_dash="dash", line_color="#3B82F6", 
                                 annotation_text="DINGIN (<25°C)")
                fig_temp.add_hline(y=28, line_dash="dash", line_color="#EF4444",
                                 annotation_text="PANAS (>28°C)")
                st.plotly_chart(fig_temp, use_container_width=True)
            
            with col_chart2:
                fig_hum = px.line(
                    history_df,
                    x='timestamp',
                    y='humidity',
                    title='Humidity Trend',
                    labels={'humidity': 'Humidity (%)', 'timestamp': 'Time'}
                )
                fig_hum.update_traces(line_color='#3B82F6', line_width=3)
                st.plotly_chart(fig_hum, use_container_width=True)
        else:
            st.info("📈 Connect to ESP32 to see live charts")
    
    with tab2:
        # ML Performance Tab
        if st.session_state.app_state["ml_performance"]:
            # Accuracy chart
            models = list(st.session_state.app_state["ml_performance"].keys())
            accuracies = [st.session_state.app_state["ml_performance"][m]["accuracy"] for m in models]
            
            fig_acc = go.Figure(data=[
                go.Bar(
                    x=models,
                    y=accuracies,
                    marker_color=['#3B82F6', '#10B981', '#F59E0B'],
                    text=[f"{acc:.1%}" for acc in accuracies],
                    textposition='auto'
                )
            ])
            
            fig_acc.update_layout(
                title='Model Accuracy Comparison',
                yaxis_title='Accuracy',
                yaxis_range=[0, 1],
                height=300
            )
            
            st.plotly_chart(fig_acc, use_container_width=True)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            col_cm1, col_cm2, col_cm3 = st.columns(3)
            
            for idx, model_name in enumerate(models):
                with [col_cm1, col_cm2, col_cm3][idx]:
                    info = st.session_state.app_state["ml_performance"][model_name]
                    cm = confusion_matrix(info["y_test"], info["y_pred"])
                    
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        color_continuous_scale='Blues',
                        title=f'{model_name}',
                        labels=dict(x="Predicted", y="Actual", color="Count")
                    )
                    
                    fig_cm.update_layout(height=250)
                    st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("📊 Train ML models to see performance metrics")
    
    with tab3:
        # Data History Tab
        if st.session_state.app_state["csv_records"]:
            df = pd.DataFrame(st.session_state.app_state["csv_records"])
            
            # Style the dataframe
            def color_status(val):
                if val == 'DINGIN':
                    return 'background-color: #DBEAFE; color: #1E40AF;'
                elif val == 'PANAS':
                    return 'background-color: #FEE2E2; color: #991B1B;'
                else:
                    return 'background-color: #D1FAE5; color: #065F46;'
            
            styled_df = df.style.applymap(color_status, subset=['status'])
            
            st.dataframe(styled_df, use_container_width=True, height=300)
            
            # Download button
            if st.button("📥 Download CSV Data", use_container_width=True):
                csv_file = "iot_data.csv"
                if os.path.exists(csv_file):
                    with open(csv_file, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="Click to Download",
                        data=csv_data,
                        file_name=f"iot_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("📋 No data collected yet")
    
    with tab4:
        # System Logs Tab
        log_container = st.container(height=400)
        
        with log_container:
            for log in st.session_state.app_state["system_logs"][:30]:
                st.markdown(f"""
                <div style="color: {log['color']}; font-family: 'Courier New', monospace; 
                         font-size: 0.85rem; padding: 0.3rem 0; border-bottom: 1px solid #E5E7EB;">
                    {log['icon']} <strong>[{log['timestamp']}]</strong> {log['message']}
                </div>
                """, unsafe_allow_html=True)
    
    # Auto-refresh
    if st.session_state.app_state["auto_refresh"] and st.session_state.app_state["mqtt_connected"]:
        time.sleep(5)
        st.rerun()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()

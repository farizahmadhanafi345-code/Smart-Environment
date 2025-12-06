# dashboard_realtime_ml_complete.py
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HIVEMQ REAL ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "fariz_device_main",
    "password": "F4riz#Device2025!",
    "use_ssl": True,
    "keepalive": 20
}

# Topics REAL dari ESP32 Anda
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# Path untuk menyimpan model ML
ML_MODELS_DIR = "ml_models"
os.makedirs(ML_MODELS_DIR, exist_ok=True)

# ==================== INISIALISASI STATE ====================
if 'system_state' not in st.session_state:
    st.session_state.system_state = {
        # Data Sensor REAL dari ESP32
        "temperature": 0.0,
        "humidity": 0.0,
        "timestamp": "",
        "status": "WAITING",
        "status_code": -1,
        
        # Status Koneksi
        "mqtt_connected": False,
        "data_points_received": 0,
        
        # Data History
        "history": [],
        "csv_data": [],
        
        # ML Models State
        "ml_models_loaded": False,
        "ml_training_data": [],
        "ml_predictions": {},
        "ml_models_info": {},
        
        # Kontrol LED
        "led_status": "off",
        
        # System Logs
        "system_logs": [],
        
        # Statistics
        "start_time": datetime.now()
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# ==================== SISTEM ML ====================
class MLSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.trained = False
        self.load_models()
    
    def load_models(self):
        """Load model ML yang sudah ada"""
        try:
            model_files = {
                "KNN": "knn_model.pkl",
                "Decision Tree": "dt_model.pkl", 
                "Logistic Regression": "lr_model.pkl"
            }
            
            for name, filename in model_files.items():
                model_path = os.path.join(ML_MODELS_DIR, filename)
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[name] = pickle.load(f)
            
            # Load scaler jika ada
            scaler_path = os.path.join(ML_MODELS_DIR, "scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if self.models:
                st.session_state.system_state["ml_models_loaded"] = True
                add_log(f"✅ Loaded {len(self.models)} ML models", "success")
            else:
                add_log("⚠️ No ML models found", "warning")
                
        except Exception as e:
            add_log(f"❌ Error loading ML models: {e}", "error")
    
    def prepare_training_data(self):
        """Siapkan data untuk training"""
        if len(st.session_state.system_state["ml_training_data"]) < 10:
            return None, None, "Need at least 10 samples for training"
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(st.session_state.system_state["ml_training_data"])
            
            # Features dan labels
            X = df[['temperature', 'humidity']].values
            y = df['status_code'].values
            
            return X, y, None
            
        except Exception as e:
            return None, None, str(e)
    
    def train_models(self):
        """Train semua model ML"""
        try:
            X, y, error = self.prepare_training_data()
            if error:
                return False, error
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models dengan parameter optimal
            models_config = {
                "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance'),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
            }
            
            results = {}
            
            for name, model in models_config.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store model
                self.models[name] = model
                results[name] = {
                    'accuracy': accuracy,
                    'model': model,
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                
                # Save model to file
                model_path = os.path.join(ML_MODELS_DIR, f"{name.lower().replace(' ', '_')}_model.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scaler
            scaler_path = os.path.join(ML_MODELS_DIR, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            self.trained = True
            st.session_state.system_state["ml_models_loaded"] = True
            
            # Simpan info model
            st.session_state.system_state["ml_models_info"] = results
            
            return True, f"Trained {len(results)} models with accuracy: " + \
                   ", ".join([f"{name}: {res['accuracy']:.2%}" for name, res in results.items()])
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def predict(self, temperature, humidity):
        """Predict dengan semua model"""
        if not self.models:
            return {}
        
        try:
            # Prepare features
            features = np.array([[temperature, humidity]])
            features_scaled = self.scaler.transform(features)
            
            predictions = {}
            
            for name, model in self.models.items():
                # Predict
                pred_code = model.predict(features_scaled)[0]
                
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(features_scaled)[0]
                    confidence = probs[pred_code]
                else:
                    confidence = 1.0
                    probs = [0, 0, 0]
                
                # Map to label
                label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
                label = label_map.get(pred_code, 'UNKNOWN')
                
                predictions[name] = {
                    'label': label,
                    'confidence': float(confidence),
                    'label_code': int(pred_code),
                    'probabilities': {
                        'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                        'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                        'PANAS': float(probs[2]) if len(probs) > 2 else 0
                    }
                }
            
            return predictions
            
        except Exception as e:
            add_log(f"❌ Prediction error: {e}", "error")
            return {}

# ==================== FUNGSI UTILITAS ====================
def add_log(message, type="info"):
    """Tambahkan log ke sistem"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "mqtt": "📡",
        "sensor": "🌡️",
        "ml": "🤖"
    }
    
    icon = icons.get(type, "📝")
    log_entry = f"[{timestamp}] {icon} {message}"
    
    st.session_state.system_state["system_logs"].insert(0, log_entry)
    
    # Batasi jumlah log
    if len(st.session_state.system_state["system_logs"]) > 30:
        st.session_state.system_state["system_logs"] = st.session_state.system_state["system_logs"][:30]

def determine_status(temperature):
    """Tentukan status berdasarkan suhu"""
    if temperature < 25:
        return "DINGIN 🥶", "#3498db", 0
    elif temperature > 28:
        return "PANAS 🔥", "#e74c3c", 2
    else:
        return "NORMAL ✅", "#2ecc71", 1

def save_to_csv(temperature, humidity, status):
    """Simpan data ke CSV"""
    try:
        csv_file = "iot_data.csv"
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Write header jika file baru
            if not file_exists:
                writer.writerow(['timestamp', 'temperature', 'humidity', 'status', 'status_code'])
            
            # Write data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status_text, _, status_code = determine_status(temperature)
            
            writer.writerow([
                timestamp,
                round(temperature, 2),
                round(humidity, 2),
                status_text,
                status_code
            ])
        
        # Tambah ke csv_data untuk display
        st.session_state.system_state["csv_data"].append({
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'status': status_text
        })
        
        # Batasi data di memory
        if len(st.session_state.system_state["csv_data"]) > 100:
            st.session_state.system_state["csv_data"] = st.session_state.system_state["csv_data"][-100:]
        
        return True
        
    except Exception as e:
        add_log(f"❌ CSV save error: {e}", "error")
        return False

# ==================== MQTT FUNCTIONS ====================
def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.system_state["mqtt_connected"] = True
        client.subscribe(DHT_TOPIC)
        add_log("✅ Connected to HiveMQ", "mqtt")
    else:
        st.session_state.system_state["mqtt_connected"] = False
        add_log(f"❌ Connection failed: {rc}", "error")

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima data REAL dari ESP32"""
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        # Tentukan status
        status_text, status_color, status_code = determine_status(temperature)
        
        # Update data real-time
        st.session_state.system_state.update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "status": status_text,
            "status_code": status_code,
            "data_points_received": st.session_state.system_state["data_points_received"] + 1
        })
        
        # Tambah ke history untuk chart
        history_entry = {
            "timestamp": datetime.now(),
            "temperature": temperature,
            "humidity": humidity,
            "status": status_text,
            "color": status_color
        }
        st.session_state.system_state["history"].append(history_entry)
        
        # Batasi history
        if len(st.session_state.system_state["history"]) > 100:
            st.session_state.system_state["history"] = st.session_state.system_state["history"][-100:]
        
        # Simpan ke CSV
        save_to_csv(temperature, humidity, status_text)
        
        # Tambah ke data training ML
        st.session_state.system_state["ml_training_data"].append({
            'temperature': temperature,
            'humidity': humidity,
            'status_code': status_code
        })
        
        # Lakukan prediksi ML jika model sudah loaded
        if 'ml_system' in st.session_state and st.session_state.system_state["ml_models_loaded"]:
            predictions = st.session_state.ml_system.predict(temperature, humidity)
            st.session_state.system_state["ml_predictions"] = predictions
        
        add_log(f"📡 Data: {temperature}°C, {humidity}% → {status_text}", "sensor")
        
    except Exception as e:
        add_log(f"❌ Error processing MQTT: {e}", "error")

def connect_mqtt():
    """Connect ke MQTT broker"""
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
        
        client.on_connect = on_mqtt_connect
        client.on_message = on_mqtt_message
        
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], 60)
        client.loop_start()
        
        st.session_state.mqtt_client = client
        time.sleep(2)
        
        return True
        
    except Exception as e:
        add_log(f"❌ MQTT Connection failed: {e}", "error")
        return False

def send_led_command(command):
    """Kirim perintah ke LED ESP32"""
    if st.session_state.mqtt_client and st.session_state.system_state["mqtt_connected"]:
        try:
            st.session_state.mqtt_client.publish(LED_TOPIC, command)
            st.session_state.system_state["led_status"] = command
            add_log(f"💡 LED command sent: {command}", "info")
            return True
        except:
            return False
    return False

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="ESP32 Real-Time ML Dashboard",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inisialisasi ML system
    if 'ml_system' not in st.session_state:
        st.session_state.ml_system = MLSystem()
    
    # Custom CSS
    st.markdown("""
    <style>
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .data-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .ml-model-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        border: 2px solid #E5E7EB;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .log-container {
        background: #1F2937;
        color: #E5E7EB;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .btn-ml {
        padding: 0.8rem;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .btn-green { background: #10B981; color: white; }
    .btn-blue { background: #3B82F6; color: white; }
    .btn-orange { background: #F59E0B; color: white; }
    .btn-red { background: #EF4444; color: white; }
    .btn-gray { background: #6B7280; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 2.8rem;">🤖 ESP32 REAL-TIME ML DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">KNN • Decision Tree • Logistic Regression</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 Connection Control")
        
        # Status koneksi
        col_conn1, col_conn2 = st.columns(2)
        with col_conn1:
            if st.session_state.system_state["mqtt_connected"]:
                st.success("🟢 CONNECTED")
            else:
                st.error("🔴 DISCONNECTED")
        
        with col_conn2:
            uptime = datetime.now() - st.session_state.system_state["start_time"]
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            st.caption(f"Uptime: {hours:02d}:{minutes:02d}")
        
        # Tombol koneksi
        if st.button("🔗 Connect to ESP32", type="primary", use_container_width=True):
            with st.spinner("Connecting..."):
                if connect_mqtt():
                    st.success("✅ Connected!")
                    time.sleep(1)
                    st.rerun()
        
        st.markdown("---")
        
        # ML Controls
        st.header("🤖 ML Controls")
        
        # Status model ML
        if st.session_state.system_state["ml_models_loaded"]:
            st.success(f"✅ {len(st.session_state.ml_system.models)} models loaded")
        else:
            st.warning("⚠️ No ML models loaded")
        
        # Train models button
        if st.button("🏋️ Train ML Models", type="secondary", use_container_width=True):
            if len(st.session_state.system_state["ml_training_data"]) >= 10:
                with st.spinner("Training ML models..."):
                    success, message = st.session_state.ml_system.train_models()
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                st.rerun()
            else:
                st.warning(f"Need at least 10 samples (have {len(st.session_state.system_state['ml_training_data'])})")
        
        # Manual training data
        st.markdown("---")
        st.subheader("🧪 Add Training Data")
        
        col_temp, col_hum = st.columns(2)
        with col_temp:
            manual_temp = st.number_input("Temp (°C)", 15.0, 35.0, 25.0, 0.5)
        with col_hum:
            manual_hum = st.number_input("Hum (%)", 30.0, 90.0, 65.0, 1.0)
        
        col_label1, col_label2, col_label3 = st.columns(3)
        with col_label1:
            if st.button("🥶", use_container_width=True):
                status_text, _, status_code = determine_status(manual_temp)
                st.session_state.system_state["ml_training_data"].append({
                    'temperature': manual_temp,
                    'humidity': manual_hum,
                    'status_code': status_code
                })
                st.success(f"Added {status_text}")
        
        with col_label2:
            if st.button("✅", use_container_width=True):
                st.session_state.system_state["ml_training_data"].append({
                    'temperature': manual_temp,
                    'humidity': manual_hum,
                    'status_code': 1
                })
                st.success("Added NORMAL")
        
        with col_label3:
            if st.button("🔥", use_container_width=True):
                status_text, _, status_code = determine_status(manual_temp)
                st.session_state.system_state["ml_training_data"].append({
                    'temperature': manual_temp,
                    'humidity': manual_hum,
                    'status_code': status_code
                })
                st.success(f"Added {status_text}")
        
        st.markdown("---")
        
        # LED Control
        st.header("💡 LED Control")
        
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
        
        st.markdown(f"**LED Status:** {st.session_state.system_state['led_status'].upper()}")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", value=True)
    
    # ============ MAIN DASHBOARD ============
    
    # Row 1: Live Sensor Data
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.system_state["temperature"]
        _, temp_color, _ = determine_status(temp)
        
        st.markdown(f"""
        <div class="data-card" style="border-left-color: {temp_color};">
            <h3>🌡️ TEMPERATURE</h3>
            <h1 class="pulse-animation" style="color: {temp_color}; font-size: 2.5rem; margin: 0.5rem 0;">
                {temp:.1f} °C
            </h1>
            <p style="color: #6B7280;">From ESP32 DHT11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.system_state["humidity"]
        hum_color = "#3498db"
        
        st.markdown(f"""
        <div class="data-card" style="border-left-color: {hum_color};">
            <h3>💧 HUMIDITY</h3>
            <h1 class="pulse-animation" style="color: {hum_color}; font-size: 2.5rem; margin: 0.5rem 0;">
                {hum:.1f} %
            </h1>
            <p style="color: #6B7280;">From ESP32 DHT11</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.system_state["status"]
        _, status_color, _ = determine_status(st.session_state.system_state["temperature"])
        
        st.markdown(f"""
        <div class="data-card" style="border-left-color: {status_color};">
            <h3>🏷️ ROOM STATUS</h3>
            <h1 style="color: {status_color}; font-size: 2.2rem; margin: 0.5rem 0;">
                {status}
            </h1>
            <p style="color: #6B7280;">Based on temperature</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_points = st.session_state.system_state["data_points_received"]
        
        st.markdown(f"""
        <div class="data-card" style="border-left-color: #8B5CF6;">
            <h3>📊 DATA STATS</h3>
            <h1 style="color: #8B5CF6; font-size: 2.5rem; margin: 0.5rem 0;">
                {data_points}
            </h1>
            <p style="color: #6B7280;">Points received</p>
            <p style="color: #6B7280; font-size: 0.9rem;">Last: {st.session_state.system_state['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: ML Predictions
    st.subheader("🔮 ML Model Predictions")
    
    if st.session_state.system_state["ml_models_loaded"]:
        # Get current predictions
        predictions = st.session_state.system_state.get("ml_predictions", {})
        
        if predictions:
            col_knn, col_dt, col_lr = st.columns(3)
            
            # KNN Prediction
            with col_knn:
                if "KNN" in predictions:
                    pred = predictions["KNN"]
                    color = "#3B82F6" if pred['label'] == 'DINGIN' else "#10B981" if pred['label'] == 'NORMAL' else "#EF4444"
                    
                    st.markdown(f"""
                    <div class="ml-model-card">
                        <h3 style="color: {color};">K-Nearest Neighbors</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                        <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                        <div style="margin-top: 0.5rem;">
                            <small>DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                            <small>NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                            <small>PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Decision Tree Prediction
            with col_dt:
                if "Decision Tree" in predictions:
                    pred = predictions["Decision Tree"]
                    color = "#3B82F6" if pred['label'] == 'DINGIN' else "#10B981" if pred['label'] == 'NORMAL' else "#EF4444"
                    
                    st.markdown(f"""
                    <div class="ml-model-card">
                        <h3 style="color: {color};">Decision Tree</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                        <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                        <div style="margin-top: 0.5rem;">
                            <small>DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                            <small>NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                            <small>PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Logistic Regression Prediction
            with col_lr:
                if "Logistic Regression" in predictions:
                    pred = predictions["Logistic Regression"]
                    color = "#3B82F6" if pred['label'] == 'DINGIN' else "#10B981" if pred['label'] == 'NORMAL' else "#EF4444"
                    
                    st.markdown(f"""
                    <div class="ml-model-card">
                        <h3 style="color: {color};">Logistic Regression</h3>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                        <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                        <div style="margin-top: 0.5rem;">
                            <small>DINGIN: {pred['probabilities']['DINGIN']:.1%}</small><br>
                            <small>NORMAL: {pred['probabilities']['NORMAL']:.1%}</small><br>
                            <small>PANAS: {pred['probabilities']['PANAS']:.1%}</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("⏳ Waiting for data to make predictions")
    else:
        st.warning("🤖 ML models not loaded. Train models first in sidebar.")
    
    st.markdown("---")
    
    # Row 3: Real-time Charts
    st.subheader("📈 Real-time Charts")
    
    if len(st.session_state.system_state["history"]) > 1:
        # Prepare data
        history_df = pd.DataFrame(st.session_state.system_state["history"])
        
        tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Humidity Trend", "ML Training Data"])
        
        with tab1:
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8, color='#e74c3c')
            ))
            
            # Add threshold lines
            fig_temp.add_hline(y=25, line_dash="dash", line_color="#3498db", 
                             annotation_text="DINGIN (<25°C)")
            fig_temp.add_hline(y=28, line_dash="dash", line_color="#e74c3c",
                             annotation_text="PANAS (>28°C)")
            
            fig_temp.update_layout(
                title="Temperature Trend",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_hum = go.Figure()
            
            fig_hum.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['humidity'],
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8, color='#3498db')
            ))
            
            fig_hum.update_layout(
                title="Humidity Trend",
                xaxis_title="Time",
                yaxis_title="Humidity (%)",
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_hum, use_container_width=True)
        
        with tab3:
            if len(st.session_state.system_state["ml_training_data"]) > 0:
                ml_df = pd.DataFrame(st.session_state.system_state["ml_training_data"])
                
                fig_scatter = px.scatter(
                    ml_df,
                    x='temperature',
                    y='humidity',
                    color=ml_df['status_code'].map({0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}),
                    title='ML Training Data Distribution',
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    }
                )
                
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("No ML training data collected yet")
    else:
        st.info("⏳ Collecting data... Charts will appear here")
    
    st.markdown("---")
    
    # Row 4: ML Model Performance
    st.subheader("📊 ML Model Performance")
    
    if st.session_state.system_state["ml_models_info"]:
        # Create performance chart
        models = list(st.session_state.system_state["ml_models_info"].keys())
        accuracies = [st.session_state.system_state["ml_models_info"][m]['accuracy'] for m in models]
        
        fig = go.Figure(data=[
            go.Bar(x=models, y=accuracies, marker_color=['#3B82F6', '#10B981', '#F59E0B'])
        ])
        
        fig.update_layout(
            title='Model Accuracy Comparison',
            yaxis_title='Accuracy',
            yaxis_range=[0, 1],
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show confusion matrices
        st.subheader("Confusion Matrices")
        col_cm1, col_cm2, col_cm3 = st.columns(3)
        
        for idx, (name, info) in enumerate(st.session_state.system_state["ml_models_info"].items()):
            with [col_cm1, col_cm2, col_cm3][idx]:
                cm = confusion_matrix(info['y_test'], info['y_pred'])
                
                fig_cm = px.imshow(
                    cm,
                    text_auto=True,
                    color_continuous_scale='Blues',
                    title=f'{name}'
                )
                
                fig_cm.update_layout(
                    xaxis_title='Predicted',
                    yaxis_title='Actual',
                    height=250
                )
                
                st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("---")
    
    # Row 5: System Logs
    st.subheader("📝 System Logs")
    
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    
    # Display logs
    logs = st.session_state.system_state["system_logs"]
    for log in logs[:20]:
        # Color coding
        if "✅" in log:
            st.markdown(f'<span style="color: #2ecc71;">{log}</span>', unsafe_allow_html=True)
        elif "❌" in log or "error" in log.lower():
            st.markdown(f'<span style="color: #e74c3c;">{log}</span>', unsafe_allow_html=True)
        elif "⚠️" in log:
            st.markdown(f'<span style="color: #f39c12;">{log}</span>', unsafe_allow_html=True)
        elif "📡" in log:
            st.markdown(f'<span style="color: #9b59b6;">{log}</span>', unsafe_allow_html=True)
        elif "🤖" in log:
            st.markdown(f'<span style="color: #3498db;">{log}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color: #ecf0f1;">{log}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear Logs", use_container_width=True):
        st.session_state.system_state["system_logs"] = []
        st.rerun()
    
    # Auto-refresh
    if auto_refresh and st.session_state.system_state["mqtt_connected"]:
        time.sleep(3)
        try:
            st.rerun()
        except:
            pass

if __name__ == "__main__":
    main()

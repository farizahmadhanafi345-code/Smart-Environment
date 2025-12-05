import streamlit as st
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
import warnings
import ssl
import threading
warnings.filterwarnings('ignore')

# =============== KONFIGURASI HIVEMQ REAL ANDA ===============
MQTT_CONFIG = {
    "host": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1764923408610",
    "password": "9y&f74G1*pWSD.tQdXa@",
    "use_ssl": True
}

# Topics REAL dari perangkat IoT Anda
PUB_TOPIC_DHT = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"  # Topic tempat ESP32 mengirim data
SUB_TOPIC_LED = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"   # Topic untuk kontrol LED ke ESP32

# =============== INISIALISASI STATE GLOBAL ===============
if 'sensor_live_data' not in st.session_state:
    st.session_state.sensor_live_data = {
        "temperature": 0.0,
        "humidity": 0.0,
        "timestamp": "",
        "led_status": "off",
        "led_mode": "manual",
        "last_update": None,
        "mqtt_connected": False,
        "data_received": False
    }

if 'history' not in st.session_state:
    st.session_state.history = {
        "timestamps": deque(maxlen=100),
        "temperatures": deque(maxlen=100),
        "humidities": deque(maxlen=100),
        "led_commands": deque(maxlen=50),
        "predictions": deque(maxlen=50)
    }

if 'ml_system' not in st.session_state:
    st.session_state.ml_system = None

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# =============== SISTEM MACHINE LEARNING REAL ===============
class RealMLSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.dataset = []
        self.is_trained = False
        self.load_or_init_model()
    
    def load_or_init_model(self):
        """Load model yang sudah ada atau buat model baru"""
        try:
            if os.path.exists('real_iot_model.pkl'):
                with open('real_iot_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.label_encoder = model_data['label_encoder']
                self.is_trained = True
                print("✅ Model loaded from file")
            else:
                self.create_initial_model()
        except:
            self.create_initial_model()
    
    def create_initial_model(self):
        """Buat model awal berdasarkan aturan suhu"""
        print("🔄 Creating initial model...")
        
        # Generate training data berdasarkan aturan fisik
        # Dingin (<25°C) -> Biru
        # Normal (25-28°C) -> Hijau
        # Panas (>28°C) -> Merah
        
        np.random.seed(42)
        n_samples = 300
        
        temperatures = []
        humidities = []
        labels = []
        
        # Data untuk kondisi dingin
        for _ in range(n_samples // 3):
            temp = np.random.uniform(15, 25)
            hum = np.random.uniform(40, 80)
            temperatures.append(temp)
            humidities.append(hum)
            labels.append('biru')
        
        # Data untuk kondisi normal
        for _ in range(n_samples // 3):
            temp = np.random.uniform(25, 28)
            hum = np.random.uniform(40, 80)
            temperatures.append(temp)
            humidities.append(hum)
            labels.append('hijau')
        
        # Data untuk kondisi panas
        for _ in range(n_samples // 3):
            temp = np.random.uniform(28, 35)
            hum = np.random.uniform(30, 70)
            temperatures.append(temp)
            humidities.append(hum)
            labels.append('merah')
        
        X = np.column_stack([temperatures, humidities])
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Save initial model
        self.save_model()
        print("✅ Initial model created and saved")
    
    def predict(self, temperature, humidity):
        """Prediksi warna LED berdasarkan data sensor"""
        if not self.is_trained:
            # Fallback ke aturan sederhana jika model belum ada
            if temperature < 25:
                return 'biru', 0.9
            elif temperature > 28:
                return 'merah', 0.9
            else:
                return 'hijau', 0.9
        
        try:
            # Prepare input
            X = np.array([[temperature, humidity]])
            X_scaled = self.scaler.transform(X)
            
            # Predict
            prediction_encoded = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Get confidence
            confidence = np.max(probabilities)
            
            # Decode prediction
            led_color = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return led_color, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            # Fallback
            if temperature < 25:
                return 'biru', 0.8
            elif temperature > 28:
                return 'merah', 0.8
            else:
                return 'hijau', 0.8
    
    def add_training_data(self, temperature, humidity, led_color, is_correct=True):
        """Tambahkan data training baru dari penggunaan real"""
        self.dataset.append({
            'timestamp': datetime.now(),
            'temperature': temperature,
            'humidity': humidity,
            'led_color': led_color,
            'is_correct': is_correct
        })
        
        # Auto-save dataset
        if len(self.dataset) % 10 == 0:
            self.save_dataset()
    
    def retrain_model(self):
        """Retrain model dengan data baru"""
        if len(self.dataset) < 10:
            return False, "Butuh minimal 10 data untuk training"
        
        try:
            # Convert to arrays
            temperatures = [d['temperature'] for d in self.dataset]
            humidities = [d['humidity'] for d in self.dataset]
            labels = [d['led_color'] for d in self.dataset]
            
            X = np.column_stack([temperatures, humidities])
            y = self.label_encoder.fit_transform(labels)
            
            # Scale
            X_scaled = self.scaler.fit_transform(X)
            
            # Train new model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            self.model.fit(X_scaled, y)
            self.is_trained = True
            
            # Save updated model
            self.save_model()
            
            return True, f"Model retrained with {len(self.dataset)} samples"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def save_model(self):
        """Save model ke file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'dataset': self.dataset,
            'trained_at': datetime.now()
        }
        
        with open('real_iot_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def save_dataset(self):
        """Save dataset ke CSV"""
        if self.dataset:
            df = pd.DataFrame(self.dataset)
            df.to_csv('real_iot_dataset.csv', index=False)

# =============== FUNGSI MQTT REAL ===============
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    """Callback ketika terhubung ke HiveMQ"""
    if rc == 0:
        st.session_state.sensor_live_data["mqtt_connected"] = True
        print(f"✅ Connected to HiveMQ Cloud!")
        print(f"📡 Subscribing to: {PUB_TOPIC_DHT}")
        
        # Subscribe ke topic sensor
        client.subscribe(PUB_TOPIC_DHT)
        
    else:
        st.session_state.sensor_live_data["mqtt_connected"] = False
        print(f"❌ Connection failed with code: {rc}")

def on_mqtt_disconnect(client, userdata, rc, properties=None):
    """Callback ketika terputus dari MQTT"""
    st.session_state.sensor_live_data["mqtt_connected"] = False
    print(f"⚠️ Disconnected from MQTT")

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima data dari perangkat IoT REAL"""
    try:
        # Decode message
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        print(f"📨 Received REAL data from {msg.topic}: {data}")
        
        # Update sensor data
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        st.session_state.sensor_live_data.update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "last_update": datetime.now(),
            "data_received": True
        })
        
        # Add to history
        st.session_state.history["timestamps"].append(
            datetime.now().strftime("%H:%M:%S")
        )
        st.session_state.history["temperatures"].append(temperature)
        st.session_state.history["humidities"].append(humidity)
        
        # Jika mode AI aktif, lakukan prediksi dan kontrol
        if st.session_state.sensor_live_data["led_mode"] == "ai":
            if st.session_state.ml_system and st.session_state.ml_system.is_trained:
                led_color, confidence = st.session_state.ml_system.predict(
                    temperature, humidity
                )
                
                # Update display
                st.session_state.sensor_live_data["led_status"] = led_color
                
                # Kirim perintah ke device jika berbeda dengan status sebelumnya
                current_led = st.session_state.sensor_live_data.get("led_status", "off")
                if led_color != current_led:
                    send_led_command(led_color)
                
                # Simpan prediksi
                st.session_state.history["predictions"].append({
                    "timestamp": datetime.now(),
                    "temperature": temperature,
                    "humidity": humidity,
                    "predicted_led": led_color,
                    "confidence": confidence
                })
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        print(f"Raw payload: {msg.payload}")
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

def connect_to_hivemq():
    """Connect ke HiveMQ Cloud"""
    try:
        # Create MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        # Setup SSL/TLS
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLS)
        
        # Set callbacks
        client.on_connect = on_mqtt_connect
        client.on_disconnect = on_mqtt_disconnect
        client.on_message = on_mqtt_message
        
        # Connect
        client.connect(MQTT_CONFIG["host"], MQTT_CONFIG["port"], 60)
        
        # Start loop in background
        client.loop_start()
        
        # Tunggu koneksi
        time.sleep(3)
        
        # Simpan client di session state
        st.session_state.mqtt_client = client
        
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to connect to HiveMQ: {str(e)}")
        return False

def disconnect_from_hivemq():
    """Disconnect dari HiveMQ"""
    if st.session_state.mqtt_client:
        try:
            st.session_state.mqtt_client.loop_stop()
            st.session_state.mqtt_client.disconnect()
            st.session_state.sensor_live_data["mqtt_connected"] = False
            print("🔌 Disconnected from HiveMQ")
            return True
        except:
            return False
    return False

def send_led_command(command):
    """Kirim perintah ke perangkat IoT REAL"""
    if not st.session_state.mqtt_client:
        st.warning("⚠️ MQTT not connected")
        return False
    
    if not st.session_state.sensor_live_data["mqtt_connected"]:
        st.warning("⚠️ Not connected to HiveMQ")
        return False
    
    try:
        # Kirim perintah ke ESP32
        st.session_state.mqtt_client.publish(SUB_TOPIC_LED, command)
        
        # Update status lokal
        st.session_state.sensor_live_data["led_status"] = command
        
        # Simpan ke history
        st.session_state.history["led_commands"].append({
            "timestamp": datetime.now(),
            "command": command,
            "mode": st.session_state.sensor_live_data["led_mode"]
        })
        
        print(f"💡 Command sent to ESP32: {command}")
        return True
        
    except Exception as e:
        st.error(f"❌ Failed to send command: {str(e)}")
        return False

# =============== FUNGSI UTAMA STREAMLIT ===============
def main():
    st.set_page_config(
        page_title="IoT Dashboard REAL with HiveMQ & AI",
        page_icon="🌡️",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .real-time-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .sensor-value {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .sensor-label {
        font-size: 1rem;
        text-align: center;
        opacity: 0.9;
    }
    .status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-disconnected {
        color: #FF5722;
        font-weight: bold;
    }
    .ai-active {
        border: 3px solid #4CAF50;
        padding: 10px;
        border-radius: 10px;
        background-color: #E8F5E9;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="real-time-card">
        <h1 style="text-align: center; margin: 0;">🌡️ IoT REAL-TIME DASHBOARD</h1>
        <p style="text-align: center; opacity: 0.9;">Live Data from ESP32 via HiveMQ Cloud | AI Control System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Kontrol
    with st.sidebar:
        st.header("🔗 HiveMQ Connection")
        
        # Status koneksi
        col_status1, col_status2 = st.columns(2)
        with col_status1:
            if st.session_state.sensor_live_data["mqtt_connected"]:
                st.success("🟢 CONNECTED")
            else:
                st.error("🔴 DISCONNECTED")
        
        with col_status2:
            if st.session_state.sensor_live_data["data_received"]:
                last_update = st.session_state.sensor_live_data.get("timestamp", "N/A")
                st.caption(f"Last: {last_update}")
        
        # Tombol koneksi
        col_conn1, col_conn2 = st.columns(2)
        with col_conn1:
            if st.button("🔗 Connect HiveMQ", type="primary", use_container_width=True):
                with st.spinner("Connecting to HiveMQ Cloud..."):
                    if connect_to_hivemq():
                        st.success("✅ Connected!")
                        time.sleep(1)
                        st.rerun()
        
        with col_conn2:
            if st.button("🔌 Disconnect", use_container_width=True):
                disconnect_from_hivemq()
                st.warning("Disconnected from HiveMQ")
                time.sleep(1)
                st.rerun()
        
        st.markdown("---")
        
        # MODE KONTROL
        st.header("🎛️ Control Mode")
        
        # Pilih mode
        mode_options = ["manual", "auto", "ai"]
        selected_mode = st.selectbox(
            "Select Control Mode",
            mode_options,
            index=mode_options.index(st.session_state.sensor_live_data.get("led_mode", "manual"))
        )
        
        if selected_mode != st.session_state.sensor_live_data["led_mode"]:
            st.session_state.sensor_live_data["led_mode"] = selected_mode
            st.rerun()
        
        st.markdown(f"**Active Mode:** `{selected_mode.upper()}`")
        
        # KONTROL MANUAL (hanya tampil jika mode manual)
        if selected_mode == "manual":
            st.subheader("🎨 Manual LED Control")
            
            col_led1, col_led2 = st.columns(2)
            
            with col_led1:
                if st.button("🔴 RED", use_container_width=True):
                    if send_led_command("merah"):
                        st.success("Red LED activated")
                    else:
                        st.error("Failed to send command")
                
                if st.button("🟢 GREEN", use_container_width=True):
                    if send_led_command("hijau"):
                        st.success("Green LED activated")
                    else:
                        st.error("Failed to send command")
            
            with col_led2:
                if st.button("🟡 YELLOW", use_container_width=True):
                    if send_led_command("kuning"):
                        st.success("Yellow LED activated")
                    else:
                        st.error("Failed to send command")
                
                if st.button("⚫ OFF", use_container_width=True):
                    if send_led_command("off"):
                        st.success("LED turned off")
                    else:
                        st.error("Failed to send command")
        
        # AI CONTROL PANEL
        if selected_mode == "ai":
            st.subheader("🧠 AI Control System")
            
            # Initialize ML system jika belum ada
            if st.session_state.ml_system is None:
                st.session_state.ml_system = RealMLSystem()
            
            # Status model AI
            if st.session_state.ml_system.is_trained:
                st.success("✅ AI Model is Ready")
                dataset_size = len(st.session_state.ml_system.dataset)
                st.caption(f"Training data: {dataset_size} samples")
            else:
                st.warning("⚠️ AI Model needs training")
            
            # Training controls
            st.markdown("---")
            st.subheader("🤖 AI Training")
            
            if st.button("🎓 Train AI Model", use_container_width=True):
                if st.session_state.ml_system:
                    success, message = st.session_state.ml_system.retrain_model()
                    if success:
                        st.success(message)
                    else:
                        st.warning(message)
            
            # Add training data dari kondisi saat ini
            if st.button("➕ Add Current Data as Training", use_container_width=True):
                if st.session_state.ml_system and st.session_state.sensor_live_data["data_received"]:
                    temp = st.session_state.sensor_live_data["temperature"]
                    hum = st.session_state.sensor_live_data["humidity"]
                    led = st.session_state.sensor_live_data["led_status"]
                    
                    st.session_state.ml_system.add_training_data(temp, hum, led)
                    st.success(f"Added: {temp}°C, {hum}% → LED {led}")
        
        st.markdown("---")
        
        # SYSTEM INFO
        st.header("📡 System Info")
        
        st.info(f"""
        **HiveMQ Host:** {MQTT_CONFIG['host']}
        **Sensor Topic:** `{PUB_TOPIC_DHT}`
        **Control Topic:** `{SUB_TOPIC_LED}`
        **Data Points:** {len(st.session_state.history['temperatures'])}
        """)
    
    # =============== MAIN DASHBOARD ===============
    # Row 1: REAL-TIME SENSOR DATA
    st.subheader("📡 LIVE SENSOR DATA FROM ESP32")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.sensor_live_data["temperature"]
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #FF5722;">
            <div class="sensor-label">🌡️ TEMPERATURE</div>
            <div class="sensor-value">{temp:.1f} °C</div>
            <div style="font-size: 0.8rem; color: #666;">Real from DHT11</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.sensor_live_data["humidity"]
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #2196F3;">
            <div class="sensor-label">💧 HUMIDITY</div>
            <div class="sensor-value">{hum:.1f} %</div>
            <div style="font-size: 0.8rem; color: #666;">Real from DHT11</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        led_status = st.session_state.sensor_live_data["led_status"].upper()
        led_mode = st.session_state.sensor_live_data["led_mode"].upper()
        
        # Color mapping
        color_icons = {
            'merah': '🔴',
            'hijau': '🟢',
            'kuning': '🟡',
            'biru': '🔵',
            'off': '⚫'
        }
        
        icon = color_icons.get(st.session_state.sensor_live_data["led_status"].lower(), '💡')
        
        st.markdown(f"""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #4CAF50;">
            <div class="sensor-label">{icon} LED STATUS</div>
            <div class="sensor-value">{led_status}</div>
            <div style="font-size: 0.8rem; color: #666;">Mode: {led_mode}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.sensor_live_data["data_received"]:
            time_diff = datetime.now() - st.session_state.sensor_live_data["last_update"]
            seconds_ago = time_diff.total_seconds()
            
            if seconds_ago < 5:
                status_color = "#4CAF50"
                status_text = "LIVE"
            elif seconds_ago < 30:
                status_color = "#FFC107"
                status_text = f"{int(seconds_ago)}s ago"
            else:
                status_color = "#FF5722"
                status_text = "STALE"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid {status_color};">
                <div class="sensor-label">🕐 DATA STATUS</div>
                <div class="sensor-value" style="color: {status_color};">{status_text}</div>
                <div style="font-size: 0.8rem; color: #666;">Last: {st.session_state.sensor_live_data['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 10px; border-left: 5px solid #9E9E9E;">
                <div class="sensor-label">🕐 DATA STATUS</div>
                <div class="sensor-value" style="color: #9E9E9E;">WAITING</div>
                <div style="font-size: 0.8rem; color: #666;">No data received</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: REAL-TIME CHARTS
    st.subheader("📈 LIVE DATA TRENDS")
    
    if len(st.session_state.history["temperatures"]) > 1:
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Temperature Trend', 'Humidity Trend'),
            vertical_spacing=0.15
        )
        
        # Temperature chart
        fig.add_trace(
            go.Scatter(
                x=list(st.session_state.history["timestamps"]),
                y=list(st.session_state.history["temperatures"]),
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#FF5722', width=3),
                marker=dict(size=8, color='#FF5722')
            ),
            row=1, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=25, line_dash="dash", line_color="blue", 
                     annotation_text="Cold Threshold", row=1, col=1)
        fig.add_hline(y=28, line_dash="dash", line_color="red",
                     annotation_text="Hot Threshold", row=1, col=1)
        
        # Humidity chart
        fig.add_trace(
            go.Scatter(
                x=list(st.session_state.history["timestamps"]),
                y=list(st.session_state.history["humidities"]),
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#2196F3', width=3),
                marker=dict(size=8, color='#2196F3')
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            template="plotly_white",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Waiting for sensor data... Connect to HiveMQ and ensure ESP32 is sending data")
    
    # Row 3: AI PREDICTION MAP (jika mode AI)
    if st.session_state.sensor_live_data["led_mode"] == "ai" and st.session_state.ml_system:
        st.subheader("🧠 AI PREDICTION MAP")
        
        # Create prediction grid
        temp_range = np.linspace(15, 35, 15)
        hum_range = np.linspace(30, 90, 15)
        
        predictions_grid = []
        
        for t in temp_range:
            row = []
            for h in hum_range:
                pred, _ = st.session_state.ml_system.predict(t, h)
                # Map to numerical values
                mapping = {'merah': 0, 'hijau': 1, 'biru': 2, 'kuning': 3, 'off': 4}
                row.append(mapping.get(pred, 1))
            predictions_grid.append(row)
        
        # Create heatmap
        fig_map = go.Figure(data=go.Heatmap(
            z=predictions_grid,
            x=hum_range,
            y=temp_range,
            colorscale=['#FF5722', '#4CAF50', '#2196F3', '#FFC107', '#9E9E9E'],
            colorbar=dict(
                title="LED Color",
                tickvals=[0, 1, 2, 3, 4],
                ticktext=["RED", "GREEN", "BLUE", "YELLOW", "OFF"]
            )
        ))
        
        # Add current sensor point
        if st.session_state.sensor_live_data["data_received"]:
            fig_map.add_trace(go.Scatter(
                x=[st.session_state.sensor_live_data["humidity"]],
                y=[st.session_state.sensor_live_data["temperature"]],
                mode='markers',
                marker=dict(
                    size=20,
                    color='white',
                    line=dict(width=3, color='black')
                ),
                name='Current Sensor'
            ))
        
        fig_map.update_layout(
            title="AI Decision Map (Temperature vs Humidity)",
            xaxis_title="Humidity (%)",
            yaxis_title="Temperature (°C)",
            height=400
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Show current prediction info
        if st.session_state.sensor_live_data["data_received"]:
            temp = st.session_state.sensor_live_data["temperature"]
            hum = st.session_state.sensor_live_data["humidity"]
            
            predicted_led, confidence = st.session_state.ml_system.predict(temp, hum)
            
            col_pred1, col_pred2 = st.columns(2)
            
            with col_pred1:
                st.info(f"""
                **Current AI Prediction:**
                
                🌡️ Temperature: {temp:.1f}°C
                💧 Humidity: {hum:.1f}%
                🎯 Predicted LED: **{predicted_led.upper()}**
                """)
            
            with col_pred2:
                # Confidence bar
                confidence_pct = confidence * 100
                st.metric("AI Confidence", f"{confidence_pct:.1f}%")
                
                # Progress bar
                st.progress(confidence)
                
                if confidence > 0.8:
                    st.success("High confidence prediction")
                elif confidence > 0.6:
                    st.warning("Medium confidence prediction")
                else:
                    st.error("Low confidence prediction")
    
    # Row 4: DATA HISTORY & LOGS
    st.subheader("📋 DATA HISTORY & LOGS")
    
    tab_logs1, tab_logs2, tab_logs3 = st.tabs(["Sensor History", "LED Commands", "System Logs"])
    
    with tab_logs1:
        if len(st.session_state.history["temperatures"]) > 0:
            # Create history dataframe
            history_df = pd.DataFrame({
                'Time': list(st.session_state.history["timestamps"]),
                'Temperature (°C)': list(st.session_state.history["temperatures"]),
                'Humidity (%)': list(st.session_state.history["humidities"])
            })
            
            st.dataframe(history_df.tail(20), use_container_width=True)
            
            # Statistics
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            with col_stat1:
                if len(st.session_state.history["temperatures"]) > 0:
                    avg_temp = np.mean(list(st.session_state.history["temperatures"]))
                    st.metric("Avg Temperature", f"{avg_temp:.1f}°C")
            
            with col_stat2:
                if len(st.session_state.history["humidities"]) > 0:
                    avg_hum = np.mean(list(st.session_state.history["humidities"]))
                    st.metric("Avg Humidity", f"{avg_hum:.1f}%")
            
            with col_stat3:
                st.metric("Data Points", len(st.session_state.history["timestamps"]))
        else:
            st.info("No sensor history available yet")
    
    with tab_logs2:
        if st.session_state.history["led_commands"]:
            commands_df = pd.DataFrame(st.session_state.history["led_commands"])
            st.dataframe(commands_df.tail(10), use_container_width=True)
        else:
            st.info("No LED commands sent yet")
    
    with tab_logs3:
        # System status
        st.write("**System Status:**")
        
        status_items = [
            ("HiveMQ Connection", "🟢 Connected" if st.session_state.sensor_live_data["mqtt_connected"] else "🔴 Disconnected"),
            ("Data Receiving", "🟢 Active" if st.session_state.sensor_live_data["data_received"] else "🟡 Waiting"),
            ("AI Model", "🟢 Trained" if st.session_state.ml_system and st.session_state.ml_system.is_trained else "🟡 Not Trained"),
            ("Control Mode", st.session_state.sensor_live_data["led_mode"].upper()),
            ("Active LED", st.session_state.sensor_live_data["led_status"].upper())
        ]
        
        for label, value in status_items:
            st.write(f"**{label}:** {value}")
    
    # Auto-refresh setiap 3 detik untuk data real-time
    time.sleep(3)
    st.rerun()

# =============== INITIALIZE AND RUN ===============
if __name__ == "__main__":
    # Initialize ML system
    if st.session_state.ml_system is None:
        st.session_state.ml_system = RealMLSystem()
    
    # Run main app
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Application error: {str(e)}")
        st.info("Please refresh the page and try again")

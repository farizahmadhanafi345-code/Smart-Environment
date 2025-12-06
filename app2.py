# dashboard_realtime_stable.py
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
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HIVEMQ REAL ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "fariz_device_main",
    "password": "F4riz#Device2025!",
    "use_ssl": True,
    "keepalive": 60,  # Increased for stability
    "clean_session": True
}

# Topics REAL dari ESP32 Anda
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# ==================== GLOBAL VARIABLES ====================
CONNECTION_TIMEOUT = 15
RECONNECT_INTERVAL = 5

# ==================== INISIALISASI STATE ====================
if 'iot_system' not in st.session_state:
    st.session_state.iot_system = {
        # Data Sensor REAL
        "temperature": 25.0,
        "humidity": 65.0,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "status": "NORMAL",
        "status_code": 1,
        
        # Connection Management
        "mqtt_connected": False,
        "connection_status": "disconnected",  # disconnected, connecting, connected
        "last_connect_time": None,
        "last_disconnect_time": None,
        "connection_errors": [],
        "reconnect_count": 0,
        
        # Data Storage
        "history": [],
        "csv_data": [],
        "data_count": 0,
        "last_data_time": None,
        
        # ML System
        "ml_models": {},
        "ml_loaded": False,
        "ml_training_data": [],
        "ml_predictions": {},
        "ml_accuracy": {},
        
        # Control
        "led_status": "off",
        "auto_refresh": True,
        
        # Logs
        "system_logs": [],
        "start_time": datetime.now()
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

if 'connection_thread' not in st.session_state:
    st.session_state.connection_thread = None

# ==================== ENHANCED LOGGING ====================
def system_log(message, level="INFO"):
    """Enhanced logging with levels and persistence"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    level_icons = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "MQTT": "📡",
        "SENSOR": "🌡️",
        "ML": "🤖"
    }
    
    icon = level_icons.get(level, "📝")
    
    if level == "ERROR":
        color = "#ef4444"
    elif level == "WARNING":
        color = "#f59e0b"
    elif level == "SUCCESS":
        color = "#10b981"
    elif level == "MQTT":
        color = "#3b82f6"
    else:
        color = "#6b7280"
    
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "level": level,
        "icon": icon,
        "color": color
    }
    
    # Store in session state
    st.session_state.iot_system["system_logs"].insert(0, log_entry)
    
    # Limit logs to 100 entries
    if len(st.session_state.iot_system["system_logs"]) > 100:
        st.session_state.iot_system["system_logs"] = st.session_state.iot_system["system_logs"][:100]
    
    # Also print to console for debugging
    print(f"[{timestamp}] {icon} {message}")

# ==================== MQTT MANAGER (REVISED) ====================
class MQTTManager:
    def __init__(self):
        self.client = None
        self.connected = False
        self.connecting = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
    def create_client(self):
        """Create MQTT client with proper configuration"""
        try:
            # Generate unique client ID
            client_id = f"dashboard_{int(time.time())}"
            
            # Create client with clean session
            self.client = mqtt.Client(
                client_id=client_id,
                clean_session=MQTT_CONFIG.get("clean_session", True),
                protocol=mqtt.MQTTv311
            )
            
            # Set username and password
            self.client.username_pw_set(
                MQTT_CONFIG["username"],
                MQTT_CONFIG["password"]
            )
            
            # Configure TLS/SSL
            if MQTT_CONFIG["use_ssl"]:
                self.client.tls_set(
                    tls_version=ssl.PROTOCOL_TLS,
                    cert_reqs=ssl.CERT_NONE
                )
                self.client.tls_insecure_set(True)  # For testing only
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            self.client.on_log = self.on_log
            
            # Set Last Will Testament
            self.client.will_set(
                "dashboard/status",
                json.dumps({
                    "status": "offline",
                    "timestamp": datetime.now().isoformat(),
                    "client": client_id
                }),
                qos=1,
                retain=True
            )
            
            system_log(f"MQTT client created: {client_id}", "MQTT")
            return True
            
        except Exception as e:
            system_log(f"Failed to create MQTT client: {str(e)}", "ERROR")
            return False
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback when connected to broker"""
        connection_result = {
            0: "Connection successful",
            1: "Incorrect protocol version",
            2: "Invalid client identifier",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        
        if rc == 0:
            self.connected = True
            self.connecting = False
            self.reconnect_attempts = 0
            
            # Update session state
            st.session_state.iot_system["mqtt_connected"] = True
            st.session_state.iot_system["connection_status"] = "connected"
            st.session_state.iot_system["last_connect_time"] = datetime.now()
            
            # Subscribe to topics
            client.subscribe(DHT_TOPIC, qos=1)
            system_log(f"✅ Connected to HiveMQ! Subscribed to: {DHT_TOPIC}", "SUCCESS")
            
            # Publish connection status
            client.publish(
                "dashboard/status",
                json.dumps({
                    "status": "online",
                    "timestamp": datetime.now().isoformat(),
                    "client": client._client_id.decode() if hasattr(client._client_id, 'decode') else str(client._client_id)
                }),
                qos=1,
                retain=True
            )
            
        else:
            error_msg = connection_result.get(rc, f"Unknown error code: {rc}")
            self.connected = False
            st.session_state.iot_system["mqtt_connected"] = False
            st.session_state.iot_system["connection_status"] = "error"
            
            system_log(f"❌ Connection failed: {error_msg}", "ERROR")
            
            # Store error
            st.session_state.iot_system["connection_errors"].append({
                "time": datetime.now(),
                "error": error_msg,
                "code": rc
            })
    
    def on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from broker"""
        self.connected = False
        st.session_state.iot_system["mqtt_connected"] = False
        st.session_state.iot_system["connection_status"] = "disconnected"
        st.session_state.iot_system["last_disconnect_time"] = datetime.now()
        
        if rc == 0:
            system_log("🔌 Disconnected normally", "INFO")
        else:
            system_log(f"⚠️ Unexpected disconnection (code: {rc})", "WARNING")
            self.attempt_reconnect()
    
    def on_message(self, client, userdata, msg):
        """Callback when message received"""
        try:
            # Parse JSON message
            payload = msg.payload.decode('utf-8')
            data = json.loads(payload)
            
            temperature = float(data.get('temperature', 0))
            humidity = float(data.get('humidity', 0))
            
            # Validate data range
            if not (0 <= temperature <= 50) or not (0 <= humidity <= 100):
                system_log(f"Invalid data range: {temperature}°C, {humidity}%", "WARNING")
                return
            
            # Determine status
            status, color, code = self.determine_status(temperature)
            
            # Update system state
            st.session_state.iot_system.update({
                "temperature": temperature,
                "humidity": humidity,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": status,
                "status_code": code,
                "data_count": st.session_state.iot_system["data_count"] + 1,
                "last_data_time": datetime.now()
            })
            
            # Add to history
            history_entry = {
                "timestamp": datetime.now(),
                "temperature": temperature,
                "humidity": humidity,
                "status": status,
                "color": color
            }
            st.session_state.iot_system["history"].append(history_entry)
            
            # Limit history size
            if len(st.session_state.iot_system["history"]) > 200:
                st.session_state.iot_system["history"] = st.session_state.iot_system["history"][-200:]
            
            # Save to CSV
            self.save_to_csv(temperature, humidity, status, code)
            
            # Add to ML training data
            st.session_state.iot_system["ml_training_data"].append({
                "temperature": temperature,
                "humidity": humidity,
                "status_code": code
            })
            
            # Make ML predictions if models are loaded
            if st.session_state.iot_system["ml_loaded"]:
                self.make_ml_predictions(temperature, humidity)
            
            system_log(f"📊 Data: {temperature:.1f}°C, {humidity:.1f}% → {status}", "SENSOR")
            
        except json.JSONDecodeError as e:
            system_log(f"Invalid JSON: {str(e)}", "ERROR")
        except Exception as e:
            system_log(f"Message processing error: {str(e)}", "ERROR")
    
    def on_log(self, client, userdata, level, buf):
        """Callback for MQTT logging"""
        if level == mqtt.MQTT_LOG_INFO:
            system_log(f"MQTT Info: {buf}", "INFO")
        elif level == mqtt.MQTT_LOG_WARNING:
            system_log(f"MQTT Warning: {buf}", "WARNING")
        elif level == mqtt.MQTT_LOG_ERR:
            system_log(f"MQTT Error: {buf}", "ERROR")
    
    def determine_status(self, temperature):
        """Determine room status based on temperature"""
        if temperature < 25:
            return "DINGIN 🥶", "#3498db", 0
        elif temperature > 28:
            return "PANAS 🔥", "#e74c3c", 2
        else:
            return "NORMAL ✅", "#2ecc71", 1
    
    def save_to_csv(self, temperature, humidity, status, code):
        """Save data to CSV file"""
        try:
            csv_file = "iot_data_real.csv"
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
            st.session_state.iot_system["csv_data"].append({
                "timestamp": timestamp,
                "temperature": temperature,
                "humidity": humidity,
                "status": status
            })
            
            if len(st.session_state.iot_system["csv_data"]) > 50:
                st.session_state.iot_system["csv_data"] = st.session_state.iot_system["csv_data"][-50:]
                
        except Exception as e:
            system_log(f"CSV save error: {str(e)}", "ERROR")
    
    def attempt_reconnect(self):
        """Attempt to reconnect to MQTT broker"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            system_log("Max reconnect attempts reached", "ERROR")
            return
        
        self.reconnect_attempts += 1
        st.session_state.iot_system["reconnect_count"] = self.reconnect_attempts
        
        delay = min(2 ** self.reconnect_attempts, 60)  # Exponential backoff
        
        system_log(f"Reconnect attempt {self.reconnect_attempts} in {delay}s...", "WARNING")
        
        def reconnect():
            time.sleep(delay)
            if not self.connected:
                self.connect()
        
        threading.Thread(target=reconnect, daemon=True).start()
    
    def connect(self):
        """Connect to MQTT broker"""
        if self.connecting or self.connected:
            return False
        
        try:
            self.connecting = True
            st.session_state.iot_system["connection_status"] = "connecting"
            
            system_log(f"Connecting to {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}...", "MQTT")
            
            if not self.client:
                if not self.create_client():
                    return False
            
            # Connect with timeout
            self.client.connect(
                MQTT_CONFIG["broker"],
                MQTT_CONFIG["port"],
                keepalive=MQTT_CONFIG["keepalive"]
            )
            
            # Start network loop
            self.client.loop_start()
            
            # Wait for connection
            start_time = time.time()
            while time.time() - start_time < CONNECTION_TIMEOUT:
                if self.connected:
                    return True
                time.sleep(0.1)
            
            # Timeout reached
            if not self.connected:
                system_log("Connection timeout", "ERROR")
                self.client.loop_stop()
                self.connecting = False
                return False
            
            return True
            
        except Exception as e:
            system_log(f"Connection error: {str(e)}", "ERROR")
            self.connecting = False
            st.session_state.iot_system["connection_status"] = "error"
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                self.connected = False
                self.connecting = False
                st.session_state.iot_system["mqtt_connected"] = False
                st.session_state.iot_system["connection_status"] = "disconnected"
                system_log("Disconnected from MQTT", "INFO")
                return True
        except:
            pass
        return False
    
    def send_led_command(self, command):
        """Send LED command to ESP32"""
        if self.connected and self.client:
            try:
                self.client.publish(LED_TOPIC, command, qos=1)
                st.session_state.iot_system["led_status"] = command
                system_log(f"LED command sent: {command}", "SUCCESS")
                return True
            except Exception as e:
                system_log(f"Failed to send LED command: {str(e)}", "ERROR")
                return False
        return False
    
    def make_ml_predictions(self, temperature, humidity):
        """Make ML predictions using loaded models"""
        try:
            # This is a placeholder - implement actual ML prediction
            predictions = {}
            
            # Example prediction logic (replace with actual ML)
            if temperature < 25:
                predictions["KNN"] = {"label": "DINGIN", "confidence": 0.95}
                predictions["DT"] = {"label": "DINGIN", "confidence": 0.92}
                predictions["LR"] = {"label": "DINGIN", "confidence": 0.90}
            elif temperature > 28:
                predictions["KNN"] = {"label": "PANAS", "confidence": 0.93}
                predictions["DT"] = {"label": "PANAS", "confidence": 0.91}
                predictions["LR"] = {"label": "PANAS", "confidence": 0.89}
            else:
                predictions["KNN"] = {"label": "NORMAL", "confidence": 0.94}
                predictions["DT"] = {"label": "NORMAL", "confidence": 0.92}
                predictions["LR"] = {"label": "NORMAL", "confidence": 0.90}
            
            st.session_state.iot_system["ml_predictions"] = predictions
            
        except Exception as e:
            system_log(f"ML prediction error: {str(e)}", "ERROR")

# ==================== INITIALIZE MQTT MANAGER ====================
if 'mqtt_manager' not in st.session_state:
    st.session_state.mqtt_manager = MQTTManager()

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="ESP32 IoT Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .status-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .connection-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.9rem;
    }
    
    .connected { background: #10b981; color: white; }
    .connecting { background: #f59e0b; color: white; }
    .disconnected { background: #ef4444; color: white; }
    
    .log-entry {
        padding: 0.5rem;
        border-bottom: 1px solid #374151;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }
    
    .data-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
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
        <h1 style="margin: 0; font-size: 2.8rem;">🌡️ ESP32 REAL-TIME DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Stable Connection • Live Data • ML Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 Connection Management")
        
        # Connection status with badge
        status = st.session_state.iot_system["connection_status"]
        badge_class = {
            "connected": "connected",
            "connecting": "connecting", 
            "disconnected": "disconnected",
            "error": "disconnected"
        }.get(status, "disconnected")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f'<div class="connection-badge {badge_class}">{status.upper()}</div>', unsafe_allow_html=True)
        
        with col2:
            if st.session_state.iot_system["last_connect_time"]:
                st.caption(f"Last: {st.session_state.iot_system['last_connect_time'].strftime('%H:%M:%S')}")
        
        # Connection buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔗 Connect", type="primary", use_container_width=True,
                        disabled=st.session_state.iot_system["mqtt_connected"]):
                with st.spinner("Connecting..."):
                    if st.session_state.mqtt_manager.connect():
                        st.success("Connected!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Connection failed")
        
        with col_btn2:
            if st.button("🔌 Disconnect", use_container_width=True,
                        disabled=not st.session_state.iot_system["mqtt_connected"]):
                if st.session_state.mqtt_manager.disconnect():
                    st.warning("Disconnected")
                    time.sleep(1)
                    st.rerun()
        
        # Connection info
        with st.expander("Connection Details"):
            st.write(f"**Broker:** {MQTT_CONFIG['broker']}")
            st.write(f"**Port:** {MQTT_CONFIG['port']}")
            st.write(f"**Topic:** {DHT_TOPIC}")
            st.write(f"**Data Points:** {st.session_state.iot_system['data_count']}")
            st.write(f"**Reconnects:** {st.session_state.iot_system['reconnect_count']}")
            
            if st.session_state.iot_system["connection_errors"]:
                st.write("**Recent Errors:**")
                for error in st.session_state.iot_system["connection_errors"][-3:]:
                    st.error(f"{error['time'].strftime('%H:%M:%S')}: {error['error']}")
        
        st.markdown("---")
        
        # LED Control
        st.header("💡 LED Control")
        
        col_led1, col_led2 = st.columns(2)
        with col_led1:
            if st.button("🔴 RED", use_container_width=True,
                        disabled=not st.session_state.iot_system["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("merah")
                st.success("RED sent")
            
            if st.button("🟢 GREEN", use_container_width=True,
                        disabled=not st.session_state.iot_system["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("hijau")
                st.success("GREEN sent")
        
        with col_led2:
            if st.button("🟡 YELLOW", use_container_width=True,
                        disabled=not st.session_state.iot_system["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("kuning")
                st.success("YELLOW sent")
            
            if st.button("⚫ OFF", use_container_width=True,
                        disabled=not st.session_state.iot_system["mqtt_connected"]):
                st.session_state.mqtt_manager.send_led_command("off")
                st.success("OFF sent")
        
        st.write(f"**Current:** {st.session_state.iot_system['led_status'].upper()}")
        
        st.markdown("---")
        
        # System Controls
        st.header("⚙️ System Controls")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (5s)", 
                                  value=st.session_state.iot_system["auto_refresh"])
        st.session_state.iot_system["auto_refresh"] = auto_refresh
        
        if st.button("🔄 Manual Refresh", use_container_width=True):
            st.rerun()
        
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.iot_system["system_logs"] = []
            st.rerun()
    
    # ============ MAIN DASHBOARD ============
    
    # Row 1: Live Data Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.iot_system["temperature"]
        _, temp_color, _ = st.session_state.mqtt_manager.determine_status(temp)
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {temp_color};">
            <h3>🌡️ TEMPERATURE</h3>
            <div class="data-value pulse" style="color: {temp_color};">{temp:.1f}°C</div>
            <p style="color: #6b7280; margin: 0;">Real-time from ESP32</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.iot_system["humidity"]
        hum_color = "#3b82f6"
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {hum_color};">
            <h3>💧 HUMIDITY</h3>
            <div class="data-value pulse" style="color: {hum_color};">{hum:.1f}%</div>
            <p style="color: #6b7280; margin: 0;">Real-time from ESP32</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.iot_system["status"]
        _, status_color, _ = st.session_state.mqtt_manager.determine_status(temp)
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {status_color};">
            <h3>🏷️ ROOM STATUS</h3>
            <div class="data-value" style="color: {status_color}; font-size: 2rem;">{status}</div>
            <p style="color: #6b7280; margin: 0;">Based on temperature</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        data_count = st.session_state.iot_system["data_count"]
        stats_color = "#8b5cf6"
        
        st.markdown(f"""
        <div class="status-card" style="border-left-color: {stats_color};">
            <h3>📊 STATISTICS</h3>
            <div class="data-value" style="color: {stats_color};">{data_count}</div>
            <p style="color: #6b7280; margin: 0;">Data points received</p>
            <p style="color: #9ca3af; font-size: 0.9rem; margin: 0;">Last: {st.session_state.iot_system['timestamp']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Charts and Visualizations
    if len(st.session_state.iot_system["history"]) > 1:
        st.subheader("📈 Real-time Charts")
        
        history_df = pd.DataFrame(st.session_state.iot_system["history"])
        
        tab1, tab2 = st.tabs(["Temperature Trend", "Humidity Trend"])
        
        with tab1:
            fig_temp = go.Figure()
            fig_temp.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#ef4444', width=3),
                marker=dict(size=8)
            ))
            
            # Add threshold lines
            fig_temp.add_hline(y=25, line_dash="dash", line_color="#3b82f6", 
                             annotation_text="DINGIN (<25°C)")
            fig_temp.add_hline(y=28, line_dash="dash", line_color="#ef4444",
                             annotation_text="PANAS (>28°C)")
            
            fig_temp.update_layout(
                title="Temperature Trend",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['humidity'],
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8)
            ))
            
            fig_hum.update_layout(
                title="Humidity Trend",
                xaxis_title="Time",
                yaxis_title="Humidity (%)",
                height=400
            )
            
            st.plotly_chart(fig_hum, use_container_width=True)
    else:
        st.info("📊 Waiting for data... Connect to ESP32 to see charts")
    
    st.markdown("---")
    
    # Row 3: Recent Data and Logs
    col_data, col_logs = st.columns([1, 1])
    
    with col_data:
        st.subheader("📋 Recent Data")
        
        if st.session_state.iot_system["csv_data"]:
            df = pd.DataFrame(st.session_state.iot_system["csv_data"][-10:])
            st.dataframe(df, use_container_width=True, height=300)
            
            # Download button
            if st.button("📥 Download CSV", use_container_width=True):
                csv_file = "iot_data_real.csv"
                if os.path.exists(csv_file):
                    with open(csv_file, 'r') as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="Download Full Dataset",
                        data=csv_data,
                        file_name=f"iot_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        else:
            st.info("No data collected yet")
    
    with col_logs:
        st.subheader("📝 System Logs")
        
        log_container = st.container(height=300)
        
        with log_container:
            for log in st.session_state.iot_system["system_logs"][:20]:
                st.markdown(f"""
                <div style="color: {log['color']}; font-family: 'Courier New', monospace; 
                         font-size: 0.85rem; padding: 0.2rem 0;">
                    {log['icon']} [{log['timestamp']}] {log['message']}
                </div>
                """, unsafe_allow_html=True)
    
    # Auto-refresh logic
    if st.session_state.iot_system["auto_refresh"] and st.session_state.iot_system["mqtt_connected"]:
        time.sleep(5)
        try:
            st.rerun()
        except:
            pass

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    # Try to auto-connect on startup
    if not st.session_state.iot_system["mqtt_connected"]:
        # Don't auto-connect immediately, let user initiate
        pass
    
    try:
        main()
    except Exception as e:
        st.error(f"Dashboard error: {str(e)}")
        st.info("Please refresh the page")

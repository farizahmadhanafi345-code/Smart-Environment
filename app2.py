# dashboard_realtime_esp32.py
import streamlit as st
import paho.mqtt.client as mqtt
import json
import time
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import ssl
import threading
import csv
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI REAL HIVEMQ ANDA ====================
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

# File CSV untuk menyimpan data REAL
CSV_FILE = "iot_data_real.csv"

# ==================== INISIALISASI STATE REAL ====================
if 'real_system' not in st.session_state:
    st.session_state.real_system = {
        # Data Sensor REAL dari ESP32
        "temperature": 0.0,
        "humidity": 0.0,
        "timestamp": "",
        "status": "WAITING",
        "status_code": -1,
        
        # Status Koneksi
        "mqtt_connected": False,
        "connection_attempts": 0,
        "last_mqtt_error": "",
        
        # Data History REAL
        "history": [],
        "csv_data": [],
        
        # Kontrol LED REAL
        "led_status": "off",
        "led_mode": "manual",
        
        # System Logs REAL
        "system_logs": [],
        
        # Statistics
        "data_points_received": 0,
        "start_time": datetime.now(),
        "last_csv_save": None
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# ==================== FUNGSI UTILITAS REAL ====================
def add_real_log(message, type="info"):
    """Tambahkan log REAL dengan timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "mqtt": "📡",
        "sensor": "🌡️",
        "led": "💡",
        "csv": "💾"
    }
    
    icon = icons.get(type, "📝")
    log_entry = f"[{timestamp}] {icon} {message}"
    
    st.session_state.real_system["system_logs"].insert(0, log_entry)
    
    # Batasi jumlah log
    if len(st.session_state.real_system["system_logs"]) > 50:
        st.session_state.real_system["system_logs"] = st.session_state.real_system["system_logs"][:50]
    
    # Print ke console untuk debugging
    print(f"REAL LOG: {log_entry}")

def determine_status(temperature):
    """Tentukan status REAL berdasarkan suhu (sesuai ESP32 Anda)"""
    if temperature < 22:
        return "DINGIN 🥶", "#3498db", 0
    elif temperature > 25:
        return "PANAS 🔥", "#e74c3c", 2
    else:
        return "NORMAL ✅", "#2ecc71", 1

def save_to_csv_real(temperature, humidity, status):
    """Simpan data REAL ke CSV"""
    try:
        file_exists = os.path.exists(CSV_FILE)
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            
            # Write header jika file baru
            if not file_exists:
                writer.writerow([
                    'timestamp', 'temperature', 'humidity', 
                    'status', 'status_code', 'source'
                ])
                add_real_log("Created new CSV file", "csv")
            
            # Write data
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            status_text, _, status_code = determine_status(temperature)
            
            writer.writerow([
                timestamp,
                round(temperature, 2),
                round(humidity, 2),
                status_text,
                status_code,
                'ESP32_REAL'
            ])
        
        # Update last save time
        st.session_state.real_system["last_csv_save"] = datetime.now()
        
        # Tambah ke csv_data untuk display
        st.session_state.real_system["csv_data"].append({
            'timestamp': timestamp,
            'temperature': temperature,
            'humidity': humidity,
            'status': status_text
        })
        
        # Batasi data di memory
        if len(st.session_state.real_system["csv_data"]) > 100:
            st.session_state.real_system["csv_data"] = st.session_state.real_system["csv_data"][-100:]
        
        add_real_log(f"Saved to CSV: {temperature}°C, {humidity}%", "csv")
        return True
        
    except Exception as e:
        add_real_log(f"CSV save error: {str(e)}", "error")
        return False

def load_csv_real():
    """Load data REAL dari CSV"""
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE, delimiter=';')
            add_real_log(f"Loaded {len(df)} records from CSV", "success")
            return df
        else:
            add_real_log("No CSV file found", "warning")
            return pd.DataFrame()
    except Exception as e:
        add_real_log(f"Error loading CSV: {str(e)}", "error")
        return pd.DataFrame()

def download_csv_real():
    """Download data REAL sebagai CSV"""
    try:
        if os.path.exists(CSV_FILE):
            with open(CSV_FILE, 'r', encoding='utf-8') as f:
                csv_content = f.read()
            return csv_content
        return ""
    except:
        return ""

# ==================== MQTT FUNCTIONS REAL ====================
def on_mqtt_connect_real(client, userdata, flags, rc):
    """Callback ketika BERHASIL terhubung ke HiveMQ REAL"""
    print(f"REAL: MQTT Connect callback, rc={rc}")
    
    if rc == 0:
        st.session_state.real_system["mqtt_connected"] = True
        st.session_state.real_system["connection_attempts"] = 0
        st.session_state.real_system["last_mqtt_error"] = ""
        
        # Subscribe ke topic sensor REAL
        client.subscribe(DHT_TOPIC)
        add_real_log(f"✅ CONNECTED to HiveMQ Cloud", "success")
        add_real_log(f"Subscribed to: {DHT_TOPIC}", "mqtt")
        
    else:
        st.session_state.real_system["mqtt_connected"] = False
        error_msgs = {
            1: "Unacceptable protocol version",
            2: "Identifier rejected", 
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        error_msg = error_msgs.get(rc, f"Unknown error {rc}")
        st.session_state.real_system["last_mqtt_error"] = error_msg
        add_real_log(f"❌ Connection failed: {error_msg}", "error")

def on_mqtt_disconnect_real(client, userdata, rc):
    """Callback ketika terputus dari MQTT"""
    print(f"REAL: MQTT Disconnect callback, rc={rc}")
    st.session_state.real_system["mqtt_connected"] = False
    add_real_log("⚠️ Disconnected from HiveMQ", "warning")
    
    # Attempt reconnect
    if rc != 0:
        threading.Thread(target=reconnect_mqtt_real, daemon=True).start()

def on_mqtt_message_real(client, userdata, msg):
    """Callback ketika menerima data REAL dari ESP32"""
    try:
        # Parse JSON dari ESP32 REAL
        payload = msg.payload.decode('utf-8')
        print(f"REAL: Received MQTT message: {payload}")
        
        data = json.loads(payload)
        
        # Ekstrak data sensor REAL
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        # Validasi data
        if not (0 <= temperature <= 50) or not (0 <= humidity <= 100):
            add_real_log(f"Invalid sensor data: {temperature}°C, {humidity}%", "warning")
            return
        
        # Update system state
        current_time = datetime.now()
        
        # Tentukan status
        status_text, status_color, status_code = determine_status(temperature)
        
        # Update data real-time
        st.session_state.real_system.update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": current_time.strftime("%H:%M:%S.%f")[:-3],
            "status": status_text,
            "status_code": status_code,
            "data_points_received": st.session_state.real_system["data_points_received"] + 1
        })
        
        # Tambah ke history untuk chart
        history_entry = {
            "timestamp": current_time,
            "temperature": temperature,
            "humidity": humidity,
            "status": status_text,
            "color": status_color
        }
        st.session_state.real_system["history"].append(history_entry)
        
        # Batasi history
        if len(st.session_state.real_system["history"]) > 100:
            st.session_state.real_system["history"] = st.session_state.real_system["history"][-100:]
        
        # Simpan ke CSV
        save_to_csv_real(temperature, humidity, status_text)
        
        # Log data received
        add_real_log(f"Sensor: {temperature:.1f}°C, {humidity:.1f}% → {status_text}", "sensor")
        
        # Trigger refresh UI
        if hasattr(st, 'rerun'):
            # Don't rerun too frequently
            last_rerun = st.session_state.real_system.get("last_rerun")
            if last_rerun is None or (current_time - last_rerun).seconds >= 2:
                st.session_state.real_system["last_rerun"] = current_time
                try:
                    st.rerun()
                except:
                    pass
        
    except json.JSONDecodeError as e:
        add_real_log(f"Invalid JSON from ESP32: {e}", "error")
        print(f"REAL: JSON Error: {e}")
    except Exception as e:
        add_real_log(f"Error processing message: {e}", "error")
        print(f"REAL: Processing Error: {e}")

def reconnect_mqtt_real():
    """Reconnect ke MQTT dengan exponential backoff"""
    if st.session_state.mqtt_client:
        attempts = st.session_state.real_system["connection_attempts"]
        backoff = min(2 ** attempts, 30)  # Max 30 seconds
        
        add_real_log(f"Reconnection attempt {attempts + 1} in {backoff}s...", "warning")
        time.sleep(backoff)
        st.session_state.real_system["connection_attempts"] += 1
        
        try:
            st.session_state.mqtt_client.reconnect()
        except Exception as e:
            add_real_log(f"Reconnect failed: {e}", "error")

def connect_to_hivemq_real():
    """Connect ke HiveMQ Cloud REAL"""
    try:
        if st.session_state.mqtt_client and st.session_state.real_system["mqtt_connected"]:
            add_real_log("Already connected to HiveMQ", "info")
            return True
        
        add_real_log("Connecting to HiveMQ Cloud...", "mqtt")
        print(f"REAL: Attempting connection to {MQTT_CONFIG['broker']}:{MQTT_CONFIG['port']}")
        
        # Create MQTT client
        client_id = f"dashboard_real_{int(time.time())}"
        client = mqtt.Client(client_id=client_id)
        
        # Set credentials REAL
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        # Setup SSL/TLS
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
            client.tls_insecure_set(True)  # Allow self-signed for debugging
        
        # Set callbacks REAL
        client.on_connect = on_mqtt_connect_real
        client.on_disconnect = on_mqtt_disconnect_real
        client.on_message = on_mqtt_message_real
        
        # Set will message
        client.will_set("dashboard/status", json.dumps({
            "device": "dashboard_real",
            "status": "disconnected",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Connect dengan timeout
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=20)
        
        # Start loop
        client.loop_start()
        
        # Tunggu koneksi
        for _ in range(15):  # Wait up to 7.5 seconds
            if st.session_state.real_system["mqtt_connected"]:
                break
            time.sleep(0.5)
        
        st.session_state.mqtt_client = client
        
        if st.session_state.real_system["mqtt_connected"]:
            add_real_log("✅ Successfully connected to ESP32 via HiveMQ", "success")
            return True
        else:
            error_msg = st.session_state.real_system.get("last_mqtt_error", "Unknown error")
            add_real_log(f"❌ Failed to connect: {error_msg}", "error")
            return False
        
    except Exception as e:
        error_msg = str(e)
        add_real_log(f"❌ Connection exception: {error_msg}", "error")
        print(f"REAL: Connection Exception: {error_msg}")
        return False

def send_led_command_real(command):
    """Kirim perintah ke LED ESP32 REAL"""
    if not st.session_state.mqtt_client or not st.session_state.real_system["mqtt_connected"]:
        add_real_log("Cannot send LED command: MQTT not connected", "error")
        return False
    
    try:
        # Kirim perintah ke ESP32
        st.session_state.mqtt_client.publish(LED_TOPIC, command)
        
        # Update LED status
        st.session_state.real_system["led_status"] = command
        
        add_real_log(f"LED command sent: {command}", "led")
        return True
        
    except Exception as e:
        add_real_log(f"Failed to send LED command: {e}", "error")
        return False

# ==================== STREAMLIT UI REAL ====================
def main():
    st.set_page_config(
        page_title="ESP32 REAL-TIME Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS untuk dashboard REAL
    st.markdown("""
    <style>
    .real-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    
    .real-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    .data-value {
        font-size: 2.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .status-connected {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-disconnected {
        background: linear-gradient(135deg, #EF4444, #DC2626);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .log-container-real {
        background: #1a202c;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.85rem;
        border: 1px solid #4a5568;
    }
    
    .btn-led-real {
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
    
    .btn-led-real:hover {
        transform: scale(1.05);
        opacity: 0.9;
    }
    
    .btn-red { background: #EF4444; color: white; }
    .btn-green { background: #10B981; color: white; }
    .btn-yellow { background: #F59E0B; color: white; }
    .btn-blue { background: #3B82F6; color: white; }
    .btn-gray { background: #6B7280; color: white; }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header REAL
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    ">
        <h1 style="margin: 0; font-size: 2.8rem;">🌡️ ESP32 REAL-TIME DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Live Data from Physical ESP32 DHT11 via HiveMQ Cloud</p>
        <div style="margin-top: 1rem;">
            <span class="{'status-connected' if st.session_state.real_system['mqtt_connected'] else 'status-disconnected'}">
                {'🟢 REAL DATA CONNECTED' if st.session_state.real_system['mqtt_connected'] else '🔴 DISCONNECTED'}
            </span>
            <span style="margin: 0 1rem;">•</span>
            <span>Broker: {MQTT_CONFIG['broker']}</span>
            <span style="margin: 0 1rem;">•</span>
            <span>Topic: {DHT_TOPIC}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Kontrol REAL
    with st.sidebar:
        st.markdown("### 🔗 MQTT CONNECTION")
        
        # Status koneksi detail
        col_conn1, col_conn2 = st.columns([1, 2])
        with col_conn1:
            if st.session_state.real_system["mqtt_connected"]:
                st.markdown('<div class="status-connected">🟢 ONLINE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-disconnected">🔴 OFFLINE</div>', unsafe_allow_html=True)
        
        with col_conn2:
            uptime = datetime.now() - st.session_state.real_system["start_time"]
            hours = uptime.seconds // 3600
            minutes = (uptime.seconds % 3600) // 60
            st.caption(f"Uptime: {hours:02d}:{minutes:02d}")
            st.caption(f"Data Points: {st.session_state.real_system['data_points_received']}")
        
        # Tombol koneksi REAL
        if st.button("🔗 CONNECT TO ESP32", type="primary", use_container_width=True, 
                    disabled=st.session_state.real_system["mqtt_connected"]):
            with st.spinner("Connecting to ESP32 via HiveMQ..."):
                if connect_to_hivemq_real():
                    st.success("✅ Connected to ESP32!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Connection failed")
                    if st.session_state.real_system["last_mqtt_error"]:
                        st.error(f"Error: {st.session_state.real_system['last_mqtt_error']}")
        
        if st.button("🔌 DISCONNECT", use_container_width=True, 
                    disabled=not st.session_state.real_system["mqtt_connected"]):
            if st.session_state.mqtt_client:
                st.session_state.mqtt_client.loop_stop()
                st.session_state.mqtt_client.disconnect()
                st.session_state.real_system["mqtt_connected"] = False
                st.warning("Disconnected from ESP32")
                st.rerun()
        
        st.markdown("---")
        
        # Kontrol LED REAL
        st.markdown("### 💡 LED CONTROL (REAL)")
        
        col_led1, col_led2 = st.columns(2)
        with col_led1:
            if st.button("🔴 RED", use_container_width=True, 
                        disabled=not st.session_state.real_system["mqtt_connected"]):
                send_led_command_real("merah")
                st.success("Sent: RED")
            
            if st.button("🟢 GREEN", use_container_width=True,
                        disabled=not st.session_state.real_system["mqtt_connected"]):
                send_led_command_real("hijau")
                st.success("Sent: GREEN")
        
        with col_led2:
            if st.button("🟡 YELLOW", use_container_width=True,
                        disabled=not st.session_state.real_system["mqtt_connected"]):
                send_led_command_real("kuning")
                st.success("Sent: YELLOW")
            
            if st.button("⚫ OFF", use_container_width=True,
                        disabled=not st.session_state.real_system["mqtt_connected"]):
                send_led_command_real("off")
                st.success("Sent: OFF")
        
        st.markdown(f"**Current LED:** {st.session_state.real_system['led_status'].upper()}")
        
        st.markdown("---")
        
        # CSV Operations REAL
        st.markdown("### 💾 DATA MANAGEMENT")
        
        if st.button("📂 LOAD CSV DATA", use_container_width=True):
            df = load_csv_real()
            if not df.empty:
                st.success(f"Loaded {len(df)} records")
            else:
                st.warning("No CSV data found")
        
        if st.button("🗑️ CLEAR HISTORY", use_container_width=True):
            st.session_state.real_system["history"] = []
            st.session_state.real_system["csv_data"] = []
            st.success("History cleared")
            st.rerun()
        
        st.markdown("---")
        
        # System Info REAL
        st.markdown("### 📊 SYSTEM INFO")
        
        info_items = [
            ("Data Points", st.session_state.real_system["data_points_received"]),
            ("Last Update", st.session_state.real_system["timestamp"] or "Never"),
            ("CSV Records", len(st.session_state.real_system["csv_data"])),
            ("History Points", len(st.session_state.real_system["history"]))
        ]
        
        for label, value in info_items:
            st.write(f"**{label}:** {value}")
        
        # Auto-refresh
        auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", value=True)
    
    # ============ MAIN DASHBOARD - DATA REAL ============
    
    # Row 1: Live Sensor Data REAL
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.session_state.real_system["temperature"]
        _, temp_color, _ = determine_status(temp)
        
        st.markdown(f"""
        <div class="real-card" style="border-left-color: {temp_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                🌡️ REAL TEMPERATURE
            </div>
            <div class="data-value pulse-animation" style="color: {temp_color};">
                {temp:.1f} °C
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                From ESP32 DHT11 • {st.session_state.real_system['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.real_system["humidity"]
        hum_color = "#3B82F6"
        
        st.markdown(f"""
        <div class="real-card" style="border-left-color: {hum_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                💧 REAL HUMIDITY
            </div>
            <div class="data-value pulse-animation" style="color: {hum_color};">
                {hum:.1f} %
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                From ESP32 DHT11 • {st.session_state.real_system['timestamp']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = st.session_state.real_system["status"]
        _, status_color, _ = determine_status(st.session_state.real_system["temperature"])
        
        st.markdown(f"""
        <div class="real-card" style="border-left-color: {status_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                🏷️ ROOM STATUS
            </div>
            <div class="data-value" style="color: {status_color};">
                {status}
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                Based on real temperature • Auto-updating
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Real-time Charts (HANYA jika ada data REAL)
    if len(st.session_state.real_system["history"]) > 0:
        st.subheader("📈 REAL-TIME CHARTS FROM ESP32")
        
        # Prepare data untuk plotting
        history_df = pd.DataFrame(st.session_state.real_system["history"])
        
        tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Humidity Trend", "Live Scatter"])
        
        with tab1:
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=history_df['timestamp'],
                y=history_df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8, color='#e74c3c'),
                hovertemplate='<b>%{x}</b><br>Temperature: %{y:.1f}°C<extra></extra>'
            ))
            
            # Add threshold lines
            fig_temp.add_hline(y=22, line_dash="dash", line_color="#3498db", 
                             annotation_text="DINGIN (<22°C)")
            fig_temp.add_hline(y=25, line_dash="dash", line_color="#e74c3c",
                             annotation_text="PANAS (>25°C)")
            
            fig_temp.update_layout(
                title="Temperature Trend - REAL DATA from ESP32",
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
                marker=dict(size=8, color='#3498db'),
                hovertemplate='<b>%{x}</b><br>Humidity: %{y:.1f}%<extra></extra>'
            ))
            
            fig_hum.update_layout(
                title="Humidity Trend - REAL DATA from ESP32",
                xaxis_title="Time",
                yaxis_title="Humidity (%)",
                height=400,
                hovermode="x unified"
            )
            
            st.plotly_chart(fig_hum, use_container_width=True)
        
        with tab3:
            fig_scatter = px.scatter(
                history_df,
                x='temperature',
                y='humidity',
                color='status',
                color_discrete_map={
                    'DINGIN 🥶': '#3498db',
                    'NORMAL ✅': '#2ecc71',
                    'PANAS 🔥': '#e74c3c'
                },
                title="Temperature vs Humidity - REAL DATA",
                hover_data=['timestamp']
            )
            
            fig_scatter.update_traces(marker=dict(size=12))
            fig_scatter.update_layout(height=400)
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        if st.session_state.real_system["mqtt_connected"]:
            st.info("⏳ Waiting for data from ESP32... Make sure ESP32 is publishing to correct topic.")
        else:
            st.warning("🔌 Not connected to ESP32. Click 'CONNECT TO ESP32' to start receiving real data.")
    
    st.markdown("---")
    
    # Row 3: CSV Data & System Logs
    col_data, col_logs = st.columns([2, 1])
    
    with col_data:
        st.subheader("📋 REAL DATA HISTORY")
        
        if len(st.session_state.real_system["csv_data"]) > 0:
            # Buat dataframe dari csv_data
            csv_df = pd.DataFrame(st.session_state.real_system["csv_data"])
            
            # Tampilkan tabel
            st.dataframe(
                csv_df.tail(10)[::-1],  # Show last 10, newest first
                use_container_width=True,
                height=350
            )
            
            # Download button
            csv_content = download_csv_real()
            if csv_content:
                st.download_button(
                    label="📥 DOWNLOAD ALL DATA",
                    data=csv_content,
                    file_name=f"esp32_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("No real data collected yet. Connect to ESP32 to start collecting.")
    
    with col_logs:
        st.subheader("📝 REAL-TIME SYSTEM LOGS")
        
        st.markdown('<div class="log-container-real">', unsafe_allow_html=True)
        
        # Display logs (newest first)
        for log in st.session_state.real_system["system_logs"][:15]:
            # Color coding
            if "✅" in log or "CONNECTED" in log:
                st.markdown(f'<span style="color: #2ecc71;">{log}</span>', unsafe_allow_html=True)
            elif "❌" in log or "failed" in log.lower() or "error" in log.lower():
                st.markdown(f'<span style="color: #e74c3c;">{log}</span>', unsafe_allow_html=True)
            elif "⚠️" in log or "Warning" in log:
                st.markdown(f'<span style="color: #f39c12;">{log}</span>', unsafe_allow_html=True)
            elif "📡" in log:
                st.markdown(f'<span style="color: #9b59b6;">{log}</span>', unsafe_allow_html=True)
            elif "🌡️" in log:
                st.markdown(f'<span style="color: #e74c3c;">{log}</span>', unsafe_allow_html=True)
            elif "💡" in log:
                st.markdown(f'<span style="color: #f39c12;">{log}</span>', unsafe_allow_html=True)
            elif "💾" in log:
                st.markdown(f'<span style="color: #3498db;">{log}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color: #ecf0f1;">{log}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🗑️ CLEAR LOGS", use_container_width=True):
            st.session_state.real_system["system_logs"] = []
            st.rerun()
    
    st.markdown("---")
    
    # Row 4: Connection Status Detail
    st.subheader("🔍 CONNECTION DETAILS")
    
    col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
    
    with col_detail1:
        st.metric("MQTT Status", 
                 "CONNECTED" if st.session_state.real_system["mqtt_connected"] else "DISCONNECTED",
                 "🟢" if st.session_state.real_system["mqtt_connected"] else "🔴")
    
    with col_detail2:
        st.metric("Data Received", 
                 st.session_state.real_system["data_points_received"],
                 "points")
    
    with col_detail3:
        if st.session_state.real_system["last_csv_save"]:
            time_since = datetime.now() - st.session_state.real_system["last_csv_save"]
            st.metric("Last Save", 
                     f"{time_since.seconds}s ago",
                     "CSV")
        else:
            st.metric("Last Save", "Never", "CSV")
    
    with col_detail4:
        history_count = len(st.session_state.real_system["history"])
        st.metric("Live History", 
                 history_count,
                 "points in memory")
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem;">
        <p><strong>🌐 ESP32 REAL-TIME MONITORING SYSTEM</strong></p>
        <p>Connected to <strong>Physical ESP32 with DHT11</strong> via <strong>HiveMQ Cloud</strong></p>
        <p>Broker: <code>{MQTT_CONFIG['broker']}</code> | Topic: <code>{DHT_TOPIC}</code></p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">© 2024 IoT Real-Time System • 100% Real Hardware Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh jika diaktifkan
    if auto_refresh and st.session_state.real_system["mqtt_connected"]:
        time.sleep(3)
        try:
            st.rerun()
        except:
            pass

# ==================== RUN REAL DASHBOARD ====================
if __name__ == "__main__":
    # Initial connection attempt
    if not st.session_state.real_system["mqtt_connected"] and MQTT_CONFIG:
        # Try to connect in background
        threading.Thread(target=connect_to_hivemq_real, daemon=True).start()
    
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Dashboard Error: {str(e)}")
        st.info("Please check your internet connection and ESP32 status.")


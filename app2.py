# dashboard_real_iot.py
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
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI REAL HIVEMQ ANDA ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1760514170127",
    "password": "0r8ULyh9&duT1,BHg%.M",
    "use_ssl": True,
    "keepalive": 20
}

# Topics REAL dari ESP32 Anda
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# ==================== INISIALISASI STATE GLOBAL ====================
if 'iot_system' not in st.session_state:
    st.session_state.iot_system = {
        # Sensor Data
        "temperature": 0.0,
        "humidity": 0.0,
        "timestamp": "",
        "label": "WAITING",
        "label_encoded": -1,
        "last_update": None,
        
        # MQTT Status
        "mqtt_connected": False,
        "data_received": False,
        "connection_attempts": 0,
        
        # LED Control
        "led_status": "off",
        "led_mode": "auto",  # auto, manual, ai
        
        # Data History
        "data_points": 0,
        "uptime": datetime.now(),
        
        # System Logs
        "system_logs": []
    }

if 'data_history' not in st.session_state:
    st.session_state.data_history = {
        "timestamps": [],
        "temperatures": [],
        "humidities": [],
        "labels": [],
        "led_commands": []
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# ==================== FUNGSI UTILITAS ====================
def add_system_log(message, type="info"):
    """Tambahkan log ke sistem dengan timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "mqtt": "📡",
        "sensor": "🌡️",
        "led": "💡"
    }
    
    icon = icons.get(type, "📝")
    log_entry = f"[{timestamp}] {icon} {message}"
    
    st.session_state.iot_system["system_logs"].insert(0, log_entry)
    
    # Batasi jumlah log
    if len(st.session_state.iot_system["system_logs"]) > 50:
        st.session_state.iot_system["system_logs"] = st.session_state.iot_system["system_logs"][:50]
    
    # Print ke console untuk debugging
    print(log_entry)

def determine_label(temperature):
    """Tentukan label berdasarkan suhu REAL (sesuai ESP32 Anda)"""
    # SESUAI DENGAN KODE ESP32 ANDA: <22°C = DINGIN, 22-25°C = NORMAL, >25°C = PANAS
    if temperature < 22:
        return "DINGIN 🥶", 0
    elif temperature > 25:
        return "PANAS 🔥", 2
    else:
        return "NORMAL ✅", 1

def calculate_uptime():
    """Hitung waktu system telah berjalan"""
    if st.session_state.iot_system["uptime"]:
        uptime = datetime.now() - st.session_state.iot_system["uptime"]
        hours = uptime.seconds // 3600
        minutes = (uptime.seconds % 3600) // 60
        seconds = uptime.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return "00:00:00"

# ==================== MQTT FUNCTIONS (REAL CONNECTION) ====================
def on_mqtt_connect(client, userdata, flags, rc):
    """Callback ketika BERHASIL terhubung ke HiveMQ REAL"""
    if rc == 0:
        st.session_state.iot_system["mqtt_connected"] = True
        st.session_state.iot_system["connection_attempts"] = 0
        
        # Subscribe ke topic sensor REAL
        client.subscribe(DHT_TOPIC)
        add_system_log(f"Connected to HiveMQ Cloud", "success")
        add_system_log(f"Subscribed to: {DHT_TOPIC}", "mqtt")
        
        # Publish connection status
        connect_msg = {
            "device": "dashboard",
            "status": "connected",
            "timestamp": datetime.now().isoformat()
        }
        client.publish("dashboard/status", json.dumps(connect_msg))
        
    else:
        st.session_state.iot_system["mqtt_connected"] = False
        error_msgs = {
            1: "Unacceptable protocol version",
            2: "Identifier rejected",
            3: "Server unavailable",
            4: "Bad username or password",
            5: "Not authorized"
        }
        error_msg = error_msgs.get(rc, f"Unknown error {rc}")
        add_system_log(f"Connection failed: {error_msg}", "error")

def on_mqtt_disconnect(client, userdata, rc):
    """Callback ketika terputus dari MQTT"""
    st.session_state.iot_system["mqtt_connected"] = False
    add_system_log("Disconnected from HiveMQ", "warning")
    
    # Attempt reconnect in background thread
    if rc != 0:
        threading.Thread(target=reconnect_mqtt, daemon=True).start()

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima data REAL dari ESP32"""
    try:
        # Parse JSON dari ESP32 REAL
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        # Ekstrak data sensor REAL
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        sensor_type = data.get('sensor', 'DHT11')
        
        # Validasi data
        if not (0 <= temperature <= 50) or not (0 <= humidity <= 100):
            add_system_log(f"Invalid sensor data: {temperature}°C, {humidity}%", "warning")
            return
        
        # Update system state
        current_time = datetime.now()
        
        # Tentukan label
        label, label_encoded = determine_label(temperature)
        
        # Update sensor data
        st.session_state.iot_system.update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": current_time.strftime("%H:%M:%S"),
            "last_update": current_time,
            "label": label,
            "label_encoded": label_encoded,
            "data_received": True,
            "data_points": st.session_state.iot_system["data_points"] + 1
        })
        
        # Tambah ke history
        st.session_state.data_history["timestamps"].append(current_time)
        st.session_state.data_history["temperatures"].append(temperature)
        st.session_state.data_history["humidities"].append(humidity)
        st.session_state.data_history["labels"].append(label)
        
        # Batasi history size
        max_history = 100
        for key in st.session_state.data_history:
            if len(st.session_state.data_history[key]) > max_history:
                st.session_state.data_history[key] = st.session_state.data_history[key][-max_history:]
        
        # Auto LED control jika mode auto
        if st.session_state.iot_system["led_mode"] == "auto":
            if temperature < 22:
                send_led_command("hijau")  # Dingin → Hijau
            elif temperature > 25:
                send_led_command("merah")  # Panas → Merah
            else:
                send_led_command("kuning")  # Normal → Kuning
        
        # Log data received
        add_system_log(f"Sensor: {temperature:.1f}°C, {humidity:.1f}% → {label}", "sensor")
        
    except json.JSONDecodeError as e:
        add_system_log(f"Invalid JSON from ESP32: {e}", "error")
    except Exception as e:
        add_system_log(f"Error processing message: {e}", "error")

def reconnect_mqtt():
    """Reconnect ke MQTT dengan exponential backoff"""
    if st.session_state.mqtt_client:
        attempts = st.session_state.iot_system["connection_attempts"]
        backoff = min(2 ** attempts, 30)  # Max 30 seconds
        
        time.sleep(backoff)
        st.session_state.iot_system["connection_attempts"] += 1
        
        try:
            add_system_log(f"Reconnection attempt {attempts + 1}...", "warning")
            st.session_state.mqtt_client.reconnect()
        except:
            pass

def connect_to_hivemq():
    """Connect ke HiveMQ Cloud REAL"""
    try:
        if st.session_state.mqtt_client and st.session_state.iot_system["mqtt_connected"]:
            return True
        
        add_system_log("Connecting to HiveMQ Cloud...", "mqtt")
        
        # Create MQTT client
        client_id = f"dashboard_{int(time.time())}"
        client = mqtt.Client(client_id=client_id)
        
        # Set credentials
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        # Setup SSL/TLS
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
        
        # Set callbacks
        client.on_connect = on_mqtt_connect
        client.on_disconnect = on_mqtt_disconnect
        client.on_message = on_mqtt_message
        
        # Set will message
        client.will_set("dashboard/status", json.dumps({
            "device": "dashboard",
            "status": "disconnected",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Connect dengan timeout
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=20)
        
        # Start loop
        client.loop_start()
        
        # Tunggu koneksi
        for _ in range(10):  # Wait up to 5 seconds
            if st.session_state.iot_system["mqtt_connected"]:
                break
            time.sleep(0.5)
        
        st.session_state.mqtt_client = client
        return st.session_state.iot_system["mqtt_connected"]
        
    except Exception as e:
        add_system_log(f"Connection failed: {str(e)}", "error")
        return False

def send_led_command(command):
    """Kirim perintah ke LED ESP32 REAL"""
    if not st.session_state.mqtt_client or not st.session_state.iot_system["mqtt_connected"]:
        add_system_log("Cannot send LED command: MQTT not connected", "error")
        return False
    
    try:
        # Kirim perintah ke ESP32
        st.session_state.mqtt_client.publish(LED_TOPIC, command)
        
        # Update LED status
        st.session_state.iot_system["led_status"] = command
        
        # Simpan ke history
        st.session_state.data_history["led_commands"].append({
            "timestamp": datetime.now(),
            "command": command,
            "mode": st.session_state.iot_system["led_mode"]
        })
        
        add_system_log(f"LED command sent: {command}", "led")
        return True
        
    except Exception as e:
        add_system_log(f"Failed to send LED command: {e}", "error")
        return False

# ==================== STREAMLIT UI (REAL DASHBOARD) ====================
def main():
    st.set_page_config(
        page_title="IoT REAL-TIME Dashboard",
        page_icon="🌡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS untuk dashboard REAL
    st.markdown("""
    <style>
    .real-time-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        text-align: center;
    }
    
    .sensor-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        border-left: 6px solid;
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .sensor-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .sensor-value {
        font-size: 2.8rem;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .status-connected {
        background: #10B981;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-disconnected {
        background: #EF4444;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    
    .label-dingin { color: #3B82F6; }
    .label-normal { color: #10B981; }
    .label-panas { color: #EF4444; }
    
    .log-container {
        background: #1F2937;
        color: #E5E7EB;
        padding: 15px;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        max-height: 300px;
        overflow-y: auto;
        font-size: 0.85rem;
    }
    
    .data-point {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .btn-led {
        padding: 12px 20px;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-bottom: 10px;
    }
    
    .btn-led:hover {
        transform: scale(1.05);
    }
    
    .btn-red { background: #EF4444; color: white; }
    .btn-green { background: #10B981; color: white; }
    .btn-yellow { background: #F59E0B; color: white; }
    .btn-blue { background: #3B82F6; color: white; }
    .btn-gray { background: #6B7280; color: white; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header REAL
    st.markdown("""
    <div class="real-time-card">
        <h1 style="margin: 0; font-size: 2.5rem;">🌡️ IOT REAL-TIME DASHBOARD</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Live Monitoring from ESP32 DHT11 via HiveMQ Cloud</p>
        <div style="margin-top: 15px;">
            <span class="status-connected">REAL DATA</span>
            <span style="margin: 0 10px;">•</span>
            <span>ESP32 DHT11</span>
            <span style="margin: 0 10px;">•</span>
            <span>HiveMQ Cloud</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Kontrol REAL
    with st.sidebar:
        st.markdown("### 🔗 MQTT CONNECTION")
        
        # Status koneksi REAL
        col_status1, col_status2 = st.columns([1, 2])
        with col_status1:
            if st.session_state.iot_system["mqtt_connected"]:
                st.markdown('<div class="status-connected">🟢 ONLINE</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-disconnected">🔴 OFFLINE</div>', unsafe_allow_html=True)
        
        with col_status2:
            uptime = calculate_uptime()
            st.caption(f"Uptime: {uptime}")
        
        # Tombol koneksi REAL
        if st.button("🔗 CONNECT HIVEMQ", type="primary", use_container_width=True):
            with st.spinner("Connecting to HiveMQ Cloud..."):
                if connect_to_hivemq():
                    st.success("✅ Connected to ESP32!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Connection failed")
        
        if st.button("🔌 DISCONNECT", use_container_width=True):
            if st.session_state.mqtt_client:
                st.session_state.mqtt_client.loop_stop()
                st.session_state.iot_system["mqtt_connected"] = False
                st.warning("Disconnected from HiveMQ")
                st.rerun()
        
        st.markdown("---")
        
        # Kontrol LED REAL
        st.markdown("### 💡 LED CONTROL")
        
        # Mode kontrol
        mode = st.radio("Control Mode", ["Auto", "Manual"], 
                       index=1 if st.session_state.iot_system["led_mode"] == "manual" else 0)
        
        if mode == "Manual":
            st.session_state.iot_system["led_mode"] = "manual"
            
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
        else:
            st.session_state.iot_system["led_mode"] = "auto"
            st.info("LED control: AUTO (based on temperature)")
        
        st.markdown("---")
        
        # System Info REAL
        st.markdown("### 📊 SYSTEM INFO")
        
        info_items = [
            ("Data Points", st.session_state.iot_system["data_points"]),
            ("Current LED", st.session_state.iot_system["led_status"].upper()),
            ("Control Mode", st.session_state.iot_system["led_mode"].upper()),
            ("Last Update", st.session_state.iot_system["timestamp"] or "N/A")
        ]
        
        for label, value in info_items:
            st.write(f"**{label}:** {value}")
        
        st.markdown("---")
        
        # Refresh control
        auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", value=True)
        
        if st.button("🔄 MANUAL REFRESH", use_container_width=True):
            st.rerun()
    
    # ============ MAIN DASHBOARD - DATA REAL ============
    
    # Row 1: Sensor Data REAL
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp = st.session_state.iot_system["temperature"]
        temp_color = "#EF4444"  # Red for temperature
        
        st.markdown(f"""
        <div class="sensor-card" style="border-left-color: {temp_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                🌡️ REAL TEMPERATURE
            </div>
            <div class="sensor-value data-point" style="color: {temp_color};">
                {temp:.1f} °C
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                From ESP32 DHT11
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        hum = st.session_state.iot_system["humidity"]
        hum_color = "#3B82F6"  # Blue for humidity
        
        st.markdown(f"""
        <div class="sensor-card" style="border-left-color: {hum_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                💧 REAL HUMIDITY
            </div>
            <div class="sensor-value data-point" style="color: {hum_color};">
                {hum:.1f} %
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                From ESP32 DHT11
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        label = st.session_state.iot_system["label"]
        label_class = label.split()[0].lower()
        
        if "DINGIN" in label:
            label_color = "#3B82F6"
            icon = "🥶"
        elif "PANAS" in label:
            label_color = "#EF4444"
            icon = "🔥"
        else:
            label_color = "#10B981"
            icon = "✅"
        
        st.markdown(f"""
        <div class="sensor-card" style="border-left-color: {label_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                🏷️ ROOM STATUS
            </div>
            <div class="sensor-value" style="color: {label_color};">
                {icon} {label}
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                Based on temperature
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        # Data freshness indicator
        if st.session_state.iot_system["last_update"]:
            time_diff = datetime.now() - st.session_state.iot_system["last_update"]
            seconds = time_diff.total_seconds()
            
            if seconds < 5:
                status_color = "#10B981"
                status_text = "LIVE"
                status_icon = "🟢"
            elif seconds < 30:
                status_color = "#F59E0B"
                status_text = f"{int(seconds)}s"
                status_icon = "🟡"
            else:
                status_color = "#EF4444"
                status_text = f"{int(seconds)}s"
                status_icon = "🔴"
        else:
            status_color = "#6B7280"
            status_text = "WAITING"
            status_icon = "⚪"
        
        st.markdown(f"""
        <div class="sensor-card" style="border-left-color: {status_color};">
            <div style="font-size: 1.2rem; color: #6B7280; margin-bottom: 10px;">
                🕐 DATA STATUS
            </div>
            <div class="sensor-value" style="color: {status_color};">
                {status_icon} {status_text}
            </div>
            <div style="color: #6B7280; font-size: 0.9rem;">
                Last: {st.session_state.iot_system['timestamp'] or 'No data'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Real-time Charts (HANYA jika ada data REAL)
    if len(st.session_state.data_history["timestamps"]) > 1:
        st.markdown("### 📈 REAL-TIME SENSOR CHARTS")
        
        # Prepare data untuk plotting
        history_df = pd.DataFrame({
            'Time': [t.strftime("%H:%M:%S") for t in st.session_state.data_history["timestamps"]],
            'Temperature': st.session_state.data_history["temperatures"],
            'Humidity': st.session_state.data_history["humidities"],
            'Label': st.session_state.data_history["labels"]
        })
        
        # Buat tabs untuk berbagai chart
        tab1, tab2, tab3 = st.tabs(["Temperature Trend", "Humidity Trend", "Live Scatter"])
        
        with tab1:
            fig_temp = go.Figure()
            
            fig_temp.add_trace(go.Scatter(
                x=history_df['Time'],
                y=history_df['Temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#EF4444', width=3),
                marker=dict(size=8, color='#EF4444'),
                hovertemplate='<b>%{x}</b><br>Temperature: %{y:.1f}°C<extra></extra>'
            ))
            
            # Add threshold lines
            fig_temp.add_hline(y=22, line_dash="dash", line_color="#3B82F6", 
                             annotation_text="DINGIN (<22°C)")
            fig_temp.add_hline(y=25, line_dash="dash", line_color="#EF4444",
                             annotation_text="PANAS (>25°C)")
            
            fig_temp.update_layout(
                title="Temperature Trend - REAL DATA",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                height=400,
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig_temp, use_container_width=True)
        
        with tab2:
            fig_hum = go.Figure()
            
            fig_hum.add_trace(go.Scatter(
                x=history_df['Time'],
                y=history_df['Humidity'],
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#3B82F6', width=3),
                marker=dict(size=8, color='#3B82F6'),
                hovertemplate='<b>%{x}</b><br>Humidity: %{y:.1f}%<extra></extra>'
            ))
            
            fig_hum.update_layout(
                title="Humidity Trend - REAL DATA",
                xaxis_title="Time",
                yaxis_title="Humidity (%)",
                height=400,
                hovermode="x unified",
                showlegend=True
            )
            
            st.plotly_chart(fig_hum, use_container_width=True)
        
        with tab3:
            fig_scatter = px.scatter(
                history_df,
                x='Temperature',
                y='Humidity',
                color='Label',
                color_discrete_map={
                    'DINGIN 🥶': '#3B82F6',
                    'NORMAL ✅': '#10B981',
                    'PANAS 🔥': '#EF4444'
                },
                title="Temperature vs Humidity - REAL DATA",
                hover_data=['Time']
            )
            
            fig_scatter.update_traces(marker=dict(size=12))
            fig_scatter.update_layout(height=400)
            
            st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("⏳ Waiting for real-time data from ESP32... Connect to HiveMQ to see live charts.")
    
    st.markdown("---")
    
    # Row 3: Data Table & System Logs
    col_data, col_logs = st.columns([2, 1])
    
    with col_data:
        st.markdown("### 📋 RECENT DATA HISTORY")
        
        if len(st.session_state.data_history["timestamps"]) > 0:
            # Create dataframe dari history data
            recent_data = pd.DataFrame({
                'Timestamp': [t.strftime("%H:%M:%S") for t in st.session_state.data_history["timestamps"][-10:]],
                'Temperature (°C)': st.session_state.data_history["temperatures"][-10:],
                'Humidity (%)': st.session_state.data_history["humidities"][-10:],
                'Status': st.session_state.data_history["labels"][-10:]
            })
            
            # Style the dataframe
            def color_status(val):
                if 'DINGIN' in val:
                    color = '#3B82F6'
                elif 'PANAS' in val:
                    color = '#EF4444'
                else:
                    color = '#10B981'
                return f'color: {color}; font-weight: bold'
            
            styled_df = recent_data.style.applymap(color_status, subset=['Status'])
            
            # Display table
            st.dataframe(styled_df, use_container_width=True, height=350)
            
            # Download button untuk data REAL
            csv_data = recent_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Recent Data",
                data=csv_data,
                file_name=f"iot_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No data received yet")
    
    with col_logs:
        st.markdown("### 📝 SYSTEM LOGS")
        
        # Log container
        st.markdown('<div class="log-container">', unsafe_allow_html=True)
        
        # Display logs (newest first)
        for log in st.session_state.iot_system["system_logs"][:15]:
            # Color code berdasarkan tipe log
            if "✅" in log or "Connected" in log:
                st.markdown(f'<span style="color: #10B981;">{log}</span>', unsafe_allow_html=True)
            elif "❌" in log or "Error" in log or "failed" in log:
                st.markdown(f'<span style="color: #EF4444;">{log}</span>', unsafe_allow_html=True)
            elif "⚠️" in log or "Warning" in log:
                st.markdown(f'<span style="color: #F59E0B;">{log}</span>', unsafe_allow_html=True)
            elif "📡" in log:
                st.markdown(f'<span style="color: #8B5CF6;">{log}</span>', unsafe_allow_html=True)
            elif "🌡️" in log:
                st.markdown(f'<span style="color: #EF4444;">{log}</span>', unsafe_allow_html=True)
            elif "💡" in log:
                st.markdown(f'<span style="color: #F59E0B;">{log}</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span style="color: #E5E7EB;">{log}</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Clear logs button
        if st.button("🗑️ Clear Logs", use_container_width=True):
            st.session_state.iot_system["system_logs"] = []
            st.rerun()
    
    st.markdown("---")
    
    # Row 4: System Statistics
    st.markdown("### 📊 SYSTEM STATISTICS")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        data_rate = st.session_state.iot_system["data_points"] / max(1, (datetime.now() - st.session_state.iot_system["uptime"]).seconds)
        st.metric("📈 Data Rate", f"{data_rate:.2f}/s", "Samples per second")
    
    with col_stat2:
        if len(st.session_state.data_history["temperatures"]) > 0:
            avg_temp = np.mean(st.session_state.data_history["temperatures"])
            st.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°C", "Historical average")
        else:
            st.metric("🌡️ Avg Temp", "0.0°C", "No data")
    
    with col_stat3:
        if len(st.session_state.data_history["humidities"]) > 0:
            avg_hum = np.mean(st.session_state.data_history["humidities"])
            st.metric("💧 Avg Hum", f"{avg_hum:.1f}%", "Historical average")
        else:
            st.metric("💧 Avg Hum", "0.0%", "No data")
    
    with col_stat4:
        if len(st.session_state.data_history["labels"]) > 0:
            # Hitung distribusi label
            labels = st.session_state.data_history["labels"]
            dingin_count = sum(1 for l in labels if 'DINGIN' in l)
            normal_count = sum(1 for l in labels if 'NORMAL' in l)
            panas_count = sum(1 for l in labels if 'PANAS' in l)
            
            total = len(labels)
            if total > 0:
                dominant = max([('DINGIN', dingin_count), ('NORMAL', normal_count), ('PANAS', panas_count)], 
                             key=lambda x: x[1])
                st.metric("🏷️ Dominant", dominant[0], f"{dominant[1]}/{total}")
            else:
                st.metric("🏷️ Dominant", "N/A", "No data")
        else:
            st.metric("🏷️ Dominant", "N/A", "No data")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; padding: 20px;">
        <p style="font-size: 1.1rem;"><strong>🌐 IOT REAL-TIME MONITORING SYSTEM v2.0</strong></p>
        <p>Connected to <strong>ESP32 DHT11</strong> via <strong>HiveMQ Cloud</strong> | Data updates every 2 seconds</p>
        <p>Broker: <code>f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud</code> | Topic: <code>sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht</code></p>
        <p style="margin-top: 15px; font-size: 0.9rem;">© 2024 IoT Smart Monitoring | 100% Real Data from Physical Sensors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Auto-refresh jika diaktifkan
    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ==================== RUN REAL DASHBOARD ====================
if __name__ == "__main__":
    # Initial connection attempt
    if not st.session_state.iot_system["mqtt_connected"]:
        # Try to connect in background
        threading.Thread(target=connect_to_hivemq, daemon=True).start()
    
    try:
        main()
    except Exception as e:
        st.error(f"⚠️ Dashboard Error: {str(e)}")
        st.info("Please refresh the page or check your internet connection.")

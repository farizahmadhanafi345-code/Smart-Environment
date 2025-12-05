# app_streamlit.py
import streamlit as st
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import threading
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigurasi MQTT
MQTT_CONFIG = {
    "host": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1764923408610",
    "password": "9y&f74G1*pWSD.tQdXa@",
    "use_ssl": True
}

# Topics
PUB_TOPIC_DHT = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
SUB_TOPIC_LED = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# Session State untuk penyimpanan data
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        "temperature": 0,
        "humidity": 0,
        "timestamp": "",
        "status": "Menunggu data...",
        "led_mode": "auto",
        "led_status": "hijau",
        "history": [],  # Menyimpan riwayat data
        "mqtt_connected": False
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# Callback MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.sensor_data["mqtt_connected"] = True
        print("✅ Connected to MQTT Broker!")
        client.subscribe(PUB_TOPIC_DHT)
        print(f"📡 Subscribed to: {PUB_TOPIC_DHT}")
    else:
        st.session_state.sensor_data["mqtt_connected"] = False
        print(f"❌ Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    try:
        if msg.topic == PUB_TOPIC_DHT:
            data = json.loads(msg.payload.decode())
            
            # Update sensor data
            timestamp = datetime.now()
            st.session_state.sensor_data["temperature"] = data.get("temperature", 0)
            st.session_state.sensor_data["humidity"] = data.get("humidity", 0)
            st.session_state.sensor_data["timestamp"] = timestamp.strftime("%H:%M:%S")
            
            # Determinstatus based on temperature
            temp = st.session_state.sensor_data["temperature"]
            if temp < 25:
                status = "DINGIN"
            elif temp > 28:
                status = "PANAS"
            else:
                status = "NORMAL"
            st.session_state.sensor_data["status"] = status
            
            # Tambahkan ke riwayat (maksimal 50 data)
            history_entry = {
                "timestamp": timestamp,
                "temperature": temp,
                "humidity": st.session_state.sensor_data["humidity"],
                "status": status
            }
            st.session_state.sensor_data["history"].append(history_entry)
            
            # Batasi riwayat menjadi 50 entri terakhir
            if len(st.session_state.sensor_data["history"]) > 50:
                st.session_state.sensor_data["history"] = st.session_state.sensor_data["history"][-50:]
                
            print(f"📊 Data updated: {st.session_state.sensor_data['temperature']}°C, {st.session_state.sensor_data['humidity']}%")
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")

# Setup MQTT
def setup_mqtt():
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        if MQTT_CONFIG["use_ssl"]:
            mqtt_client.tls_set()
        
        mqtt_client.on_connect = on_connect
        mqtt_client.on_message = on_message
        
        mqtt_client.connect(MQTT_CONFIG["host"], MQTT_CONFIG["port"], 60)
        mqtt_client.loop_start()
        
        st.session_state.mqtt_client = mqtt_client
        return True
        
    except Exception as e:
        print(f"❌ MQTT setup error: {e}")
        st.session_state.sensor_data["mqtt_connected"] = False
        return False

def send_led_command(command):
    if st.session_state.mqtt_client and st.session_state.mqtt_client.is_connected():
        st.session_state.mqtt_client.publish(SUB_TOPIC_LED, command)
        print(f"💡 LED command sent: {command}")
        
        # Update LED status
        if command == "auto":
            st.session_state.sensor_data["led_mode"] = "auto"
        else:
            st.session_state.sensor_data["led_mode"] = "manual"
            st.session_state.sensor_data["led_status"] = command
            
        return True
    return False

# Fungsi untuk visualisasi
def create_temperature_gauge(temp):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = temp,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Suhu (°C)"},
        gauge = {
            'axis': {'range': [None, 40]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 25], 'color': "lightblue"},
                {'range': [25, 28], 'color': "lightgreen"},
                {'range': [28, 40], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': temp
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def create_humidity_gauge(humidity):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = humidity,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Kelembaban (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightyellow"},
                {'range': [30, 70], 'color': "lightgreen"},
                {'range': [70, 100], 'color': "blue"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

# Layout Streamlit
def main():
    st.set_page_config(
        page_title="MQTT Sensor Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Kontrol Dashboard")
        
        # Status koneksi
        st.subheader("Status Koneksi")
        if st.session_state.sensor_data["mqtt_connected"]:
            st.success("✅ MQTT Terhubung")
        else:
            st.error("❌ MQTT Terputus")
            if st.button("Coba Hubungkan Ulang"):
                setup_mqtt()
        
        # Kontrol LED
        st.subheader("🎛️ Kontrol LED")
        led_mode = st.radio(
            "Mode LED:",
            ["Auto", "Manual"],
            index=0 if st.session_state.sensor_data["led_mode"] == "auto" else 1
        )
        
        if led_mode == "Manual":
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔴 Merah", use_container_width=True):
                    send_led_command("merah")
            with col2:
                if st.button("🟢 Hijau", use_container_width=True):
                    send_led_command("hijau")
            
            col3, col4 = st.columns(2)
            with col3:
                if st.button("🟡 Kuning", use_container_width=True):
                    send_led_command("kuning")
            with col4:
                if st.button("⚫ Off", use_container_width=True):
                    send_led_command("off")
        else:
            if st.button("🔄 Mode Auto", use_container_width=True):
                send_led_command("auto")
        
        st.divider()
        st.subheader("📋 Informasi")
        st.write(f"**Topik Subscribe:** `{PUB_TOPIC_DHT}`")
        st.write(f"**Topik Publish:** `{SUB_TOPIC_LED}`")
        st.write(f"**Broker:** {MQTT_CONFIG['host']}")
        st.write(f"**Update Terakhir:** {st.session_state.sensor_data['timestamp']}")
    
    # Main content
    st.title("📊 Dashboard Monitoring Sensor MQTT")
    st.markdown("---")
    
    # Metrik utama
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temp = st.session_state.sensor_data["temperature"]
        status = st.session_state.sensor_data["status"]
        
        # Warna berdasarkan status
        if status == "DINGIN":
            color = "blue"
        elif status == "PANAS":
            color = "red"
        else:
            color = "green"
        
        st.metric(
            label="Suhu",
            value=f"{temp}°C",
            delta=status,
            delta_color="off"  # We'll handle color with custom CSS
        )
        
    with col2:
        st.metric(
            label="Kelembaban",
            value=f"{st.session_state.sensor_data['humidity']}%",
            delta="Normal" if 30 <= st.session_state.sensor_data['humidity'] <= 70 else "Ekstrem"
        )
        
    with col3:
        mode_icon = "🔄" if st.session_state.sensor_data["led_mode"] == "auto" else "✋"
        led_icon = {
            "merah": "🔴",
            "hijau": "🟢", 
            "kuning": "🟡",
            "off": "⚫"
        }.get(st.session_state.sensor_data["led_status"], "⚫")
        
        st.metric(
            label="Status LED",
            value=f"{mode_icon} {led_icon}",
            delta=f"Mode: {st.session_state.sensor_data['led_mode']}"
        )
    
    # Visualisasi gauge
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(create_temperature_gauge(st.session_state.sensor_data["temperature"]), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(create_humidity_gauge(st.session_state.sensor_data["humidity"]), 
                       use_container_width=True)
    
    # Riwayat data
    st.subheader("📈 Riwayat Data")
    
    if st.session_state.sensor_data["history"]:
        # Konversi ke DataFrame
        df = pd.DataFrame(st.session_state.sensor_data["history"])
        
        # Chart suhu dan kelembaban
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["temperature"], 
                      name="Suhu (°C)", line=dict(color="red")),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df["timestamp"], y=df["humidity"], 
                      name="Kelembaban (%)", line=dict(color="blue")),
            secondary_y=True,
        )
        
        fig.update_layout(
            title="Trend Suhu dan Kelembaban",
            xaxis_title="Waktu",
            height=400
        )
        
        fig.update_yaxes(title_text="Suhu (°C)", secondary_y=False)
        fig.update_yaxes(title_text="Kelembaban (%)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabel data terbaru
        st.subheader("📋 Data Terkini")
        recent_data = df.tail(10).sort_values("timestamp", ascending=False)
        st.dataframe(
            recent_data[["timestamp", "temperature", "humidity", "status"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("⏳ Menunggu data sensor...")
    
    # Auto-refresh setiap 2 detik
    time.sleep(2)
    st.rerun()

if __name__ == '__main__':
    # Setup MQTT saat pertama kali run
    if st.session_state.mqtt_client is None:
        setup_mqtt()
    
    main()

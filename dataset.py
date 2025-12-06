import streamlit as st
import paho.mqtt.client as mqtt
import json
import csv
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import ssl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px                 # <-- ditambahkan
import threading
import warnings
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI HIVEMQ REAL ====================
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1764923408610",
    "password": "9y&f74G1*pWSD.tQdXa@",
    "use_ssl": True,
    "keepalive": 20   # <-- keepalive lebih kecil agar tidak cepat disconnect
}

# Topics REAL dari ESP32 Anda
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
LED_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/sub/led"

# ==================== KONFIGURASI DATASET ====================
DATASET_CONFIG = {
    "folder_path": "IoT_Dataset",
    "filename": "sensor_data.csv",
    "target_records": 50,  # Target jumlah data untuk dikumpulkan
    "sampling_interval": 2,  # Interval sampling (detik)
    "auto_label": True,  # Otomatis beri label berdasarkan suhu
    "temperature_thresholds": {
        "cold": 25,     # < 25°C = DINGIN
        "normal_low": 25,  # 25°C = batas bawah normal
        "normal_high": 28, # 28°C = batas atas normal
        "hot": 28       # > 28°C = PANAS
    }
}

# ==================== INISIALISASI STATE ====================
if 'data_collector' not in st.session_state:
    st.session_state.data_collector = {
        "is_collecting": False,
        "records_collected": 0,
        "last_record": None,
        "start_time": None,
        "mqtt_connected": False,
        "csv_initialized": False,
        "live_data": {
            "temperature": 0.0,
            "humidity": 0.0,
            "timestamp": "",
            "label": "WAITING",
            "label_encoded": -1
        },
        "data_history": [],
        "status_messages": []
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# Flags untuk mencegah double-start/duplicate reconnect thread
if 'mqtt_thread_started' not in st.session_state:
    st.session_state.mqtt_thread_started = False
if 'mqtt_reconnect_in_progress' not in st.session_state:
    st.session_state.mqtt_reconnect_in_progress = False

# ==================== SISTEM DATA COLLECTOR ====================
class RealTimeDataCollector:
    def __init__(self, config):
        self.config = config
        self.csv_path = os.path.join(config["folder_path"], config["filename"])
        self.setup_csv()
    
    def setup_csv(self):
        """Setup folder dan file CSV untuk penyimpanan data"""
        try:
            # Buat folder jika belum ada
            if not os.path.exists(self.config["folder_path"]):
                os.makedirs(self.config["folder_path"])
                self.log_message(f"📁 Folder created: {self.config['folder_path']}")
            
            # Cek apakah file CSV sudah ada
            file_exists = os.path.exists(self.csv_path)
            
            if not file_exists:
                # Buat file CSV baru dengan header
                with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f, delimiter=';')
                    writer.writerow([
                        'timestamp',        # Format: YYYY-MM-DD HH:MM:SS
                        'temperature',      # °C
                        'humidity',         # %
                        'label',            # DINGIN/NORMAL/PANAS
                        'label_encoded',    # 0/1/2
                        'collection_session', # Session identifier
                        'device_id'         # ESP32 Identifier
                    ])
                self.log_message(f"✅ CSV file created: {self.csv_path}")
                st.session_state.data_collector["csv_initialized"] = True
            else:
                # Hitung jumlah data yang sudah ada
                with open(self.csv_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    existing_count = len(lines) - 1  # minus header
                
                self.log_message(f"📊 Existing records: {existing_count}")
                st.session_state.data_collector["records_collected"] = existing_count
                st.session_state.data_collector["csv_initialized"] = True
                
                # Tampilkan beberapa data terakhir
                if existing_count > 0:
                    self.log_message("📄 Last 3 records:")
                    for line in lines[max(1, len(lines)-3):]:
                        self.log_message(f"   {line.strip()}")
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error setting up CSV: {str(e)}")
            return False
    
    def determine_label(self, temperature):
        """Tentukan label berdasarkan suhu"""
        thresholds = self.config["temperature_thresholds"]
        
        if temperature < thresholds["cold"]:
            return "DINGIN", 0
        elif temperature > thresholds["hot"]:
            return "PANAS", 2
        else:
            return "NORMAL", 1
    
    def save_data(self, temperature, humidity, label=None, label_encoded=None):
        """Simpan data ke CSV"""
        try:
            # Tentukan label jika tidak diberikan
            if label is None or label_encoded is None:
                label, label_encoded = self.determine_label(temperature)
            
            # Prepare data row
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            csv_data = [
                timestamp,
                round(float(temperature), 2),
                round(float(humidity), 2),
                label,
                label_encoded,
                datetime.now().strftime('%Y%m%d_%H%M'),  # Session ID
                "ESP32_DHT11"  # Device identifier
            ]
            
            # Save to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                writer.writerow(csv_data)
            
            # Update state
            st.session_state.data_collector["records_collected"] += 1
            st.session_state.data_collector["last_record"] = {
                "timestamp": timestamp,
                "temperature": temperature,
                "humidity": humidity,
                "label": label
            }
            
            # Tambah ke history untuk visualisasi
            st.session_state.data_collector["data_history"].append({
                "timestamp": datetime.now(),
                "temperature": temperature,
                "humidity": humidity,
                "label": label
            })
            
            # Log message
            self.log_message(f"📥 Saved: {temperature}°C, {humidity}%, Label: {label}")
            
            return True
            
        except Exception as e:
            self.log_message(f"❌ Error saving data: {str(e)}")
            return False
    
    def get_dataset_info(self):
        """Dapatkan informasi tentang dataset yang terkumpul"""
        try:
            if not os.path.exists(self.csv_path):
                return {"total_records": 0, "label_distribution": {}}
            
            df = pd.read_csv(self.csv_path, delimiter=';')
            
            info = {
                "total_records": len(df),
                "file_size": os.path.getsize(self.csv_path),
                "date_range": {
                    "start": df['timestamp'].min() if len(df) > 0 else "N/A",
                    "end": df['timestamp'].max() if len(df) > 0 else "N/A"
                },
                "temperature_stats": {
                    "min": df['temperature'].min() if len(df) > 0 else 0,
                    "max": df['temperature'].max() if len(df) > 0 else 0,
                    "mean": df['temperature'].mean() if len(df) > 0 else 0,
                    "std": df['temperature'].std() if len(df) > 0 else 0
                },
                "label_distribution": df['label'].value_counts().to_dict() if 'label' in df.columns else {},
                "sessions": df['collection_session'].nunique() if 'collection_session' in df.columns else 0
            }
            
            return info
            
        except Exception as e:
            self.log_message(f"❌ Error getting dataset info: {str(e)}")
            return {"total_records": 0, "label_distribution": {}}
    
    def export_for_training(self):
        """Export dataset untuk training ML"""
        try:
            if not os.path.exists(self.csv_path):
                return False, "Dataset file not found"
            
            df = pd.read_csv(self.csv_path, delimiter=';')
            
            if len(df) < 10:
                return False, f"Need at least 10 records, only have {len(df)}"
            
            # Export untuk berbagai format
            export_dir = os.path.join(self.config["folder_path"], "exports")
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # 1. CSV untuk scikit-learn
            csv_export_path = os.path.join(export_dir, "dataset_training.csv")
            df.to_csv(csv_export_path, index=False)
            
            # 2. JSON untuk web apps
            json_export_path = os.path.join(export_dir, "dataset_info.json")
            dataset_info = {
                "total_samples": len(df),
                "features": ["temperature", "humidity"],
                "labels": df['label'].unique().tolist(),
                "label_distribution": df['label'].value_counts().to_dict(),
                "temperature_range": [df['temperature'].min(), df['temperature'].max()],
                "humidity_range": [df['humidity'].min(), df['humidity'].max()],
                "export_date": datetime.now().isoformat()
            }
            
            import json
            with open(json_export_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            # 3. TXT untuk edge devices
            txt_export_path = os.path.join(export_dir, "dataset_edge.txt")
            with open(txt_export_path, 'w') as f:
                f.write("// IoT Sensor Dataset for Edge ML\n")
                f.write(f"// Generated: {datetime.now()}\n")
                f.write(f"// Samples: {len(df)}\n\n")
                
                for _, row in df.iterrows():
                    f.write(f"{row['temperature']},{row['humidity']},{row['label_encoded']}\n")
            
            self.log_message(f"✅ Dataset exported to {export_dir}")
            return True, f"Dataset exported with {len(df)} samples"
            
        except Exception as e:
            return False, f"Export error: {str(e)}"
    
    def log_message(self, message):
        """Log message untuk ditampilkan di UI"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Simpan di session state
        st.session_state.data_collector["status_messages"].append(log_entry)
        
        # Batasi jumlah message
        if len(st.session_state.data_collector["status_messages"]) > 20:
            st.session_state.data_collector["status_messages"] = st.session_state.data_collector["status_messages"][-20:]
        
        # Print ke console juga
        print(log_entry)

# ==================== FUNGSI MQTT (DISEMPURNAKAN) ====================
def on_mqtt_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        st.session_state.data_collector["mqtt_connected"] = True
        st.session_state.mqtt_client = client
        print("✅ Connected to HiveMQ Cloud!")
        # Subscribe ke topic DHT
        client.subscribe(DHT_TOPIC)
        print(f"📡 Subscribed to: {DHT_TOPIC}")
        # Reset reconnect flag
        st.session_state.mqtt_reconnect_in_progress = False
    else:
        st.session_state.data_collector["mqtt_connected"] = False
        print(f"❌ Connection failed with code: {rc}")

def on_mqtt_disconnect(client, userdata, rc):
    # Called when disconnected
    st.session_state.data_collector["mqtt_connected"] = False
    print(f"⚠ MQTT Disconnected. RC={rc}")
    # Start background reconnect loop (if not already started)
    if not st.session_state.mqtt_reconnect_in_progress:
        st.session_state.mqtt_reconnect_in_progress = True
        def reconnect_loop(c):
            backoff = 1
            while not st.session_state.data_collector["mqtt_connected"]:
                try:
                    print("🔁 Attempting MQTT reconnect...")
                    c.reconnect()
                    # if reconnect succeeded, on_connect will set mqtt_connected True
                    time.sleep(1)
                    backoff = 1
                    break
                except Exception as e:
                    print(f"❌ Reconnect failed: {e} — retry in {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 20)  # exponential backoff up to 20s
            st.session_state.mqtt_reconnect_in_progress = False

        threading.Thread(target=reconnect_loop, args=(client,), daemon=True).start()

def on_mqtt_message(client, userdata, msg):
    """Callback ketika menerima data sensor dari ESP32"""
    try:
        # Parse data JSON dari ESP32
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        print(f"📨 Received: {temperature}°C, {humidity}%")
        
        # Update live data display (safely)
        st.session_state.data_collector["live_data"].update({
            "temperature": temperature,
            "humidity": humidity,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Tentukan label
        label, label_encoded = st.session_state.collector.determine_label(temperature)
        st.session_state.data_collector["live_data"]["label"] = label
        st.session_state.data_collector["live_data"]["label_encoded"] = label_encoded
        
        # Simpan data jika sedang collecting
        if st.session_state.data_collector["is_collecting"]:
            # Cek interval sampling
            last_save_time = st.session_state.data_collector.get("last_save_time")
            current_time = time.time()
            
            if last_save_time is None or (current_time - last_save_time) >= DATASET_CONFIG["sampling_interval"]:
                success = st.session_state.collector.save_data(temperature, humidity, label, label_encoded)
                if success:
                    st.session_state.data_collector["last_save_time"] = current_time
                    
                    # Cek jika sudah mencapai target
                    if st.session_state.data_collector["records_collected"] >= DATASET_CONFIG["target_records"]:
                        st.session_state.data_collector["is_collecting"] = False
                        st.session_state.collector.log_message(f"🎯 Target reached: {DATASET_CONFIG['target_records']} records!")
        
    except Exception as e:
        print(f"❌ Error processing MQTT message: {e}")

def connect_mqtt():
    """Connect ke HiveMQ Cloud - improved & safe to call multiple times"""
    try:
        # Jika client sudah ada dan connected, langsung return True
        client = st.session_state.get("mqtt_client")
        if client and st.session_state.data_collector["mqtt_connected"]:
            print("MQTT already connected")
            return True
        
        # Create new client
        client = mqtt.Client()  # use default client (can add client_id if needed)
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        if MQTT_CONFIG.get("use_ssl", False):
            # Use TLSv1_2 (most compatible)
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
        
        client.on_connect = on_mqtt_connect
        # attach disconnect callback
        client.on_disconnect = on_mqtt_disconnect
        client.on_message = on_mqtt_message
        
        # Connect with a smaller keepalive
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=MQTT_CONFIG.get("keepalive", 20))
        
        # Start network loop if not started before
        # To avoid double loop_start, check flag
        if not st.session_state.mqtt_thread_started:
            client.loop_start()
            st.session_state.mqtt_thread_started = True
        
        # Save client into session_state
        st.session_state.mqtt_client = client
        
        # Small wait to allow on_connect to be called
        time.sleep(1.5)
        return True
        
    except Exception as e:
        print(f"❌ MQTT Connection failed: {e}")
        return False

def send_led_command(command):
    """Kirim perintah ke LED ESP32"""
    if st.session_state.mqtt_client and st.session_state.data_collector["mqtt_connected"]:
        try:
            st.session_state.mqtt_client.publish(LED_TOPIC, command)
            print(f"💡 LED command sent: {command}")
            return True
        except Exception as e:
            print(f"❌ Publish failed: {e}")
            return False
    return False

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Real-Time IoT Data Collector",
        page_icon="📊",
        layout="wide"
    )
    
    # Inisialisasi collector
    if 'collector' not in st.session_state:
        st.session_state.collector = RealTimeDataCollector(DATASET_CONFIG)
    
    # Custom CSS
    st.markdown("""
    <style>
    .data-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin-bottom: 15px;
    }
    .collecting-active {
        border-left: 5px solid #4CAF50 !important;
        background: #E8F5E9 !important;
    }
    .status-connected {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-disconnected {
        color: #FF5722;
        font-weight: bold;
    }
    .log-container {
        background: #2d3748;
        color: white;
        padding: 15px;
        border-radius: 10px;
        font-family: monospace;
        max-height: 300px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 Real-Time IoT Data Collector")
    st.markdown("### Collect DHT11 Sensor Data from ESP32 for Machine Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("🔗 Connection Control")
        
        # Status koneksi
        col_conn1, col_conn2 = st.columns(2)
        with col_conn1:
            if st.session_state.data_collector["mqtt_connected"]:
                st.success("🟢 CONNECTED")
            else:
                st.error("🔴 DISCONNECTED")
        
        # Tombol koneksi
        if st.button("🔗 Connect to HiveMQ", type="primary", use_container_width=True):
            with st.spinner("Connecting to HiveMQ Cloud..."):
                if connect_mqtt():
                    st.success("✅ Connected!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Could not connect. Check credentials or network.")
        
        if st.button("🔌 Disconnect", use_container_width=True):
            if st.session_state.mqtt_client:
                try:
                    st.session_state.mqtt_client.loop_stop()
                except:
                    pass
                st.session_state.data_collector["mqtt_connected"] = False
                st.session_state.mqtt_thread_started = False
                st.warning("Disconnected from HiveMQ")
                st.rerun()
        
        st.markdown("---")
        
        # Dataset Configuration
        st.header("⚙️ Dataset Settings")
        
        target_records = st.number_input(
            "Target Records",
            min_value=10,
            max_value=1000,
            value=DATASET_CONFIG["target_records"],
            step=10
        )
        DATASET_CONFIG["target_records"] = target_records
        
        sampling_interval = st.slider(
            "Sampling Interval (seconds)",
            min_value=1,
            max_value=30,
            value=DATASET_CONFIG["sampling_interval"],
            step=1
        )
        DATASET_CONFIG["sampling_interval"] = sampling_interval
        
        st.markdown("---")
        
        # Data Collection Control
        st.header("🎯 Data Collection")
        
        if not st.session_state.data_collector["is_collecting"]:
            if st.button("▶️ Start Collecting", type="primary", use_container_width=True):
                if st.session_state.data_collector["mqtt_connected"]:
                    st.session_state.data_collector["is_collecting"] = True
                    st.session_state.data_collector["start_time"] = datetime.now()
                    st.session_state.collector.log_message("🚀 Started data collection!")
                    st.rerun()
                else:
                    st.warning("Please connect to HiveMQ first")
        else:
            if st.button("⏹️ Stop Collecting", type="secondary", use_container_width=True):
                st.session_state.data_collector["is_collecting"] = False
                st.session_state.collector.log_message("🛑 Stopped data collection")
                st.rerun()
        
        st.markdown("---")
        
        # Manual LED Control
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
        
        st.markdown("---")
        
        # Export Options
        st.header("📤 Export Data")
        
        if st.button("💾 Export for Training", use_container_width=True):
            success, message = st.session_state.collector.export_for_training()
            if success:
                st.success(message)
            else:
                st.warning(message)
        
        if st.button("📊 Show Dataset Info", use_container_width=True):
            dataset_info = st.session_state.collector.get_dataset_info()
            st.json(dataset_info)
    
    # ==================== MAIN CONTENT ====================
    # Row 1: Live Data Display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="data-card {'collecting-active' if st.session_state.data_collector['is_collecting'] else ''}">
            <h4>📡 Live Sensor Data</h4>
            <p><strong>Temperature:</strong> {st.session_state.data_collector['live_data']['temperature']:.1f} °C</p>
            <p><strong>Humidity:</strong> {st.session_state.data_collector['live_data']['humidity']:.1f} %</p>
            <p><strong>Label:</strong> {st.session_state.data_collector['live_data']['label']}</p>
            <p><small>Last update: {st.session_state.data_collector['live_data']['timestamp']}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        records_collected = st.session_state.data_collector["records_collected"]
        target = DATASET_CONFIG["target_records"]
        progress = min(records_collected / target, 1.0)
        
        st.markdown(f"""
        <div class="data-card">
            <h4>📊 Collection Progress</h4>
            <p><strong>Collected:</strong> {records_collected} / {target} records</p>
            <div style="background: #e0e0e0; border-radius: 5px; height: 20px; margin: 10px 0;">
                <div style="background: #4CAF50; width: {progress*100}%; height: 100%; border-radius: 5px;"></div>
            </div>
            <p><strong>Status:</strong> {'🟢 COLLECTING' if st.session_state.data_collector['is_collecting'] else '⏸️ PAUSED'}</p>
            {f'<p><strong>Time elapsed:</strong> {(datetime.now() - st.session_state.data_collector["start_time"]).seconds} seconds</p>' if st.session_state.data_collector['start_time'] else ''}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Dataset info
        dataset_info = st.session_state.collector.get_dataset_info()
        
        st.markdown(f"""
        <div class="data-card">
            <h4>💾 Dataset Info</h4>
            <p><strong>Total Records:</strong> {dataset_info.get('total_records', 0)}</p>
            <p><strong>File Size:</strong> {dataset_info.get('file_size', 0):,} bytes</p>
            <p><strong>Sessions:</strong> {dataset_info.get('sessions', 0)}</p>
            <p><strong>Temperature Range:</strong> {dataset_info.get('temperature_stats', {}).get('min', 0):.1f} - {dataset_info.get('temperature_stats', {}).get('max', 0):.1f} °C</p>
            <p><small>File: {st.session_state.collector.csv_path}</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Real-time Charts
    st.subheader("📈 Real-Time Data Visualization")
    
    if len(st.session_state.data_collector["data_history"]) > 1:
        # Prepare data for plotting
        history_df = pd.DataFrame(st.session_state.data_collector["data_history"])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature Trend', 'Humidity Trend', 
                          'Temperature Distribution', 'Label Distribution'),
            vertical_spacing=0.15,
            horizontal_spacing=0.15
        )
        
        # Temperature trend
        fig.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='#FF5722', width=2)
            ),
            row=1, col=1
        )
        
        # Humidity trend
        fig.add_trace(
            go.Scatter(
                x=history_df['timestamp'],
                y=history_df['humidity'],
                mode='lines+markers',
                name='Humidity',
                line=dict(color='#2196F3', width=2)
            ),
            row=1, col=2
        )
        
        # Temperature histogram
        fig.add_trace(
            go.Histogram(
                x=history_df['temperature'],
                nbinsx=20,
                name='Temperature',
                marker_color='#FF5722'
            ),
            row=2, col=1
        )
        
        # Label distribution
        label_counts = history_df['label'].value_counts()
        fig.add_trace(
            go.Bar(
                x=label_counts.index,
                y=label_counts.values,
                name='Labels',
                marker_color=['#2196F3', '#4CAF50', '#FF5722']  # blue, green, red
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("⏳ Collecting data... Visualizations will appear here")
    
    # Row 3: Data Table and Logs
    col_table, col_logs = st.columns([1, 1])
    
    with col_table:
        st.subheader("📋 Recent Data")
        
        if len(st.session_state.data_collector["data_history"]) > 0:
            # Show last 10 records
            recent_data = pd.DataFrame(st.session_state.data_collector["data_history"][-10:])
            
            # Format timestamp
            recent_data['timestamp'] = recent_data['timestamp'].dt.strftime('%H:%M:%S')
            
            st.dataframe(
                recent_data[['timestamp', 'temperature', 'humidity', 'label']],
                use_container_width=True,
                height=300
            )
            
            # Download button
            csv_data = recent_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Recent Data",
                data=csv_data,
                file_name="recent_sensor_data.csv",
                mime="text/csv"
            )
        else:
            st.info("No data collected yet")
    
    with col_logs:
        st.subheader("📝 System Logs")
        
        # Log container
        st.markdown("""
        <div class="log-container">
        """, unsafe_allow_html=True)
        
        # Display logs in reverse order (newest first)
        logs = st.session_state.data_collector["status_messages"][::-1]
        for log in logs[:15]:  # Show last 15 logs
            st.text(log)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Clear logs button
        if st.button("🗑️ Clear Logs"):
            st.session_state.data_collector["status_messages"] = []
            st.rerun()
    
    st.markdown("---")
    
    # Row 4: Dataset Analysis
    st.subheader("🔍 Dataset Analysis")
    
    dataset_info = st.session_state.collector.get_dataset_info()
    if dataset_info["total_records"] > 0:
        col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
        
        with col_analysis1:
            # Label distribution pie chart
            if dataset_info["label_distribution"]:
                fig_pie = go.Figure(data=[go.Pie(
                    labels=list(dataset_info["label_distribution"].keys()),
                    values=list(dataset_info["label_distribution"].values()),
                    hole=.3,
                    marker_colors=['#2196F3', '#4CAF50', '#FF5722']
                )])
                fig_pie.update_layout(title="Label Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_analysis2:
            # Temperature vs Humidity scatter
            if len(st.session_state.data_collector["data_history"]) > 0:
                scatter_df = pd.DataFrame(st.session_state.data_collector["data_history"])
                
                fig_scatter = px.scatter(
                    scatter_df,
                    x='temperature',
                    y='humidity',
                    color='label',
                    title='Temperature vs Humidity',
                    color_discrete_map={
                        'DINGIN': '#2196F3',
                        'NORMAL': '#4CAF50',
                        'PANAS': '#FF5722'
                    }
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col_analysis3:
            # Statistics
            st.write("**Dataset Statistics:**")
            
            stats_items = [
                ("Total Records", dataset_info["total_records"]),
                ("Temperature Mean", f"{dataset_info['temperature_stats']['mean']:.1f}°C"),
                ("Temperature Std", f"{dataset_info['temperature_stats']['std']:.1f}°C"),
                ("Date Range", f"{dataset_info['date_range']['start']} to {dataset_info['date_range']['end']}")
            ]
            
            for label, value in stats_items:
                st.write(f"**{label}:** {value}")
            
            # Label counts
            st.write("**Label Counts:**")
            for label, count in dataset_info["label_distribution"].items():
                percentage = (count / dataset_info["total_records"]) * 100
                st.write(f"{label}: {count} ({percentage:.1f}%)")
    
    # Auto-refresh jika sedang collecting
    if st.session_state.data_collector["is_collecting"]:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()


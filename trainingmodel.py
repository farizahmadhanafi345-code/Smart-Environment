# ml_training_live.py
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
import ssl
import threading
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "IoT_Dataset")
CSV_FILE = os.path.join(DATA_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(DATA_DIR, "ml_models_live")
REPORTS_DIR = os.path.join(DATA_DIR, "ml_reports_live")

# Konfigurasi HiveMQ REAL
MQTT_CONFIG = {
    "broker": "f44c5a09b28447449642c2c62b63bba7.s1.eu.hivemq.cloud",
    "port": 8883,
    "username": "hivemq.webclient.1764923408610",
    "password": "9y&f74G1*pWSD.tQdXa@",
    "use_ssl": True,
    "keepalive": 20
}

# Topics
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
TRAINING_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_training"

# Buat folder
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==================== INISIALISASI STATE ====================
if 'training_system' not in st.session_state:
    st.session_state.training_system = {
        "is_training": False,
        "training_progress": 0,
        "training_status": "IDLE",
        "models_trained": [],
        "last_training_time": None,
        "training_data_count": 0,
        "live_data_received": 0,
        "mqtt_connected": False,
        "live_predictions": [],
        "training_logs": []
    }

if 'mqtt_client' not in st.session_state:
    st.session_state.mqtt_client = None

# ==================== SISTEM TRAINING LIVE ====================
class LiveTrainingSystem:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.dataset = pd.DataFrame()
        self.is_trained = False
        self.load_existing_data()
    
    def load_existing_data(self):
        """Load existing dataset dari CSV"""
        try:
            if os.path.exists(CSV_FILE):
                self.dataset = pd.read_csv(CSV_FILE, delimiter=';')
                st.success(f"📊 Dataset loaded: {len(self.dataset)} records")
                
                # Konversi timestamp
                if 'timestamp' in self.dataset.columns:
                    self.dataset['timestamp'] = pd.to_datetime(self.dataset['timestamp'])
                    self.dataset['hour'] = self.dataset['timestamp'].dt.hour
                    self.dataset['minute'] = self.dataset['timestamp'].dt.minute
            else:
                st.warning("📝 No existing dataset found")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
    
    def add_live_data(self, temperature, humidity, label=None, label_encoded=None):
        """Tambahkan data live ke dataset"""
        try:
            # Tentukan label jika tidak diberikan
            if label is None or label_encoded is None:
                if temperature < 25:
                    label = "DINGIN"
                    label_encoded = 0
                elif temperature > 28:
                    label = "PANAS"
                    label_encoded = 2
                else:
                    label = "NORMAL"
                    label_encoded = 1
            
            # Tambah data baru
            new_data = {
                'timestamp': datetime.now(),
                'temperature': temperature,
                'humidity': humidity,
                'label': label,
                'label_encoded': label_encoded,
                'collection_session': 'live_training',
                'device_id': 'ESP32_LIVE',
                'hour': datetime.now().hour,
                'minute': datetime.now().minute
            }
            
            # Tambah ke dataset
            self.dataset = pd.concat([self.dataset, pd.DataFrame([new_data])], ignore_index=True)
            
            # Update count
            st.session_state.training_system["live_data_received"] += 1
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error adding live data: {e}")
            return False
    
    def prepare_features(self):
        """Siapkan features untuk training"""
        if self.dataset.empty:
            return None, None
        
        try:
            # Features
            feature_cols = ['temperature', 'humidity']
            if 'hour' in self.dataset.columns:
                feature_cols.append('hour')
            if 'minute' in self.dataset.columns:
                feature_cols.append('minute')
            
            X = self.dataset[feature_cols].values
            y = self.dataset['label_encoded'].values
            
            return X, y
            
        except Exception as e:
            st.error(f"❌ Error preparing features: {e}")
            return None, None
    
    def train_models(self):
        """Train semua model ML"""
        try:
            # Prepare data
            X, y = self.prepare_features()
            if X is None or y is None:
                return False, "No data available for training"
            
            # Cek jumlah data
            if len(X) < 10:
                return False, f"Need at least 10 samples, only have {len(X)}"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models
            models_config = {
                'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
                'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            results = {}
            
            for name, model in models_config.items():
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                
                # Store results
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_test': y_test
                }
                
                # Update session state
                st.session_state.training_system["models_trained"].append(name)
            
            # Save models
            self.models = results
            
            # Save to files
            self.save_models()
            
            return True, f"Trained {len(results)} models with {len(X)} samples"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def save_models(self):
        """Save models ke file"""
        try:
            # Save scaler
            scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save models
            for name, result in self.models.items():
                model_path = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(result['model'], f)
            
            # Save metadata
            metadata = {
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'models_trained': list(self.models.keys()),
                'dataset_size': len(self.dataset),
                'performance': {},
                'features_used': ['temperature', 'humidity', 'hour', 'minute']
            }
            
            for name, result in self.models.items():
                metadata['performance'][name] = {
                    'accuracy': float(result['accuracy']),
                    'precision': float(result['precision']),
                    'recall': float(result['recall']),
                    'f1_score': float(result['f1_score']),
                    'cv_mean': float(result['cv_mean']),
                    'cv_std': float(result['cv_std'])
                }
            
            metadata_path = os.path.join(MODELS_DIR, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # Save dataset
            dataset_path = os.path.join(DATA_DIR, "training_dataset.csv")
            self.dataset.to_csv(dataset_path, index=False)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error saving models: {e}")
            return False
    
    def predict_live(self, temperature, humidity):
        """Predict dengan model yang sudah trained"""
        if not self.models:
            return {}
        
        try:
            # Prepare features
            hour = datetime.now().hour
            minute = datetime.now().minute
            
            features = np.array([[temperature, humidity, hour, minute]])
            features_scaled = self.scaler.transform(features)
            
            predictions = {}
            
            for name, result in self.models.items():
                model = result['model']
                
                # Predict
                pred_code = model.predict(features_scaled)[0]
                
                # Get probabilities
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
                    'probabilities': {
                        'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                        'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                        'PANAS': float(probs[2]) if len(probs) > 2 else 0
                    }
                }
            
            return predictions
            
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
            return {}

# ==================== FUNGSI MQTT ====================
def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        st.session_state.training_system["mqtt_connected"] = True
        client.subscribe(DHT_TOPIC)
        add_training_log("✅ Connected to HiveMQ for live training")
    else:
        st.session_state.training_system["mqtt_connected"] = False

def on_mqtt_message(client, userdata, msg):
    """Receive live data untuk training"""
    try:
        data = json.loads(msg.payload.decode('utf-8'))
        
        temperature = float(data.get('temperature', 0))
        humidity = float(data.get('humidity', 0))
        
        # Add to training system
        if 'training_system_obj' in st.session_state:
            training_system = st.session_state.training_system_obj
            training_system.add_live_data(temperature, humidity)
        
        add_training_log(f"📥 Live data: {temperature}°C, {humidity}%")
        
    except Exception as e:
        add_training_log(f"❌ Error processing MQTT: {e}")

def connect_mqtt_training():
    """Connect ke MQTT untuk live training"""
    try:
        client = mqtt.Client()
        client.username_pw_set(MQTT_CONFIG["username"], MQTT_CONFIG["password"])
        
        if MQTT_CONFIG["use_ssl"]:
            client.tls_set(tls_version=ssl.PROTOCOL_TLSv1_2)
        
        client.on_connect = on_mqtt_connect
        client.on_message = on_mqtt_message
        
        client.connect(MQTT_CONFIG["broker"], MQTT_CONFIG["port"], keepalive=20)
        client.loop_start()
        
        st.session_state.mqtt_client = client
        time.sleep(2)
        
        return True
        
    except Exception as e:
        add_training_log(f"❌ MQTT Connection failed: {e}")
        return False

def add_training_log(message):
    """Tambahkan log ke training system"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    
    st.session_state.training_system["training_logs"].append(log_entry)
    
    # Batasi jumlah log
    if len(st.session_state.training_system["training_logs"]) > 30:
        st.session_state.training_system["training_logs"] = st.session_state.training_system["training_logs"][-30:]

# ==================== STREAMLIT UI ====================
def main():
    st.set_page_config(
        page_title="Live ML Training Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    # Inisialisasi training system
    if 'training_system_obj' not in st.session_state:
        st.session_state.training_system_obj = LiveTrainingSystem()
    
    # Custom CSS
    st.markdown("""
    <style>
    .training-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 6px solid #3B82F6;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    .live-training-active {
        border-left: 6px solid #10B981 !important;
        background: #F0F9FF;
    }
    
    .model-card {
        background: white;
        padding: 1rem;
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
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🤖 Live ML Training Dashboard")
    st.markdown("### Real-time Machine Learning Training from IoT Sensor Data")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Training Control")
        
        # Status koneksi
        col_conn1, col_conn2 = st.columns(2)
        with col_conn1:
            if st.session_state.training_system["mqtt_connected"]:
                st.success("🟢 CONNECTED")
            else:
                st.error("🔴 DISCONNECTED")
        
        # Tombol koneksi
        if st.button("🔗 Connect to HiveMQ", type="primary", use_container_width=True):
            with st.spinner("Connecting..."):
                if connect_mqtt_training():
                    st.success("Connected for live training!")
                    st.rerun()
        
        st.markdown("---")
        
        # Training controls
        st.subheader("🏋️ Training Operations")
        
        # Load existing data
        if st.button("📂 Load Existing Dataset", use_container_width=True):
            st.session_state.training_system_obj.load_existing_data()
            st.success("Dataset loaded!")
            st.rerun()
        
        # Start training
        if st.button("🚀 Start Live Training", type="primary", use_container_width=True):
            st.session_state.training_system["is_training"] = True
            add_training_log("🚀 Starting live training...")
            
            with st.spinner("Training in progress..."):
                success, message = st.session_state.training_system_obj.train_models()
                
                if success:
                    st.session_state.training_system["is_training"] = False
                    st.session_state.training_system["last_training_time"] = datetime.now()
                    st.success(message)
                else:
                    st.error(message)
            
            st.rerun()
        
        # Manual data input
        st.markdown("---")
        st.subheader("🧪 Manual Data Input")
        
        manual_temp = st.number_input("Temperature (°C)", value=25.0, min_value=0.0, max_value=50.0)
        manual_hum = st.number_input("Humidity (%)", value=65.0, min_value=0.0, max_value=100.0)
        
        col_label1, col_label2, col_label3 = st.columns(3)
        with col_label1:
            if st.button("🥶 DINGIN", use_container_width=True):
                st.session_state.training_system_obj.add_live_data(manual_temp, manual_hum, "DINGIN", 0)
                st.success("Added DINGIN sample")
        
        with col_label2:
            if st.button("✅ NORMAL", use_container_width=True):
                st.session_state.training_system_obj.add_live_data(manual_temp, manual_hum, "NORMAL", 1)
                st.success("Added NORMAL sample")
        
        with col_label3:
            if st.button("🔥 PANAS", use_container_width=True):
                st.session_state.training_system_obj.add_live_data(manual_temp, manual_hum, "PANAS", 2)
                st.success("Added PANAS sample")
        
        st.markdown("---")
        
        # Refresh button
        if st.button("🔄 Refresh Dashboard", use_container_width=True):
            st.rerun()
    
    # ============ MAIN CONTENT ============
    # Row 1: System Status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dataset_size = len(st.session_state.training_system_obj.dataset)
        st.markdown(f"""
        <div class="training-card">
            <h3>📊 Dataset Size</h3>
            <h1 style="color: #3B82F6;">{dataset_size}</h1>
            <p>Total training samples</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        live_data = st.session_state.training_system["live_data_received"]
        st.markdown(f"""
        <div class="training-card">
            <h3>📡 Live Data</h3>
            <h1 style="color: #10B981;">{live_data}</h1>
            <p>Real-time samples received</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        models_trained = len(st.session_state.training_system["models_trained"])
        st.markdown(f"""
        <div class="training-card">
            <h3>🤖 Models Trained</h3>
            <h1 style="color: #8B5CF6;">{models_trained}</h1>
            <p>ML models available</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if st.session_state.training_system["last_training_time"]:
            last_train = st.session_state.training_system["last_training_time"].strftime("%H:%M")
            st.markdown(f"""
            <div class="training-card">
                <h3>🕒 Last Training</h3>
                <h1 style="color: #F59E0B;">{last_train}</h1>
                <p>Most recent training</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="training-card">
                <h3>🕒 Last Training</h3>
                <h1 style="color: #6B7280;">Never</h1>
                <p>No training completed yet</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Row 2: Live Training Status
    st.subheader("🔄 Live Training Status")
    
    if st.session_state.training_system["is_training"]:
        st.warning("🏃 Training in progress...")
        
        # Progress bar
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress.progress(i + 1)
    else:
        if st.session_state.training_system["models_trained"]:
            st.success(f"✅ Training complete! {len(st.session_state.training_system['models_trained'])} models trained")
        else:
            st.info("⏳ Ready for training. Connect to HiveMQ and start training.")
    
    st.markdown("---")
    
    # Row 3: Model Performance
    st.subheader("📊 Model Performance")
    
    if st.session_state.training_system_obj.models:
        # Create performance chart
        models_list = list(st.session_state.training_system_obj.models.keys())
        accuracy_scores = [st.session_state.training_system_obj.models[m]['accuracy'] for m in models_list]
        f1_scores = [st.session_state.training_system_obj.models[m]['f1_score'] for m in models_list]
        
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=models_list, y=accuracy_scores, marker_color='#3B82F6'),
            go.Bar(name='F1-Score', x=models_list, y=f1_scores, marker_color='#10B981')
        ])
        
        fig.update_layout(
            title='Model Performance Comparison',
            barmode='group',
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details in columns
        model_cols = st.columns(len(models_list))
        
        for idx, (name, result) in enumerate(st.session_state.training_system_obj.models.items()):
            with model_cols[idx]:
                st.markdown(f"""
                <div class="model-card">
                    <h4>{name}</h4>
                    <p><strong>Accuracy:</strong> {result['accuracy']:.3f}</p>
                    <p><strong>F1-Score:</strong> {result['f1_score']:.3f}</p>
                    <p><strong>Precision:</strong> {result['precision']:.3f}</p>
                    <p><strong>Recall:</strong> {result['recall']:.3f}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No models trained yet. Start training to see performance metrics.")
    
    st.markdown("---")
    
    # Row 4: Live Prediction Test
    st.subheader("🧪 Live Prediction Test")
    
    col_test1, col_test2 = st.columns(2)
    
    with col_test1:
        test_temp = st.slider("Test Temperature (°C)", 15.0, 35.0, 25.0, 0.5)
    
    with col_test2:
        test_hum = st.slider("Test Humidity (%)", 30.0, 90.0, 65.0, 1.0)
    
    if st.button("🔮 Test Prediction", type="primary"):
        predictions = st.session_state.training_system_obj.predict_live(test_temp, test_hum)
        
        if predictions:
            st.success("Predictions generated!")
            
            # Display predictions
            pred_cols = st.columns(len(predictions))
            
            for idx, (model_name, pred) in enumerate(predictions.items()):
                with pred_cols[idx]:
                    # Determine color
                    if pred['label'] == 'DINGIN':
                        color = '#3B82F6'
                    elif pred['label'] == 'PANAS':
                        color = '#EF4444'
                    else:
                        color = '#10B981'
                    
                    st.markdown(f"""
                    <div style="
                        background: white;
                        padding: 1rem;
                        border-radius: 10px;
                        border-left: 5px solid {color};
                        text-align: center;
                    ">
                        <h4 style="color: {color};">{model_name}</h4>
                        <h2 style="color: {color}; margin: 0.5rem 0;">{pred['label']}</h2>
                        <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No trained models available for prediction")
    
    st.markdown("---")
    
    # Row 5: Training Logs
    st.subheader("📝 Training Logs")
    
    st.markdown('<div class="log-container">', unsafe_allow_html=True)
    
    # Display logs (newest first)
    logs = st.session_state.training_system["training_logs"][::-1]
    for log in logs[:20]:
        # Color code
        if "✅" in log or "success" in log.lower():
            st.markdown(f'<span style="color: #10B981;">{log}</span>', unsafe_allow_html=True)
        elif "❌" in log or "error" in log.lower():
            st.markdown(f'<span style="color: #EF4444;">{log}</span>', unsafe_allow_html=True)
        elif "⚠️" in log or "warning" in log.lower():
            st.markdown(f'<span style="color: #F59E0B;">{log}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span style="color: #E5E7EB;">{log}</span>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear logs button
    if st.button("🗑️ Clear Logs"):
        st.session_state.training_system["training_logs"] = []
        st.rerun()
    
    # Auto-refresh jika sedang training
    if st.session_state.training_system["is_training"]:
        time.sleep(1)
        st.rerun()

if __name__ == "__main__":
    main()

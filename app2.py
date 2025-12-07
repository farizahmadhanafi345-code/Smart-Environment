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
import threading
import queue

# ==================== KONFIGURASI ====================
BASE_DIR = "Data_Collector"
CSV_FILE = os.path.join(BASE_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# MQTT Configuration
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
PREDICTION_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_prediction"

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="DHT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== MQTT MANAGER ====================
class MQTTManager:
    def __init__(self):
        self.client = None
        self.connected = False
        self.message_queue = queue.Queue()
        self.received_data = []
        self.max_history = 100  # Max messages to keep in history
        self.lock = threading.Lock()
        
    def connect(self):
        try:
            self.client = mqtt.Client()
            self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
            self.client.tls_set()
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.on_disconnect = self.on_disconnect
            
            # Connect
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # Start thread
            self.mqtt_thread = threading.Thread(target=self.client.loop_forever, daemon=True)
            self.mqtt_thread.start()
            
            time.sleep(2)  # Wait for connection
            return True
            
        except Exception as e:
            st.error(f"MQTT Connection failed: {e}")
            return False
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(DHT_TOPIC)
            client.subscribe(PREDICTION_TOPIC)
        else:
            self.connected = False
    
    def on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            data['topic'] = msg.topic
            data['received_time'] = datetime.now().strftime('%H:%M:%S')
            
            with self.lock:
                self.message_queue.put(data)
                self.received_data.append(data)
                
                # Keep only last N messages
                if len(self.received_data) > self.max_history:
                    self.received_data = self.received_data[-self.max_history:]
                    
        except Exception as e:
            print(f"Error processing MQTT message: {e}")
    
    def on_disconnect(self, client, userdata, rc):
        self.connected = False
    
    def get_latest_message(self):
        """Get latest message from queue"""
        try:
            with self.lock:
                if not self.message_queue.empty():
                    return self.message_queue.get_nowait()
        except:
            pass
        return None
    
    def get_all_messages(self):
        """Get all received messages"""
        with self.lock:
            return self.received_data.copy()
    
    def publish_prediction(self, prediction_data):
        """Publish prediction to MQTT"""
        if self.connected and self.client:
            try:
                payload = json.dumps(prediction_data)
                result = self.client.publish(PREDICTION_TOPIC, payload, qos=1)
                return result.rc == mqtt.MQTT_ERR_SUCCESS
            except Exception as e:
                st.error(f"Publish error: {e}")
        return False
    
    def disconnect(self):
        if self.client and self.connected:
            self.client.disconnect()
            self.connected = False

# ==================== INITIALIZE MQTT ====================
@st.cache_resource
def init_mqtt():
    """Initialize MQTT connection"""
    mqtt_manager = MQTTManager()
    return mqtt_manager

# ==================== LOAD DATA & MODELS ====================
@st.cache_data
def load_data():
    """Load dataset"""
    if not os.path.exists(CSV_FILE):
        return None
    
    try:
        df = pd.read_csv(CSV_FILE, delimiter=';')
        
        # Process timestamp
        if 'timestamp' in df.columns:
            df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
        else:
            current_time = datetime.now()
            df['hour'] = current_time.hour
            df['minute'] = current_time.minute
            df['second'] = current_time.second
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_models():
    """Load ML models with caching"""
    models = {}
    metadata = {}
    
    try:
        # Load scaler
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        
        # Load metadata
        metadata_path = os.path.join(MODELS_DIR, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load individual models
        model_files = {
            'Decision Tree': 'decision_tree.pkl',
            'K-Nearest Neighbors': 'k_nearest_neighbors.pkl',
            'Logistic Regression': 'logistic_regression.pkl',
            'Dummy Classifier': 'dummy_classifier.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    models[name] = pickle.load(f)
        
        return models, scaler, metadata
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# ==================== PREDICTION FUNCTIONS ====================
def predict_temperature(models, scaler, temperature, humidity, hour=None, minute=None):
    """Predict with all models"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    
    try:
        features_scaled = scaler.transform(features)
    except:
        features_scaled = features
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            pred_code = model.predict(features_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code] if len(probs) > pred_code else 1.0
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(pred_code, 'UNKNOWN')
            
            predictions[model_name] = {
                'label': label,
                'confidence': float(confidence),
                'probabilities': {
                    'DINGIN': float(probs[0]) if len(probs) > 0 else 0,
                    'NORMAL': float(probs[1]) if len(probs) > 1 else 0,
                    'PANAS': float(probs[2]) if len(probs) > 2 else 0
                },
                'label_encoded': int(pred_code),
                'color': get_label_color(label)
            }
            
        except Exception as e:
            predictions[model_name] = {
                'label': 'ERROR',
                'confidence': 0.0,
                'error': str(e),
                'color': '#f39c12'
            }
    
    return predictions

def get_label_color(label):
    """Get color based on label"""
    colors = {
        'DINGIN': '#3498db',
        'NORMAL': '#2ecc71',
        'PANAS': '#e74c3c',
        'UNKNOWN': '#95a5a6',
        'ERROR': '#f39c12'
    }
    return colors.get(label, '#95a5a6')

# ==================== SIDEBAR ====================
def sidebar_controls(mqtt_manager):
    """Sidebar controls"""
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # MQTT Status
    st.sidebar.subheader("📡 MQTT Connection")
    
    if mqtt_manager.connected:
        st.sidebar.success("✅ Connected to HiveMQ")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 Refresh MQTT"):
                st.rerun()
        with col2:
            if st.button("📡 Disconnect"):
                mqtt_manager.disconnect()
                st.rerun()
    else:
        st.sidebar.error("❌ Not connected")
        if st.sidebar.button("🔗 Connect to HiveMQ"):
            if mqtt_manager.connect():
                st.sidebar.success("Connected!")
                st.rerun()
    
    # Real-time controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 Real-time Mode")
    
    real_time_mode = st.sidebar.checkbox("Enable Real-time", value=False)
    auto_update = st.sidebar.checkbox("Auto-update", value=True)
    
    if auto_update and real_time_mode:
        update_interval = st.sidebar.slider("Update (seconds)", 1, 30, 5)
        st.sidebar.caption(f"Next update in {update_interval}s")
    
    st.sidebar.markdown("---")
    
    # Model selection
    st.sidebar.subheader("🤖 Model Selection")
    show_dt = st.sidebar.checkbox("Decision Tree", value=True)
    show_knn = st.sidebar.checkbox("K-Nearest Neighbors", value=True)
    show_lr = st.sidebar.checkbox("Logistic Regression", value=True)
    
    st.sidebar.markdown("---")
    
    # Manual prediction
    st.sidebar.subheader("🔮 Manual Prediction")
    manual_temp = st.sidebar.slider("Temperature (°C)", 15.0, 35.0, 24.0, 0.5)
    manual_hum = st.sidebar.slider("Humidity (%)", 30.0, 90.0, 65.0, 1.0)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        manual_hour = st.number_input("Hour", 0, 23, datetime.now().hour)
    with col2:
        manual_minute = st.number_input("Minute", 0, 59, datetime.now().minute)
    
    # Publish prediction
    if st.sidebar.button("📤 Publish to MQTT", use_container_width=True):
        st.sidebar.info("Predictions will be published to MQTT")
    
    st.sidebar.markdown("---")
    
    # Time range for historical data
    st.sidebar.subheader("📅 Time Range")
    days_back = st.sidebar.slider("Days to display", 1, 30, 7)
    
    # Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    return {
        'models': {'Decision Tree': show_dt, 'K-Nearest Neighbors': show_knn, 'Logistic Regression': show_lr},
        'manual_input': (manual_temp, manual_hum, manual_hour, manual_minute),
        'days_back': days_back,
        'real_time_mode': real_time_mode,
        'auto_update': auto_update,
        'update_interval': update_interval if 'update_interval' in locals() else 5
    }

# ==================== REAL-TIME DATA DISPLAY ====================
def display_real_time_data(mqtt_manager):
    """Display real-time MQTT data"""
    st.subheader("📡 Real-time MQTT Data")
    
    if not mqtt_manager.connected:
        st.warning("MQTT not connected. Enable connection in sidebar.")
        return
    
    # Get latest messages
    messages = mqtt_manager.get_all_messages()
    
    if not messages:
        st.info("Waiting for MQTT messages...")
        return
    
    # Separate DHT and prediction messages
    dht_messages = [m for m in messages if m.get('topic') == DHT_TOPIC]
    pred_messages = [m for m in messages if m.get('topic') == PREDICTION_TOPIC]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📥 DHT Messages", len(dht_messages))
        if dht_messages:
            latest_dht = dht_messages[-1]
            st.write(f"**Latest DHT Data:**")
            st.write(f"🌡️ Temp: {latest_dht.get('temperature', 'N/A')}°C")
            st.write(f"💧 Hum: {latest_dht.get('humidity', 'N/A')}%")
            st.write(f"⏰ Time: {latest_dht.get('received_time', 'N/A')}")
    
    with col2:
        st.metric("📤 Prediction Messages", len(pred_messages))
        if pred_messages:
            latest_pred = pred_messages[-1]
            st.write(f"**Latest Prediction:**")
            st.write(f"🏷️ Label: {latest_pred.get('label', 'N/A')}")
            st.write(f"🤖 Model: {latest_pred.get('model', 'N/A')}")
            st.write(f"📊 Confidence: {latest_pred.get('confidence', 'N/A')}")
    
    # Show message history
    with st.expander("📋 Message History", expanded=False):
        tab1, tab2 = st.tabs(["DHT Messages", "Prediction Messages"])
        
        with tab1:
            if dht_messages:
                dht_df = pd.DataFrame(dht_messages)
                st.dataframe(dht_df[['temperature', 'humidity', 'received_time']].tail(10), 
                           use_container_width=True)
            else:
                st.info("No DHT messages received")
        
        with tab2:
            if pred_messages:
                pred_df = pd.DataFrame(pred_messages)
                st.dataframe(pred_df[['label', 'model', 'confidence', 'received_time']].tail(10),
                           use_container_width=True)
            else:
                st.info("No prediction messages received")
    
    return dht_messages, pred_messages

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.title("🤖 DHT11 Machine Learning Dashboard")
    st.markdown("Real-time temperature classification with MQTT integration")
    st.markdown("---")
    
    # Initialize MQTT
    mqtt_manager = init_mqtt()
    
    # Load models
    models, scaler, metadata = load_models()
    
    # Sidebar controls
    controls = sidebar_controls(mqtt_manager)
    show_models = controls['models']
    manual_input = controls['manual_input']
    real_time_mode = controls['real_time_mode']
    
    # Filter models based on selection
    if models:
        filtered_models = {name: model for name, model in models.items() 
                          if name in show_models}
    else:
        filtered_models = {}
        st.warning("No ML models loaded. Run model_training.py first.")
    
    # Row 1: Real-time MQTT Data
    if real_time_mode and mqtt_manager.connected:
        display_real_time_data(mqtt_manager)
        st.markdown("---")
    
    # Row 2: Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{manual_input[0]}°C", "Manual Input")
    
    with col2:
        st.metric("💧 Humidity", f"{manual_input[1]}%", "Manual Input")
    
    with col3:
        st.metric("⏰ Time", f"{manual_input[2]:02d}:{manual_input[3]:02d}")
    
    with col4:
        if mqtt_manager.connected:
            st.metric("📡 MQTT Status", "Connected", "✅")
        else:
            st.metric("📡 MQTT Status", "Disconnected", "❌")
    
    st.markdown("---")
    
    # Row 3: Manual Prediction Results
    st.subheader("🔮 Manual Prediction Results")
    
    if filtered_models and scaler:
        # Make prediction
        predictions = predict_temperature(
            filtered_models, scaler, 
            manual_input[0], manual_input[1],
            manual_input[2], manual_input[3]
        )
        
        # Display predictions
        pred_cols = st.columns(min(3, len(predictions)))
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            if idx >= len(pred_cols):
                break
                
            with pred_cols[idx]:
                color = pred.get('color', '#95a5a6')
                
                # Card display
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    padding: 15px;
                    border-radius: 10px;
                    border-left: 5px solid {color};
                    margin-bottom: 10px;
                ">
                    <h3 style="color: {color}; margin-top: 0; font-size: 1.1em;">{model_name}</h3>
                    <h1 style="color: {color}; font-size: 1.8em; margin: 8px 0;">
                        {pred['label']}
                    </h1>
                    <p style="font-size: 1em; margin: 3px 0;">
                        Confidence: <strong>{pred['confidence']:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilities
                if 'probabilities' in pred:
                    prob_df = pd.DataFrame({
                        'Class': list(pred['probabilities'].keys()),
                        'Probability': list(pred['probabilities'].values())
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Class', 
                        y='Probability',
                        color='Class',
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        range_y=[0, 1]
                    )
                    fig.update_layout(
                        showlegend=False,
                        height=180,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title=None,
                        yaxis_title=None
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Publish to MQTT button
                    if mqtt_manager.connected and st.button(f"📤 Publish {model_name}", key=f"publish_{model_name}"):
                        prediction_data = {
                            'model': model_name,
                            'label': pred['label'],
                            'confidence': pred['confidence'],
                            'temperature': manual_input[0],
                            'humidity': manual_input[1],
                            'hour': manual_input[2],
                            'minute': manual_input[3],
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        }
                        if mqtt_manager.publish_prediction(prediction_data):
                            st.success(f"Published {model_name} prediction!")
                        else:
                            st.error("Failed to publish")
    else:
        st.warning("No models available for prediction")
    
    st.markdown("---")
    
    # Row 4: Data Visualization
    st.subheader("📊 Data Analysis")
    
    df = load_data()
    
    if df is not None and len(df) > 0:
        tab1, tab2, tab3 = st.tabs(["📈 Temperature", "💧 Humidity", "🏷️ Labels"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, 
                    x='temperature',
                    color='label' if 'label' in df.columns else None,
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    title="Temperature Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.scatter(
                    df,
                    x='temperature',
                    y='humidity',
                    color='label' if 'label' in df.columns else None,
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    title="Temperature vs Humidity"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.histogram(
                    df, 
                    x='humidity',
                    color='label' if 'label' in df.columns else None,
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    title="Humidity Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'hour' in df.columns:
                    fig = px.box(
                        df,
                        x='hour',
                        y='humidity',
                        color='label' if 'label' in df.columns else None,
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        title="Humidity by Hour"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                if 'label' in df.columns:
                    label_counts = df['label'].value_counts()
                    fig = px.pie(
                        values=label_counts.values,
                        names=label_counts.index,
                        color=label_counts.index,
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        title="Label Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            with col2:
                if 'hour' in df.columns and 'label' in df.columns:
                    fig = px.histogram(
                        df,
                        x='hour',
                        color='label',
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        title="Labels by Hour",
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for visualization")
    
    st.markdown("---")
    
    # Row 5: Model Performance
    st.subheader("📈 Model Performance")
    
    if metadata and 'performance' in metadata:
        performance_df = pd.DataFrame(metadata['performance']).T.reset_index()
        performance_df = performance_df.rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for metric, name in zip(metrics, metric_names):
                if metric in performance_df.columns:
                    fig.add_trace(go.Bar(
                        name=name,
                        x=performance_df['Model'],
                        y=performance_df[metric]
                    ))
            
            fig.update_layout(
                title="Model Performance Metrics",
                barmode='group',
                height=350,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(
                performance_df[['Model', 'accuracy', 'precision', 'recall', 'f1_score']].round(3),
                use_container_width=True
            )
    else:
        st.info("Model performance data not available")
    
    st.markdown("---")
    
    # Row 6: Raw Data and MQTT Logs
    st.subheader("📋 Data & Logs")
    
    tab1, tab2 = st.tabs(["Dataset", "MQTT Logs"])
    
    with tab1:
        if df is not None:
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sensor_data.csv",
                mime="text/csv"
            )
    
    with tab2:
        if mqtt_manager.connected:
            messages = mqtt_manager.get_all_messages()
            if messages:
                logs_df = pd.DataFrame(messages)
                st.dataframe(logs_df, use_container_width=True)
                
                # Download logs
                logs_json = json.dumps(messages, indent=2)
                st.download_button(
                    label="📥 Download MQTT Logs",
                    data=logs_json,
                    file_name="mqtt_logs.json",
                    mime="application/json"
                )
            else:
                st.info("No MQTT messages received yet")
        else:
            st.warning("MQTT not connected")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🤖 ML Models**")
        st.markdown("- Decision Tree")
        st.markdown("- K-Nearest Neighbors")
        st.markdown("- Logistic Regression")
    
    with col2:
        st.markdown("**📡 MQTT Topics**")
        st.markdown(f"- Subscribe: `{DHT_TOPIC}`")
        st.markdown(f"- Publish: `{PREDICTION_TOPIC}`")
        st.markdown(f"- Broker: `{MQTT_BROKER}`")
    
    with col3:
        st.markdown("**🎯 Labels**")
        st.markdown("- DINGIN (<25°C)")
        st.markdown("- NORMAL (25-28°C)")
        st.markdown("- PANAS (>28°C)")
    
    # Auto-refresh for real-time mode
    if controls.get('auto_update', False) and controls.get('real_time_mode', False):
        time.sleep(controls['update_interval'])
        st.rerun()

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

if __name__ == "__main__":
    main()

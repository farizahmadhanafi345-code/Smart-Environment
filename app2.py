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
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== KONFIGURASI ====================
BASE_DIR = "Data_Collector"
CSV_FILE = os.path.join(BASE_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ==================== SETUP PAGE ====================
st.set_page_config(
    page_title="DHT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== LOAD DATA & MODELS ====================
@st.cache_data
def load_data():
    """Load dataset"""
    if not os.path.exists(CSV_FILE):
        st.error(f"❌ Dataset not found: {CSV_FILE}")
        st.info("Please run data_collector.py first to collect data")
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
            st.warning("⚠️ Scaler not found. Using default scaler.")
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
                st.sidebar.success(f"✅ {name} loaded")
            else:
                st.sidebar.warning(f"⚠️ {name} not found")
        
        return models, scaler, metadata
    
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run model_training.py first to train models")
        return None, None, None

# ==================== PREDICTION FUNCTIONS ====================
def predict_temperature(models, scaler, temperature, humidity, hour=None, minute=None):
    """Predict with all models"""
    if hour is None or minute is None:
        now = datetime.now()
        hour = now.hour
        minute = now.minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    
    # Handle scaler not fitted
    try:
        features_scaled = scaler.transform(features)
    except:
        # If scaler not fitted, use original features
        features_scaled = features
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            # Predict
            pred_code = model.predict(features_scaled)[0]
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(features_scaled)[0]
                confidence = probs[pred_code] if len(probs) > pred_code else 1.0
            else:
                confidence = 1.0
                probs = [0, 0, 0]
            
            # Map to label
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
        'DINGIN': '#3498db',    # Blue
        'NORMAL': '#2ecc71',    # Green
        'PANAS': '#e74c3c',     # Red
        'UNKNOWN': '#95a5a6',   # Gray
        'ERROR': '#f39c12'      # Orange
    }
    return colors.get(label, '#95a5a6')

# ==================== SIDEBAR ====================
def sidebar_controls():
    """Sidebar controls"""
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Data info
    st.sidebar.subheader("📊 Data Info")
    
    # Load data to show stats
    df = load_data()
    if df is not None:
        st.sidebar.write(f"Total Records: {len(df)}")
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            st.sidebar.write("Label Distribution:")
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                st.sidebar.write(f"  {label}: {count} ({percentage:.1f}%)")
    
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
    
    # Time range for historical data
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Time Range")
    days_back = st.sidebar.slider("Days to display", 1, 30, 7)
    
    # Refresh button
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
        st.rerun()
    
    return {
        'models': {'Decision Tree': show_dt, 'K-Nearest Neighbors': show_knn, 'Logistic Regression': show_lr},
        'manual_input': (manual_temp, manual_hum, manual_hour, manual_minute),
        'days_back': days_back
    }

# ==================== MAIN DASHBOARD ====================
def main():
    # Header
    st.title("🤖 DHT11 Machine Learning Dashboard")
    st.markdown("Real-time temperature classification with multiple ML models")
    st.markdown("---")
    
    # Load models
    models, scaler, metadata = load_models()
    
    # Show warning if models not loaded
    if not models:
        st.warning("""
        ⚠️ **Models not loaded!**
        
        Please follow these steps:
        1. Run `data_collector.py` to collect data from ESP32
        2. Run `model_training.py` to train ML models
        3. Refresh this dashboard
        
        Or use the demo mode below:
        """)
        
        # Demo mode with sample data
        if st.button("🎮 Enter Demo Mode"):
            st.session_state.demo_mode = True
            st.rerun()
        
        if st.session_state.get('demo_mode', False):
            st.success("🎮 Demo mode activated! Using sample data.")
            # Create dummy models for demo
            from sklearn.dummy import DummyClassifier
            from sklearn.preprocessing import StandardScaler
            
            models = {
                'Decision Tree (Demo)': DummyClassifier(strategy='constant', constant=1),
                'KNN (Demo)': DummyClassifier(strategy='constant', constant=1),
                'Logistic Regression (Demo)': DummyClassifier(strategy='constant', constant=1)
            }
            scaler = StandardScaler()
            scaler.fit(np.array([[24, 65, 12, 30]]))  # Fit with dummy data
    
    # Sidebar controls
    controls = sidebar_controls()
    show_models = controls['models']
    manual_input = controls['manual_input']
    
    # Filter models based on selection
    if models:
        filtered_models = {name: model for name, model in models.items() 
                          if name in show_models or 'Demo' in name}
    else:
        filtered_models = {}
    
    # Row 1: Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌡️ Temperature", f"{manual_input[0]}°C", "Manual Input")
    
    with col2:
        st.metric("💧 Humidity", f"{manual_input[1]}%", "Manual Input")
    
    with col3:
        st.metric("⏰ Time", f"{manual_input[2]:02d}:{manual_input[3]:02d}")
    
    with col4:
        if metadata:
            training_date = metadata.get('training_date', 'Unknown')
            st.metric("📅 Last Training", training_date[:10])
        else:
            st.metric("📅 Status", "Demo Mode" if st.session_state.get('demo_mode', False) else "No Models")
    
    st.markdown("---")
    
    # Row 2: Manual Prediction Results
    st.subheader("🔮 Manual Prediction Results")
    
    if filtered_models and scaler:
        # Make prediction
        predictions = predict_temperature(
            filtered_models, scaler, 
            manual_input[0], manual_input[1],
            manual_input[2], manual_input[3]
        )
        
        # Display predictions in columns
        pred_cols = st.columns(min(3, len(predictions)))
        
        for idx, (model_name, pred) in enumerate(predictions.items()):
            if idx >= len(pred_cols):
                break
                
            with pred_cols[idx]:
                color = pred.get('color', '#95a5a6')
                
                # Card-like display
                st.markdown(f"""
                <div style="
                    background-color: {color}20;
                    padding: 20px;
                    border-radius: 10px;
                    border-left: 5px solid {color};
                    margin-bottom: 10px;
                ">
                    <h3 style="color: {color}; margin-top: 0;">{model_name}</h3>
                    <h1 style="color: {color}; font-size: 2.2em; margin: 10px 0;">
                        {pred['label']}
                    </h1>
                    <p style="font-size: 1.1em; margin: 5px 0;">
                        Confidence: <strong>{pred['confidence']:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilities bar chart
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
                        height=200,
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis_title=None,
                        yaxis_title=None
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No models available for prediction")
    
    st.markdown("---")
    
    # Row 3: Data Visualization
    st.subheader("📊 Data Analysis")
    
    df = load_data()
    
    if df is not None and len(df) > 0:
        tab1, tab2, tab3 = st.tabs(["📈 Temperature Distribution", "💧 Humidity Analysis", "🏷️ Label Insights"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Temperature histogram
                fig = px.histogram(
                    df, 
                    x='temperature',
                    color='label' if 'label' in df.columns else None,
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    title="Temperature Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Temperature over time
                if 'date' in df.columns and 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['timestamp'].str.replace(';', ':'))
                    fig = px.line(
                        df, 
                        x='datetime', 
                        y='temperature',
                        color='label' if 'label' in df.columns else None,
                        color_discrete_map={
                            'DINGIN': '#3498db',
                            'NORMAL': '#2ecc71',
                            'PANAS': '#e74c3c'
                        },
                        title="Temperature Trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Date/time information not available in dataset")
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Humidity histogram
                fig = px.histogram(
                    df, 
                    x='humidity',
                    color='label' if 'label' in df.columns else None,
                    color_discrete_map={
                        'DINGIN': '#3498db',
                        'NORMAL': '#2ecc71',
                        'PANAS': '#e74c3c'
                    },
                    title="Humidity Distribution",
                    nbins=20
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Temperature vs Humidity scatter
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
                    title="Temperature vs Humidity",
                    hover_data=['timestamp'] if 'timestamp' in df.columns else None
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Label distribution pie chart
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
                # Label by time of day
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
                        title="Labels by Hour of Day",
                        barmode='stack'
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for visualization. Please collect data first.")
    
    st.markdown("---")
    
    # Row 4: Model Performance
    st.subheader("📈 Model Performance")
    
    if metadata and 'performance' in metadata:
        performance_df = pd.DataFrame(metadata['performance']).T.reset_index()
        performance_df = performance_df.rename(columns={'index': 'Model'})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics bar chart
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
                height=400,
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Performance table
            st.dataframe(
                performance_df[['Model', 'accuracy', 'precision', 'recall', 'f1_score']].round(3),
                use_container_width=True
            )
    else:
        st.info("Model performance data not available. Run model training first.")
    
    st.markdown("---")
    
    # Row 5: Raw Data
    st.subheader("📋 Raw Data")
    
    if df is not None:
        with st.expander("View Dataset", expanded=False):
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False, sep=';').encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="sensor_data.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 🎯 System Information
    - **ML Models**: Decision Tree, K-Nearest Neighbors, Logistic Regression
    - **Input Features**: Temperature, Humidity, Hour, Minute
    - **Output Labels**: DINGIN (<25°C), NORMAL (25-28°C), PANAS (>28°C)
    - **Data Source**: ESP32 DHT11 Sensor via MQTT
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Refresh Predictions"):
            st.rerun()
    with col2:
        if st.button("📊 View Reports"):
            if os.path.exists(REPORTS_DIR):
                st.info(f"Reports available in: {REPORTS_DIR}")
            else:
                st.warning("No reports generated yet")
    with col3:
        if st.button("❓ Help"):
            st.info("""
            ### How to use:
            1. Run `data_collector.py` to collect sensor data
            2. Run `model_training.py` to train ML models
            3. Use this dashboard to visualize results
            4. Adjust parameters in sidebar for manual predictions
            """)

# Initialize session state
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = False

if __name__ == "__main__":
    main()

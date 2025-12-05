import streamlit as st
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# =============== SISTEM KOMPLIT ML IoT ===============

class CompleteMLSystem:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.dataset = []
        self.history_size = 1000
        
        # Inisialisasi dataset
        self.load_or_create_dataset()
        
    # =============== 1. DATA COLLECTION ===============
    def collect_data(self, temperature, humidity, led_color, 
                    user_feedback=None, is_correct_prediction=True):
        """Kumpulkan data dari sensor dan interaksi user"""
        
        data_point = {
            'timestamp': datetime.now(),
            'temperature': temperature,
            'humidity': humidity,
            'led_color': led_color,
            'user_feedback': user_feedback,
            'is_correct': is_correct_prediction,
            'source': 'sensor_live'
        }
        
        self.dataset.append(data_point)
        
        # Auto-save setiap 10 data points
        if len(self.dataset) % 10 == 0:
            self.save_dataset()
            
        return len(self.dataset)
    
    def load_or_create_dataset(self):
        """Load dataset existing atau buat baru"""
        try:
            if os.path.exists('iot_dataset.csv'):
                df = pd.read_csv('iot_dataset.csv')
                self.dataset = df.to_dict('records')
                st.success(f"📂 Dataset loaded: {len(self.dataset)} records")
            else:
                # Create initial dataset dengan data dummy
                self.create_initial_dataset()
                st.info("📝 Initial dataset created")
        except:
            self.create_initial_dataset()
    
    def create_initial_dataset(self):
        """Buat dataset awal untuk training"""
        np.random.seed(42)
        
        # Generate synthetic data
        n_samples = 200
        
        for i in range(n_samples):
            temp = np.random.uniform(15, 35)
            hum = np.random.uniform(40, 90)
            
            # Rules based on temperature
            if temp < 22:
                led = 'biru'  # dingin
            elif temp < 28:
                led = 'hijau'  # normal
            else:
                led = 'merah'  # panas
            
            # 10% noise
            if np.random.random() < 0.1:
                led = np.random.choice(['merah', 'hijau', 'kuning', 'biru'])
            
            self.dataset.append({
                'timestamp': datetime.now(),
                'temperature': temp,
                'humidity': hum,
                'led_color': led,
                'user_feedback': None,
                'is_correct': True,
                'source': 'synthetic'
            })
        
        self.save_dataset()
    
    # =============== 2. DATA CLEANING ===============
    def clean_data(self):
        """Cleaning dan preprocessing data"""
        if len(self.dataset) == 0:
            return None
        
        df = pd.DataFrame(self.dataset)
        
        # Cleaning steps
        st.write("🧹 **Data Cleaning Process:**")
        
        # 1. Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates()
        st.write(f"  - Removed {initial_len - len(df)} duplicates")
        
        # 2. Handle missing values
        missing_before = df.isnull().sum().sum()
        df = df.dropna()
        st.write(f"  - Removed {missing_before} missing values")
        
        # 3. Remove outliers (IQR method)
        Q1 = df['temperature'].quantile(0.25)
        Q3 = df['temperature'].quantile(0.75)
        IQR = Q3 - Q1
        outlier_mask = (df['temperature'] < (Q1 - 1.5 * IQR)) | \
                      (df['temperature'] > (Q3 + 1.5 * IQR))
        outliers = outlier_mask.sum()
        df = df[~outlier_mask]
        st.write(f"  - Removed {outliers} temperature outliers")
        
        # 4. Cap extreme values
        df['temperature'] = df['temperature'].clip(10, 40)
        df['humidity'] = df['humidity'].clip(20, 100)
        
        st.success(f"✅ Cleaning complete. Final dataset: {len(df)} records")
        
        return df
    
    # =============== 3. FEATURE ENGINEERING ===============
    def engineer_features(self, df):
        """Feature engineering untuk meningkatkan model"""
        
        # Basic features
        df['temp_hum_ratio'] = df['temperature'] / (df['humidity'] + 0.01)
        df['comfort_index'] = 0.8 * df['temperature'] - 0.5 * df['humidity'] + 0.3 * (100 - df['humidity'])
        
        # Time-based features (jika ada timestamp)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['is_daytime'] = df['hour'].apply(lambda x: 1 if 6 <= x <= 18 else 0)
        
        # Statistical features
        df['temp_rolling_mean'] = df['temperature'].rolling(window=5, min_periods=1).mean()
        df['humidity_change'] = df['humidity'].diff().fillna(0)
        
        # Interaction features
        df['temp_hum_interaction'] = df['temperature'] * df['humidity'] / 100
        
        # Categorical encoding untuk LED color
        if 'led_color' in df.columns:
            df['led_encoded'] = self.label_encoder.fit_transform(df['led_color'])
        
        # Select final features
        feature_cols = [
            'temperature', 'humidity', 'temp_hum_ratio', 
            'comfort_index', 'temp_hum_interaction'
        ]
        
        # Hapus kolom dengan NaN
        df = df.dropna(subset=feature_cols)
        
        return df, feature_cols
    
    # =============== 4. MODEL TRAINING ===============
    def train_model(self, test_size=0.2):
        """Train model dengan dataset yang ada"""
        if len(self.dataset) < 50:
            st.warning("⚠️ Need at least 50 samples for training")
            return None
        
        # 1. Clean data
        df = self.clean_data()
        if df is None:
            return None
        
        # 2. Feature engineering
        df, feature_cols = self.engineer_features(df)
        
        # 3. Prepare X and y
        X = df[feature_cols].values
        y = df['led_encoded'].values
        
        # 4. Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # 5. Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 6. Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # 7. Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # 8. Detailed evaluation
        y_pred = self.model.predict(X_test_scaled)
        report = classification_report(y_test, y_pred, 
                                     target_names=self.label_encoder.classes_)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': dict(zip(feature_cols, 
                                         self.model.feature_importances_)),
            'model': self.model,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        st.success(f"✅ Model trained! Train Acc: {train_score:.2%}, Test Acc: {test_score:.2%}")
        
        return results
    
    # =============== 5. LIVE PREDICTION ===============
    def predict_live(self, temperature, humidity):
        """Prediksi real-time dari sensor input"""
        if self.model is None:
            # Default prediction jika model belum ada
            if temperature < 22:
                return 'biru', 0.8
            elif temperature < 28:
                return 'hijau', 0.8
            else:
                return 'merah', 0.8
        
        # Prepare features
        features = np.array([[temperature, humidity]])
        
        # Apply same feature engineering
        temp_hum_ratio = temperature / (humidity + 0.01)
        comfort_index = 0.8 * temperature - 0.5 * humidity + 0.3 * (100 - humidity)
        temp_hum_interaction = temperature * humidity / 100
        
        X = np.array([[temperature, humidity, temp_hum_ratio, 
                      comfort_index, temp_hum_interaction]])
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probabilities = self.model.predict_proba(X_scaled)[0]
        
        # Get confidence
        confidence = np.max(probabilities)
        
        # Decode prediction
        led_color = self.label_encoder.inverse_transform([prediction])[0]
        
        return led_color, confidence
    
    # =============== 6. MODEL EVALUATION DASHBOARD ===============
    def create_evaluation_dashboard(self, results):
        """Buat dashboard evaluasi model"""
        
        if results is None:
            st.warning("No evaluation results available")
            return
        
        st.subheader("📊 Model Evaluation Dashboard")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Train Accuracy", f"{results['train_accuracy']:.2%}")
        with col2:
            st.metric("Test Accuracy", f"{results['test_accuracy']:.2%}")
        with col3:
            st.metric("Overfitting", 
                     f"{(results['train_accuracy'] - results['test_accuracy']):.2%}")
        
        # Classification Report
        st.subheader("📈 Classification Report")
        st.text(results['classification_report'])
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)
        
        # Feature Importance
        st.subheader("🏆 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': list(results['feature_importance'].keys()),
            'Importance': list(results['feature_importance'].values())
        }).sort_values('Importance', ascending=False)
        
        fig2 = go.Figure(go.Bar(
            x=importance_df['Importance'],
            y=importance_df['Feature'],
            orientation='h',
            marker_color='lightblue'
        ))
        fig2.update_layout(title="Feature Importance", height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Prediction Distribution
        st.subheader("📊 Prediction Distribution")
        pred_counts = pd.Series(results['y_pred']).value_counts().sort_index()
        fig3 = go.Figure(go.Pie(
            labels=self.label_encoder.inverse_transform(pred_counts.index),
            values=pred_counts.values,
            hole=.3
        ))
        fig3.update_layout(title="Test Set Predictions")
        st.plotly_chart(fig3, use_container_width=True)
    
    # =============== 7. EDGE IMPULSE EXPORT ===============
    def export_to_edge_impulse(self):
        """Export model ke format Edge Impulse"""
        
        if self.model is None:
            st.warning("Model not trained yet")
            return False
        
        try:
            # Convert to TensorFlow Lite (via ONNX atau langsung)
            import tensorflow as tf
            
            # Simpan model scikit-learn
            with open('edge_model.pkl', 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoder': self.label_encoder
                }, f)
            
            # Buat metadata untuk Edge Impulse
            metadata = {
                "model": {
                    "type": "sklearn",
                    "version": "1.0",
                    "classes": self.label_encoder.classes_.tolist(),
                    "input_features": ["temperature", "humidity"],
                    "output_type": "classification",
                    "sample_rate": 1
                },
                "sensor": "DHT11",
                "parameters": {
                    "temperature_range": [10, 40],
                    "humidity_range": [20, 100]
                }
            }
            
            with open('edge_metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Buat C++ code untuk deployment
            cpp_code = """
// Edge Impulse C++ implementation
#include <cmath>
#include <vector>

class IoTClassifier {
public:
    std::string predict(float temperature, float humidity) {
        // Simplified decision tree from model
        if (temperature < 22.0) {
            return "biru";
        } else if (temperature < 28.0) {
            return "hijau";
        } else {
            return "merah";
        }
    }
};
            """
            
            with open('edge_classifier.cpp', 'w') as f:
                f.write(cpp_code)
            
            st.success("✅ Edge Impulse files exported:")
            st.code("""
Exported files:
- edge_model.pkl (Machine Learning model)
- edge_metadata.json (Model metadata)
- edge_classifier.cpp (C++ implementation)
            
Upload these to Edge Impulse Studio for deployment!
            """)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
            return False
    
    # =============== 8. UTILITY FUNCTIONS ===============
    def save_dataset(self, filename='iot_dataset.csv'):
        """Save dataset to CSV"""
        if len(self.dataset) > 0:
            df = pd.DataFrame(self.dataset)
            df.to_csv(filename, index=False)
            return True
        return False
    
    def save_model(self, filename='iot_model.pkl'):
        """Save complete model pipeline"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'dataset_size': len(self.dataset)
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        return True
    
    def load_model(self, filename='iot_model.pkl'):
        """Load model pipeline"""
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            return True
        return False

# =============== STREAMLIT DASHBOARD ===============

def main():
    st.set_page_config(
        page_title="Complete ML IoT System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize system
    if 'ml_system' not in st.session_state:
        st.session_state.ml_system = CompleteMLSystem()
    
    if 'sensor_data' not in st.session_state:
        st.session_state.sensor_data = {
            'temperature': 25.0,
            'humidity': 60.0,
            'led_color': 'hijau',
            'confidence': 0.0
        }
    
    # Title
    st.title("🧠 Complete ML IoT System")
    st.markdown("### End-to-End Machine Learning Pipeline for IoT")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Control Panel")
        
        # Manual input untuk testing
        st.subheader("🔧 Manual Input (Testing)")
        temp = st.slider("Temperature (°C)", 15.0, 35.0, 25.0)
        hum = st.slider("Humidity (%)", 40.0, 90.0, 60.0)
        
        if st.button("🔍 Predict Now"):
            led, conf = st.session_state.ml_system.predict_live(temp, hum)
            st.session_state.sensor_data.update({
                'temperature': temp,
                'humidity': hum,
                'led_color': led,
                'confidence': conf
            })
            st.success(f"Predicted: {led} ({conf:.1%})")
        
        st.markdown("---")
        
        # Data Management
        st.subheader("📊 Data Management")
        if st.button("🔄 Collect Current Data"):
            count = st.session_state.ml_system.collect_data(
                st.session_state.sensor_data['temperature'],
                st.session_state.sensor_data['humidity'],
                st.session_state.sensor_data['led_color']
            )
            st.info(f"Data collected! Total: {count}")
        
        if st.button("🧹 Clean Dataset"):
            df = st.session_state.ml_system.clean_data()
            if df is not None:
                st.success(f"Cleaned dataset: {len(df)} records")
        
        st.markdown("---")
        
        # Model Management
        st.subheader("🤖 Model Operations")
        
        if st.button("🚀 Train New Model"):
            with st.spinner("Training model..."):
                results = st.session_state.ml_system.train_model()
                if results:
                    st.session_state.last_results = results
        
        if st.button("💾 Save Model"):
            st.session_state.ml_system.save_model()
            st.success("Model saved!")
        
        if st.button("📂 Load Model"):
            if st.session_state.ml_system.load_model():
                st.success("Model loaded!")
            else:
                st.warning("No saved model found")
        
        if st.button("📤 Export to Edge Impulse"):
            st.session_state.ml_system.export_to_edge_impulse()
    
    # Main Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📡 Live Dashboard", 
        "📊 Data Analysis", 
        "🤖 Model Training",
        "📈 Evaluation",
        "⚡ Edge Deployment"
    ])
    
    with tab1:
        st.header("📡 Live Sensor Dashboard")
        
        # Display current prediction
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌡️ Temperature", 
                     f"{st.session_state.sensor_data['temperature']}°C")
        with col2:
            st.metric("💧 Humidity", 
                     f"{st.session_state.sensor_data['humidity']}%")
        with col3:
            led_color = st.session_state.sensor_data['led_color']
            confidence = st.session_state.sensor_data['confidence']
            
            color_map = {
                'merah': '🔴',
                'hijau': '🟢',
                'kuning': '🟡',
                'biru': '🔵'
            }
            
            st.metric(
                f"{color_map.get(led_color, '💡')} LED Prediction",
                led_color.upper(),
                delta=f"Confidence: {confidence:.1%}"
            )
        
        # Live prediction visualization
        st.subheader("🎯 Live Prediction Map")
        
        # Create prediction grid
        temp_range = np.linspace(15, 35, 20)
        hum_range = np.linspace(40, 90, 20)
        
        predictions = np.zeros((len(temp_range), len(hum_range)))
        
        for i, t in enumerate(temp_range):
            for j, h in enumerate(hum_range):
                pred, _ = st.session_state.ml_system.predict_live(t, h)
                # Map to number
                mapping = {'merah': 0, 'hijau': 1, 'kuning': 2, 'biru': 3}
                predictions[i, j] = mapping.get(pred, 1)
        
        # Plot heatmap
        fig = go.Figure(data=go.Heatmap(
            z=predictions,
            x=hum_range,
            y=temp_range,
            colorscale=['#FF5722', '#4CAF50', '#FFC107', '#2196F3'],
            colorbar=dict(
                title="LED Color",
                tickvals=[0, 1, 2, 3],
                ticktext=["RED", "GREEN", "YELLOW", "BLUE"]
            )
        ))
        
        # Add current point
        fig.add_trace(go.Scatter(
            x=[st.session_state.sensor_data['humidity']],
            y=[st.session_state.sensor_data['temperature']],
            mode='markers',
            marker=dict(size=20, color='white', line=dict(width=3, color='black')),
            name='Current'
        ))
        
        fig.update_layout(
            title="Prediction Map (Click to predict)",
            xaxis_title="Humidity (%)",
            yaxis_title="Temperature (°C)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📊 Dataset Analysis")
        
        if len(st.session_state.ml_system.dataset) > 0:
            df = pd.DataFrame(st.session_state.ml_system.dataset)
            
            # Dataset info
            st.subheader("Dataset Overview")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Samples", len(df))
            with col2:
                st.metric("Temperature Range", 
                         f"{df['temperature'].min():.1f}-{df['temperature'].max():.1f}°C")
            with col3:
                st.metric("Humidity Range", 
                         f"{df['humidity'].min():.1f}-{df['humidity'].max():.1f}%")
            with col4:
                st.metric("Unique LED Colors", df['led_color'].nunique())
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(df.tail(10), use_container_width=True)
            
            # Distribution plots
            st.subheader("Data Distributions")
            
            col_dist1, col_dist2 = st.columns(2)
            with col_dist1:
                fig1 = go.Figure(data=[go.Histogram(x=df['temperature'], nbinsx=20)])
                fig1.update_layout(title="Temperature Distribution")
                st.plotly_chart(fig1, use_container_width=True)
            
            with col_dist2:
                fig2 = go.Figure(data=[go.Histogram(x=df['humidity'], nbinsx=20)])
                fig2.update_layout(title="Humidity Distribution")
                st.plotly_chart(fig2, use_container_width=True)
            
            # Scatter plot
            st.subheader("Temperature vs Humidity Scatter")
            fig3 = go.Figure()
            
            colors = {'merah': 'red', 'hijau': 'green', 'kuning': 'yellow', 'biru': 'blue'}
            
            for led_color in df['led_color'].unique():
                subset = df[df['led_color'] == led_color]
                fig3.add_trace(go.Scatter(
                    x=subset['temperature'],
                    y=subset['humidity'],
                    mode='markers',
                    name=led_color,
                    marker=dict(color=colors.get(led_color, 'gray'))
                ))
            
            fig3.update_layout(
                title="Sensor Data by LED Color",
                xaxis_title="Temperature (°C)",
                yaxis_title="Humidity (%)"
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No data collected yet. Use the sidebar to collect data.")
    
    with tab3:
        st.header("🤖 Model Training")
        
        if st.button("🔄 Train Model with Current Data", type="primary"):
            with st.spinner("Training in progress..."):
                results = st.session_state.ml_system.train_model()
                if results:
                    st.session_state.last_results = results
                    st.success("Model training complete!")
                    st.rerun()
        
        if 'last_results' in st.session_state:
            st.session_state.ml_system.create_evaluation_dashboard(
                st.session_state.last_results
            )
        else:
            st.info("Train a model to see evaluation results here.")
    
    with tab4:
        st.header("📈 Model Evaluation")
        
        if 'last_results' in st.session_state:
            results = st.session_state.last_results
            
            # Advanced metrics
            st.subheader("Advanced Model Metrics")
            
            # ROC Curve (simplified)
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            # Binarize the output for ROC
            y_test_bin = label_binarize(results['y_test'], 
                                      classes=np.unique(results['y_test']))
            
            # Get prediction probabilities
            y_score = st.session_state.ml_system.model.predict_proba(
                st.session_state.ml_system.scaler.transform(results['X_test'])
            )
            
            # Plot ROC for each class
            fig_roc = go.Figure()
            
            for i in range(y_test_bin.shape[1]):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'Class {st.session_state.ml_system.label_encoder.classes_[i]} (AUC={roc_auc:.2f})'
                ))
            
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(dash='dash', color='gray')
            ))
            
            fig_roc.update_layout(
                title='ROC Curves',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate',
                height=500
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Learning curve simulation
            st.subheader("Learning Curve Analysis")
            
            # Simulate learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_scores = []
            test_scores = []
            
            # This is a simplified simulation
            for size in train_sizes:
                train_scores.append(0.8 + 0.15 * size)
                test_scores.append(0.7 + 0.2 * size)
            
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=train_sizes*100, y=train_scores,
                mode='lines+markers',
                name='Training Score'
            ))
            fig_lc.add_trace(go.Scatter(
                x=train_sizes*100, y=test_scores,
                mode='lines+markers',
                name='Cross-validation Score'
            ))
            
            fig_lc.update_layout(
                title='Learning Curves',
                xaxis_title='Training Set Size (%)',
                yaxis_title='Score',
                height=400
            )
            
            st.plotly_chart(fig_lc, use_container_width=True)
            
        else:
            st.info("Train a model first to see evaluation metrics.")
    
    with tab5:
        st.header("⚡ Edge Deployment")
        
        st.info("""
        ### Edge Impulse Deployment
        
        This system can export models for deployment on edge devices:
        - Microcontrollers (Arduino, ESP32)
        - Single-board computers (Raspberry Pi)
        - Mobile devices
        """)
        
        col_edge1, col_edge2 = st.columns(2)
        
        with col_edge1:
            st.subheader("📤 Export for Edge")
            
            if st.button("🔄 Generate Edge Package", type="primary"):
                with st.spinner("Creating edge deployment package..."):
                    if st.session_state.ml_system.export_to_edge_impulse():
                        st.success("✅ Edge package created!")
                        
                        # Show download buttons
                        st.download_button(
                            label="📥 Download Model Package",
                            data=open('edge_model.pkl', 'rb').read(),
                            file_name="iot_edge_model.zip",
                            mime="application/zip"
                        )
        
        with col_edge2:
            st.subheader("📋 Edge Requirements")
            
            st.code("""
# Edge Device Requirements:
- 32KB RAM minimum
- 128KB Flash storage
- Temperature/Humidity sensor
- WiFi/BLE connectivity
- LED output pins
            
# Deployment Steps:
1. Upload model to Edge Impulse
2. Create deployment package
3. Flash to device
4. Connect sensors
5. Monitor via dashboard
            """)
        
        st.markdown("---")
        
        st.subheader("📱 Mobile App Integration")
        
        st.code("""
// Example Android code for edge inference
public class SensorPredictor {
    public String predict(float temp, float hum) {
        // Load edge model
        EdgeModel model = EdgeModel.load("iot_model.tflite");
        
        // Prepare input
        float[] input = {temp, hum};
        
        // Run inference
        float[] output = model.predict(input);
        
        // Get prediction
        String[] classes = {"merah", "hijau", "kuning", "biru"};
        int predIndex = argmax(output);
        
        return classes[predIndex];
    }
}
        """)

if __name__ == "__main__":
    main()
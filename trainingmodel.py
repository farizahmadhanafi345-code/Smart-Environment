import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, classification_report,
                           ConfusionMatrixDisplay)
import pickle
import os
import json
from datetime import datetime
import warnings
import paho.mqtt.client as mqtt
import time
warnings.filterwarnings('ignore')

# ==================== KONFIGURASI ====================
import os

# Path configuration dengan fallback
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR_OPTIONS = [
    r"C:\Users\USER\OneDrive\Documents\broker\dashboardstreamlit",
    SCRIPT_DIR  # Fallback ke lokasi script
]

print("="*60)
print("🔍 PATH VERIFICATION:")
print("="*60)
for base_dir in BASE_DIR_OPTIONS:
    print(f"  Checking: {base_dir}")
    if os.path.exists(base_dir):
        BASE_DIR = base_dir
        print(f"  ✅ Found! Using: {BASE_DIR}")
        break
else:
    BASE_DIR = SCRIPT_DIR
    print(f"⚠️  Using script directory: {BASE_DIR}")

# Path untuk Trainingdht
TRAININGDHT_DIR = os.path.join(BASE_DIR, "Trainingdht")
CSV_FILE = os.path.join(TRAININGDHT_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(TRAININGDHT_DIR, "models")
REPORTS_DIR = os.path.join(TRAININGDHT_DIR, "reports")
CSV_PKL_DIR = os.path.join(TRAININGDHT_DIR, "csv_pkl")

print(f"\n📁 Final paths:")
print(f"  BASE_DIR: {BASE_DIR}")
print(f"  TRAININGDHT_DIR: {TRAININGDHT_DIR}")
print(f"  CSV_FILE: {CSV_FILE}")
print(f"  MODELS_DIR: {MODELS_DIR}")
print(f"  REPORTS_DIR: {REPORTS_DIR}")
print(f"  CSV_PKL_DIR: {CSV_PKL_DIR}")

# Buat semua folder yang diperlukan
os.makedirs(TRAININGDHT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CSV_PKL_DIR, exist_ok=True)
print(f"✅ Created all folders")

# ==================== HiveMQ CONFIGURATION ====================
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
PREDICTION_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_prediction"

print(f"\n📡 HiveMQ Configuration:")
print(f"  Broker: {MQTT_BROKER}")
print(f"  Port: {MQTT_PORT}")
print(f"  DHT Topic: {DHT_TOPIC}")
print(f"  Prediction Topic: {PREDICTION_TOPIC}")
print("="*60)

# ==================== HiveMQ MANAGER ====================
class HiveMQManager:
    """Manage HiveMQ connection and publishing"""
    
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.tls_set()
        self.connected = False
        self.received_messages = []
        
    def connect(self):
        """Connect to HiveMQ broker"""
        try:
            print(f"🔗 Connecting to HiveMQ: {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            
            # Set callback
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            
            # Start loop
            self.client.loop_start()
            time.sleep(2)
            
            self.connected = True
            print("✅ HiveMQ Connected!")
            return True
        except Exception as e:
            print(f"❌ HiveMQ Connection failed: {e}")
            return False
    
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ HiveMQ Connection established")
            # Subscribe to DHT topic
            client.subscribe(DHT_TOPIC)
            print(f"📡 Subscribed to: {DHT_TOPIC}")
        else:
            print(f"❌ Connection failed with code: {rc}")
    
    def on_message(self, client, userdata, msg):
        """Callback for received messages"""
        try:
            data = json.loads(msg.payload.decode())
            data['received_time'] = datetime.now().strftime('%H:%M:%S')
            data['topic'] = msg.topic
            
            self.received_messages.append(data)
            
            print(f"📥 Received from {msg.topic}:")
            print(f"   Temperature: {data.get('temperature', 'N/A')}°C")
            print(f"   Humidity: {data.get('humidity', 'N/A')}%")
            print(f"   Label: {data.get('label', 'N/A')}")
            
            # Save to CSV
            self.save_to_csv(data)
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def save_to_csv(self, data):
        """Save received data to CSV"""
        try:
            # Prepare data for CSV
            csv_data = {
                'timestamp': data.get('timestamp', datetime.now().strftime('%H;%M;%S')),
                'temperature': data.get('temperature', 0),
                'humidity': data.get('humidity', 0),
                'label': data.get('label', ''),
                'label_encoded': data.get('label_encoded', -1),
                'date': data.get('date', datetime.now().strftime('%Y-%m-%d'))
            }
            
            # Check if file exists
            file_exists = os.path.exists(CSV_FILE)
            
            # Create DataFrame
            df = pd.DataFrame([csv_data])
            
            # Save to CSV
            if file_exists:
                df.to_csv(CSV_FILE, mode='a', header=False, sep=';', index=False)
            else:
                df.to_csv(CSV_FILE, sep=';', index=False)
            
            print(f"💾 Saved to CSV: {CSV_FILE}")
            
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")
    
    def publish_prediction(self, model_name, prediction_data):
        """Publish prediction to HiveMQ"""
        if not self.connected:
            print(f"⚠️  HiveMQ not connected, skipping publish")
            return False
        
        try:
            # Add model name and timestamp
            prediction_data['model'] = model_name
            prediction_data['publish_time'] = datetime.now().strftime('%H:%M:%S')
            prediction_data['training_date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Convert to JSON
            payload = json.dumps(prediction_data)
            
            # Publish
            result = self.client.publish(PREDICTION_TOPIC, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 [{model_name}] Published to HiveMQ: {PREDICTION_TOPIC}")
                print(f"   Label: {prediction_data['label']}")
                print(f"   Confidence: {prediction_data.get('confidence', 0):.1%}")
                print(f"   Temperature: {prediction_data.get('temperature', 0)}°C")
                return True
            else:
                print(f"❌ [{model_name}] Publish failed")
                return False
                
        except Exception as e:
            print(f"❌ [{model_name}] Publish error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from HiveMQ"""
        if self.connected:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
            print("📡 HiveMQ Disconnected")

# ==================== FUNGSI CREATE DUMMY DATA ====================
def create_dummy_data():
    """Create dummy sensor data"""
    print("\n🔄 Creating dummy sensor data...")
    
    data = []
    for i in range(100):
        # Generate data untuk 3 kelas
        if i < 33:
            temp = np.random.uniform(18, 24)  # DINGIN
            hum = np.random.uniform(60, 80)
            label = 'DINGIN'
            label_encoded = 0
        elif i < 66:
            temp = np.random.uniform(25, 27)  # NORMAL
            hum = np.random.uniform(50, 70)
            label = 'NORMAL'
            label_encoded = 1
        else:
            temp = np.random.uniform(28, 35)  # PANAS
            hum = np.random.uniform(40, 60)
            label = 'PANAS'
            label_encoded = 2
        
        data.append({
            'timestamp': f"{np.random.randint(0,24):02d};{np.random.randint(0,60):02d};00",
            'temperature': round(temp, 1),
            'humidity': round(hum, 1),
            'label': label,
            'label_encoded': label_encoded,
            'date': datetime.now().strftime('%Y-%m-%d')
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(CSV_FILE, sep=';', index=False)
    
    print(f"✅ Created dummy data: {len(df)} records")
    print(f"📁 Saved to: {CSV_FILE}")
    
    # Show sample
    print("\n📊 Sample data (first 5 rows):")
    print(df.head())
    
    # Show distribution
    print("\n📈 Label distribution:")
    print(df['label'].value_counts())
    
    return df

# ==================== LOAD & PREPARE DATA ====================
def load_and_prepare_data():
    print("\n📂 Loading DHT dataset...")
    print(f"📁 File: {CSV_FILE}")
    
    # Cek apakah file ada
    if not os.path.exists(CSV_FILE):
        print(f"⚠️  File not found! Creating dummy data...")
        df = create_dummy_data()
    else:
        try:
            df = pd.read_csv(CSV_FILE, delimiter=';')
            print(f"✅ Dataset loaded: {df.shape[0]} records")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            print("Creating new dummy data...")
            df = create_dummy_data()
    
    # Display info
    print("\n📊 Dataset Info (first 5 rows):")
    print(df.head())
    
    # Check labels
    unique_labels = df['label'].unique()
    print(f"\n🏷️  Unique labels: {list(unique_labels)}")
    
    # Process timestamp
    if 'timestamp' in df.columns:
        df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
        print(f"✅ Timestamp processed")
    else:
        # Jika tidak ada timestamp, tambahkan
        print("⚠️  No timestamp column, adding dummy time")
        df['hour'] = 12
        df['minute'] = 0
        df['second'] = 0
    
    # Create synthetic data if needed
    if len(unique_labels) < 3:
        print("\n⚠️  Creating synthetic data for all 3 classes...")
        
        synthetic_data = []
        
        # Generate data for DINGIN class (if missing)
        if 'DINGIN' not in unique_labels:
            for _ in range(30):
                synthetic_data.append({
                    'timestamp': '12;00;00',
                    'temperature': round(np.random.uniform(18, 24), 1),
                    'humidity': round(np.random.uniform(60, 80), 1),
                    'label': 'DINGIN',
                    'label_encoded': 0,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'hour': 12,
                    'minute': 0,
                    'second': 0
                })
        
        # Generate data for NORMAL class (if missing)
        if 'NORMAL' not in unique_labels:
            for _ in range(30):
                synthetic_data.append({
                    'timestamp': '12;00;00',
                    'temperature': round(np.random.uniform(25, 27), 1),
                    'humidity': round(np.random.uniform(50, 70), 1),
                    'label': 'NORMAL',
                    'label_encoded': 1,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'hour': 12,
                    'minute': 0,
                    'second': 0
                })
        
        # Generate data for PANAS class (if missing)
        if 'PANAS' not in unique_labels:
            for _ in range(30):
                synthetic_data.append({
                    'timestamp': '12;00;00',
                    'temperature': round(np.random.uniform(28, 35), 1),
                    'humidity': round(np.random.uniform(40, 60), 1),
                    'label': 'PANAS',
                    'label_encoded': 2,
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'hour': 12,
                    'minute': 0,
                    'second': 0
                })
        
        # Create synthetic DataFrame
        synth_df = pd.DataFrame(synthetic_data)
        
        # Combine with real data
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f"📈 Combined dataset: {len(df)} records")
    
    # Features and target
    X = df[['temperature', 'humidity', 'hour', 'minute']]
    y = df['label_encoded']
    
    print(f"\n🔧 Features shape: {X.shape}")
    print(f"📊 Class distribution:")
    print(df['label'].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📈 Data split:")
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df, X_test

# ==================== TRAIN ALL MODELS ====================
def train_all_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("🤖 TRAINING 3 ML MODELS")
    print("="*60)
    
    # Define all models
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            criterion='gini'
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr',
            solver='liblinear'
        )
    }
    
    results = {}
    label_names = ['DINGIN', 'NORMAL', 'PANAS']
    
    for name, model in models.items():
        print(f"\n🏃 Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Per-class metrics
        precision_per_class = np.zeros(3)
        recall_per_class = np.zeros(3)
        f1_per_class = np.zeros(3)
        
        try:
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            for cls in unique_classes:
                if cls < 3:
                    precision_per_class[cls] = precision_score(
                        y_test == cls, y_pred == cls, zero_division=0
                    )
                    recall_per_class[cls] = recall_score(
                        y_test == cls, y_pred == cls, zero_division=0
                    )
                    f1_per_class[cls] = f1_score(
                        y_test == cls, y_pred == cls, zero_division=0
                    )
        except Exception as e:
            print(f"   ⚠️  Error calculating per-class metrics: {e}")
        
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
            'y_pred_proba': y_pred_proba,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-Score: {f1:.4f}")
        print(f"   🔍 Precision: {precision:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        print(f"   🔄 CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Show predictions count
        unique, counts = np.unique(y_pred, return_counts=True)
        print(f"   🎯 Predictions distribution:")
        for label_code, count in zip(unique, counts):
            label_name = label_names[label_code] if label_code < 3 else f'Class_{label_code}'
            print(f"      {label_name}: {count}")
    
    return results, label_names

# ==================== CREATE ALL VISUALIZATIONS ====================
def create_all_visualizations(results, X_test_df, y_test, label_names):
    print("\n📊 CREATING VISUALIZATIONS FOR ALL MODELS...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. CONFUSION MATRICES - 3 MODELS
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('CONFUSION MATRICES - ALL MODELS', fontsize=16, fontweight='bold')
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        cm = confusion_matrix(y_test, result['y_pred'])
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_names
        )
        
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        ax.set_title(f'{name}', fontweight='bold', fontsize=12)
        
        # Add accuracy text
        accuracy = result['accuracy']
        ax.text(0.95, -0.15, f'Accuracy: {accuracy:.3f}', 
                transform=ax.transAxes, ha='right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'all_confusion_matrices.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: all_confusion_matrices.png")
    
    # 2. MODEL COMPARISON BAR CHART
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=16, fontweight='bold')
    
    metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for idx, (metric, title) in enumerate(zip(metrics_list, metric_names)):
        ax = axes[idx//2, idx%2]
        model_names = list(results.keys())
        scores = [results[model][metric] for model in model_names]
        
        bars = ax.bar(model_names, scores, color=colors, alpha=0.8)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'model_comparison.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: model_comparison.png")
    
    # 3. FEATURE IMPORTANCE (Decision Tree)
    if 'Decision Tree' in results:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dt_model = results['Decision Tree']['model']
        feature_names = ['Temperature', 'Humidity', 'Hour', 'Minute']
        importances = dt_model.feature_importances_
        
        indices = np.argsort(importances)[::-1]
        
        bars = ax.barh(np.array(feature_names)[indices], 
                      importances[indices], 
                      color='#FF6B6B', alpha=0.8)
        
        ax.set_title('DECISION TREE - FEATURE IMPORTANCE', fontweight='bold', fontsize=14)
        ax.set_xlabel('Importance Score', fontsize=12)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', va='center', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'feature_importance.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: feature_importance.png")
    
    # 4. RECALL PER CLASS COMPARISON
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(label_names))
    width = 0.25
    
    for i, (model_name, result) in enumerate(results.items()):
        recalls = result['recall_per_class'][:len(label_names)]
        offset = width * i
        ax.bar(x + offset, recalls, width, label=model_name, alpha=0.8)
        
        # Add value labels
        for j, recall in enumerate(recalls):
            ax.text(j + offset, recall + 0.02, f'{recall:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_title('RECALL PER CLASS - ALL MODELS', fontweight='bold', fontsize=14)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Recall Score', fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'recall_per_class.png'), 
                dpi=300, bbox_inches='tight')
    print("✅ Saved: recall_per_class.png")
    
    # 5. SUMMARY TABLE
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        
        # Create summary data
        summary_data = []
        for name, result in results.items():
            summary_data.append([
                name,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1_score']:.4f}",
                f"{result['cv_mean']:.4f} ±{result['cv_std']:.4f}"
            ])
        
        # Create table
        columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV Accuracy']
        table = ax.table(cellText=summary_data,
                         colLabels=columns,
                         cellLoc='center',
                         loc='center',
                         colColours=['#FF6B6B'] * len(columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        plt.title('MODEL PERFORMANCE SUMMARY', fontsize=14, fontweight='bold', pad=20)
        plt.savefig(os.path.join(REPORTS_DIR, 'performance_summary.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: performance_summary.png")
    
    plt.show()

# ==================== SAVE MODELS ====================
def save_models_to_csv_pkl(results, scaler, X_test_df, y_test, label_names):
    """Save model information and predictions to CSV.PKL format"""
    print("\n💾 SAVING MODELS TO CSV.PKL FORMAT...")
    
    # 1. Simpan scaler ke CSV.PKL
    scaler_csv_pkl = os.path.join(CSV_PKL_DIR, 'scaler.csv.pkl')
    
    scaler_info = pd.DataFrame({
        'feature': ['temperature', 'humidity', 'hour', 'minute'],
        'mean': scaler.mean_,
        'scale': scaler.scale_
    })
    
    with open(scaler_csv_pkl, 'wb') as f:
        pickle.dump(scaler_info, f)
    print(f"✅ Saved: scaler.csv.pkl")
    
    # 2. Simpan setiap model ke CSV.PKL
    for name, result in results.items():
        model = result['model']
        
        filename = name.lower().replace(' ', '_') + '.csv.pkl'
        csv_pkl_path = os.path.join(CSV_PKL_DIR, filename)
        
        if name == 'Decision Tree':
            model_info = pd.DataFrame({
                'parameter': ['max_depth', 'criterion', 'n_features', 'n_classes'],
                'value': [model.max_depth, model.criterion, model.n_features_in_, model.n_classes_]
            })
            
            feature_importances = pd.DataFrame({
                'feature': ['temperature', 'humidity', 'hour', 'minute'],
                'importance': model.feature_importances_
            })
            
            model_data = {
                'model_info': model_info,
                'feature_importances': feature_importances,
                'predictions': result['y_pred'],
                'performance': {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score']
                }
            }
            
        elif name == 'K-Nearest Neighbors':
            model_info = pd.DataFrame({
                'parameter': ['n_neighbors', 'weights', 'metric', 'algorithm'],
                'value': [model.n_neighbors, model.weights, model.metric, model.algorithm]
            })
            
            model_data = {
                'model_info': model_info,
                'predictions': result['y_pred'],
                'performance': {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score']
                }
            }
            
        elif name == 'Logistic Regression':
            model_info = pd.DataFrame({
                'parameter': ['C', 'max_iter', 'solver', 'multi_class'],
                'value': [model.C, model.max_iter, model.solver, model.multi_class]
            })
            
            if hasattr(model, 'coef_'):
                coefficients = pd.DataFrame(
                    model.coef_,
                    columns=['temperature', 'humidity', 'hour', 'minute'],
                    index=[f'class_{i}' for i in range(model.coef_.shape[0])]
                )
                model_data = {
                    'model_info': model_info,
                    'coefficients': coefficients,
                    'predictions': result['y_pred'],
                    'performance': {
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score']
                    }
                }
            else:
                model_data = {
                    'model_info': model_info,
                    'predictions': result['y_pred'],
                    'performance': {
                        'accuracy': result['accuracy'],
                        'f1_score': result['f1_score']
                    }
                }
        
        with open(csv_pkl_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Saved: {filename}")
    
    # 3. Simpan semua predictions
    all_predictions_data = {}
    for name, result in results.items():
        all_predictions_data[name] = {
            'predictions': result['y_pred'],
            'accuracy': result['accuracy'],
            'actual_labels': y_test
        }
    
    predictions_csv_pkl = os.path.join(CSV_PKL_DIR, 'all_predictions.csv.pkl')
    with open(predictions_csv_pkl, 'wb') as f:
        pickle.dump(all_predictions_data, f)
    print(f"✅ Saved: all_predictions.csv.pkl")
    
    # 4. Simpan test data
    test_data_csv_pkl = os.path.join(CSV_PKL_DIR, 'test_data.csv.pkl')
    test_data = {
        'features': X_test_df,
        'labels': y_test,
        'label_names': label_names
    }
    with open(test_data_csv_pkl, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"✅ Saved: test_data.csv.pkl")
    
    # 5. Simpan metadata
    metadata_csv_pkl = os.path.join(CSV_PKL_DIR, 'metadata.csv.pkl')
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': list(results.keys()),
        'features_used': ['temperature', 'humidity', 'hour', 'minute'],
        'label_mapping': {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
    }
    with open(metadata_csv_pkl, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✅ Saved: metadata.csv.pkl")
    
    print(f"\n📁 All CSV.PKL files saved to: {CSV_PKL_DIR}")
    
    return True

def save_all_models(results, scaler):
    print("\n💾 SAVING ALL MODELS (PICKLE FORMAT)...")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler.pkl")
    
    # Save all models individually
    for name, result in results.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(MODELS_DIR, filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        
        print(f"✅ Saved: {filename}")
    
    # Save ensemble model
    ensemble_path = os.path.join(MODELS_DIR, 'all_models.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(results, f)
    print("✅ Saved: all_models.pkl (ensemble)")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': list(results.keys()),
        'performance': {},
        'hivemq_config': {
            'broker': MQTT_BROKER,
            'dht_topic': DHT_TOPIC,
            'prediction_topic': PREDICTION_TOPIC
        },
        'note': 'All 3 models trained with HiveMQ integration'
    }
    
    for name, result in results.items():
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
    
    print("✅ Saved: metadata.json")
    
    return metadata

# ==================== PREDICT & PUBLISH ====================
def predict_new_data(models, scaler, temperature, humidity, hour=None, minute=None):
    """Predict label for new sensor data using all models"""
    
    if hour is None:
        hour = datetime.now().hour
    if minute is None:
        minute = datetime.now().minute
    
    features = np.array([[temperature, humidity, hour, minute]])
    features_scaled = scaler.transform(features)
    
    predictions = {}
    
    for model_name, model_data in models.items():
        model = model_data['model']
        
        try:
            prediction = model.predict(features_scaled)[0]
            
            confidence = 1.0
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features_scaled)[0]
                    if prediction < len(probabilities):
                        confidence = probabilities[prediction]
                except:
                    confidence = 1.0
            
            label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
            label = label_map.get(prediction, f'Class_{prediction}')
            
            predictions[model_name] = {
                'label': label,
                'label_encoded': int(prediction),
                'confidence': float(confidence),
                'temperature': float(temperature),
                'humidity': float(humidity),
                'hour': hour,
                'minute': minute
            }
            
        except Exception as e:
            print(f"❌ Error predicting with {model_name}: {e}")
            predictions[model_name] = {
                'label': 'ERROR',
                'label_encoded': -1,
                'confidence': 0.0,
                'temperature': float(temperature),
                'humidity': float(humidity),
                'hour': hour,
                'minute': minute
            }
    
    return predictions

def test_and_publish_to_hivemq(results, scaler, hivemq_manager):
    print("\n" + "="*60)
    print("📤 TESTING & PUBLISHING TO HiveMQ")
    print("="*60)
    
    # Test cases
    test_cases = [
        (18.0, 75.0, 14, 30, "DINGIN"),
        (20.5, 70.0, 10, 15, "DINGIN"),
        (23.0, 65.0, 12, 45, "NORMAL"),
        (25.5, 60.0, 15, 20, "NORMAL"),
        (27.0, 55.0, 18, 10, "PANAS"),
        (29.0, 50.0, 20, 0, "PANAS"),
        (32.0, 45.0, 22, 30, "PANAS"),
    ]
    
    all_predictions = []
    
    for temp, hum, hour, minute, expected in test_cases:
        print(f"\n🌡️  Test: {temp}°C, {hum}%, {hour:02d}:{minute:02d} (Expected: {expected})")
        print("-" * 50)
        
        predictions = predict_new_data(results, scaler, temp, hum, hour, minute)
        
        test_predictions = []
        
        for model_name, pred_data in predictions.items():
            label = pred_data['label']
            confidence = pred_data['confidence']
            is_correct = label == expected
            
            # Prepare data for HiveMQ
            prediction_data = {
                'label': label,
                'label_encoded': pred_data['label_encoded'],
                'temperature': float(temp),
                'humidity': float(hum),
                'hour': hour,
                'minute': minute,
                'confidence': float(confidence),
                'expected': expected,
                'is_correct': is_correct
            }
            
            # Publish to HiveMQ
            if hivemq_manager.connected:
                success = hivemq_manager.publish_prediction(model_name, prediction_data)
                if success:
                    print(f"   📤 Published {model_name} to HiveMQ")
            
            test_predictions.append({
                'model': model_name,
                'prediction': label,
                'confidence': confidence,
                'correct': is_correct
            })
            
            print(f"   [{model_name:20}] → {label:10} ({confidence:.1%}) {'✅' if is_correct else '❌'}")
        
        all_predictions.append(test_predictions)
    
    # Summary
    print("\n" + "="*60)
    print("📊 PUBLISHING SUMMARY")
    print("="*60)
    
    for model_name in results.keys():
        correct_count = 0
        total_count = 0
        
        for test_set in all_predictions:
            for pred in test_set:
                if pred['model'] == model_name:
                    total_count += 1
                    if pred['correct']:
                        correct_count += 1
        
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"{model_name:20}: {correct_count}/{total_count} correct ({accuracy:.1%})")
    
    print(f"\n📡 All predictions published to: {PREDICTION_TOPIC}")
    
    return all_predictions

# ==================== MAIN PROGRAM ====================
def main():
    print("\n" + "="*60)
    print("🚀 ESP32 DHT11 ML TRAINING SYSTEM WITH HiveMQ")
    print("="*60)
    print(f"📁 Base Directory: {BASE_DIR}")
    print(f"📁 TrainingDHT: {TRAININGDHT_DIR}")
    print(f"🤖 Models: Decision Tree, KNN, Logistic Regression")
    print(f"📡 HiveMQ Broker: {MQTT_BROKER}")
    print("="*60)
    
    # Initialize HiveMQ Manager
    hivemq_manager = HiveMQManager()
    
    try:
        # Option: Connect to HiveMQ to collect real-time data
        print("\n🔗 Would you like to connect to HiveMQ to collect real-time data?")
        print("   1. Yes - Connect and collect data")
        print("   2. No  - Use existing CSV data")
        
        choice = input("Enter choice (1/2): ").strip()
        
        if choice == '1':
            print("\n📡 Connecting to HiveMQ...")
            if hivemq_manager.connect():
                print("\n✅ Connected to HiveMQ!")
                print("📥 Listening for sensor data...")
                print("⏳ Collecting data for 30 seconds...")
                
                # Collect data for 30 seconds
                time.sleep(30)
                
                print(f"\n📊 Collected {len(hivemq_manager.received_messages)} messages")
                hivemq_manager.disconnect()
            else:
                print("❌ Failed to connect to HiveMQ")
                print("📁 Using existing CSV data instead...")
        
        # 1. Load and prepare data
        data_result = load_and_prepare_data()
        if data_result is None:
            print("❌ Failed to load data")
            return
        
        X_train, X_test, y_train, y_test, scaler, df, X_test_df = data_result
        
        # 2. Train all 3 models
        results, label_names = train_all_models(X_train, X_test, y_train, y_test)
        
        # 3. Create visualizations
        create_all_visualizations(results, X_test_df, y_test, label_names)
        
        # 4. Save all models
        metadata = save_all_models(results, scaler)
        
        # 5. Save models to CSV.PKL format
        save_models_to_csv_pkl(results, scaler, X_test_df, y_test, label_names)
        
        # 6. Connect to HiveMQ for publishing predictions
        print("\n🔗 Connecting to HiveMQ for publishing predictions...")
        if hivemq_manager.connect():
            print("✅ Connected to HiveMQ for publishing!")
            
            # 7. Test and publish to HiveMQ
            test_and_publish_to_hivemq(results, scaler, hivemq_manager)
            
            # Disconnect
            hivemq_manager.disconnect()
        
        # 8. Show final summary
        print("\n" + "="*60)
        print("🏆 TRAINING COMPLETE - ALL 3 MODELS")
        print("="*60)
        
        print(f"\n📊 Models trained and saved:")
        for model_name in results.keys():
            acc = results[model_name]['accuracy']
            f1 = results[model_name]['f1_score']
            print(f"   • {model_name:25} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_f1 = results[best_model]['f1_score']
        
        print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
        
        print(f"\n📁 Files saved:")
        print(f"   • Regular models: {MODELS_DIR}/")
        print(f"   • CSV.PKL models: {CSV_PKL_DIR}/")
        print(f"   • Reports: {REPORTS_DIR}/")
        
        print(f"\n📡 HiveMQ Configuration:")
        print(f"   • Broker: {MQTT_BROKER}")
        print(f"   • Subscribe Topic: {DHT_TOPIC}")
        print(f"   • Publish Topic: {PREDICTION_TOPIC}")
        
        print(f"\n🔧 How to use models:")
        print(f"   1. Load models: pickle.load('model.pkl')")
        print(f"   2. Load scaler: pickle.load('scaler.pkl')")
        print(f"   3. Make predictions: model.predict(scaler.transform(features))")
        print(f"   4. Check metadata.json for HiveMQ configuration")
        
        print("\n✅ All 3 models ready for deployment with HiveMQ integration!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect HiveMQ
        if hivemq_manager.connected:
            hivemq_manager.disconnect()

if __name__ == "__main__":
    main()

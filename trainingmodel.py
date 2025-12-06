import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
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
BASE_DIR = "Data_Collector"
CSV_FILE = os.path.join(BASE_DIR, "sensor_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# MQTT Configuration
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"

# Topics
SUB_TOPIC_DHT = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"
PUB_TOPIC_PREDICTION = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/ml_prediction"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ==================== MQTT MANAGER ====================
class MQTTManager:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.client.tls_set()
        self.connected = False
        
    def connect(self):
        try:
            print(f"🔗 Connecting to MQTT: {MQTT_BROKER}:{MQTT_PORT}")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            time.sleep(2)
            self.connected = True
            print("✅ MQTT Connected!")
            return True
        except Exception as e:
            print(f"❌ MQTT Connection failed: {e}")
            return False
    
    def publish_prediction(self, model_name, prediction_data):
        if not self.connected:
            print(f"⚠️  MQTT not connected, skipping publish")
            return False
        
        try:
            prediction_data['model'] = model_name
            prediction_data['publish_time'] = datetime.now().strftime('%H:%M:%S')
            
            payload = json.dumps(prediction_data)
            result = self.client.publish(PUB_TOPIC_PREDICTION, payload, qos=1)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"📤 [{model_name}] Published: {prediction_data['label']}")
                return True
            else:
                print(f"❌ [{model_name}] Publish failed")
                return False
                
        except Exception as e:
            print(f"❌ [{model_name}] Publish error: {e}")
            return False
    
    def subscribe_to_sensor_data(self):
        if self.connected:
            self.client.subscribe(SUB_TOPIC_DHT)
            print(f"📡 Subscribed to: {SUB_TOPIC_DHT}")
            return True
        return False
    
    def disconnect(self):
        if self.connected:
            self.client.disconnect()
            self.connected = False
            print("📡 MQTT Disconnected")

# ==================== LOAD & PREPARE DATA ====================
def load_and_prepare_data():
    print("📂 Loading DHT dataset...")
    print(f"📁 File: {CSV_FILE}")
    
    try:
        if not os.path.exists(CSV_FILE):
            print(f"❌ File not found! Please run ESP32 data collector first")
            print(f"📝 Expected format: timestamp;temperature;humidity;label;label_encoded;date")
            return None
        
        df = pd.read_csv(CSV_FILE, delimiter=';')
        print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} columns")
        
        # Display first few rows
        print("\n📊 First 5 rows of dataset:")
        print(df.head())
        
        # Check required columns
        required_cols = ['timestamp', 'temperature', 'humidity', 'label', 'label_encoded']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            print("Please check CSV format from ESP32")
            return None
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None
    
    # Display dataset info
    print("\n📊 Dataset Info:")
    print(f"   Total records: {len(df)}")
    if 'date' in df.columns:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Check labels
    unique_labels = df['label'].unique()
    print(f"\n🏷️  Unique labels: {list(unique_labels)}")
    
    # Check label distribution
    print(f"\n📈 Label distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count} records ({percentage:.1f}%)")
    
    # Process timestamp
    print("\n🔧 Processing timestamp...")
    try:
        # Split timestamp into hour, minute, second
        if 'timestamp' in df.columns:
            df[['hour', 'minute', 'second']] = df['timestamp'].str.split(';', expand=True).astype(int)
            print(f"   Timestamp processed successfully")
        else:
            print("⚠️  Timestamp column not found, using current time")
            current_time = datetime.now()
            df['hour'] = current_time.hour
            df['minute'] = current_time.minute
            df['second'] = current_time.second
    except Exception as e:
        print(f"❌ Error processing timestamp: {e}")
        print("⚠️  Using current time instead")
        current_time = datetime.now()
        df['hour'] = current_time.hour
        df['minute'] = current_time.minute
        df['second'] = current_time.second
    
    # Features and target
    X = df[['temperature', 'humidity', 'hour', 'minute']]
    y = df['label_encoded']
    
    print(f"\n🔧 Features shape: {X.shape}")
    print("📊 Feature statistics:")
    print(X.describe())
    
    # Check for class imbalance
    print(f"\n⚖️  Class balance check:")
    class_counts = y.value_counts()
    for class_id, count in class_counts.items():
        if 'label' in df.columns:
            matching_rows = df[df['label_encoded'] == class_id]
            if len(matching_rows) > 0:
                label_name = matching_rows['label'].iloc[0]
            else:
                label_name = f'Class_{class_id}'
        else:
            label_name = f'Class_{class_id}'
        percentage = (count / len(y)) * 100
        print(f"   Class {class_id} ({label_name}): {count} samples ({percentage:.1f}%)")
    
    # Split data - use stratify only if we have at least 2 classes
    if len(class_counts) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    else:
        print("⚠️  Only 1 class in data, cannot stratify")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📈 Data split:")
    print(f"   Training: {X_train.shape[0]} samples ({len(np.unique(y_train))} classes)")
    print(f"   Testing: {X_test.shape[0]} samples ({len(np.unique(y_test))} classes)")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, df, X_test

# ==================== TRAIN ALL MODELS ====================
def train_all_models(X_train, X_test, y_train, y_test):
    print("\n" + "="*60)
    print("🤖 TRAINING ML MODELS")
    print("="*60)
    
    results = {}
    label_names = ['DINGIN', 'NORMAL', 'PANAS']
    
    # Cek jumlah kelas unik dalam data training
    unique_classes_train = np.unique(y_train)
    num_classes_train = len(unique_classes_train)
    print(f"📊 Classes in training data: {unique_classes_train}")
    print(f"📊 Number of unique classes: {num_classes_train}")
    
    # Define models based on available classes
    models = {}
    
    # Always include Decision Tree
    print("\n➕ Adding Decision Tree...")
    models['Decision Tree'] = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        criterion='gini',
        min_samples_split=10,
        min_samples_leaf=5
    )
    
    # Always include KNN
    print("➕ Adding K-Nearest Neighbors...")
    models['K-Nearest Neighbors'] = KNeighborsClassifier(
        n_neighbors=min(5, len(X_train)),
        weights='distance',
        metric='euclidean',
        algorithm='auto'
    )
    
    # Only include Logistic Regression if we have at least 2 classes
    if num_classes_train >= 2:
        print("➕ Adding Logistic Regression...")
        models['Logistic Regression'] = LogisticRegression(
            random_state=42,
            max_iter=1000,
            multi_class='ovr',
            solver='liblinear',
            C=1.0
        )
    else:
        print("⚠️  Logistic Regression skipped: Need at least 2 classes in data")
        print("➕ Adding Dummy Classifier as alternative...")
        models['Dummy Classifier'] = DummyClassifier(strategy='most_frequent')
    
    print(f"\n📋 Total models to train: {len(models)}")
    
    for name, model in models.items():
        print(f"\n🏃 Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # For weighted metrics, handle single class case
            if num_classes_train >= 2:
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            else:
                # For single class, all predictions should be correct
                precision = 1.0 if accuracy == 1.0 else 0.0
                recall = 1.0 if accuracy == 1.0 else 0.0
                f1 = 1.0 if accuracy == 1.0 else 0.0
            
            # Cross-validation - adjust folds based on sample size
            cv_folds = min(5, len(X_train))
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            except:
                cv_mean = accuracy
                cv_std = 0.0
            
            # Per-class metrics for all possible classes (0, 1, 2)
            all_classes = [0, 1, 2]
            
            precision_per_class = []
            recall_per_class = []
            f1_per_class = []
            
            for class_label in all_classes:
                # Create binary arrays for this class
                y_test_binary = (y_test == class_label).astype(int)
                y_pred_binary = (y_pred == class_label).astype(int)
                
                if np.sum(y_test_binary) > 0 or np.sum(y_pred_binary) > 0:
                    try:
                        prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
                        rec = recall_score(y_test_binary, y_pred_binary, zero_division=0)
                        f1c = f1_score(y_test_binary, y_pred_binary, zero_division=0)
                    except:
                        prec = rec = f1c = 0.0
                else:
                    # If class doesn't exist, set metrics to 0
                    prec = rec = f1c = 0.0
                
                precision_per_class.append(prec)
                recall_per_class.append(rec)
                f1_per_class.append(f1c)
            
            # Convert to numpy arrays
            precision_per_class = np.array(precision_per_class)
            recall_per_class = np.array(recall_per_class)
            f1_per_class = np.array(f1_per_class)
            
            # Detailed classification report
            try:
                # Get classes present in y_test or y_pred
                present_classes = np.unique(np.concatenate([y_test.values if hasattr(y_test, 'values') else y_test, 
                                                          y_pred]))
                present_classes = sorted(present_classes)
                
                # Filter label names to only those present
                present_labels = []
                for cls in present_classes:
                    if cls < len(label_names):
                        present_labels.append(label_names[cls])
                    else:
                        present_labels.append(f'Class_{cls}')
                
                report = classification_report(y_test, y_pred, 
                                              labels=present_classes,
                                              target_names=present_labels,
                                              output_dict=True,
                                              zero_division=0)
            except Exception as e:
                print(f"   ⚠️  Classification report warning: {e}")
                report = {
                    'accuracy': accuracy,
                    'weighted avg': {
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1
                    }
                }
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class,
                'classification_report': report
            }
            
            print(f"   ✅ Accuracy: {accuracy:.4f}")
            print(f"   📊 F1-Score: {f1:.4f}")
            print(f"   🔍 Precision: {precision:.4f}")
            print(f"   📈 Recall: {recall:.4f}")
            print(f"   🔄 CV Accuracy: {cv_mean:.4f} (±{cv_std:.4f})")
            
            # Show predictions count
            unique, counts = np.unique(y_pred, return_counts=True)
            print(f"   🎯 Predictions distribution:")
            for label_code, count in zip(unique, counts):
                if label_code < len(label_names):
                    label_name = label_names[label_code]
                else:
                    label_name = f'Class_{label_code}'
                print(f"      {label_name}: {count}")
                
        except Exception as e:
            print(f"❌ Error training {name}: {e}")
            print(f"   ⚠️  Skipping this model")
    
    return results, label_names

# ==================== CREATE ALL VISUALIZATIONS ====================
def create_all_visualizations(results, X_test_df, y_test, label_names):
    if not results:
        print("❌ No models trained, skipping visualizations")
        return
    
    print("\n📊 CREATING VISUALIZATIONS FOR ALL MODELS...")
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. CONFUSION MATRICES
    num_models = len(results)
    if num_models > 0:
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 5))
        if num_models == 1:
            axes = [axes]
        
        fig.suptitle('CONFUSION MATRICES - ALL MODELS', fontsize=16, fontweight='bold')
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            cm = confusion_matrix(y_test, result['y_pred'])
            
            # Get unique labels that exist in data
            unique_preds = np.unique(result['y_pred'])
            unique_tests = np.unique(y_test)
            all_unique = np.unique(np.concatenate([unique_preds, unique_tests]))
            all_unique = sorted(all_unique)
            
            # Create display labels
            display_labels = []
            for i in all_unique:
                if i < len(label_names):
                    display_labels.append(label_names[i])
                else:
                    display_labels.append(f'Class_{i}')
            
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=display_labels
            )
            
            disp.plot(cmap='Blues', ax=ax, values_format='d')
            ax.set_title(f'{name}', fontweight='bold', fontsize=12)
            
            accuracy = result['accuracy']
            ax.text(0.95, -0.15, f'Accuracy: {accuracy:.3f}', 
                    transform=ax.transAxes, ha='right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'all_confusion_matrices.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: all_confusion_matrices.png")
    
    # 2. MODEL COMPARISON BAR CHART
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MODEL PERFORMANCE COMPARISON', fontsize=16, fontweight='bold')
        
        metrics_list = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        
        for idx, (metric, title) in enumerate(zip(metrics_list, metric_names)):
            ax = axes[idx//2, idx%2]
            model_names = list(results.keys())
            scores = [results[model][metric] for model in model_names]
            
            bars = ax.bar(model_names, scores, color=colors[:len(model_names)], alpha=0.8)
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
        
        # Sort by importance
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
    if len(results) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use actual label names that exist
        actual_labels = []
        for i in range(len(label_names)):
            # Check if this class exists in any model's predictions
            class_exists = False
            for result in results.values():
                if len(result['recall_per_class']) > i:
                    class_exists = True
                    break
            if class_exists:
                actual_labels.append(label_names[i])
            else:
                actual_labels.append(f'Class_{i}')
        
        x = np.arange(len(actual_labels))
        if len(actual_labels) > 0:
            width = 0.8 / len(results) if len(results) > 0 else 0.25
            
            for i, (model_name, result) in enumerate(results.items()):
                # Make sure we have enough values
                recalls = result['recall_per_class']
                if len(recalls) < len(actual_labels):
                    # Pad with zeros if needed
                    recalls = np.pad(recalls, (0, len(actual_labels) - len(recalls)), 'constant')
                
                offset = width * i
                ax.bar(x + offset, recalls[:len(actual_labels)], width, label=model_name, alpha=0.8)
                
                for j, recall in enumerate(recalls[:len(actual_labels)]):
                    ax.text(j + offset, recall + 0.02, f'{recall:.3f}', 
                           ha='center', va='bottom', fontsize=9)
            
            ax.set_title('RECALL PER CLASS - ALL MODELS', fontweight='bold', fontsize=14)
            ax.set_xlabel('Class', fontsize=12)
            ax.set_ylabel('Recall Score', fontsize=12)
            ax.set_xticks(x + width * (len(results)-1)/2 if len(results) > 1 else x)
            ax.set_xticklabels(actual_labels)
            ax.legend()
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(REPORTS_DIR, 'recall_per_class.png'), 
                        dpi=300, bbox_inches='tight')
            print("✅ Saved: recall_per_class.png")
    
    # 5. CORRELATION HEATMAP
    fig, ax = plt.subplots(figsize=(8, 6))
    
    try:
        # Combine features for correlation
        correlation_data = pd.DataFrame(X_test_df, columns=['temperature', 'humidity', 'hour', 'minute'])
        correlation_data['label'] = y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else y_test
        
        # Calculate correlation
        corr = correlation_data.corr()
        
        # Plot heatmap
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        
        ax.set_title('FEATURE CORRELATION MATRIX', fontweight='bold', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'correlation_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: correlation_heatmap.png")
    except Exception as e:
        print(f"⚠️  Could not create correlation heatmap: {e}")
    
    # 6. SCATTER PLOT TEMP vs HUMIDITY
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # Create color map for labels
        colors = ['blue', 'green', 'red']  # DINGIN, NORMAL, PANAS
        label_colors = []
        for label in y_test:
            if label < len(colors):
                label_colors.append(colors[label])
            else:
                label_colors.append('gray')
        
        scatter = ax.scatter(X_test_df['temperature'], X_test_df['humidity'], 
                            c=label_colors, alpha=0.6, s=50)
        
        ax.set_title('TEMPERATURE vs HUMIDITY (Colored by Label)', fontweight='bold', fontsize=14)
        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Humidity (%)', fontsize=12)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = []
        for i in range(min(3, len(label_names))):
            legend_elements.append(Patch(facecolor=colors[i], label=label_names[i]))
        if len(label_names) > 3:
            legend_elements.append(Patch(facecolor='gray', label='Other'))
        ax.legend(handles=legend_elements, title='Labels')
        
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, 'temp_vs_humidity.png'), 
                    dpi=300, bbox_inches='tight')
        print("✅ Saved: temp_vs_humidity.png")
    except Exception as e:
        print(f"⚠️  Could not create scatter plot: {e}")
    
    # 7. SUMMARY TABLE
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

# ==================== SAVE ALL MODELS ====================
def save_all_models(results, scaler):
    if not results:
        print("❌ No models to save")
        return None
    
    print("\n💾 SAVING ALL MODELS...")
    
    # Save scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print("✅ Saved: scaler.pkl")
    
    # Save all models individually
    for name, result in results.items():
        # Create filename
        filename = name.lower().replace(' ', '_') + '.pkl'
        model_path = os.path.join(MODELS_DIR, filename)
        
        with open(model_path, 'wb') as f:
            pickle.dump(result['model'], f)
        
        print(f"✅ Saved: {filename}")
    
    # Save ensemble model (all models together)
    ensemble_path = os.path.join(MODELS_DIR, 'all_models.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(results, f)
    print("✅ Saved: all_models.pkl (ensemble)")
    
    # Save metadata
    metadata = {
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models_trained': list(results.keys()),
        'label_mapping': {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'},
        'features': ['temperature', 'humidity', 'hour', 'minute'],
        'performance': {},
        'mqtt_config': {
            'broker': MQTT_BROKER,
            'topic_prediction': PUB_TOPIC_PREDICTION,
            'topic_sensor': SUB_TOPIC_DHT
        },
        'note': 'Models trained on ESP32 DHT11 data'
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

# ==================== PREDICTION FUNCTIONS ====================
def predict_new_data(models, scaler, temperature, humidity, hour=None, minute=None):
    """Predict label for new sensor data"""
    
    # Use current time if not provided
    if hour is None:
        hour = datetime.now().hour
    if minute is None:
        minute = datetime.now().minute
    
    # Prepare features
    features = np.array([[temperature, humidity, hour, minute]])
    features_scaled = scaler.transform(features)
    
    predictions = {}
    
    for model_name, model_data in models.items():
        model = model_data['model']
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
        else:
            confidence = 1.0
            probabilities = [0, 0, 0]
        
        # Map to label
        label_map = {0: 'DINGIN', 1: 'NORMAL', 2: 'PANAS'}
        label = label_map.get(prediction, 'UNKNOWN')
        
        predictions[model_name] = {
            'label': label,
            'label_encoded': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist() if hasattr(model, 'predict_proba') else None
        }
    
    return predictions

# ==================== TEST & PUBLISH TO MQTT ====================
def test_and_publish_all_models(results, scaler, mqtt_manager):
    if not results:
        print("❌ No models to test")
        return
    
    print("\n" + "="*60)
    print("📤 TESTING & PUBLISHING ALL MODELS TO MQTT")
    print("="*60)
    
    # Test cases based on temperature ranges
    test_cases = [
        # (temp, hum, hour, minute, expected)
        (18.0, 75.0, 14, 30, "DINGIN"),    # < 25°C
        (20.5, 70.0, 10, 15, "DINGIN"),    # < 25°C
        (22.0, 65.0, 12, 45, "DINGIN"),    # < 25°C
        (24.0, 60.0, 15, 20, "DINGIN"),    # < 25°C
        (25.0, 58.0, 16, 30, "NORMAL"),    # = 25°C
        (26.0, 55.0, 18, 10, "NORMAL"),    # 25-28°C
        (27.0, 53.0, 19, 45, "NORMAL"),    # 25-28°C
        (28.0, 50.0, 20, 0, "PANAS"),      # = 28°C
        (29.0, 48.0, 21, 15, "PANAS"),     # > 28°C
        (32.0, 45.0, 22, 30, "PANAS"),     # > 28°C
    ]
    
    all_predictions = []
    
    for temp, hum, hour, minute, expected in test_cases:
        print(f"\n🌡️  Test: {temp}°C, {hum}%, {hour:02d}:{minute:02d} (Expected: {expected})")
        print("-" * 50)
        
        # Get predictions from all models
        predictions = predict_new_data(results, scaler, temp, hum, hour, minute)
        
        test_predictions = []
        
        for model_name, pred_data in predictions.items():
            label = pred_data['label']
            confidence = pred_data['confidence']
            is_correct = label == expected
            
            # Prepare data for MQTT
            prediction_data = {
                'label': label,
                'label_encoded': pred_data['label_encoded'],
                'temperature': float(temp),
                'humidity': float(hum),
                'hour': hour,
                'minute': minute,
                'confidence': float(confidence),
                'expected': expected,
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'is_correct': is_correct
            }
            
            # Publish to MQTT
            if mqtt_manager.connected:
                mqtt_manager.publish_prediction(model_name, prediction_data)
            
            # Store for display
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
    
    print(f"\n📡 All predictions published to: {PUB_TOPIC_PREDICTION}")
    print("💡 ESP32 can subscribe to see ML predictions")

# ==================== REAL-TIME PREDICTION MODE ====================
def real_time_prediction_mode(results, scaler, mqtt_manager):
    """Run in real-time mode, listening for sensor data"""
    if not results:
        print("❌ No models available for real-time prediction")
        return
    
    print("\n" + "="*60)
    print("🌐 REAL-TIME PREDICTION MODE")
    print("="*60)
    print("Listening for sensor data from ESP32...")
    print("Press Ctrl+C to exit")
    
    # Store received data
    received_data = []
    
    def on_message(client, userdata, msg):
        """Callback for MQTT messages"""
        try:
            data = json.loads(msg.payload.decode())
            
            # Extract sensor data
            temperature = data.get('temperature', 0)
            humidity = data.get('humidity', 0)
            timestamp = data.get('timestamp', '')
            
            # Parse timestamp to get hour and minute
            if ';' in timestamp:
                hour, minute, second = map(int, timestamp.split(';'))
            else:
                hour = datetime.now().hour
                minute = datetime.now().minute
            
            print(f"\n📥 Received sensor data:")
            print(f"   🌡️  Temp: {temperature}°C")
            print(f"   💧 Hum: {humidity}%")
            print(f"   ⏰ Time: {hour:02d}:{minute:02d}")
            
            # Get predictions from all models
            predictions = predict_new_data(results, scaler, temperature, humidity, hour, minute)
            
            # Publish predictions
            for model_name, pred_data in predictions.items():
                prediction_payload = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'hour': hour,
                    'minute': minute,
                    'prediction': pred_data['label'],
                    'confidence': pred_data['confidence'],
                    'model': model_name,
                    'sensor_timestamp': timestamp,
                    'prediction_time': datetime.now().strftime('%H:%M:%S')
                }
                
                mqtt_manager.publish_prediction(model_name, prediction_payload)
            
            # Store for logging
            received_data.append({
                'timestamp': datetime.now(),
                'temperature': temperature,
                'humidity': humidity,
                'predictions': predictions
            })
            
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    # Set message callback
    mqtt_manager.client.on_message = on_message
    
    # Subscribe to sensor data
    mqtt_manager.client.subscribe(SUB_TOPIC_DHT)
    
    try:
        # Keep listening
        mqtt_manager.client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Real-time mode stopped")
        
        # Save received data
        if received_data:
            log_file = os.path.join(REPORTS_DIR, 'real_time_predictions.json')
            with open(log_file, 'w') as f:
                # Convert datetime to string for JSON
                log_data = []
                for item in received_data:
                    item_copy = item.copy()
                    item_copy['timestamp'] = item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    log_data.append(item_copy)
                
                json.dump(log_data, f, indent=4)
            print(f"📝 Saved real-time predictions to: {log_file}")

# ==================== MAIN PROGRAM ====================
def main():
    print("\n" + "="*60)
    print("🚀 ESP32 DHT11 ML TRAINING SYSTEM")
    print("="*60)
    print(f"📁 Data from: {CSV_FILE}")
    print(f"🤖 Models: Decision Tree, KNN, Logistic Regression")
    print(f"📡 MQTT Broker: {MQTT_BROKER}")
    print("="*60)
    
    # Initialize MQTT
    mqtt_manager = MQTTManager()
    mqtt_manager.connect()
    
    try:
        # 1. Load and prepare data
        data_result = load_and_prepare_data()
        if data_result is None:
            print("❌ Failed to load data")
            return
        
        X_train, X_test, y_train, y_test, scaler, df, X_test_df = data_result
        
        # 2. Train models (will handle single class data)
        results, label_names = train_all_models(X_train, X_test, y_train, y_test)
        
        if not results:
            print("❌ No models were successfully trained")
            return
        
        # 3. Create visualizations
        create_all_visualizations(results, X_test_df, y_test, label_names)
        
        # 4. Save all models
        metadata = save_all_models(results, scaler)
        
        # 5. Test and publish to MQTT
        test_and_publish_all_models(results, scaler, mqtt_manager)
        
        # 6. Show final summary
        print("\n" + "="*60)
        print("🏆 TRAINING COMPLETE")
        print("="*60)
        
        print(f"\n📊 Models trained and saved:")
        for model_name in results.keys():
            acc = results[model_name]['accuracy']
            f1 = results[model_name]['f1_score']
            print(f"   • {model_name:25} → Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
        # Find best model by F1-score if we have at least 2 models
        if len(results) >= 2:
            best_model = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
            best_f1 = results[best_model]['f1_score']
            print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
        
        print(f"\n📁 Files saved:")
        print(f"   • Models: {MODELS_DIR}/")
        print(f"   • Reports: {REPORTS_DIR}/")
        
        print(f"\n📡 MQTT Configuration:")
        print(f"   • Broker: {MQTT_BROKER}")
        print(f"   • Topic (Receive): {SUB_TOPIC_DHT}")
        print(f"   • Topic (Send): {PUB_TOPIC_PREDICTION}")
        
        print(f"\n🔧 How to use models:")
        print(f"   1. Load models: pickle.load('model.pkl')")
        print(f"   2. Load scaler: pickle.load('scaler.pkl')")
        print(f"   3. Make predictions: model.predict(scaler.transform(features))")
        print(f"   4. Check metadata.json for details")
        
        # Ask if want to run real-time mode
        print("\n" + "="*60)
        response = input("🎯 Run real-time prediction mode? (y/n): ").lower()
        
        if response == 'y':
            real_time_prediction_mode(results, scaler, mqtt_manager)
        
        print("\n✅ All models ready for deployment!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Disconnect MQTT
        mqtt_manager.disconnect()

if __name__ == "__main__":
    main()

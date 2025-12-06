import paho.mqtt.client as mqtt
import json
import csv
import time
from datetime import datetime
import os
import sys

# ==================== KONFIGURASI ====================
MQTT_BROKER = "76c4ab43d10547d5a223d4648d43ceb6.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "hivemq.webclient.1764923408610"
MQTT_PASSWORD = "9y&f74G1*pWSD.tQdXa@"
DHT_TOPIC = "sic/dibimbing/kelompok-SENSOR/FARIZ/pub/dht"

# File Configuration
FOLDER_PATH = "Data_Collector"
CSV_FILENAME = "sensor_data.csv"
CSV_PATH = os.path.join(FOLDER_PATH, CSV_FILENAME)

# Target pengumpulan data
TARGET_RECORDS = 15
current_record_count = 0

# ==================== SETUP CSV ====================
def setup_csv():
    """Buat file CSV dengan header"""
    if not os.path.exists(FOLDER_PATH):
        os.makedirs(FOLDER_PATH)
        print(f"📁 Created folder: {FOLDER_PATH}")
    
    file_exists = os.path.exists(CSV_PATH)
    
    if not file_exists:
        with open(CSV_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                'timestamp',
                'temperature',
                'humidity',
                'label',
                'label_encoded',
                'date'
            ])
        print(f"✅ Created new CSV file: {CSV_PATH}")
    else:
        # Hitung data yang sudah ada
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            existing_count = len(lines) - 1
            global current_record_count
            current_record_count = existing_count
            
            print(f"📊 Existing records: {existing_count}")
            print(f"🎯 Target: {TARGET_RECORDS}")
            
            if existing_count > 0:
                print("📄 Last 3 records:")
                for line in lines[max(1, len(lines)-3):]:
                    print(f"   {line.strip()}")
    
    return current_record_count

# ==================== TENTUKAN LABEL ====================
def determine_label(temperature):
    if temperature < 25.0:
        return "DINGIN", 0
    elif temperature > 28.0:
        return "PANAS", 2
    else:
        return "NORMAL", 1

# ==================== MQTT CALLBACKS ====================
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("✅ Connected to HiveMQ Cloud!")
        print(f"📡 Subscribing to topic: {DHT_TOPIC}")
        client.subscribe(DHT_TOPIC)
        print("⏳ Waiting for DHT data...")
        print("=" * 50)
    else:
        print(f"❌ Connection failed with code: {rc}")

def on_message(client, userdata, msg):
    global current_record_count
    
    try:
        # Parse JSON data dari ESP32
        data = json.loads(msg.payload.decode())
        
        # Ekstrak data
        temperature = data.get('temperature', 0)
        humidity = data.get('humidity', 0)
        label = data.get('label', '')
        label_encoded = data.get('label_encoded', 1)
        timestamp = data.get('timestamp', '')
        date = data.get('date', '')
        record_num = data.get('record', 0)
        
        # Jika timestamp kosong, buat timestamp baru
        if not timestamp:
            now = datetime.now()
            timestamp = now.strftime('%H;%M;%S')
            date = now.strftime('%Y-%m-%d')
        
        # Jika label kosong, tentukan label
        if not label:
            label, label_encoded = determine_label(temperature)
        
        # Simpan ke CSV
        with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow([
                timestamp,
                round(float(temperature), 2),
                round(float(humidity), 2),
                label,
                label_encoded,
                date
            ])
        
        current_record_count += 1
        
        # Print ke console
        print(f"\n📥 Data #{record_num} saved:")
        print(f"   ⏰ Time: {timestamp}")
        print(f"   📅 Date: {date}")
        print(f"   🌡️  Temp: {temperature}°C")
        print(f"   💧 Hum: {humidity}%")
        print(f"   🏷️  Label: {label} ({label_encoded})")
        print(f"   📊 Progress: {current_record_count}/{TARGET_RECORDS}")
        print("-" * 40)
        
        # Cek apakah sudah mencapai target
        if current_record_count >= TARGET_RECORDS:
            print("\n🎯 Reached 15 records! Stopping...")
            client.disconnect()
            show_summary()
            
    except Exception as e:
        print(f"❌ Error processing message: {e}")

# ==================== TAMPILKAN SUMMARY ====================
def show_summary():
    print("\n" + "="*50)
    print("📊 COLLECTION SUMMARY")
    print("="*50)
    
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_records = len(lines) - 1
            
            print(f"✅ Total records collected: {total_records}")
            print(f"💾 File saved: {CSV_PATH}")
            
            # Hitung distribusi label
            label_counts = {"DINGIN": 0, "NORMAL": 0, "PANAS": 0}
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(';')
                if len(parts) >= 4:
                    label = parts[3]
                    if label in label_counts:
                        label_counts[label] += 1
            
            print(f"\n🏷️  Label distribution:")
            for label, count in label_counts.items():
                percentage = (count / total_records * 100) if total_records > 0 else 0
                print(f"   {label}: {count} ({percentage:.1f}%)")
            
            # Tampilkan semua data
            print(f"\n📄 All collected data:")
            for i, line in enumerate(lines[1:], 1):
                print(f"   {i}. {line.strip()}")
    
    print("\n🎉 Data collection complete!")
    print("="*50)

# ==================== PROGRAM UTAMA ====================
def main():
    print("\n" + "="*50)
    print("🌡️  DHT11 DATA COLLECTOR (Python Version)")
    print("🎯 Target: 15 Records from ESP32")
    print("="*50)
    
    # Setup CSV file
    setup_csv()
    
    # Jika sudah mencapai target, tampilkan summary
    if current_record_count >= TARGET_RECORDS:
        print("⚠ Already have 15+ records!")
        show_summary()
        return
    
    # Buat MQTT client
    client = mqtt.Client()
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    
    # Set callback functions
    client.on_connect = on_connect
    client.on_message = on_message
    
    # Connect ke broker
    try:
        print("🔗 Connecting to HiveMQ...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return
    
    # Mulai mengumpulkan data
    print(f"\n🎯 Need: {TARGET_RECORDS - current_record_count} more records")
    print("Press Ctrl+C to stop early")
    print("="*50)
    
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        client.disconnect()

# ==================== JALANKAN PROGRAM ====================
if __name__ == "__main__":
    main()

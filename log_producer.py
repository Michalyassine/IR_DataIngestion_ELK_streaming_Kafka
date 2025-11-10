import os
import re
from kafka import KafkaProducer
import json

# --- Configuration ---
LOGS_ROOT_DIR = 'Data'
KAFKA_BROKER = 'localhost:9094'  
KAFKA_TOPIC = 'log-topic'
# Regex to extract the date (YYYY-MM-DD) from the folder name (log-YYYY-MM-DD)
DATE_FOLDER_PATTERN = re.compile(r'log-(\d{4}-\d{2}-\d{2})')

def create_kafka_producer():
    """Initializes and returns a Kafka Producer configured for JSON serialization."""
    print(f"Attempting to connect to Kafka broker at {KAFKA_BROKER}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=[KAFKA_BROKER],
            # Use JSON serializer to send structured data
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            retries=5,
            retry_backoff_ms=500
        )
        print("Kafka Producer connected successfully.")
        return producer
    except Exception as e:
        print(f"Error connecting to Kafka: {e}")
        return None

def extract_date_from_path(root_path):
    """Extracts the date string (YYYY-MM-DD) from the directory name."""
    folder_name = os.path.basename(root_path)
    match = DATE_FOLDER_PATTERN.search(folder_name)
    if match:
        return match.group(1) # Returns the date part
    return None

def process_log_files(producer):
    """Recursively walks the log directory and sends lines along with metadata to Kafka."""
    if not os.path.isdir(LOGS_ROOT_DIR):
        print(f"Error: Directory '{LOGS_ROOT_DIR}' not found.")
        return

    total_lines = 0
    total_files = 0
    
    for root, dirs, files in os.walk(LOGS_ROOT_DIR):
        if root != LOGS_ROOT_DIR:
            log_date = extract_date_from_path(root)
            if not log_date:
                print(f"Skipping folder: {root}. Date format not found.")
                continue

            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    print(f"\nProcessing file: {file_path} (Date: {log_date})")
                    file_lines = 0
                    total_files += 1
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            for line in f:
                                # SIMPLIFIED PAYLOAD: Raw line and folder date
                                payload = {
                                    "raw_line": line.strip(),
                                    "folder_date": log_date,
                                    "source_file": filename
                                }
                                
                                producer.send(KAFKA_TOPIC, value=payload)
                                total_lines += 1
                                file_lines += 1
                                
                        print(f"  Successfully sent {file_lines} lines.")
                        
                    except Exception as e:
                        print(f"  Failed to read or send lines from {file_path}: {e}")

    producer.flush() 
    print(f"\n--- Production Complete ---")
    print(f"Total lines sent to Kafka topic '{KAFKA_TOPIC}': {total_lines}")

if __name__ == "__main__":
    producer = create_kafka_producer()
    if producer:
        process_log_files(producer)
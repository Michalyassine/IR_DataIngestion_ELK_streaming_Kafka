# Firefox Build Logs - ML Anomaly Detection Pipeline

A real-time log processing and anomaly detection system for Firefox build logs using ELK Stack, Kafka, and Machine Learning.

## Architecture
  ![Architecture Diagram](https://github.com/Michalyassine/IR_DataIngestion_ELK_streaming_Kafka/blob/main/End-to-End%20Data%20Pipeline%20ELK.png)

- **Kafka**: Message streaming for log ingestion
- **Logstash**: Log parsing and transformation
- **Elasticsearch**: Data storage and indexing
- **Kibana**: Data visualization and dashboards
- **ML Service**: Isolation Forest anomaly detection

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Michalyassine/IR_DataIngestion_ELK_streaming_Kafka.git
cd IR_DataIngestion_ELK_streaming_Kafka
```

### 2. Start Infrastructure

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps
```

**Expected services:**
- Kafka (port 9094)
- Elasticsearch (port 9200)
- Logstash (port 9600)
- Kibana (port 5601)

### 3. Prepare Log Data

Create log files in the expected structure:

```bash
mkdir -p Data/log-2018-06-08
# Place your .txt log files in Data/log-2018-06-08/
# Example: 129557113_2018-06-08-00-28-22.txt
```

**Log format expected:**
```
16:31:01 INFO - Starting test: test_example
16:31:02 ERROR - Test failed with timeout
```

### 4. Create Kafka Topic

```bash
# Create the log topic
docker exec -it kafka kafka-topics --create \
  --topic log-topic \
  --bootstrap-server localhost:9094 \
  --partitions 1 \
  --replication-factor 1

# Verify topic creation
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9094
```

### 5. Install Python Dependencies

```bash
pip install kafka-python
```

### 6. Start Log Producer

```bash
# Send logs to Kafka
python log_producer.py
```

### 7. Verify Data Flow

**Check Elasticsearch:**
```bash
curl "http://localhost:9200/mozilla-log-*/_count"
```

**Access Kibana:**
- Open: http://localhost:5601
- Create index pattern: `mozilla-log-*`
- Set time field: `@timestamp`

### 8. Start ML Anomaly Detection

```bash
# Install ML dependencies
cd ml/
pip install -r requirements.txt

# Train initial model
python train_model.py

# Start real-time anomaly detection
python anomaly_detector.py
```

## Project Structure

```
├── Data/                          # Log files directory
│   └── log-2018-06-08/           # Date-based folders
├── config/
│   └── logstash.conf             # Logstash pipeline config
├── ml/
│   ├── anomaly_detector.py       # Real-time ML service
│   ├── train_model.py            # Model training script
│   ├── requirements.txt          # ML dependencies
│   └── Dockerfile               # ML service container
├── docker-compose.yml            # Infrastructure setup
└── log_producer.py              # Kafka log producer
```

## Usage

### Monitor Logs in Kibana
1. Go to http://localhost:5601
2. Navigate to **Discover**
3. View real-time logs with fields:
   - `log_level` (INFO, ERROR, WARN)
   - `event_message`
   - `folder_date`
   - `source_file`

### View Anomalies
- Anomalies are stored in `firefox-ml-anomalies-*` indices
- Check ML service logs: `docker logs ml-service`

## Troubleshooting

**Kafka Connection Issues:**
```bash
# Check Kafka is running
docker exec -it kafka kafka-broker-api-versions --bootstrap-server localhost:9094
```

**Elasticsearch Issues:**
```bash
# Check ES health
curl "http://localhost:9200/_cluster/health"
```

**No Data in Kibana:**
- Verify log files are in correct format
- Check Logstash logs: `docker logs logstash`
- Ensure date folders match pattern: `log-YYYY-MM-DD`

## Configuration

### Environment Variables
- `KAFKA_BROKER`: Kafka connection (default: localhost:9094)
- `ES_HOST`: Elasticsearch host (default: localhost)
- `DETECTION_INTERVAL`: ML detection interval in seconds (default: 60)

### Log File Requirements
- Files must be `.txt` format
- Place in `Data/log-YYYY-MM-DD/` folders
- Log format: `TIME LEVEL - MESSAGE`

## Next Steps

1. **Add more log files** to `Data/` directory
2. **Create Kibana dashboards** for visualization
3. **Configure ML alerts** for critical anomalies
4. **Scale with additional Kafka partitions** for high volume
"""
Firefox Build Logs - ML Anomaly Detection Service
Detects anomalies using Isolation Forest and LSTM Autoencoder
"""

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FirefoxAnomalyDetector:
    """ML-based anomaly detection for Firefox build logs"""
    
    def __init__(self, es_host='localhost', es_port=9200):
        """Initialize Elasticsearch connection and ML models"""
        self.es = Elasticsearch([f'http://{es_host}:{es_port}'])
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        # NEW: Precision tracking
        self.metrics_history = {
            'detections': [],
            'false_positives': 0,
            'true_positives': 0,
            'false_negatives': 0,
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
    def extract_features(self, time_window_minutes=5, lookback_hours=1):
        """
        Extract features from Elasticsearch for anomaly detection
        
        Args:
            time_window_minutes: Aggregation window in minutes
            lookback_hours: How many hours back to look
            
        Returns:
            pandas.DataFrame with engineered features
        """
        logger.info(f"Extracting features for {time_window_minutes}min window, {lookback_hours}h lookback")
        
        try:
            # Query Elasticsearch
            query = {
                "query": {
                    "range": {
                        "@timestamp": {
                            "gte": f"now-{lookback_hours}h",
                            "lte": "now"
                        }
                    }
                },
                "aggs": {
                    "time_buckets": {
                        "date_histogram": {
                            "field": "@timestamp",
                            "fixed_interval": f"{time_window_minutes}m"
                        },
                        "aggs": {
                            "avg_duration": {"avg": {"field": "test_duration"}},
                            "max_duration": {"max": {"field": "test_duration"}},
                            "min_duration": {"min": {"field": "test_duration"}},
                            "failed_tests": {
                                "filter": {"term": {"test_status": "failed"}}
                            },
                            "passed_tests": {
                                "filter": {"term": {"test_status": "passed"}}
                            },
                            "unique_tests": {
                                "cardinality": {"field": "test_name.keyword"}
                            },
                            "warning_sum": {"sum": {"field": "warning_count"}},
                            "error_sum": {"sum": {"field": "error_count"}}
                        }
                    }
                }
            }
            
            response = self.es.search(
                index="firefox-logs-*",
                size=0,
                query=query["query"],
                aggs=query["aggs"]
            )
            
            buckets = response['aggregations']['time_buckets']['buckets']
            
            if not buckets:
                logger.warning("No data found in Elasticsearch. Waiting for logs to be indexed...")
                return pd.DataFrame()
            
            # Build feature dataframe
            features_list = []
            for bucket in buckets:
                timestamp = pd.to_datetime(bucket['key_as_string'])
                total_tests = bucket['doc_count']
                failed = bucket['failed_tests']['doc_count']
                passed = bucket['passed_tests']['doc_count']
                
                features = {
                    'timestamp': timestamp,
                    'total_tests': total_tests,
                    'avg_duration': bucket['avg_duration']['value'] or 0,
                    'max_duration': bucket['max_duration']['value'] or 0,
                    'min_duration': bucket['min_duration']['value'] or 0,
                    'failure_rate': (failed / total_tests * 100) if total_tests > 0 else 0,
                    'pass_rate': (passed / total_tests * 100) if total_tests > 0 else 0,
                    'unique_tests': bucket['unique_tests']['value'],
                    'warning_count': bucket['warning_sum']['value'] or 0,
                    'error_count': bucket['error_sum']['value'] or 0,
                    'hour_of_day': timestamp.hour,
                    'day_of_week': timestamp.dayofweek,
                    'is_weekend': 1 if timestamp.dayofweek >= 5 else 0
                }
                features_list.append(features)
            
            df = pd.DataFrame(features_list)
            
            # Add rolling features
            if len(df) >= 3:
                df['avg_duration_rolling_mean'] = df['avg_duration'].rolling(3, min_periods=1).mean()
                df['failure_rate_rolling_mean'] = df['failure_rate'].rolling(3, min_periods=1).mean()
                df['avg_duration_std'] = df['avg_duration'].rolling(3, min_periods=1).std().fillna(0)
            
            logger.info(f"Extracted {len(df)} feature vectors")
            return df
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()

    def train_model(self, contamination=0.01, training_window_hours=1):
        """
        Train Isolation Forest model
        
        Args:
            contamination: Expected proportion of anomalies (0.01 = 1%)
            training_window_hours: Hours of historical data to use
        """
        logger.info(f"Training Isolation Forest model (window: {training_window_hours}h)")
        
        try:
            # Use 1-minute buckets for more data points
            df = self.extract_features(time_window_minutes=1, lookback_hours=training_window_hours)
            
            # Require minimum 3 samples
            if len(df) < 3:
                logger.warning(f"Not enough data to train model (found {len(df)} samples, need 3+). Waiting for more data...")
                return False
            
            # Select feature columns
            self.feature_names = [
                'total_tests', 'avg_duration', 'max_duration', 'min_duration',
                'failure_rate', 'pass_rate', 'unique_tests', 'warning_count',
                'error_count', 'hour_of_day', 'day_of_week', 'is_weekend'
            ]
            
            if 'avg_duration_rolling_mean' in df.columns:
                self.feature_names.extend(['avg_duration_rolling_mean', 'failure_rate_rolling_mean', 'avg_duration_std'])
            
            X = df[self.feature_names].fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            self.model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                n_jobs=-1
            )
            
            self.model.fit(X_scaled)
            logger.info("Model training completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False

    def calculate_precision_metrics(self):
        """
        Calculate precision, recall, and F1-score
        
        Returns:
            dict: Model performance metrics
        """
        tp = self.metrics_history['true_positives']
        fp = self.metrics_history['false_positives']
        fn = self.metrics_history['false_negatives']
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False Positive Rate
        fpr = fp / (fp + tp) if (fp + tp) > 0 else 0
        
        metrics = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1, 4),
            'false_positive_rate': round(fpr, 4),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_detections': tp + fp
        }
        
        # Store in history
        self.metrics_history['precision'].append(precision)
        self.metrics_history['recall'].append(recall)
        self.metrics_history['f1_score'].append(f1)
        
        logger.info(f"Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
        
        return metrics
    
    def mark_anomaly_feedback(self, anomaly_id, is_true_positive):
        """
        Record feedback on whether an anomaly was correctly identified
        
        Args:
            anomaly_id: ID of the anomaly document
            is_true_positive: True if anomaly was real, False if false positive
        """
        try:
            if is_true_positive:
                self.metrics_history['true_positives'] += 1
            else:
                self.metrics_history['false_positives'] += 1
            
            # Update document with feedback
            self.es.update(
                index=f"firefox-ml-anomalies-*",
                id=anomaly_id,
                body={
                    "doc": {
                        "is_verified": True,
                        "is_true_positive": is_true_positive,
                        "verified_at": datetime.now().isoformat()
                    }
                }
            )
            
            logger.info(f"Feedback recorded: {'True Positive' if is_true_positive else 'False Positive'}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    def evaluate_model_performance(self, test_window_hours=24):
        """
        Evaluate model performance on recent data with known labels
        
        Args:
            test_window_hours: Hours of data to evaluate
            
        Returns:
            dict: Performance metrics
        """
        logger.info(f"Evaluating model performance on last {test_window_hours}h")
        
        try:
            # Query for verified anomalies (with feedback)
            query = {
                "size": 1000,
                "query": {
                    "bool": {
                        "must": [
                            {"range": {"@timestamp": {"gte": f"now-{test_window_hours}h"}}},
                            {"exists": {"field": "is_verified"}}
                        ]
                    }
                }
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            tp = 0
            fp = 0
            
            for hit in response['hits']['hits']:
                if hit['_source'].get('is_true_positive', False):
                    tp += 1
                else:
                    fp += 1
            
            # Update metrics
            self.metrics_history['true_positives'] = tp
            self.metrics_history['false_positives'] = fp
            
            # Calculate metrics
            metrics = self.calculate_precision_metrics()
            
            # Index performance report
            self.index_performance_report(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return {}
    
    def index_performance_report(self, metrics):
        """
        Index model performance metrics to Elasticsearch
        
        Args:
            metrics: Performance metrics dictionary
        """
        try:
            report = {
                '@timestamp': datetime.now().isoformat(),
                'model_name': 'isolation_forest_v1',
                'metrics': metrics,
                'model_params': {
                    'n_estimators': 100,
                    'contamination': 0.01,
                    'training_window_hours': 1
                }
            }
            
            self.es.index(
                index=f"firefox-ml-performance-{datetime.now().strftime('%Y.%m')}",
                document=report
            )
            
            logger.info("Performance report indexed")
            
        except Exception as e:
            logger.error(f"Error indexing performance report: {e}")
    
    def optimize_threshold(self, target_precision=0.85):
        """
        Optimize anomaly score threshold to achieve target precision
        
        Args:
            target_precision: Desired precision level (0-1)
            
        Returns:
            float: Optimized threshold
        """
        logger.info(f"Optimizing threshold for precision >= {target_precision}")
        
        try:
            # Get recent anomalies with scores
            query = {
                "size": 1000,
                "query": {
                    "range": {"@timestamp": {"gte": "now-7d"}}
                },
                "sort": [{"anomaly_score": "desc"}]
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            scores = []
            labels = []
            
            for hit in response['hits']['hits']:
                src = hit['_source']
                if 'is_verified' in src:
                    scores.append(src['anomaly_score'])
                    labels.append(1 if src.get('is_true_positive', False) else 0)
            
            if len(scores) < 10:
                logger.warning("Not enough verified data for threshold optimization")
                return -0.5
            
            # Find optimal threshold
            best_threshold = -0.5
            best_precision = 0
            
            for threshold_score in np.linspace(min(scores), max(scores), 50):
                tp = sum(1 for s, l in zip(scores, labels) if s >= threshold_score and l == 1)
                fp = sum(1 for s, l in zip(scores, labels) if s >= threshold_score and l == 0)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                if precision >= target_precision and precision > best_precision:
                    best_precision = precision
                    best_threshold = (threshold_score - 100) / 100  # Convert to score_samples scale
            
            logger.info(f"Optimized threshold: {best_threshold:.4f} (Precision: {best_precision:.2%})")
            
            return best_threshold
            
        except Exception as e:
            logger.error(f"Error optimizing threshold: {e}")
            return -0.5

    def detect_anomalies(self, threshold_score=-0.5):
        """Detect anomalies in recent data"""
        if self.model is None:
            logger.warning("Model not trained yet, skipping detection")
            return []
        
        logger.info("Detecting anomalies")
        
        try:
            df = self.extract_features(time_window_minutes=1)
            
            if len(df) == 0:
                logger.warning("No data to analyze")
                return []
            
            X = df[self.feature_names].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            predictions = self.model.predict(X_scaled)
            anomaly_scores = self.model.score_samples(X_scaled)
            
            anomalies = []
            for idx, (pred, score) in enumerate(zip(predictions, anomaly_scores)):
                if pred == -1 or score < threshold_score:
                    anomaly = {
                        '@timestamp': df.iloc[idx]['timestamp'].isoformat(),
                        'anomaly_score': float(abs(score) * 100),
                        'raw_score': float(score),  # NEW: Store raw score
                        'anomaly_type': self._classify_anomaly(df.iloc[idx]),
                        'severity': self._calculate_severity(score),
                        'confidence': float(abs(score)),
                        'model_name': 'isolation_forest_v1',
                        'features': df.iloc[idx][self.feature_names].to_dict(),
                        'recommendation': self._generate_recommendation(df.iloc[idx]),
                        'is_verified': False,  # NEW: Awaiting verification
                        'is_true_positive': None  # NEW: To be set by feedback
                    }
                    anomalies.append(anomaly)
            
            logger.info(f"Detected {len(anomalies)} anomalies")
            
            # Track detection count
            self.metrics_history['detections'].append({
                'timestamp': datetime.now().isoformat(),
                'count': len(anomalies)
            })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error during anomaly detection: {e}")
            return []

    def run_detection_loop(self, interval_seconds=60):
        """Run continuous anomaly detection"""
        import time
        
        logger.info(f"Starting anomaly detection loop (interval: {interval_seconds}s)")
        
        # Wait for Elasticsearch to have data
        logger.info("Waiting for data to be available in Elasticsearch...")
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.es.count(index="firefox-logs-*")
                doc_count = response['count']
                
                if doc_count > 0:
                    logger.info(f"Found {doc_count} documents in Elasticsearch")
                    break
                else:
                    logger.info(f"No documents yet (attempt {retry_count + 1}/{max_retries}). Waiting 30s...")
                    time.sleep(30)
                    retry_count += 1
                    
            except Exception as e:
                logger.warning(f"Elasticsearch not ready (attempt {retry_count + 1}/{max_retries}): {e}")
                time.sleep(30)
                retry_count += 1
        
        # Initial training
        logger.info("Attempting initial model training...")
        training_success = False
        
        while not training_success:
            training_success = self.train_model()
            if not training_success:
                logger.info("Waiting 60s before retry...")
                time.sleep(60)
        
        # Main detection loop
        while True:
            try:
                # Detect anomalies
                anomalies = self.detect_anomalies()
                
                # Index to Elasticsearch
                self.index_anomalies(anomalies)
                
                # Retrain periodically (every hour)
                if datetime.now().minute == 0:
                    logger.info("Hourly retraining triggered")
                    self.train_model()
                
                time.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Stopping anomaly detection loop")
                break
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(interval_seconds)

    def get_recent_anomalies_api(self, count=10):
        """
        REST API endpoint to get recent anomalies
        
        Args:
            count: Number of anomalies to return
            
        Returns:
            List of recent anomalies with details
        """
        try:
            query = {
                "size": count,
                "sort": [{"@timestamp": {"order": "desc"}}]
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            anomalies = []
            for hit in response['hits']['hits']:
                anomalies.append(hit['_source'])
            
            return {
                "total": response['hits']['total']['value'],
                "anomalies": anomalies
            }
        except Exception as e:
            logger.error(f"Error fetching anomalies: {e}")
            return {"error": str(e)}
    
    def get_anomaly_summary_api(self):
        """
        REST API endpoint to get anomaly summary statistics
        
        Returns:
            Summary statistics of detected anomalies
        """
        try:
            # Get total count
            count_response = self.es.count(index="firefox-ml-anomalies-*")
            total = count_response['count']
            
            # Get aggregations
            query = {
                "size": 0,
                "aggs": {
                    "by_type": {
                        "terms": {"field": "anomaly_type.keyword"}
                    },
                    "by_severity": {
                        "terms": {"field": "severity.keyword"}
                    },
                    "avg_score": {
                        "avg": {"field": "anomaly_score"}
                    },
                    "max_score": {
                        "max": {"field": "anomaly_score"}
                    }
                }
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            return {
                "total_anomalies": total,
                "average_score": response['aggregations']['avg_score']['value'],
                "max_score": response['aggregations']['max_score']['value'],
                "by_type": response['aggregations']['by_type']['buckets'],
                "by_severity": response['aggregations']['by_severity']['buckets']
            }
        except Exception as e:
            logger.error(f"Error fetching summary: {e}")
            return {"error": str(e)}

    def evaluate_score_distribution(self):
        """
        Evaluate model by analyzing anomaly score distribution
        No labels required - purely statistical validation
        
        Returns:
            dict: Distribution metrics
        """
        logger.info("Evaluating anomaly score distribution")
        
        try:
            # Get all recent anomalies
            query = {
                "size": 1000,
                "query": {
                    "range": {"@timestamp": {"gte": "now-24h"}}
                }
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            scores = [hit['_source']['anomaly_score'] for hit in response['hits']['hits']]
            
            if not scores:
                return {}
            
            scores_array = np.array(scores)
            
            metrics = {
                'total_anomalies': len(scores),
                'mean_score': float(np.mean(scores_array)),
                'median_score': float(np.median(scores_array)),
                'std_score': float(np.std(scores_array)),
                'min_score': float(np.min(scores_array)),
                'max_score': float(np.max(scores_array)),
                'percentile_75': float(np.percentile(scores_array, 75)),
                'percentile_90': float(np.percentile(scores_array, 90)),
                'percentile_95': float(np.percentile(scores_array, 95)),
                'score_range': float(np.max(scores_array) - np.min(scores_array)),
                'coefficient_variation': float(np.std(scores_array) / np.mean(scores_array)) if np.mean(scores_array) > 0 else 0
            }
            
            # Quality indicators (heuristic-based)
            # Good model: High separation between normal and anomalies
            if metrics['mean_score'] > 65 and metrics['std_score'] < 20:
                metrics['quality_indicator'] = 'Good - Consistent high scores'
            elif metrics['std_score'] > 30:
                metrics['quality_indicator'] = 'Poor - High variance, may need tuning'
            elif metrics['mean_score'] < 55:
                metrics['quality_indicator'] = 'Warning - Low average scores'
            else:
                metrics['quality_indicator'] = 'Fair - Acceptable performance'
            
            logger.info(f"Score distribution: Mean={metrics['mean_score']:.2f}, Std={metrics['std_score']:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating score distribution: {e}")
            return {}

    def evaluate_temporal_stability(self):
        """
        Evaluate if model detects consistent patterns over time
        Stable detection = good model
        
        Returns:
            dict: Temporal stability metrics
        """
        logger.info("Evaluating temporal stability")
        
        try:
            # Get anomalies grouped by hour
            query = {
                "size": 0,
                "query": {
                    "range": {"@timestamp": {"gte": "now-24h"}}
                },
                "aggs": {
                    "by_hour": {
                        "date_histogram": {
                            "field": "@timestamp",
                            "fixed_interval": "1h"
                        },
                        "aggs": {
                            "avg_score": {"avg": {"field": "anomaly_score"}},
                            "count": {"value_count": {"field": "anomaly_score"}}
                        }
                    }
                }
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            buckets = response['aggregations']['by_hour']['buckets']
            
            hourly_counts = [b['doc_count'] for b in buckets if b['doc_count'] > 0]
            hourly_scores = [b['avg_score']['value'] for b in buckets if b.get('avg_score', {}).get('value')]
            
            if not hourly_counts:
                return {}
            
            metrics = {
                'hours_analyzed': len(buckets),
                'hours_with_anomalies': len(hourly_counts),
                'avg_anomalies_per_hour': float(np.mean(hourly_counts)),
                'std_anomalies_per_hour': float(np.std(hourly_counts)),
                'avg_score_variation': float(np.std(hourly_scores)) if hourly_scores else 0,
                'detection_consistency': float(1 - (np.std(hourly_counts) / np.mean(hourly_counts))) if np.mean(hourly_counts) > 0 else 0
            }
            
            # Stability indicator
            if metrics['detection_consistency'] > 0.7:
                metrics['stability_indicator'] = 'Stable - Consistent detection rate'
            elif metrics['detection_consistency'] > 0.4:
                metrics['stability_indicator'] = 'Moderate - Some variation'
            else:
                metrics['stability_indicator'] = 'Unstable - High variation in detection'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating temporal stability: {e}")
            return {}

    def evaluate_feature_importance(self):
        """
        Analyze which features contribute most to anomaly detection
        No labels needed - based on feature values in detected anomalies
        
        Returns:
            dict: Feature importance metrics
        """
        logger.info("Evaluating feature importance")
        
        try:
            # Get recent anomalies with features
            query = {
                "size": 100,
                "query": {
                    "range": {"@timestamp": {"gte": "now-24h"}}
                },
                "sort": [{"anomaly_score": "desc"}]
            }
            
            response = self.es.search(index="firefox-ml-anomalies-*", **query)
            
            # Extract features from anomalies
            feature_values = {fname: [] for fname in self.feature_names}
            
            for hit in response['hits']['hits']:
                features = hit['_source'].get('features', {})
                for fname in self.feature_names:
                    if fname in features:
                        feature_values[fname].append(features[fname])
            
            # Calculate statistics for each feature
            feature_stats = {}
            for fname, values in feature_values.items():
                if values:
                    feature_stats[fname] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'coefficient_variation': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else 0
                    }
            
            # Rank features by variation (high variation = likely important)
            ranked_features = sorted(
                feature_stats.items(),
                key=lambda x: x[1]['coefficient_variation'],
                reverse=True
            )
            
            return {
                'feature_statistics': feature_stats,
                'ranked_by_variation': [f[0] for f in ranked_features[:5]],
                'most_variable_feature': ranked_features[0][0] if ranked_features else None
            }
            
        except Exception as e:
            logger.error(f"Error evaluating feature importance: {e}")
            return {}

    def cross_validate_with_synthetic_anomalies(self):
        """
        Create synthetic anomalies and test if model detects them
        Self-validation without manual feedback
        
        Returns:
            dict: Validation metrics
        """
        logger.info("Cross-validating with synthetic anomalies")
        
        try:
            # Get normal data
            df = self.extract_features(time_window_minutes=1, lookback_hours=1)
            
            if len(df) < 10:
                return {}
            
            # Create synthetic anomalies by perturbing normal data
            synthetic_anomalies = []
            normal_samples = []
            
            for idx in range(min(10, len(df))):
                row = df.iloc[idx].copy()
                normal_samples.append(row[self.feature_names].values)
                
                # Create anomaly by multiplying key features
                anomaly_row = row.copy()
                anomaly_row['avg_duration'] *= 5  # 5x longer
                anomaly_row['failure_rate'] *= 3  # 3x more failures
                anomaly_row['warning_count'] *= 4  # 4x more warnings
                
                synthetic_anomalies.append(anomaly_row[self.feature_names].values)
            
            # Test model on synthetic data
            normal_array = np.array(normal_samples)
            anomaly_array = np.array(synthetic_anomalies)
            
            normal_scaled = self.scaler.transform(normal_array)
            anomaly_scaled = self.scaler.transform(anomaly_array)
            
            normal_predictions = self.model.predict(normal_scaled)
            anomaly_predictions = self.model.predict(anomaly_scaled)
            
            # Count correct predictions
            normal_correct = sum(1 for p in normal_predictions if p == 1)  # Should be 1 (normal)
            anomaly_correct = sum(1 for p in anomaly_predictions if p == -1)  # Should be -1 (anomaly)
            
            metrics = {
                'normal_accuracy': float(normal_correct / len(normal_predictions)),
                'anomaly_detection_rate': float(anomaly_correct / len(anomaly_predictions)),
                'overall_accuracy': float((normal_correct + anomaly_correct) / (len(normal_predictions) + len(anomaly_predictions))),
                'test_samples': len(normal_predictions) + len(anomaly_predictions)
            }
            
            if metrics['overall_accuracy'] > 0.8:
                metrics['validation_result'] = 'Pass - Good discrimination'
            elif metrics['overall_accuracy'] > 0.6:
                metrics['validation_result'] = 'Marginal - Acceptable'
            else:
                metrics['validation_result'] = 'Fail - Poor discrimination'
            
            logger.info(f"Synthetic validation: {metrics['validation_result']}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in synthetic validation: {e}")
            return {}

    def comprehensive_evaluation_no_labels(self):
        """
        Run all unsupervised evaluation methods
        No manual feedback required
        
        Returns:
            dict: Complete evaluation report
        """
        logger.info("Running comprehensive unsupervised evaluation")
        
        report = {
            '@timestamp': datetime.now().isoformat(),
            'model_name': 'isolation_forest_v1',
            'evaluation_type': 'unsupervised',
            'score_distribution': self.evaluate_score_distribution(),
            'temporal_stability': self.evaluate_temporal_stability(),
            'feature_importance': self.evaluate_feature_importance(),
            'synthetic_validation': self.cross_validate_with_synthetic_anomalies()
        }
        
        # Overall health score (0-100)
        health_components = []
        
        # Score distribution health
        score_dist = report['score_distribution']
        if score_dist:
            if score_dist.get('mean_score', 0) > 65:
                health_components.append(80)
            elif score_dist.get('mean_score', 0) > 55:
                health_components.append(60)
            else:
                health_components.append(40)
        
        # Temporal stability health
        temp_stab = report['temporal_stability']
        if temp_stab:
            health_components.append(temp_stab.get('detection_consistency', 0) * 100)
        
        # Synthetic validation health
        synth_val = report['synthetic_validation']
        if synth_val:
            health_components.append(synth_val.get('overall_accuracy', 0) * 100)
        
        report['overall_health_score'] = int(np.mean(health_components)) if health_components else 0
        
        # Index report
        try:
            self.es.index(
                index=f"firefox-ml-evaluation-{datetime.now().strftime('%Y.%m')}",
                document=report
            )
            logger.info(f"Evaluation report indexed. Health score: {report['overall_health_score']}/100")
        except Exception as e:
            logger.error(f"Error indexing evaluation report: {e}")
        
        return report


if __name__ == '__main__':
    import os
    
    # Read configuration from environment
    es_host = os.getenv('ES_HOST', 'localhost')
    es_port = int(os.getenv('ES_PORT', '9200'))
    detection_interval = int(os.getenv('DETECTION_INTERVAL', '60'))
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    training_window = int(os.getenv('TRAINING_WINDOW_HOURS', '1'))
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    logger.info(f"Starting anomaly detector with ES={es_host}:{es_port}, interval={detection_interval}s")
    
    # Run anomaly detector
    detector = FirefoxAnomalyDetector(es_host=es_host, es_port=es_port)
    detector.run_detection_loop(interval_seconds=detection_interval)
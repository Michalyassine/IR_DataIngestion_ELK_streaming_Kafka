"""
Script d'entraînement du modèle ML pour détection d'anomalies
Usage: python ml/train_model.py
"""

from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime

class ModelTrainer:
    def __init__(self, es_host="http://localhost:9200"):
        self.es = Elasticsearch([es_host])
        self.model = None
        self.scaler = None
        
    def fetch_training_data(self, days=30, max_docs=100000):
        """Récupère données d'entraînement depuis Elasticsearch"""
        print(f"🔍 Récupération des données des {days} derniers jours...")
        
        query = {
            "size": max_docs,
            "query": {
                "range": {
                    "@timestamp": {"gte": f"now-{days}d"}
                }
            },
            "sort": [{"@timestamp": "desc"}]
        }
        
        results = self.es.search(index="firefox-logs-*", body=query)
        docs = [hit["_source"] for hit in results["hits"]["hits"]]
        
        print(f"✅ {len(docs):,} documents récupérés")
        return pd.DataFrame(docs)
    
    def engineer_features(self, df):
        """Feature engineering complet"""
        print("\n🔧 Feature engineering...")
        
        features = pd.DataFrame()
        
        # 1. Temporal features
        df["@timestamp"] = pd.to_datetime(df["@timestamp"])
        features["hour"] = df["@timestamp"].dt.hour
        features["day_of_week"] = df["@timestamp"].dt.dayofweek
        
        # 2. Performance features
        features["test_duration"] = df["test_duration"].fillna(0)
        features["duration_log"] = np.log1p(features["test_duration"])
        
        # 3. Rolling statistics (par test)
        df_sorted = df.sort_values("@timestamp")
        features["duration_rolling_mean"] = df_sorted.groupby("test_name")["test_duration"].transform(
            lambda x: x.rolling(window=50, min_periods=1).mean()
        )
        features["duration_rolling_std"] = df_sorted.groupby("test_name")["test_duration"].transform(
            lambda x: x.rolling(window=50, min_periods=1).std().fillna(0)
        )
        
        # 4. Deviation from baseline
        test_medians = df.groupby("test_name")["test_duration"].median()
        features["duration_deviation"] = df.apply(
            lambda row: abs(row.get("test_duration", 0) - test_medians.get(row.get("test_name", ""), 0)),
            axis=1
        )
        
        # 5. Categorical encoding
        features["status_failed"] = (df.get("test_status", pd.Series([""] * len(df))) == "failed").astype(int)
        features["has_warnings"] = (df.get("warning_count", pd.Series([0] * len(df))) > 0).astype(int)
        
        # 6. Text features
        features["test_name_length"] = df.get("test_name", pd.Series([""] * len(df))).str.len()
        features["message_length"] = df.get("message", pd.Series([""] * len(df))).str.len()
        
        print(f"✅ {len(features.columns)} features créées: {list(features.columns)}")
        return features
    
    def train(self, features, contamination=0.01):
        """Entraîne modèle Isolation Forest"""
        print(f"\n🤖 Entraînement du modèle (contamination={contamination})...")
        
        # Normalisation
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(features.fillna(0))
        
        # Modèle
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples=256,
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_scaled)
        
        print("✅ Modèle entraîné!")
        
        # Évaluation sur données d'entraînement
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        anomaly_count = (predictions == -1).sum()
        normal_count = (predictions == 1).sum()
        
        print(f"\n📊 Résultats sur données d'entraînement:")
        print(f"   Normal: {normal_count:,} ({100*normal_count/len(predictions):.2f}%)")
        print(f"   Anomalies: {anomaly_count:,} ({100*anomaly_count/len(predictions):.2f}%)")
        print(f"   Score moyen: {scores.mean():.4f}")
        print(f"   Score min (plus anormal): {scores.min():.4f}")
        print(f"   Score max (plus normal): {scores.max():.4f}")
        
        return predictions, scores
    
    def save_model(self, model_dir="ml/models"):
        """Sauvegarde modèle et métadonnées"""
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "anomaly_detector.pkl")
        
        # Sauvegarder modèle et scaler
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler
        }, model_path)
        
        # Métadonnées
        metadata = {
            "trained_at": datetime.now().isoformat(),
            "model_type": "IsolationForest",
            "contamination": self.model.contamination,
            "n_estimators": self.model.n_estimators,
            "max_samples": self.model.max_samples,
            "features": [
                "hour", "day_of_week", "test_duration", "duration_log",
                "duration_rolling_mean", "duration_rolling_std",
                "duration_deviation", "status_failed", "has_warnings",
                "test_name_length", "message_length"
            ]
        }
        
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n💾 Modèle sauvegardé:")
        print(f"   {model_path}")
        print(f"   {metadata_path}")

def main():
    """Script principal"""
    print("=" * 70)
    print("🎓 ENTRAÎNEMENT DU MODÈLE ML - DÉTECTION D'ANOMALIES")
    print("=" * 70)
    
    # Configuration
    ELASTICSEARCH_HOST = "http://localhost:9200"
    TRAINING_DAYS = 30
    MAX_DOCS = 100000
    CONTAMINATION = 0.01  # 1% anomalies attendues
    
    # Initialisation
    trainer = ModelTrainer(es_host=ELASTICSEARCH_HOST)
    
    # 1. Récupération données
    df = trainer.fetch_training_data(days=TRAINING_DAYS, max_docs=MAX_DOCS)
    
    if len(df) == 0:
        print("❌ Aucune donnée trouvée!")
        return
    
    # 2. Feature engineering
    features = trainer.engineer_features(df)
    
    # 3. Entraînement
    predictions, scores = trainer.train(features, contamination=CONTAMINATION)
    
    # 4. Sauvegarde
    trainer.save_model()
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT TERMINÉ!")
    print("=" * 70)
    print("\n📝 Prochaines étapes:")
    print("   1. Tester le modèle: python ml/evaluate_model.py")
    print("   2. Démarrer le scorer temps réel: python ml/realtime_scorer.py")
    print("   3. Ou démarrer le service Flask: python ml/ml_service.py")

if __name__ == "__main__":
    main()

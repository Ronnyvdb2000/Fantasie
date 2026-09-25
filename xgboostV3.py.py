import os
import joblib
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
import xgboost as xgb


def get_training_data_from_supabase():
  db_url = os.environ.get("SUPABASE_DB_URL")
  if not db_url:
    raise RuntimeError("SUPABASE_DB_URL ontbreekt in de omgeving.")

  conn = psycopg2.connect(db_url)
  try:
    # Haal alle data op uit de tabel 'selecties'
    df = pd.read_sql("SELECT * FROM selecties;", conn)
  finally:
    conn.close()

  return df


def train_xgboost2():
  print("Dataset ophalen uit Supabase (tabel: selecties)...")
  df = get_training_data_from_supabase()

  if df.empty:
    print("De tabel 'selecties' is leeg.")
    return

  # Kies hier je gewenste horizon/target kolom, bijv. 'ret_20d' of 'ret_60d'
  # We maken hier een binaire target van: 1 als rendement > 0, anders 0
  target_column = (
      "ret_20d"  # Pas dit aan naar bijv. 'ret_60d' als je langer wilt meten
  )

  if target_column not in df.columns:
    print(f"Doelkolom '{target_column}' ontbreekt in de tabel.")
    return

  # Maak de target-kolom (is het rendement positief?)
  df["is_profitable"] = (df[target_column] > 0).astype(int)

  # Selecteer automatisch alle bruikbare numerieke kolommen als features,
  # behalve de target zelf, ID's of datums.
  exclude_cols = [
      "id",
      "datum",
      "created_at",
      "ticker",
      "strategie",
      "beurs",
      "grafiek",
      "parameters",
      "sector",
      "rsi_label",
      "macd_label",
      "is_profitable",
      "ret_5d",
      "ret_20d",
      "ret_60d",
  ]

  features = [
      col
      for col in df.select_dtypes(
          include=["number", "boolean"]
      ).columns  # type: ignore[attr-defined]
      if col not in exclude_cols
  ]

  print(f"Aantal geselecteerde features voor training: {len(features)}")

  # Filter rijen waar essentiële data ontbreekt
  df_clean = df.dropna(subset=features + ["is_profitable"])

  if len(df_clean) < 30:
    print(
        f"Nog niet genoeg data met ingevulde features (minimaal 30 vereist, nu"
        f" {len(df_clean)})."
    )
    return

  X = df_clean[features]
  y = df_clean["is_profitable"]

  # Train-test split (80% trainen, 20% testen)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print(f"Start training van xgboostV2 op {len(X_train)} records...")

  # XGBoost Classifier geoptimaliseerd voor financiële data
  model = xgb.XGBClassifier(
      n_estimators=150,
      learning_rate=0.03,
      max_depth=5,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
  )

  model.fit(X_train, y_train)

  # Evalueer de nauwkeurigheid op de testset
  score = model.score(X_test, y_test)
  print(f"Model xgboostV2 succesvol getraind! Test-accuratesse: {score * 100:.2f}%")

  # Sla het getrainde model op
  joblib.dump(model, "xgboostV2_model.pkl")
  print("Getraind model opgeslagen als xgboostV2_model.pkl")


if __name__ == "__main__":
  train_xgboost2()
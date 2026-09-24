9import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from supabase import create_client
import xgboost as xgb


def get_training_data_from_supabase():
  url = os.environ.get("SUPABASE_URL")
  key = os.environ.get("SUPABASE_KEY")
  supabase = create_client(url, key)

  # Haal alle evaluatielogs op uit de database
  response = supabase.table("evaluation_logs").select("*").execute()
  df = pd.DataFrame(response.data)
  return df


def train_xgboost2():
  print("Geoptimaliseerde dataset ophalen uit Supabase voor xgboostV2...")
  df = get_training_data_from_supabase()

  if df.empty or "is_profitable" not in df.columns:
    print(
        "Onvoldoende data of target-kolom 'is_profitable' ontbreekt in Supabase."
    )
    return

  # De slimme parameters (features) die zorgen voor de beste filtering
  features = [
      "win_rate_historical",
      "average_return_historical",
      "sample_size",
      "confluence_score",
      "market_volatility",
  ]

  # Filter rijen waar essentiële data ontbreekt
  df_clean = df.dropna(subset=features + ["is_profitable"])

  if len(df_clean) < 50:
    print(
        "Nog niet genoeg historische rijen met gevulde data om te trainen"
        " (minimaal 50 vereist)."
    )
    return

  X = df_clean[features]
  y = df_clean["is_profitable"]  # 1 = succesvol rendement, 0 = afvaller

  # Train-test split (80% trainen, 20% testen)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print(
      f"Start training van xgboostV2 op {len(X_train)} records met features:"
      f" {features}..."
  )

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

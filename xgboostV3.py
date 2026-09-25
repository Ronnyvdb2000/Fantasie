import os
import joblib
import pandas as pd
import psycopg2
from sklearn.model_selection import train_test_split
import xgboost as xgb


HORIZONS = ["10d", "30d", "60d"]

FEATURE_COLUMNS = [
    "atr14",
    "atr14_pct",
    "rsi14",
    "ibs",
    "ma50",
    "ma200",
    "pct_from_ma50",
    "pct_from_ma200",
    "vol_ratio_20d",
    "high52w",
    "pct_from_high52w",
]

JOIN_QUERY = """
WITH fr_dedup AS (
    SELECT DISTINCT ON (ticker, datum)
        ticker, datum, fwd_ret_10d, fwd_ret_30d, fwd_ret_60d
    FROM forward_returns
    ORDER BY ticker, datum
)
SELECT
    gt.ticker,
    gt.datum,
    gt.atr14,
    gt.atr14_pct,
    gt.rsi14,
    gt.ibs,
    gt.ma50,
    gt.ma200,
    gt.pct_from_ma50,
    gt.pct_from_ma200,
    gt.vol_ratio_20d,
    gt.high52w,
    gt.pct_from_high52w,
    fr_dedup.fwd_ret_10d,
    fr_dedup.fwd_ret_30d,
    fr_dedup.fwd_ret_60d
FROM generieke_technicals gt
JOIN fr_dedup ON fr_dedup.ticker = gt.ticker AND fr_dedup.datum = gt.datum;
"""


def get_training_data_from_supabase():
  db_url = os.environ.get("SUPABASE_DB_URL")
  if not db_url:
    raise RuntimeError("SUPABASE_DB_URL ontbreekt in de omgeving.")

  conn = psycopg2.connect(db_url)
  try:
    df = pd.read_sql(JOIN_QUERY, conn)
  finally:
    conn.close()

  return df


def train_voor_horizon(df: pd.DataFrame, horizon: str) -> None:
  target_column = f"fwd_ret_{horizon}"

  if target_column not in df.columns:
    print(f"[{horizon}] Doelkolom '{target_column}' ontbreekt in de gejoinde dataset.")
    return

  df_horizon = df.copy()
  df_horizon["is_profitable"] = (df_horizon[target_column] > 0).astype(int)

  df_clean = df_horizon.dropna(subset=FEATURE_COLUMNS + [target_column])

  if len(df_clean) < 30:
    print(
        f"[{horizon}] Nog niet genoeg data met ingevulde features (minimaal 30"
        f" vereist, nu {len(df_clean)})."
    )
    return

  X = df_clean[FEATURE_COLUMNS]
  y = df_clean["is_profitable"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print(f"[{horizon}] Start training op {len(X_train)} records...")

  model = xgb.XGBClassifier(
      n_estimators=150,
      learning_rate=0.03,
      max_depth=5,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
  )

  model.fit(X_train, y_train)

  score = model.score(X_test, y_test)
  print(f"[{horizon}] Model succesvol getraind! Test-accuratesse: {score * 100:.2f}%")

  bestandsnaam = f"xgboostV3_{horizon}_model.pkl"
  joblib.dump(model, bestandsnaam)
  print(f"[{horizon}] Getraind model opgeslagen als {bestandsnaam}")


def train_xgboost3():
  print(
      "Dataset ophalen uit Supabase (generieke_technicals + forward_returns,"
      " join op ticker+datum, horizons: " + ", ".join(HORIZONS) + ")..."
  )
  df = get_training_data_from_supabase()

  if df.empty:
    print("Geen gejoinde rijen tussen generieke_technicals en forward_returns.")
    return

  print(f"Aantal geselecteerde features per model: {len(FEATURE_COLUMNS)}")

  for horizon in HORIZONS:
    train_voor_horizon(df, horizon)


if __name__ == "__main__":
  train_xgboost3()

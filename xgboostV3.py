import os
import joblib
import pandas as pd
import psycopg2
from sklearn.metrics import roc_auc_score
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

# Aandeel van de test-set dat als "top-selectie" wordt behandeld bij het
# vergelijken van model-ranking vs. baseline-ranking (bv. 0.20 = top 20%).
TOP_N_FRACTIE = 0.20

# Enige tot nu toe bevestigd significante losse voorspeller in dit project
# (analyseer_forward_correlaties.py, rho -0.179, p_adj 0.0000): hoe verder
# een aandeel op selectiemoment ONDER zijn 50-daags gemiddelde noteert, hoe
# hoger het rendement nadien neigt te zijn. Dient hier als baseline om te
# checken of XGBoost er effectief iets aan toevoegt.
BASELINE_KOLOM = "pct_from_ma50"

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


def print_top_n_vergelijking(test_df: pd.DataFrame, target_column: str, horizon: str) -> None:
  """Vergelijkt het gemiddelde forward-rendement van de top-N% aandelen
  volgens het model met (a) de top-N% volgens de simpele baseline
  (verst onder MA50) en (b) het gemiddelde over de hele test-set.
  Dit is de metric die echt telt voor "krijg ik aandelen met een hogere
  kans op goede performantie" -- accuratesse alleen zegt dat niet."""
  n_top = max(1, int(len(test_df) * TOP_N_FRACTIE))

  model_ranking = test_df.sort_values("proba", ascending=False)
  gemiddeld_model = model_ranking.head(n_top)[target_column].mean()

  # rho is negatief: meer negatieve pct_from_ma50 (verder ONDER MA50) hoort
  # bij hoger forward-rendement, dus oplopend sorteren en de eerste n_top nemen.
  baseline_ranking = test_df.sort_values(BASELINE_KOLOM, ascending=True)
  gemiddeld_baseline = baseline_ranking.head(n_top)[target_column].mean()

  gemiddeld_alle = test_df[target_column].mean()

  print(
      f"[{horizon}] Top {n_top}/{len(test_df)} volgens model: gemiddeld"
      f" {target_column}={gemiddeld_model:.3f}%  |  volgens baseline"
      f" ({BASELINE_KOLOM}): {gemiddeld_baseline:.3f}%  |  gemiddelde van de"
      f" hele test-set: {gemiddeld_alle:.3f}%"
  )
  if gemiddeld_model <= gemiddeld_baseline:
    print(
        f"[{horizon}] LET OP: het model doet het niet beter dan de simpele"
        f" baseline ({BASELINE_KOLOM}) -- voegt op deze horizon geen"
        " aantoonbare waarde toe boven het reeds gekende signaal."
    )


def train_voor_horizon(df: pd.DataFrame, horizon: str) -> None:
  target_column = f"fwd_ret_{horizon}"

  if target_column not in df.columns:
    print(f"[{horizon}] Doelkolom '{target_column}' ontbreekt in de gejoinde dataset.")
    return

  df_horizon = df.copy()
  df_horizon["is_profitable"] = (df_horizon[target_column] > 0).astype(int)

  df_clean = df_horizon.dropna(subset=FEATURE_COLUMNS + [target_column, "datum"])

  if len(df_clean) < 30:
    print(
        f"[{horizon}] Nog niet genoeg data met ingevulde features (minimaal 30"
        f" vereist, nu {len(df_clean)})."
    )
    return

  # Tijdsgebaseerde split i.p.v. willekeurige split: bij een willekeurige
  # split lekt toekomstige marktinformatie de trainingsset in (overlappende
  # periodes, gecorreleerde aandelen), wat de test-accuratesse optimistischer
  # laat lijken dan wat je live zou halen. Hier: train op de oudste 80% van
  # de datums, test strikt op de meest recente 20%.
  df_clean = df_clean.sort_values("datum").reset_index(drop=True)
  split_idx = int(len(df_clean) * 0.8)
  split_datum = df_clean.iloc[split_idx]["datum"]

  train_df = df_clean[df_clean["datum"] <= split_datum]
  test_df = df_clean[df_clean["datum"] > split_datum]

  if len(test_df) < 10:
    print(
        f"[{horizon}] Te weinig recente data voor een betrouwbare tijds-"
        f"gebaseerde test-set (nu {len(test_df)}, minimaal 10 vereist)."
    )
    return

  if train_df["is_profitable"].nunique() < 2:
    print(
        f"[{horizon}] Trainingsset bevat maar 1 klasse (allemaal winst of"
        " allemaal verlies) -- kan geen classifier trainen op deze periode."
    )
    return

  X_train, y_train = train_df[FEATURE_COLUMNS], train_df["is_profitable"]
  X_test, y_test = test_df[FEATURE_COLUMNS], test_df["is_profitable"]

  print(
      f"[{horizon}] Start training op {len(X_train)} records (tot en met"
      f" {split_datum}), test op {len(X_test)} recentere records..."
  )

  model = xgb.XGBClassifier(
      n_estimators=150,
      learning_rate=0.03,
      max_depth=5,
      subsample=0.8,
      colsample_bytree=0.8,
      random_state=42,
  )

  model.fit(X_train, y_train)

  accuracy = model.score(X_test, y_test)
  proba = model.predict_proba(X_test)[:, 1]

  if y_test.nunique() < 2:
    auc = float("nan")
    print(
        f"[{horizon}] AUC niet berekenbaar: test-set bevat maar 1 klasse"
        " (alle labels identiek in deze periode)."
    )
  else:
    auc = roc_auc_score(y_test, proba)

  print(
      f"[{horizon}] Model getraind. Test-accuratesse: {accuracy * 100:.2f}%"
      f"  |  AUC: {auc:.3f}" + ("  (0.50 = geen edge boven toeval)" if auc == auc else "")
  )

  test_eval = test_df.copy()
  test_eval["proba"] = proba
  print_top_n_vergelijking(test_eval, target_column, horizon)

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

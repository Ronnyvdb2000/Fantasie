# features.py
"""
Feature-selectie voor de parameter-analyse en het meta-model.

ANALYSE_WHITELIST: ruimere set voor de correlatie-analyse. Bevat ook
    spiegel-features (ma200, ma50, high52w) zodat de decay-curve het hele
    trendconcept laat zien.

MODEL_FEATURES: strikte set voor het meta-model. Één representant per
    concept, geen spiegel-features (die zouden het model dubbel gewicht
    geven op hetzelfde signaal). Gebaseerd op de run van 2026-10-04 met
    --min-strategieen 3: alle features hieronder hebben >=11 strategieën
    en zijn significant op 20d.
"""

ANALYSE_WHITELIST = [
    # Trendpositie
    "pct_from_ma200", "pct_from_ma50", "pct_from_high52w",
    # Trendpositie (spiegels, alleen voor analyse)
    "ma200", "ma50", "high52w",
    # Volatiliteit
    "atr14", "atr14_pct", "atr",
    # Composite
    "score", "rank",
    # Context / controles
    "ibs", "vol_ratio_20d", "rsi14",
    "market_cap", "div_yield",
    "stop",
]

# Strikte set voor het model. Een feature per concept.
MODEL_FEATURES = [
    "pct_from_ma200",     # trendpositie lange termijn
    "pct_from_high52w",   # momentum-bevestiging
    "atr14_pct",          # volatiliteit (genormaliseerd)
    "score",              # composite van de bots zelf
    "pct_from_ma50",      # trendpositie korte termijn
    "ibs",                # intraday positie (control)
    "vol_ratio_20d",      # volume (control)
]

# Target-horizon voor het model (dagen)
MODEL_HORIZON = 20
MODEL_TARGET = f"fwd_ret_{MODEL_HORIZON}d"

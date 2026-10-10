# --- EMA8, EMA20 en afgeleiden -------------------------------------
try:
    ema8_reeks = close.ewm(span=8, adjust=False).mean()
    ema20_reeks = close.ewm(span=20, adjust=False).mean()

    ema8_laatste = float(ema8_reeks.iloc[-1]) if len(ema8_reeks) else float("nan")
    ema20_laatste = float(ema20_reeks.iloc[-1]) if len(ema20_reeks) else float("nan")
    close_laatste = float(close.iloc[-1]) if len(close) else float("nan")

    EMA8 = ema8_laatste if not math.isnan(ema8_laatste) else None
    EMA20 = ema20_laatste if not math.isnan(ema20_laatste) else None

    if EMA8 and EMA8 > 0 and not math.isnan(close_laatste):
        PCT_FROM_EMA8 = round((close_laatste - EMA8) / EMA8 * 100, 4)
    else:
        PCT_FROM_EMA8 = None

    if EMA20 and EMA20 > 0 and not math.isnan(close_laatste):
        PCT_FROM_EMA20 = round((close_laatste - EMA20) / EMA20 * 100, 4)
    else:
        PCT_FROM_EMA20 = None

    if EMA8 is not None and EMA20 is not None:
        EMA8_MINUS_EMA20 = round(EMA8 - EMA20, 4)
    else:
        EMA8_MINUS_EMA20 = None

except Exception:
    EMA8 = EMA20 = PCT_FROM_EMA8 = PCT_FROM_EMA20 = EMA8_MINUS_EMA20 = None

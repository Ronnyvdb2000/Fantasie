# Fantasie — Naamgevingsoverzicht

*Er bestond nog geen handleiding voor dit repo — dit document is dus nieuw,
niet een update van iets bestaands. Gebaseerd op een volledige doorlichting
van alle 49 Python-bestanden in de repo-root op 2026-09-24.*

## 1. Snel overzicht (het "knipperend van het oog"-plaatje)

| Prefix / patroon | Betekenis | Voorbeelden |
|---|---|---|
| `bot_00*` / `bot_01*` | Een individuele selectiestrategie die naar `selecties` logt | `bot_00kr`, `bot_01hoogl` |
| `a_*` | Combineert/rangschikt de output van meerdere bots samen | `a_trade`, `a_trade_combi` |
| `analyse_*` / `analyseer_*` | Leest data terug en meet prestatie/correlatie (geen selectie, geen logging naar `selecties`) | `analyse_forward_returns`, `analyseer_weekly_correlaties` |
| `bouw_*` | Vult een feature-store-tabel (generiek, strategie-onafhankelijk) | `bouw_forward_returns`, `bouw_generieke_technicals` |
| `weekly_*` | Wekelijkse rapportage/opvolging, los van de dagelijkse bot-strategieën | `weekly_report`, `weekly_report_opvolg` |
| `db_logger.py` | De ene gedeelde databank-schrijflaag die alle bots gebruiken | — |
| `test_*` | Eenmalige connectiviteitstestjes, geen strategie | `test_telegram`, `test_verbinding` |
| `tickers_0NN{a,d,m,x}.txt` | Per-beurs tickerlijsten (a=alle, d=?, m/x=kwaliteitsgefilterd) | `tickers_048a.txt` |

**Belangrijk:** dit is de betekenis zoals ze *nu, na het feit* uit de bestanden
af te leiden valt — het was tot nu toe nergens neergeschreven, dus nieuwe
bestanden volgden dit patroon niet altijd bewust (zie sectie 3).

## 2. Categorieën in detail

### 2.1 Individuele strategie-bots (`bot_00*`, `bot_01*`)
Elke bot screent zelfstandig een universum en logt zijn top-kandidaten naar
de gedeelde `selecties`-tabel via `db_logger.py`.

**Fundamentele screeners:** `bot_00graham`, `bot_00greenblatt`,
`bot_00oshaughnessy`, `bot_00Fisher`, `bot_01hoogl`, `bot_01kasstr`
**Technisch/kwantitatief:** `bot_00cs`, `bot_00db`, `bot_00dm`, `bot_00ms`,
`bot_00vcp`, `bot_00mr`, `bot_00supervar`, `bot_00xxxV2`,
`bot_01cointegr`, `bot_01marktsent`, `bot_01repititief`, `bot_01volhunter`,
`bot_01beurssign`
**Machine learning:** `bot_01xgboost`, `bot_01xgboostMeta`
**Geen strategie (rapportage/infra), toch `bot_`-genaamd:** `bot_00mail`,
`bot_00ultmail` (digest-mailers), `bot_041m`, `bot_041mV2` (tickerlijst-filter-infra)

### 2.2 Combineerders (`a_*`)
Lezen de gedeelde `selecties`-tabel terug en rangschikken op overlap tussen
strategieën. `a_trade` (som over lookback-periode), `a_trade_q` (= "vers
signaal", enkel de meest recente dag), `a_trade_combi` (gewogen stemming
over een vaste subset bots).

### 2.3 Analyse-scripts (`analyse_*` / `analyseer_*`)
Puur lezend, meten prestatie/correlatie, schrijven niets naar `selecties`.
`analyse_selecties`, `analyse_forward_returns`, `analyse_parameter_correlatie`,
`analyseer_weekly_correlaties`, `analyseer_forward_correlaties`.

### 2.4 Feature-store-bouwers (`bouw_*`)
Vullen een gedeelde, strategie-onafhankelijke tabel. `bouw_forward_returns`,
`bouw_generieke_technicals`.

### 2.5 Wekelijkse rapportage (`weekly_*`)
Los van de dagelijkse bot-cyclus. `weekly_report` (Hall-of-Fame, a-lijsten),
`weekly_report_x` (zelfde maar x-lijsten), `weekly_report_opvolg` (meet
week_perf van de Hall-of-Fame-tickers terug), `weekly_rep_backtest`
(backtest van weekly_report's picks), `weekly_db_opvolg` (databanklaag
voor weekly_report_opvolg, geen zelfstandig script).

### 2.6 Overig / eenmalig
`test_telegram`, `test_verbinding` (connectiviteitstests), `backtest_5jaar`,
`backtstbeter`, `tradingagents_bridge` (gekoppeld aan het gepauzeerde
TradingAgents-experiment), `top_dist_sma50` (ad hoc opzoektool op het
`generieke_technicals`-signaal).

## 3. Bekende inconsistenties (gevonden 2026-09-24)

Dit is de kern van waar je op doelde — met naam en toenaam, zodat het niet
in vaagheid blijft hangen:

- **`bot_00` vs `bot_01` betekent NIETS.** Geen enkel patroon gevonden — het
  is geen versie, geen categorie, geen datum. `bot_00graham`,
  `bot_00greenblatt` en `bot_00oshaughnessy` zijn evengoed fundamentele
  screeners als `bot_01hoogl`/`bot_01kasstr`, maar zitten toch in de
  "verkeerde" groep. Zelfs mijn eigen geheugen had dit fout genoteerd tot
  ik de repo vandaag volledig doorlichtte.
- **Inconsistente hoofdlettergebruik:** `bot_00Fisher` (hoofdletter F) is de
  enige uitzondering tussen verder allemaal lowercase namen
  (`graham`, `greenblatt`, `kr`, `db`, `dm`, `ms`, `vcp`, `cs`, `mr`).
- **`backtstbeter.py`** — een tikfout in de naam zelf ("backtst" i.p.v.
  "backtest"), en de relatie met `backtest_5jaar.py` is niet uit de naam af
  te leiden (lijkt een latere, verbeterde variant, maar welke is nu de
  "echte"?).
- **`analyse_*` vs `analyseer_*`** — exact dezelfde functiecategorie
  (lezen + prestatie meten), maar twee verschillende werkwoordsvormen
  door elkaar (`analyse_forward_returns` naast `analyseer_weekly_correlaties`).
- **`a_top_dist_sma50.py`** — de enige `a_`-naam die niet bij de
  combineer-familie (`a_trade*`) hoort; de `a_`-prefix suggereert hier ten
  onrechte een verband met die groep.
- **4 gelijkaardig klinkende `weekly_report*`-namen**
  (`weekly_report`, `weekly_report_x`, `weekly_report_opvolg`,
  `weekly_rep_backtest`) waarvan de onderlinge relatie niet uit de naam
  zelf blijkt, enkel uit de inhoud.
- **`test_telegram.py` en `test_verbinding.py`** doen vrijwel hetzelfde
  (Telegram-verbinding testen) — mogelijk overbodige duplicatie.

## 4. Voorstel: conventie voor nieuwe bestanden vanaf nu

Geen herdoop-actie van bestaande bestanden hierin vervat — enkel een
afspraak voor wat je hierna toevoegt, zodat de chaos niet verder groeit:

1. **Nieuwe strategie-bot:** `bot_NN_<naam>.py` met **doorlopende
   nummering** (geen 00/01-knip meer die niets betekent) — of, als de
   nummering sowieso al betekenisloos is, overweeg ze gewoon te laten
   vallen: `bot_<naam>.py`.
2. **Nieuwe combineerder:** `combi_<naam>.py` i.p.v. opnieuw `a_*` (die
   prefix is nu dubbelzinnig geworden door punt 3.5 hierboven).
3. **Nieuw analysescript:** kies **één** vorm en blijf erbij — voorstel:
   `analyseer_<naam>.py` (voltooid deelwoord, "wat dit doet"), dus toekomstige
   scripts niet meer `analyse_*`.
4. **Nieuwe feature-store-tabel:** `bouw_<tabelnaam>.py`, ongewijzigd.
5. **Ad hoc/eenmalig/test:** `test_<naam>.py`, en overweeg zulke bestanden
   na gebruik gewoon te verwijderen i.p.v. te laten liggen (zoals bij
   `test_diag_rank_kolommen.py` eerder al gedaan — self-deletende workflow).

---
*Wil je dat ik hier ook nog een sectie aan toevoeg met alle Supabase-tabellen
(`selecties`, `forward_returns`, `generieke_technicals`, `weekly_toppers`,
`weekly_topper_parameters`, `mr_trades`, `mr_snapshots`) en hun onderlinge
relatie? Dat stond buiten de scope van "naamgeving van bots" maar sluit er
inhoudelijk wel bij aan.*

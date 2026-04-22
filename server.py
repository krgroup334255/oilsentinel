"""
Oil Inventory Dashboard — Backend Server
Proxies EIA, NewsAPI and OpenAI (keys stay server-side).
World Bank and GDELT are called directly from the browser.

Requirements:
    pip install flask flask-cors requests apscheduler openai

Run:
    python server.py

The server starts on http://localhost:5001
It also runs a background scheduler that refreshes EIA data every 24 hours.

API Keys (set as environment variables or edit the constants below):
    export EIA_API_KEY=your_key_here       # free at https://www.eia.gov/opendata/register.php
    export NEWS_API_KEY=your_key_here      # free at https://newsapi.org/register
    export OPENAI_API_KEY=your_key_here    # https://platform.openai.com

Data sources for per-product petroleum stocks:
    USA        — EIA API v2 (weekly, free key)
    EU27+      — Eurostat nrg_stk_oilm API (monthly, no key)
    All others — JODI Oil Database CSV (monthly, no key)
"""

import os
import io
import csv
import re as _re
import time
import logging
import threading
from datetime import datetime, timezone

# Load .env file when running locally (ignored in production where env vars are set directly)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests
from flask import Flask, jsonify, abort, request as flask_request
from flask_cors import CORS
from apscheduler.schedulers.background import BackgroundScheduler
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────
EIA_API_KEY    = os.environ.get("EIA_API_KEY",    "")
NEWS_API_KEY   = os.environ.get("NEWS_API_KEY",   "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip().lstrip("=")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

EIA_BASE  = "https://api.eia.gov/v2"
NEWS_BASE = "https://newsapi.org/v2"

CACHE_TTL_SECONDS        = 24 * 60 * 60   # 24 hours
CACHE_TTL_FLOWS_SECONDS  =  6 * 60 * 60   # 6 hours for production/consumption/imports
REQUEST_TIMEOUT          = 15              # seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # allow all origins (dashboard HTML served locally)

# ── In-memory cache ──────────────────────────────────────────────────────────
_cache: dict = {}

def _cache_get(key):
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < CACHE_TTL_SECONDS:
        return entry["data"]
    return None

def _cache_set(key, data):
    _cache[key] = {"ts": time.time(), "data": data}

# ── EIA helpers ──────────────────────────────────────────────────────────────
def fetch_eia_us_stocks():
    """
    Weekly US ending stocks for crude oil and total petroleum products.
    Returns a dict keyed by product code with the latest value in MBBL.
    """
    key = "eia_stocks"
    cached = _cache_get(key)
    if cached:
        log.info("EIA stocks served from cache")
        return cached

    # Products we care about
    products = {
        "EPC0":  "Crude Oil",
        "EP00":  "Total Petroleum Products",
        "EPD0":  "Distillate Fuel",
        "EPM0":  "Motor Gasoline",
        "EPPR":  "Residual Fuel Oil",
        "EPJK":  "Jet Fuel",
    }

    results = {}
    for code, label in products.items():
        url = (
            f"{EIA_BASE}/petroleum/stoc/wstk/data/"
            f"?api_key={EIA_API_KEY}"
            f"&frequency=weekly"
            f"&data[0]=value"
            f"&facets[duoarea][]=NUS"
            f"&facets[product][]={code}"
            f"&sort[0][column]=period"
            f"&sort[0][direction]=desc"
            f"&offset=0&length=2"
        )
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
            if rows:
                latest = rows[0]
                prev   = rows[1] if len(rows) > 1 else None
                val    = float(latest.get("value") or 0)
                prev_v = float(prev.get("value") or 0) if prev else None
                results[code] = {
                    "label":   label,
                    "period":  latest.get("period"),
                    "value_mbbl": val,          # thousands of barrels
                    "value_mb":   val / 1000,   # millions of barrels
                    "units":   latest.get("units", "MBBL"),
                    "change_mbbl": round(val - prev_v, 1) if prev_v else None,
                }
        except Exception as e:
            log.warning("EIA fetch failed for %s: %s", code, e)
            results[code] = {"label": label, "error": str(e)}

        time.sleep(0.4)   # avoid EIA rate-limit

    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "EIA API v2 — Weekly Petroleum Status Report",
        "stocks": results,
    }
    _cache_set(key, out)
    log.info("EIA stocks refreshed — %d products", len(results))
    return out


def fetch_eia_us_consumption():
    """
    Latest weekly US petroleum product supplied (consumption proxy).
    """
    key = "eia_consumption"
    cached = _cache_get(key)
    if cached:
        return cached

    url = (
        f"{EIA_BASE}/petroleum/cons/wpsup/data/"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=weekly"
        f"&data[0]=value"
        f"&facets[duoarea][]=NUS"
        f"&facets[product][]=EP00"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&offset=0&length=4"
    )
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        out = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "EIA API v2 — Weekly Petroleum Supply",
            "data": rows,
        }
        _cache_set(key, out)
        return out
    except Exception as e:
        log.warning("EIA consumption fetch failed: %s", e)
        return {"error": str(e)}


def fetch_eia_imports():
    """
    Latest weekly US crude oil imports by country of origin.
    """
    key = "eia_imports"
    cached = _cache_get(key)
    if cached:
        return cached

    url = (
        f"{EIA_BASE}/petroleum/move/wkly/data/"
        f"?api_key={EIA_API_KEY}"
        f"&frequency=weekly"
        f"&data[0]=value"
        f"&facets[process][]=IMP"
        f"&facets[product][]=EPC0"
        f"&sort[0][column]=period"
        f"&sort[0][direction]=desc"
        f"&offset=0&length=30"
    )
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
        out = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "EIA API v2 — Weekly Petroleum Movement",
            "data": rows,
        }
        _cache_set(key, out)
        return out
    except Exception as e:
        log.warning("EIA imports fetch failed: %s", e)
        return {"error": str(e)}


# ── EIA per-product stocks (USA only) ────────────────────────────────────────
def fetch_eia_product_stocks_usa():
    """
    Weekly US ending stocks by product from EIA API v2.
    Returns dict with crude, gasoline, diesel, jet, fuel_oil, lpg_naphtha in Mb.
    """
    key = "eia_product_stocks_usa"
    cached = _cache_get(key)
    if cached:
        return cached

    # EIA series codes → product labels
    products = {
        "EPC0":  "crude",
        "EPM0":  "gasoline",
        "EPD0":  "diesel",
        "EPJK":  "jet",
        "EPPR":  "fuel_oil",
        "EPLLPZ": "lpg_naphtha",
    }

    result = {}
    for code, label in products.items():
        url = (
            f"{EIA_BASE}/petroleum/stoc/wstk/data/"
            f"?api_key={EIA_API_KEY}"
            f"&frequency=weekly&data[0]=value"
            f"&facets[duoarea][]=NUS&facets[product][]={code}"
            f"&sort[0][column]=period&sort[0][direction]=desc"
            f"&offset=0&length=1"
        )
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            rows = r.json().get("response", {}).get("data", [])
            if rows:
                val = float(rows[0].get("value") or 0)
                result[label] = round(val / 1000, 1)   # MBBL → Mb
        except Exception as e:
            log.warning("EIA product stock failed for %s: %s", code, e)
        time.sleep(0.3)

    out = {
        "source": "EIA API v2",
        "frequency": "weekly",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "products": result,
    }
    _cache_set(key, out)
    log.info("EIA product stocks (USA) refreshed: %s", result)
    return out


# ── Eurostat per-product stocks (EU27 + UK, Norway, Turkey) ──────────────────
# Eurostat geo codes → ISO3 mapping for countries we track
EUROSTAT_GEO_MAP = {
    "DE": "DEU", "FR": "FRA", "GB": "GBR", "ES": "ESP", "IT": "ITA",
    "NL": "NLD", "PL": "POL", "TR": "TUR", "NO": "NOR", "AU": "AUS",
    "BE": "BEL", "GR": "GRC", "PT": "PRT", "CZ": "CZE", "HU": "HUN",
    "RO": "ROU", "SK": "SVK", "FI": "FIN", "SE": "SWE", "DK": "DNK",
    "AT": "AUT", "IE": "IRL", "BG": "BGR", "HR": "HRV", "SI": "SVN",
    "LT": "LTU", "LV": "LVA", "EE": "EST", "LU": "LUX", "CY": "CYP",
    "MT": "MLT",
}

# Eurostat SIEC codes → our product labels
EUROSTAT_SIEC_MAP = {
    "O4100_TOT": "crude",
    "O4652":     "gasoline",    # motor gasoline
    "O4671":     "diesel",      # gas oil + diesel
    "O4661":     "jet",         # kerosene-type jet fuel
    "O4680":     "fuel_oil",    # fuel oil total
    "O4630":     "lpg_naphtha", # LPG
}

def fetch_eurostat_product_stocks():
    """
    Monthly petroleum product stocks from Eurostat nrg_stk_oilm.
    Returns { iso3: { crude, gasoline, diesel, jet, fuel_oil, lpg_naphtha } }
    Values in million barrels (converted from thousand tonnes using ~7.33 bbl/tonne).
    """
    key = "eurostat_product_stocks"
    cached = _cache_get(key)
    if cached:
        return cached

    EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_stk_oilm"
    # stk_flow=CLO = closing stocks on national territory
    # unit=THS_T = thousand tonnes
    siec_codes = list(EUROSTAT_SIEC_MAP.keys())
    geo_codes   = list(EUROSTAT_GEO_MAP.keys())

    # Build one query per SIEC to keep response size manageable
    country_data = {iso3: {} for iso3 in EUROSTAT_GEO_MAP.values()}

    for siec, label in EUROSTAT_SIEC_MAP.items():
        params = {
            "format": "JSON",
            "lang": "en",
            "siec": siec,
            "stk_flow": "CLO",
            "unit": "THS_T",
            "sinceTimePeriod": "2024-01",   # last ~18 months is enough
        }
        # add all geo filters
        for g in geo_codes:
            params[f"geo"] = ",".join(geo_codes)
            break  # build as single param

        try:
            r = requests.get(EUROSTAT_BASE, params=params, timeout=20)
            r.raise_for_status()
            jdata = r.json()

            # Eurostat JSON-stat structure
            dims   = jdata.get("dimension", {})
            values = jdata.get("value", {})
            geo_dim  = dims.get("geo",  {}).get("category", {}).get("index", {})
            time_dim = dims.get("time", {}).get("category", {}).get("index", {})

            if not geo_dim or not time_dim:
                continue

            # Size of each dimension
            n_geo  = len(geo_dim)
            n_time = len(time_dim)

            # For each geo, find the latest non-null value
            for geo_code, geo_idx in geo_dim.items():
                iso3 = EUROSTAT_GEO_MAP.get(geo_code)
                if not iso3:
                    continue
                # Walk time periods newest-first
                latest_val = None
                for t_period in sorted(time_dim.keys(), reverse=True):
                    t_idx = time_dim[t_period]
                    flat_idx = str(geo_idx * n_time + t_idx)
                    v = values.get(flat_idx)
                    if v is not None:
                        latest_val = v
                        break

                if latest_val is not None:
                    # Convert thousand tonnes → million barrels (~7.33 bbl/tonne)
                    mb = round(latest_val * 7.33 / 1000, 1)
                    country_data[iso3][label] = mb

        except Exception as e:
            log.warning("Eurostat fetch failed for siec=%s: %s", siec, e)
        time.sleep(0.5)

    out = {
        "source": "Eurostat nrg_stk_oilm",
        "frequency": "monthly",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "countries": country_data,
    }
    _cache_set(key, out)
    log.info("Eurostat product stocks refreshed for %d countries", len(country_data))
    return out


# ── JODI per-product stocks (90+ countries) ───────────────────────────────────
# New JODI format (2024+): ISO2 country codes, KBBL unit, CLOSTLV flow
# Primary CSV: crude oil / NGL; Secondary CSV: refined products

# JODI ENERGY_PRODUCT codes → our product labels
JODI_PRODUCT_MAP = {
    # Primary (crude)
    "CRUDEOIL": "crude",
    "TOTCRUDE": "crude",    # total crude (may overlap — handled by dedup)
    # Secondary (refined)
    "GASOLINE": "gasoline",
    "GASDIES":  "diesel",
    "JETKERO":  "jet",
    "KEROSENE": "jet",      # merge into jet
    "RESFUEL":  "fuel_oil",
    "LPG":      "lpg_naphtha",
    "NAPHTHA":  "lpg_naphtha",
}

# ISO2 (JODI REF_AREA) → ISO3
JODI_ISO2_TO_ISO3 = {
    "AE": "ARE", "AG": "ATG", "AL": "ALB", "AM": "ARM", "AO": "AGO",
    "AR": "ARG", "AT": "AUT", "AU": "AUS", "AZ": "AZE", "BD": "BGD",
    "BE": "BEL", "BG": "BGR", "BH": "BHR", "BO": "BOL", "BR": "BRA",
    "BY": "BLR", "CA": "CAN", "CH": "CHE", "CL": "CHL", "CN": "CHN",
    "CO": "COL", "CU": "CUB", "CY": "CYP", "CZ": "CZE", "DE": "DEU",
    "DK": "DNK", "DZ": "DZA", "EC": "ECU", "EE": "EST", "EG": "EGY",
    "ES": "ESP", "FI": "FIN", "FR": "FRA", "GB": "GBR", "GH": "GHA",
    "GR": "GRC", "HR": "HRV", "HU": "HUN", "ID": "IDN", "IE": "IRL",
    "IN": "IND", "IQ": "IRQ", "IR": "IRN", "IT": "ITA", "JP": "JPN",
    "KE": "KEN", "KR": "KOR", "KW": "KWT", "KZ": "KAZ", "LB": "LBN",
    "LT": "LTU", "LU": "LUX", "LV": "LVA", "LY": "LBY", "MA": "MAR",
    "MM": "MMR", "MX": "MEX", "MY": "MYS", "NG": "NGA", "NL": "NLD",
    "NO": "NOR", "NZ": "NZL", "OM": "OMN", "PA": "PAN", "PE": "PER",
    "PH": "PHL", "PK": "PAK", "PL": "POL", "PT": "PRT", "QA": "QAT",
    "RO": "ROU", "RS": "SRB", "RU": "RUS", "SA": "SAU", "SD": "SDN",
    "SE": "SWE", "SG": "SGP", "SI": "SVN", "SK": "SVK", "SN": "SEN",
    "TH": "THA", "TM": "TKM", "TN": "TUN", "TR": "TUR", "TT": "TTO",
    "UA": "UKR", "US": "USA", "UY": "URY", "UZ": "UZB", "VE": "VEN",
    "VN": "VNM", "YE": "YEM", "ZA": "ZAF",
}

def _parse_jodi_csv(csv_text, product_map):
    """Parse JODI new-format CSV. Returns {iso3: {product_label: {period: Mb}}}."""
    rows_by_cp = {}
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            if row.get("FLOW_BREAKDOWN", "").strip() != "CLOSTLV":
                continue
            if row.get("UNIT_MEASURE", "").strip() != "KBBL":
                continue
            raw_val = row.get("OBS_VALUE", "").strip()
            if raw_val in ("", "-", "x"):
                continue
            try:
                val_kbbl = float(raw_val)
            except ValueError:
                continue
            iso2  = row.get("REF_AREA", "").strip().upper()
            iso3  = JODI_ISO2_TO_ISO3.get(iso2)
            if not iso3:
                continue
            prod  = row.get("ENERGY_PRODUCT", "").strip().upper()
            label = product_map.get(prod)
            if not label:
                continue
            period = row.get("TIME_PERIOD", "").strip()
            key2 = (iso3, label)
            if key2 not in rows_by_cp:
                rows_by_cp[key2] = {}
            rows_by_cp[key2][period] = val_kbbl
    except Exception as e:
        log.warning("JODI CSV parse error: %s", e)
    return rows_by_cp


def _parse_jodi_flows_csv(csv_text):
    """
    Parse JODI CSV for flow data (production, consumption, imports).
    FLOW_BREAKDOWN codes:
      PRODREFIN  = refinery production (output of refined products)
      TOTPROD    = total production (crude + NGL)
      DEMAND     = total inland demand / consumption
      IMTOTAL    = total imports
    UNIT_MEASURE = KBBL (thousand barrels) or KBBLDAY (thousand barrels/day)
    Returns { iso3: { production_kbd, consumption_kbd, imports_kbd } }
    """
    FLOW_MAP = {
        "TOTPROD":   "production_kbd",
        "DEMAND":    "consumption_kbd",
        "IMTOTAL":   "imports_kbd",
    }
    # Accumulate latest period per country+flow
    rows = {}   # (iso3, flow) -> {period -> value_kbd}
    try:
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            flow = row.get("FLOW_BREAKDOWN", "").strip()
            field = FLOW_MAP.get(flow)
            if not field:
                continue
            unit = row.get("UNIT_MEASURE", "").strip()
            raw_val = row.get("OBS_VALUE", "").strip()
            if raw_val in ("", "-", "x"):
                continue
            try:
                val = float(raw_val)
            except ValueError:
                continue
            # Normalise to kbd
            if unit == "KBBL":
                val = val / 30.0   # monthly kbbl → kbd approx
            elif unit == "KBBLDAY":
                pass               # already kbd
            else:
                continue
            iso2 = row.get("REF_AREA", "").strip().upper()
            iso3 = JODI_ISO2_TO_ISO3.get(iso2)
            if not iso3:
                continue
            period = row.get("TIME_PERIOD", "").strip()
            key = (iso3, field)
            if key not in rows:
                rows[key] = {}
            rows[key][period] = val
    except Exception as e:
        log.warning("JODI flows CSV parse error: %s", e)

    # Pick latest period per country+flow
    result = {}
    for (iso3, field), periods in rows.items():
        if not periods:
            continue
        val = periods[sorted(periods.keys())[-1]]
        if iso3 not in result:
            result[iso3] = {}
        result[iso3][field] = round(val, 1)
    return result


def fetch_jodi_country_flows():
    """
    Fetch JODI production, consumption and imports flows for 90+ countries.
    Uses both primary (crude) and secondary (products) CSVs.
    Returns { iso3: { production_kbd, consumption_kbd, imports_kbd, source, period } }
    Cached for 6 hours.
    """
    key = "jodi_country_flows"
    cached = _cache_get(key)
    if cached and (time.time() - _cache[key]["ts"]) < CACHE_TTL_FLOWS_SECONDS:
        return cached

    year = datetime.now(timezone.utc).year
    merged = {}

    for csv_type in ["primary", "secondary"]:
        r = None
        for y in [year, year - 1]:
            url = (f"https://www.jodidata.org/_resources/files/downloads/oil-data"
                   f"/annual-csv/{csv_type}/{y}.csv")
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    log.info("JODI flows %s/%d CSV downloaded (%d bytes)", csv_type, y, len(r.content))
                    break
                else:
                    r = None
            except Exception as e:
                log.warning("JODI flows %s/%d failed: %s", csv_type, y, e)
                r = None
        if r is None:
            continue

        flows = _parse_jodi_flows_csv(r.text)
        for iso3, vals in flows.items():
            if iso3 not in merged:
                merged[iso3] = {}
            # secondary overrides primary for demand/imports (better coverage)
            merged[iso3].update(vals)

    # Basic plausibility filter — discard obviously wrong values
    clean = {}
    for iso3, vals in merged.items():
        entry = {}
        prod = vals.get("production_kbd", 0)
        cons = vals.get("consumption_kbd", 0)
        imps = vals.get("imports_kbd", 0)
        if prod >= 0 and prod < 30000:
            entry["production_kbd"] = prod
        if cons > 50 and cons < 40000:
            entry["consumption_kbd"] = cons
        if imps >= 0 and imps < 30000:
            entry["imports_kbd"] = imps
        if entry:
            entry["source"] = "JODI"
            clean[iso3] = entry

    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": "JODI Oil Database",
        "frequency": "monthly",
        "countries": clean,
    }
    _cache_set(key, out)
    log.info("JODI country flows fetched for %d countries", len(clean))
    return out


def fetch_jodi_product_stocks():
    """
    Download JODI Oil primary + secondary CSVs and extract closing stocks.
    New format (2024+): REF_AREA=ISO2, FLOW_BREAKDOWN=CLOSTLV, UNIT_MEASURE=KBBL.
    Returns { iso3: { crude, gasoline, diesel, jet, fuel_oil, lpg_naphtha } }
    Values in million barrels (KBBL / 1000).
    """
    key = "jodi_product_stocks"
    cached = _cache_get(key)
    if cached:
        return cached

    year = datetime.now(timezone.utc).year
    country_data = {}

    for csv_type in ["primary", "secondary"]:
        prod_map = {k: v for k, v in JODI_PRODUCT_MAP.items()
                    if (csv_type == "primary") == (v == "crude")}
        if not prod_map:
            # secondary maps all non-crude products
            prod_map = {k: v for k, v in JODI_PRODUCT_MAP.items() if v != "crude"}

        r = None
        for y in [year, year - 1]:
            url = f"https://www.jodidata.org/_resources/files/downloads/oil-data/annual-csv/{csv_type}/{y}.csv"
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    log.info("JODI %s/%d CSV downloaded (%d bytes)", csv_type, y, len(r.content))
                    break
                else:
                    r = None
            except Exception as e:
                log.warning("JODI %s CSV failed for %d: %s", csv_type, y, e)
                r = None
        if r is None:
            log.warning("JODI %s CSV unavailable for all years", csv_type)
            continue

        rows_by_cp = _parse_jodi_csv(r.text, prod_map)
        # Pick latest period per country+product and convert KBBL → Mb
        for (iso3, label), periods in rows_by_cp.items():
            if not periods:
                continue
            latest_val = periods[sorted(periods.keys())[-1]]
            mb = round(latest_val / 1000, 1)   # KBBL → Mb
            if iso3 not in country_data:
                country_data[iso3] = {}
            # Accumulate (e.g. KEROSENE + JETKERO both → jet)
            country_data[iso3][label] = round(
                country_data[iso3].get(label, 0) + mb, 1
            )

    out = {
        "source": "JODI Oil Database",
        "frequency": "monthly",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "countries": country_data,
    }
    _cache_set(key, out)
    log.info("JODI product stocks parsed for %d countries", len(country_data))
    return out


def fetch_all_product_stocks():
    """
    Merge EIA (USA), Eurostat (EU27+) and JODI (all others) into one dict:
    { iso3: { crude, gasoline, diesel, jet, fuel_oil, lpg_naphtha, source } }
    Priority: EIA > Eurostat > JODI
    """
    key = "all_product_stocks"
    cached = _cache_get(key)
    if cached:
        return cached

    merged = {}

    # 1. JODI — broadest coverage, lowest priority
    try:
        jodi = fetch_jodi_product_stocks()
        for iso3, prods in jodi.get("countries", {}).items():
            merged[iso3] = dict(prods)
            merged[iso3]["source"] = "JODI"
    except Exception as e:
        log.warning("JODI merge failed: %s", e)

    # 2. Eurostat — overwrites JODI for EU/European countries
    try:
        estat = fetch_eurostat_product_stocks()
        for iso3, prods in estat.get("countries", {}).items():
            if prods:   # only overwrite if Eurostat actually returned data
                merged[iso3] = dict(prods)
                merged[iso3]["source"] = "Eurostat"
    except Exception as e:
        log.warning("Eurostat merge failed: %s", e)

    # 3. EIA — overwrites everything for USA
    try:
        eia = fetch_eia_product_stocks_usa()
        if eia.get("products"):
            merged["USA"] = dict(eia["products"])
            merged["USA"]["source"] = "EIA"
    except Exception as e:
        log.warning("EIA product merge failed: %s", e)

    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "countries": merged,
    }
    _cache_set(key, out)
    log.info("All product stocks merged for %d countries", len(merged))
    return out


# ── NewsAPI helper ────────────────────────────────────────────────────────────
def fetch_oil_news(query="oil energy sanctions OPEC supply", page_size=20):
    key = f"news_{query[:20]}"
    cached = _cache_get(key)
    if cached:
        log.info("News served from cache")
        return cached

    if not NEWS_API_KEY:
        return {"error": "NEWS_API_KEY not set", "articles": []}

    url = (
        f"{NEWS_BASE}/everything"
        f"?q={requests.utils.quote(query)}"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&pageSize={page_size}"
        f"&apiKey={NEWS_API_KEY}"
    )
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        out = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "source": "NewsAPI",
            "totalResults": data.get("totalResults", 0),
            "articles": [
                {
                    "title":       a.get("title"),
                    "description": a.get("description"),
                    "url":         a.get("url"),
                    "source":      a.get("source", {}).get("name"),
                    "publishedAt": a.get("publishedAt"),
                }
                for a in data.get("articles", [])
            ],
        }
        _cache_set(key, out)
        log.info("NewsAPI refreshed — %d articles", len(out["articles"]))
        return out
    except Exception as e:
        log.warning("NewsAPI fetch failed: %s", e)
        return {"error": str(e), "articles": []}


# ── Routes ────────────────────────────────────────────────────────────────────
WAR_START_DATE = datetime(2026, 2, 28, tzinfo=timezone.utc)

@app.route("/")
def index():
    """Serve the dashboard HTML directly from Railway — no Hostinger needed."""
    import pathlib
    html_path = pathlib.Path(__file__).parent / "oil_inventory_dashboard.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})

@app.route("/api/war-status")
def war_status():
    """
    Returns current war context: days elapsed, Hormuz closure % assumption,
    and per-country stock depletion estimates based on elapsed days.
    Used by frontend to apply dynamic war-impact adjustments to country stocks.
    """
    now = datetime.now(timezone.utc)
    days_elapsed = max(0, (now - WAR_START_DATE).days)

    # Hormuz throughput reduction over time:
    # Week 1-2: partial disruption (~30% reduction)
    # Week 3-4: significant (~55% reduction)
    # Day 30+:  near-full closure (~75% reduction, some tanker rerouting)
    # Day 60+:  stabilise at ~70% as rerouting increases
    if days_elapsed <= 14:
        hormuz_closure_pct = min(30, days_elapsed * 2.1)
    elif days_elapsed <= 30:
        hormuz_closure_pct = 30 + (days_elapsed - 14) * 1.6
    elif days_elapsed <= 60:
        hormuz_closure_pct = 55 + (days_elapsed - 30) * 0.67
    else:
        hormuz_closure_pct = min(75, 75 - (days_elapsed - 60) * 0.1)  # slight recovery from rerouting

    # Consumption surge factor: panic buying + hoarding + inefficiency
    # Peaks around day 20-40, eases slightly as rationing kicks in
    if days_elapsed <= 20:
        consumption_surge_pct = days_elapsed * 0.15   # up to +3%
    elif days_elapsed <= 45:
        consumption_surge_pct = 3 + (days_elapsed - 20) * 0.08  # up to +5%
    else:
        consumption_surge_pct = max(3, 5 - (days_elapsed - 45) * 0.04)  # eases back

    return jsonify({
        "war_start":           WAR_START_DATE.date().isoformat(),
        "days_elapsed":        days_elapsed,
        "hormuz_closure_pct":  round(hormuz_closure_pct, 1),
        "consumption_surge_pct": round(consumption_surge_pct, 2),
        "as_of":               now.isoformat(),
    })

@app.route("/api/eia/stocks")
def eia_stocks():
    return jsonify(fetch_eia_us_stocks())

@app.route("/api/eia/consumption")
def eia_consumption():
    return jsonify(fetch_eia_us_consumption())

@app.route("/api/eia/imports")
def eia_imports():
    return jsonify(fetch_eia_imports())

@app.route("/api/news")
def news():
    return jsonify(fetch_oil_news())

@app.route("/api/news/geopolitical")
def news_geo():
    return jsonify(fetch_oil_news(
        query="oil Iran Russia Saudi OPEC sanctions war conflict pipeline supply",
        page_size=25
    ))

@app.route("/api/gdelt")
def gdelt_proxy():
    """
    Server-side proxy for GDELT API v2 doc search.
    Accepts query params: q (query string), startdatetime, maxrecords.
    GDELT does not send CORS headers so the browser cannot call it directly.
    """
    q             = flask_request.args.get("q", "oil+sanctions")
    startdatetime = flask_request.args.get("startdatetime", "")
    maxrecords    = flask_request.args.get("maxrecords", "10")

    cache_key = f"gdelt_{q}_{startdatetime}"
    cached = _cache_get(cache_key)
    if cached:
        return jsonify(cached)

    url = (
        "https://api.gdeltproject.org/api/v2/doc/doc"
        f"?query={q}&mode=artlist&format=json"
        f"&maxrecords={maxrecords}&sourcelang=english&sort=datedesc"
    )
    if startdatetime:
        url += f"&startdatetime={startdatetime}"

    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        # Short TTL for news (15 min)
        _cache[cache_key] = {"ts": time.time() - (CACHE_TTL_SECONDS - PRICE_CACHE_TTL), "data": data}
        return jsonify(data)
    except Exception as e:
        log.warning("GDELT proxy failed for q=%s: %s", q, e)
        return jsonify({"articles": [], "error": str(e)}), 502


@app.route("/api/summary")
def summary():
    """All data in one call — used by the dashboard on startup."""
    stocks      = fetch_eia_us_stocks()
    consumption = fetch_eia_us_consumption()
    news_geo    = fetch_oil_news(
        query="oil Iran Russia Saudi OPEC sanctions war conflict pipeline supply",
        page_size=25
    )
    return jsonify({
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "eia_stocks":      stocks,
        "eia_consumption": consumption,
        "news_geopolitical": news_geo,
    })

@app.route("/api/product-stocks")
def product_stocks():
    """Per-product petroleum stock breakdown for all countries."""
    return jsonify(fetch_all_product_stocks())

@app.route("/api/country-flows")
def country_flows():
    """
    Live production, consumption and imports flows per country from JODI.
    Used by the frontend to update the COUNTRIES array with real data
    instead of hardcoded IEA/BP baselines.
    Returns { fetched_at, countries: { iso3: { production_kbd, consumption_kbd, imports_kbd, source } } }
    """
    return jsonify(fetch_jodi_country_flows())

@app.route("/api/cache/status")
def cache_status():
    return jsonify({
        k: {
            "age_seconds": round(time.time() - v["ts"]),
            "expires_in":  max(0, CACHE_TTL_SECONDS - round(time.time() - v["ts"])),
        }
        for k, v in _cache.items()
    })

@app.route("/api/cache/clear", methods=["POST"])
def cache_clear():
    _cache.clear()
    log.info("Cache cleared manually")
    return jsonify({"cleared": True})


# ── OpenAI — per-country risk insight ────────────────────────────────────────
@app.route("/api/ai/country", methods=["POST"])
def ai_country_insight():
    """
    Accepts a JSON body with country data and returns a GPT-generated
    oil supply risk assessment for that country.

    Body fields (all optional except 'name'):
        name, iso3, days, stocks_mb, consumption_kbd, production_kbd,
        imports_kbd, nearest_spr, producer (bool)
    """
    body = flask_request.get_json(silent=True) or {}

    name            = body.get("name", "Unknown")
    days            = body.get("days", None)
    stocks_mb       = body.get("stocks_mb", None)
    consumption_kbd = body.get("consumption_kbd", 0)
    production_kbd  = body.get("production_kbd", 0)
    imports_kbd     = body.get("imports_kbd", 0)
    nearest_spr     = body.get("nearest_spr", "")
    producer        = body.get("producer", False)
    news_headlines  = body.get("news_headlines", [])   # list of headline strings

    # Cache key — per country, refreshed daily with other caches
    cache_key = f"ai_country_{body.get('iso3', name)}"
    cached = _cache_get(cache_key)
    if cached:
        log.info("AI insight served from cache for %s", name)
        return jsonify(cached)

    prompt = (
        f"Search for {name}'s counter measures to avert fuel crisis arising from iran - us war "
        f"from 28th February 2026 to now and due to closure of hormuz strait. "
        f"Get information the new source, potential barrels being sourced and if its concluded or "
        f"when its expected to be concluded and stage of the negotiation. "
        f"Respond in English only, regardless of the country. "
        f"Summarize the details in less than 100 words in a single paragraph. "
        f"At the very end of your response, on a new line, output a JSON array of up to 3 reference URLs "
        f"as: REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] "
        f"Use only real, publicly accessible URLs."
    )

    try:
        import json as _json
        # gpt-4o-search-preview has built-in web search — can find 2026 news
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
        )
        raw = response.choices[0].message.content.strip()

        # Extract REFS JSON block if present
        insight = raw
        refs = []
        if "REFS:" in raw:
            parts = raw.split("REFS:", 1)
            insight = parts[0].strip()
            try:
                refs = _json.loads(parts[1].strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []

        # Capture citations the model added as markdown links [text](url)
        import re
        if not refs:
            md_links = re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', insight)
            refs = [{"title": t, "url": u} for t, u in md_links[:3]]

        # Strip ALL URL-containing patterns from insight text:
        # 1. ([text](url)) — parenthesised citation blocks
        insight = re.sub(r'\(\[([^\]]+)\]\(https?://[^\)]+\)\)', r'', insight)
        # 2. [text](url) — plain markdown links → keep label text only
        insight = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', insight)
        # 3. bare URLs
        insight = re.sub(r'https?://\S+', '', insight)
        # 4. stray citation markers like [1], [2] left behind
        insight = re.sub(r'\[\d+\]', '', insight)
        # 5. collapse any double spaces / leading-trailing whitespace left by removals
        insight = re.sub(r'  +', ' ', insight).strip()

        # Hard-cap at 120 words
        words = insight.split()
        if len(words) > 120:
            insight = " ".join(words[:120]) + "…"

        out = {
            "country": name,
            "insight": insight,
            "refs": refs,
            "model": "gpt-4o-search-preview",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _cache_set(cache_key, out)
        log.info("AI insight generated for %s", name)
        return jsonify(out)

    except Exception as e:
        log.warning("OpenAI call failed for %s: %s", name, e)
        return jsonify({"country": name, "insight": None, "refs": [], "error": str(e)}), 500


# ── Crude oil price (yfinance primary, EIA fallback) ─────────────────────────
PRICE_CACHE_TTL = 900   # 15-minute TTL — yfinance updates frequently

# yfinance tickers: CL=F = WTI front-month futures, BZ=F = Brent front-month futures
_YF_MAP = {
    "WTI":   ("CL=F", "WTI Crude"),
    "Brent": ("BZ=F", "Brent Crude"),
}
# EIA seriesid fallback (requires full PET.SERIES.FREQ format)
_EIA_MAP = {
    "WTI":   ("PET.RWTC.D", "WTI Crude"),
    "Brent": ("PET.RBRTE.D", "Brent Crude"),
}

def _build_price_entry(closes_map, short, label, WAR_START, source):
    """
    Given an ordered list of (period_str, price) pairs, compute all price metrics.
    closes_map: list of {"period": "YYYY-MM-DD", "price": float}, sorted ascending.
    """
    if len(closes_map) < 2:
        return None
    price_now  = closes_map[-1]["price"]
    price_prev = closes_map[-2]["price"]
    period_now = closes_map[-1]["period"]

    change_1d     = round(price_now - price_prev, 2)
    change_1d_pct = round((change_1d / price_prev * 100) if price_prev else 0, 2)

    # War baseline: last trading day on or before WAR_START
    war_row = next(
        (r for r in reversed(closes_map) if r["period"] <= WAR_START), None
    )
    price_at_war = war_row["price"] if war_row else price_now
    change_since_war     = round(price_now - price_at_war, 2)
    change_since_war_pct = round(
        (change_since_war / price_at_war * 100) if price_at_war else 0, 2
    )

    # Weighted avg since war (linear ramp; de-escalation days down-weighted)
    war_rows = [r for r in closes_map if r["period"] >= WAR_START]
    daily_prices = [r["price"] for r in war_rows]
    weighted_avg = None
    if daily_prices:
        weights = []
        for i, p in enumerate(daily_prices):
            w = i + 1
            if i > 0 and daily_prices[i] < daily_prices[i-1] * 0.98:
                w *= 0.5
            weights.append(w)
        weighted_avg = round(
            sum(p * w for p, w in zip(daily_prices, weights)) / sum(weights), 2
        )

    # Sparkline: last 30 days (pre-war + post-war for context)
    sparkline = closes_map[-30:]

    return {
        "label":                 label,
        "price":                 round(price_now, 2),
        "period":                period_now,
        "change_1d":             change_1d,
        "change_1d_pct":         change_1d_pct,
        "price_at_war_start":    round(price_at_war, 2),
        "change_since_war":      change_since_war,
        "change_since_war_pct":  change_since_war_pct,
        "weighted_avg_since_war": weighted_avg,
        "daily_series":          sparkline,
        "source":                source,
    }


def fetch_crude_price():
    """
    Primary: yfinance front-month futures (CL=F WTI, BZ=F Brent) — real-time, 60-day history.
    Fallback: EIA v2 seriesid endpoint — may lag 1-2 days but is official spot price.
    Cache TTL: 15 minutes.
    """
    key = "crude_price"
    entry = _cache.get(key)
    if entry and (time.time() - entry["ts"]) < PRICE_CACHE_TTL:
        return entry["data"]

    WAR_START = "2026-02-28"
    result = {}

    # ── Primary: yfinance ───────────────────────────────────────────────────
    try:
        import yfinance as yf
        for short, (ticker_sym, label) in _YF_MAP.items():
            try:
                hist = yf.Ticker(ticker_sym).history(period="60d", interval="1d")
                if hist.empty:
                    continue
                closes = hist["Close"].dropna()
                if len(closes) < 2:
                    continue
                closes_map = sorted([
                    {"period": dt.strftime("%Y-%m-%d"), "price": float(pv)}
                    for dt, pv in closes.items()
                ], key=lambda x: x["period"])
                entry_data = _build_price_entry(closes_map, short, label, WAR_START, "yfinance")
                if entry_data:
                    result[short] = entry_data
                    log.info("yfinance %s=%.2f (%s)", short, entry_data["price"], entry_data["period"])
            except Exception as e_yf_ticker:
                log.warning("yfinance fetch failed for %s: %s", short, e_yf_ticker)
    except ImportError:
        log.warning("yfinance not installed — using EIA only")

    # ── Fallback: EIA seriesid for any missing tickers ──────────────────────
    missing = [s for s in ("WTI", "Brent") if s not in result]
    for short in missing:
        product_code, label = _EIA_MAP[short]
        url = (
            f"{EIA_BASE}/seriesid/{product_code}"
            f"?api_key={EIA_API_KEY}&num=60"
        )
        try:
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            rows = r.json().get("response", {}).get("data", [])
            if not rows:
                continue
            closes_map = sorted([
                {"period": r2["period"], "price": float(r2["value"])}
                for r2 in rows if r2.get("value")
            ], key=lambda x: x["period"])
            entry_data = _build_price_entry(closes_map, short, label, WAR_START, "EIA")
            if entry_data:
                result[short] = entry_data
                log.info("EIA fallback %s=%.2f (%s)", short, entry_data["price"], entry_data["period"])
        except Exception as e_eia:
            log.warning("EIA price fetch failed for %s: %s", short, e_eia)

    out = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "war_start": WAR_START,
        "prices": result,
    }
    _cache[key] = {"ts": time.time(), "data": out}
    log.info("Crude prices refreshed: WTI=%s Brent=%s",
             result.get("WTI", {}).get("price"), result.get("Brent", {}).get("price"))
    return out


@app.route("/api/crude-price")
def crude_price():
    return jsonify(fetch_crude_price())


# ── AI 4-scenario 7-day Brent/WTI price prediction ───────────────────────────
SCENARIO_DEFS = {
    1: {
        "label": "Full Escalation",
        "color": "#ef4444",
        "desc": (
            "Situation escalates severely: more US/allied strikes on Iran, Iran deploys"
            " proxy forces across the region, tankers are attacked in the Arabian Sea and"
            " Red Sea, the Strait of Hormuz is fully closed to commercial traffic."
            " Global oil supply loses 20–25% of seaborne crude. OPEC cannot compensate."
            " IEA record reserve release (announced March 2026) proves insufficient to"
            " offset supply shock. SPR drawdowns slow the spike but cannot cap prices."
        ),
    },
    2: {
        "label": "Status Quo — Active War",
        "color": "#f97316",
        "desc": (
            "Situation remains as of today: Iran and the US continue exchanging heavy"
            " missile and drone strikes. The Strait of Hormuz is partially operational"
            " (50–70% normal throughput). Global production is flat or marginally lower."
            " Risk premium persists. OPEC output unchanged."
            " IEA has proposed a record emergency reserves release (March 11 2026) to"
            " stabilise prices — market assessing whether this is sufficient."
        ),
    },
    3: {
        "label": "Ceasefire / De-escalation",
        "color": "#22d3ee",
        "desc": (
            "A ceasefire is agreed through mediation (Qatar/Oman/UN). Strait of Hormuz"
            " reopens to full commercial traffic within 2 weeks. Iranian and Gulf exports"
            " resume. Global supply gradually restores to pre-war levels."
            " IEA record reserve release (March 2026) combines with ceasefire to"
            " significantly deflate the war risk premium. OPEC+ maintains current output."
        ),
    },
    4: {
        "label": "Full Relief",
        "color": "#4ade80",
        "desc": (
            "Complete ceasefire and diplomatic resolution. All sanctions eased."
            " Strait of Hormuz fully operational. Iranian crude re-enters market."
            " IEA record reserve release (March 2026) floods market with additional supply."
            " Global production returns to and exceeds pre-war levels with Saudi/UAE"
            " output increase. Demand returns to pre-war trajectory. Risk premium gone."
        ),
    },
}

@app.route("/api/ai/scenario-prediction", methods=["GET"])
def ai_scenario_prediction():
    """
    Returns a 7-day Brent/WTI price outlook for one of 4 geopolitical scenarios.
    Query param: scenario=1|2|3|4
    Cache: 1 hour per scenario.
    """
    import re as _re, json as _json

    scenario_id = flask_request.args.get("scenario", "2")
    try:
        scenario_id = int(scenario_id)
        if scenario_id not in SCENARIO_DEFS:
            raise ValueError
    except (ValueError, TypeError):
        scenario_id = 2

    cache_key = f"ai_scenario_{scenario_id}"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < PRICE_CACHE_TTL:
        log.info("Scenario %d prediction served from cache", scenario_id)
        return jsonify(entry["data"])

    sc = SCENARIO_DEFS[scenario_id]
    price_data = fetch_crude_price()
    brent = price_data["prices"].get("Brent", {})
    wti   = price_data["prices"].get("WTI",   {})

    brent_now  = brent.get("price", "N/A")
    brent_war  = brent.get("price_at_war_start", "N/A")
    _brent_chg_raw = brent.get("change_since_war_pct", None)
    brent_chg  = f"{_brent_chg_raw:+.1f}" if isinstance(_brent_chg_raw, (int, float)) else "N/A"
    brent_wavg = brent.get("weighted_avg_since_war", "N/A")
    wti_now    = wti.get("price", "N/A")   # context only

    prompt = (
        f"You are an energy market analyst. All price forecasts in this task must be in "
        f"BRENT CRUDE only — do not reference or forecast WTI prices.\n\n"
        f"Iran-US war context (started Feb 28 2026).\n"
        f"Brent Crude spot price data:\n"
        f"  - Pre-war baseline (Feb 27 2026): ${brent_war}/bbl\n"
        f"  - Current spot: ${brent_now}/bbl\n"
        f"  - Change since war onset: {brent_chg}%\n"
        f"  - Weighted daily average since Feb 28 (de-escalation days down-weighted): ${brent_wavg}/bbl\n"
        f"  - WTI spot (reference only): ${wti_now}/bbl\n\n"
        f"Search for the top 10 most recent BRENT crude oil price predictions and energy news "
        f"(Reuters, Bloomberg, EIA, IEA, OPEC, S&P Global, Argus Media, ICE) from after Feb 28 2026.\n"
        f"Also specifically search for: 'IEA record reserves release March 2026' and its impact on oil prices.\n\n"
        f"Now assume the following scenario for the NEXT 7 DAYS:\n"
        f"SCENARIO {scenario_id} — {sc['label'].upper()}: {sc['desc']}\n\n"
        f"Based on the top 10 Brent news signals, the IEA reserve release news, AND the above scenario "
        f"assumptions, give a concise 7-day BRENT crude price outlook (under 100 words). "
        f"Factor in: war risk premium, IEA emergency reserve release impact, OPEC response, "
        f"SPR releases, demand destruction, shipping insurance costs, ICE Brent futures positioning. "
        f"The forecast range MUST be for BRENT CRUDE. "
        f"End your response with EXACTLY one line in this format: "
        f"'Expected Brent range: $X–$Y/bbl (7-day)'. "
        f"Then on a new line: REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] with up to 3 real URLs."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=450,
        )
        raw = response.choices[0].message.content.strip()

        insight = raw
        refs = []
        if "REFS:" in raw:
            parts = raw.split("REFS:", 1)
            insight = parts[0].strip()
            try:
                refs = _json.loads(parts[1].strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []

        if not refs:
            md_links = _re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', insight)
            refs = [{"title": t, "url": u} for t, u in md_links[:3]]

        # Strip URLs from insight text
        insight = _re.sub(r'\(\[([^\]]+)\]\(https?://[^\)]+\)\)', r'', insight)
        insight = _re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', insight)
        insight = _re.sub(r'https?://\S+', '', insight)
        insight = _re.sub(r'\[\d+\]', '', insight)
        insight = _re.sub(r'  +', ' ', insight).strip()

        # Extract price range line (matches "Expected Brent range:" or "Expected range:")
        range_match = _re.search(
            r'Expected(?:\s+Brent)?\s+range[:\s]+(\$[\d.,]+\s*[–\-—]+\s*\$[\d.,]+/bbl[^.\n]*)',
            insight, _re.IGNORECASE
        )
        price_range = ("Brent " + range_match.group(1).strip()) if range_match else None
        if range_match:
            insight = insight[:range_match.start()].strip()

        # Word cap
        words = insight.split()
        if len(words) > 110:
            insight = " ".join(words[:110]) + "…"

        out = {
            "scenario": scenario_id,
            "scenario_label": sc["label"],
            "scenario_color": sc["color"],
            "insight": insight,
            "price_range": price_range,
            "refs": refs,
            "wti_now": wti_now,
            "brent_now": brent_now,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _cache[cache_key] = {"ts": time.time(), "data": out}
        log.info("Scenario %d prediction generated (range: %s)", scenario_id, price_range)
        return jsonify(out)

    except Exception as e:
        log.warning("Scenario %d prediction failed: %s", scenario_id, e)
        return jsonify({
            "scenario": scenario_id, "insight": None,
            "price_range": None, "refs": [], "error": str(e)
        }), 500


# ── AI crude price prediction (aggregates top news + price movement) ──────────
@app.route("/api/ai/price-prediction", methods=["GET"])
def ai_price_prediction():
    """
    Fetches top oil-related news, weighs price movement since Feb 28 2026,
    and asks GPT-4o-search-preview for a short price prediction with refs.
    Cache TTL: 1 hour.
    """
    import re as _re, json as _json

    cache_key = "ai_price_prediction"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < PRICE_CACHE_TTL:
        log.info("AI price prediction served from cache")
        return jsonify(entry["data"])

    # Get current price data
    price_data = fetch_crude_price()
    wti  = price_data["prices"].get("WTI", {})
    brent = price_data["prices"].get("Brent", {})

    wti_now   = wti.get("price", "N/A")
    brent_now = brent.get("price", "N/A")
    wti_war   = wti.get("price_at_war_start", "N/A")
    _wti_chg_raw = wti.get("change_since_war_pct", None)
    wti_chg   = f"{_wti_chg_raw:+.1f}" if isinstance(_wti_chg_raw, (int, float)) else "N/A"
    wti_wavg  = wti.get("weighted_avg_since_war", "N/A")

    prompt = (
        f"Search for the top 10 most recent crude oil price predictions and news articles from "
        f"credible financial and energy sources (Reuters, Bloomberg, EIA, IEA, OPEC, S&P Global, "
        f"Argus Media, Energy Intelligence) published after 28th February 2026. "
        f"Context: Iran-US war started around 28 Feb 2026. Strait of Hormuz partially or fully "
        f"disrupted. WTI crude was ${wti_war}/bbl at war onset, now at ${wti_now}/bbl "
        f"({wti_chg}% change). Weighted daily average since war onset: ${wti_wavg}/bbl "
        f"(de-escalation days where price dropped >2%/day have been down-weighted in this average). "
        f"Brent spot: ${brent_now}/bbl. "
        f"Based on the top 10 news predictions and the actual price movement above, give a concise "
        f"forward price outlook for WTI crude for the next 30 days in under 120 words. "
        f"Factor in: (1) escalation vs de-escalation signals, (2) OPEC response, "
        f"(3) SPR release decisions, (4) demand destruction from high prices. "
        f"End with a single price range estimate: e.g. 'Expected range: $X–$Y/bbl'. "
        f"Then on a new line output: REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] "
        f"with up to 3 real source URLs."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()

        insight = raw
        refs = []
        if "REFS:" in raw:
            parts = raw.split("REFS:", 1)
            insight = parts[0].strip()
            try:
                refs = _json.loads(parts[1].strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []

        if not refs:
            md_links = _re.findall(r'\[([^\]]+)\]\((https?://[^\)]+)\)', insight)
            refs = [{"title": t, "url": u} for t, u in md_links[:3]]

        # Strip inline URLs from insight text
        insight = _re.sub(r'\(\[([^\]]+)\]\(https?://[^\)]+\)\)', r'', insight)
        insight = _re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', insight)
        insight = _re.sub(r'https?://\S+', '', insight)
        insight = _re.sub(r'\[\d+\]', '', insight)
        insight = _re.sub(r'  +', ' ', insight).strip()

        out = {
            "insight": insight,
            "refs": refs,
            "wti_now": wti_now,
            "brent_now": brent_now,
            "wti_change_pct": wti_chg,
            "weighted_avg": wti_wavg,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        _cache[cache_key] = {"ts": time.time(), "data": out}
        log.info("AI price prediction generated")
        return jsonify(out)

    except Exception as e:
        log.warning("AI price prediction failed: %s", e)
        return jsonify({"insight": None, "refs": [], "error": str(e)}), 500


@app.route("/api/ai/fuel-crisis-scan")
def ai_fuel_crisis_scan():
    """
    AI news scan: which countries have reported fuel shortages, rationing,
    outages or critical supply failures since the Iran-US war started (Feb 28 2026).
    Cached 2 hours.
    """
    import json as _json, re as _re
    cache_key = "ai_fuel_crisis_scan"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < 7200:
        log.info("Fuel crisis scan served from cache")
        return jsonify(entry["data"])

    now = datetime.now(timezone.utc)
    days_elapsed = max(0, (now - WAR_START_DATE).days)

    prompt = (
        f"Search for news from after 28th February 2026 about countries experiencing "
        f"fuel shortages, fuel rationing, fuel outages, petrol queues, diesel shortages, "
        f"energy crises or critical oil supply failures due to the Iran-US war and "
        f"Strait of Hormuz disruption. Today is {now.strftime('%B %d, %Y')} "
        f"({days_elapsed} days since the war started). "
        f"For each affected country found, provide: country name, severity "
        f"(critical/severe/moderate), what fuel type is affected, and a one-line summary. "
        f"Respond in English only. Format your response as a JSON array: "
        f"CRISIS:[{{\"country\":\"...\",\"iso3\":\"...\",\"severity\":\"critical|severe|moderate\","
        f"\"fuel_type\":\"...\",\"summary\":\"...\"}},...] "
        f"followed by REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] with up to 5 real URLs. "
        f"If no confirmed shortages are found, return CRISIS:[] with an explanation."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        raw = response.choices[0].message.content.strip()

        crisis_list = []
        refs = []
        explanation = ""

        if "CRISIS:" in raw:
            parts = raw.split("CRISIS:", 1)
            explanation = parts[0].strip()
            remainder = parts[1].strip()
            if "REFS:" in remainder:
                crisis_part, refs_part = remainder.split("REFS:", 1)
            else:
                crisis_part, refs_part = remainder, "[]"
            try:
                crisis_list = _json.loads(crisis_part.strip())
                if not isinstance(crisis_list, list):
                    crisis_list = []
            except Exception:
                crisis_list = []
            try:
                refs = _json.loads(refs_part.strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []
        else:
            explanation = raw

        out = {
            "days_elapsed":  days_elapsed,
            "crisis_count":  len(crisis_list),
            "countries":     crisis_list,
            "explanation":   explanation,
            "refs":          refs,
            "generated_at":  now.isoformat(),
            "model":         "gpt-4o-search-preview",
        }
        _cache[cache_key] = {"ts": time.time(), "data": out}
        log.info("Fuel crisis scan completed — %d countries affected", len(crisis_list))
        return jsonify(out)

    except Exception as e:
        log.warning("Fuel crisis scan failed: %s", e)
        return jsonify({"crisis_count": 0, "countries": [], "error": str(e)}), 500


@app.route("/api/ai/country-discovery")
def ai_country_discovery():
    """
    AI news scraper: discovers countries NOT already in the dashboard that are
    experiencing significant oil supply disruption, import disruption, fuel rationing,
    or energy crisis due to the Iran-US war. Returns new countries to add to the model.
    Cached 6 hours.
    """
    import json as _json, re as _re
    cache_key = "ai_country_discovery"
    entry = _cache.get(cache_key)
    if entry and (time.time() - entry["ts"]) < 21600:   # 6h TTL
        log.info("Country discovery served from cache")
        return jsonify(entry["data"])

    now = datetime.now(timezone.utc)
    days_elapsed = max(0, (now - WAR_START_DATE).days)

    # Countries already tracked (so we only return NEW ones)
    tracked = [
        "USA","CHN","IND","JPN","KOR","DEU","FRA","SAU","RUS","ARE","IRN","IRQ",
        "KWT","CAN","BRA","GBR","ESP","ITA","NLD","TUR","SGP","AUS","POL","UKR",
        "NGA","LBY","DZA","VEN","KAZ","NOR","QAT","OMN","AZE","ECU","COL","MYS",
        "PAK","BGD","EGY","THA","IDN","ZAF","ARG","MAR","LKA","LBN","YEM","MMR",
        "CUB","PER","GHA","SDN","BHR","TKM","GTM","BOL","MEX","JOR","TWN","PHL",
        "ISR","ETH","KEN","NZL","GRC","GEO","MNG","NPL",
    ]
    tracked_str = ", ".join(tracked)

    prompt = (
        f"It is {now.strftime('%B %d, %Y')}, Day {days_elapsed} of the Iran-US war (started Feb 28 2026). "
        f"The Strait of Hormuz is approximately 67% disrupted. "
        f"Search for recent news about countries that are experiencing significant oil supply problems, "
        f"fuel shortages, import disruptions, or energy crises caused by the Iran-US war and Hormuz disruption "
        f"that are NOT in this already-tracked list: {tracked_str}. "
        f"For each new country you find that has credible supply disruption news, provide: "
        f"country name, ISO-3 code, estimated oil stocks in million barrels (stocks_mb), "
        f"consumption in kbd, domestic production in kbd, imports in kbd, "
        f"Hormuz-routed imports in kbd (how much of their oil transits Hormuz), "
        f"and a one-line summary of the crisis. "
        f"Only include countries with REAL confirmed disruptions from news sources. "
        f"Respond in English only. "
        f"Format: COUNTRIES:[{{\"iso3\":\"...\",\"name\":\"...\",\"stocks_mb\":N,\"cons_kbd\":N,"
        f"\"prod_kbd\":N,\"imp_kbd\":N,\"hormuz_kbd\":N,\"flag\":\"emoji\","
        f"\"lat\":N,\"lng\":N,\"summary\":\"...\"}},...] "
        f"Then REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] with up to 5 real URLs. "
        f"If no new countries are found, return COUNTRIES:[]."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=900,
        )
        raw = response.choices[0].message.content.strip()

        new_countries = []
        refs = []

        if "COUNTRIES:" in raw:
            remainder = raw.split("COUNTRIES:", 1)[1].strip()
            if "REFS:" in remainder:
                countries_part, refs_part = remainder.split("REFS:", 1)
            else:
                countries_part, refs_part = remainder, "[]"
            try:
                new_countries = _json.loads(countries_part.strip())
                if not isinstance(new_countries, list):
                    new_countries = []
            except Exception:
                new_countries = []
            try:
                refs = _json.loads(refs_part.strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []

        out = {
            "days_elapsed":    days_elapsed,
            "new_count":       len(new_countries),
            "new_countries":   new_countries,
            "refs":            refs,
            "generated_at":    now.isoformat(),
            "model":           "gpt-4o-search-preview",
        }
        _cache[cache_key] = {"ts": time.time(), "data": out}
        log.info("Country discovery: found %d new affected countries", len(new_countries))
        return jsonify(out)

    except Exception as e:
        log.warning("Country discovery failed: %s", e)
        return jsonify({"new_count": 0, "new_countries": [], "error": str(e)}), 500


@app.route("/api/ai/weekly-assessment")
def ai_weekly_assessment():
    """
    Weekly AI assessment of global oil stockpile status.
    Uses war-impact model to compute adjusted days-of-supply per country,
    then asks GPT-4o to analyse critical/severe cases and give a geopolitical outlook.
    Cached 7 days (refreshed every Monday 08:00 UTC by scheduler).
    """
    import json as _json, re as _re

    cache_key = "ai_weekly_assessment"
    entry = _cache.get(cache_key)
    WEEKLY_TTL = 7 * 24 * 60 * 60   # 7 days
    if entry and (time.time() - entry["ts"]) < WEEKLY_TTL:
        log.info("Weekly assessment served from cache")
        return jsonify(entry["data"])

    now = datetime.now(timezone.utc)
    days_elapsed = max(0, (now - WAR_START_DATE).days)
    week_number  = (days_elapsed // 7) + 1

    # Compute Hormuz closure % (same formula as /api/war-status)
    if days_elapsed <= 14:
        hormuz_pct = min(30, days_elapsed * 2.1)
    elif days_elapsed <= 30:
        hormuz_pct = 30 + (days_elapsed - 14) * 1.6
    elif days_elapsed <= 60:
        hormuz_pct = 55 + (days_elapsed - 30) * 0.67
    else:
        hormuz_pct = min(75, 75 - (days_elapsed - 60) * 0.1)

    consumption_surge = min(5.0, days_elapsed * 0.11) if days_elapsed <= 45 else max(3.0, 5.0 - (days_elapsed - 45) * 0.04)

    # CTABLE baselines (iso3: [stocks_mb, cons_kbd, prod_kbd, imp_kbd])
    # Pulled from JODI flows cache if available, else use hardcoded reference values
    BASELINE = {
        "USA": [1650,20100,13200,6900], "CHN": [1200,16300,4200,11000],
        "IND": [102,5300,750,4800],     "JPN": [600,3100,10,3100],
        "KOR": [260,2700,5,2700],       "DEU": [90,2200,60,2200],
        "FRA": [80,1700,20,1700],       "SAU": [500,3700,12000,0],
        "RUS": [280,3700,10500,0],      "ARE": [250,1100,4000,0],
        "IRN": [200,2100,3200,0],       "IRQ": [130,880,4400,0],
        "KWT": [130,470,2700,0],        "CAN": [350,2400,5700,0],
        "BRA": [150,3300,3700,500],     "GBR": [55,1500,800,900],
        "ESP": [65,1300,30,1300],       "ITA": [62,1300,80,1250],
        "NLD": [95,900,40,1600],        "TUR": [42,1000,70,970],
        "SGP": [80,1600,0,1600],        "AUS": [40,1060,430,700],
        "PAK": [16,560,80,490],         "BGD": [8,350,10,340],
        "EGY": [45,840,560,350],        "THA": [45,1400,370,1100],
        "IDN": [42,1700,600,1200],      "ZAF": [28,620,10,630],
        "MAR": [9,280,10,280],          "LKA": [3,130,0,130],
        "LBN": [2,100,0,100],           "YEM": [3,90,30,70],
        "NGA": [65,530,1500,0],         "QAT": [85,280,1900,0],
        "OMN": [30,200,1100,0],         "MYS": [42,820,570,350],
        "JOR": [15,200,5,200],          "TWN": [90,1100,5,1100],
        "PHL": [25,450,10,430],         "ISR": [60,280,20,250],
        "ETH": [5,70,0,70],             "ZAF": [28,620,10,630],
        "KEN": [15,110,0,100],          "NZL": [15,160,20,140],
        "GRC": [30,300,10,280],         "CUB": [5,130,50,90],
        "GEO": [4,60,0,60],             "MNG": [3,30,0,30],
        "NPL": [3,40,0,40],
    }

    HORMUZ_EXPOSURE = {
        "JPN": 3050, "KOR": 2600, "CHN": 3500, "IND": 1800, "SGP": 1400,
        "THA": 900,  "IDN": 600,  "MYS": 400,  "TUR": 600,  "GBR": 200,
        "DEU": 250,  "ITA": 300,  "ESP": 220,  "FRA": 180,  "NLD": 350,
        "USA": 500,  "AUS": 180,  "EGY": 200,  "PAK": 450,  "BGD": 300,
        "LKA": 120,  "LBN": 80,   "JOR": 180,  "TWN": 900,  "PHL": 350,
        "ISR": 150,  "ETH": 50,   "ZAF": 300,  "KEN": 80,   "NZL": 100,
        "GRC": 100,  "GEO": 15,   "MNG": 20,   "NPL": 25,
    }
    HORMUZ_EXPORTERS = {"SAU": 7000, "ARE": 2500, "KWT": 1800, "IRQ": 3500, "QAT": 1200}

    hf = hormuz_pct / 100.0
    surge = consumption_surge / 100.0

    country_status = []
    for iso3, (stocks_mb, cons_kbd, prod_kbd, imp_kbd) in BASELINE.items():
        adj_stocks = stocks_mb
        adj_cons   = cons_kbd
        adj_prod   = prod_kbd

        if iso3 in HORMUZ_EXPOSURE:
            hormuz_kbd    = HORMUZ_EXPOSURE[iso3]
            daily_loss    = (hormuz_kbd * hf) / 1000
            depletion     = daily_loss * days_elapsed
            surge_draw    = (cons_kbd * surge / 1000) * days_elapsed
            adj_stocks    = max(0.5, stocks_mb - depletion - surge_draw)
            adj_cons      = round(cons_kbd * (1 + surge))

        if iso3 in HORMUZ_EXPORTERS:
            blocked_kbd   = HORMUZ_EXPORTERS[iso3] * hf
            stock_build   = (blocked_kbd * hf * days_elapsed) / 1000
            adj_stocks    = min(stocks_mb * 1.4, stocks_mb + stock_build * 0.3)
            adj_prod      = round(prod_kbd * (1 - hf * 0.15))

        gap  = adj_cons - adj_prod
        days = 999 if gap <= 0 else min(998, round((adj_stocks * 1000) / gap))

        if days < 90:   # only include countries below 90-day IEA threshold
            country_status.append({
                "iso3":       iso3,
                "stocks_mb":  round(adj_stocks, 1),
                "cons_kbd":   adj_cons,
                "prod_kbd":   adj_prod,
                "days":       days,
                "severity":   "depleted" if days <= 0 else "critical" if days < 14 else "severe" if days < 45 else "moderate",
            })

    country_status.sort(key=lambda x: x["days"])
    depleted  = [c for c in country_status if c["severity"] == "depleted"]
    critical  = [c for c in country_status if c["severity"] == "critical"]
    severe    = [c for c in country_status if c["severity"] == "severe"]
    moderate  = [c for c in country_status if c["severity"] == "moderate"]

    summary_lines = "\n".join(
        f"- {c['iso3']}: {'DEPLETED (fuel rationing / emergency imports only)' if c['days'] <= 0 else str(c['days']) + ' days remaining'} ({c['severity'].upper()}), "
        f"stocks {c['stocks_mb']}Mb, consumption {c['cons_kbd']}kbd, production {c['prod_kbd']}kbd"
        for c in country_status[:25]
    )

    depleted_str = ", ".join(c["iso3"] for c in depleted) if depleted else "none"
    prompt = (
        f"You are an energy security analyst. It is Week {week_number} of the Iran-US war "
        f"(started Feb 28 2026, {days_elapsed} days elapsed). "
        f"The Strait of Hormuz is {hormuz_pct:.0f}% closed to commercial traffic. "
        f"Consumer fuel demand is running {consumption_surge:.1f}% above pre-war levels due to hoarding.\n\n"
        f"Based on war-adjusted stockpile modelling, the following countries are below "
        f"the IEA 90-day emergency threshold:\n{summary_lines}\n\n"
        f"DEPLETED (stocks exhausted, surviving on emergency imports / rationing): {depleted_str}\n\n"
        f"Search for the latest news (Reuters, Bloomberg, IEA, EIA, S&P Global) from the past 7 days "
        f"about oil supply shortages, fuel rationing, emergency SPR releases, and country-specific "
        f"energy crises related to the Iran-US war.\n\n"
        f"Write a concise Week {week_number} Global Oil Supply Assessment (under 250 words) covering:\n"
        f"1. Which countries are DEPLETED (stocks exhausted, now on emergency imports or rationing)\n"
        f"2. Which countries face CRITICAL shortage (under 14 days)\n"
        f"3. Which countries face SEVERE shortage (14-45 days)\n"
        f"4. Key supply developments this week (SPR releases, alternative routes, OPEC response)\n"
        f"5. Overall risk trajectory for next 7 days\n"
        f"Respond in English only.\n"
        f"End with: REFS:[{{\"title\":\"...\",\"url\":\"...\"}},...] with up to 5 real URLs."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700,
        )
        raw = response.choices[0].message.content.strip()

        assessment = raw
        refs = []
        if "REFS:" in raw:
            parts = raw.split("REFS:", 1)
            assessment = parts[0].strip()
            try:
                refs = _json.loads(parts[1].strip())
                if not isinstance(refs, list):
                    refs = []
            except Exception:
                refs = []

        # Clean markdown links from assessment text
        assessment = _re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', assessment)
        assessment = _re.sub(r'https?://\S+', '', assessment)
        assessment = _re.sub(r'\[\d+\]', '', assessment)
        assessment = _re.sub(r'  +', ' ', assessment).strip()

        out = {
            "week_number":     week_number,
            "days_elapsed":    days_elapsed,
            "hormuz_pct":      round(hormuz_pct, 1),
            "assessment":      assessment,
            "depleted_count":  len(depleted),
            "critical_count":  len(critical),
            "severe_count":    len(severe),
            "moderate_count":  len(moderate),
            "depleted":        depleted,
            "critical":        critical,
            "severe":          severe,
            "moderate":        moderate,
            "refs":            refs,
            "generated_at":    now.isoformat(),
            "model":           "gpt-4o-search-preview",
        }
        _cache[cache_key] = {"ts": time.time(), "data": out}
        log.info("Weekly assessment generated — Week %d, %d critical, %d severe",
                 week_number, len(critical), len(severe))
        return jsonify(out)

    except Exception as e:
        log.warning("Weekly assessment failed: %s", e)
        return jsonify({"assessment": None, "error": str(e)}), 500


def generate_weekly_assessment():
    """Called by scheduler every Monday — pre-generates and caches weekly assessment."""
    log.info("Generating weekly AI stockpile assessment…")
    _cache.pop("ai_weekly_assessment", None)   # force fresh generation
    try:
        with app.test_request_context():
            ai_weekly_assessment()
        log.info("Weekly assessment pre-generated successfully")
    except Exception as e:
        log.warning("Weekly assessment pre-generation failed: %s", e)


# ── Background scheduler ──────────────────────────────────────────────────────
def scheduled_refresh():
    log.info("Scheduled midnight refresh starting…")
    fetch_eia_us_stocks()
    fetch_eia_us_consumption()
    fetch_eia_imports()
    fetch_all_product_stocks()
    fetch_jodi_country_flows()   # live production/consumption/imports per country
    fetch_crude_price()
    fetch_oil_news()
    fetch_oil_news(
        query="oil Iran Russia Saudi OPEC sanctions war conflict pipeline supply",
        page_size=25
    )
    log.info("Scheduled midnight refresh complete")


def midnight_ai_refresh():
    """
    Runs at midnight UTC — busts all AI caches so fresh GPT-4o searches
    are triggered on the next request (fuel crisis scan, country discovery,
    price prediction, scenario predictions).
    """
    log.info("Midnight AI cache bust starting…")
    ai_keys = [
        "ai_fuel_crisis_scan",
        "ai_country_discovery",
        "ai_price_prediction",
        "ai_scenario_1", "ai_scenario_2", "ai_scenario_3", "ai_scenario_4",
    ]
    for k in ai_keys:
        _cache.pop(k, None)
    log.info("Midnight AI cache bust complete — %d keys cleared", len(ai_keys))

    # Pre-generate fuel crisis scan and country discovery in background
    try:
        with app.test_request_context():
            ai_fuel_crisis_scan()
            log.info("Midnight: fuel crisis scan pre-generated")
    except Exception as e:
        log.warning("Midnight fuel crisis scan failed: %s", e)
    try:
        with app.test_request_context():
            ai_country_discovery()
            log.info("Midnight: country discovery pre-generated")
    except Exception as e:
        log.warning("Midnight country discovery failed: %s", e)


# ── Startup: cache warm-up + scheduler (runs under both __main__ and gunicorn) ─
log.info("Starting background cache warm-up…")
_warmup_thread = threading.Thread(target=scheduled_refresh, daemon=True)
_warmup_thread.start()

_scheduler = BackgroundScheduler()
# Midnight UTC — data refresh + AI cache bust
_scheduler.add_job(scheduled_refresh,     "cron", hour=0, minute=0,  timezone="UTC")
_scheduler.add_job(midnight_ai_refresh,   "cron", hour=0, minute=5,  timezone="UTC")
# Monday 00:10 UTC — weekly assessment (after midnight data refresh)
_scheduler.add_job(generate_weekly_assessment, "cron", day_of_week="mon", hour=0, minute=10, timezone="UTC")
_scheduler.start()
log.info("Scheduler running — midnight UTC data refresh, 00:05 AI bust, Monday 00:10 weekly assessment")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)

# app.py
# Streamlit dashboard voor ProRail Data Challenge (HGWBRN, HGWBRZ, GWBR)
# Functionaliteit:
# - CSV's inladen (sensor + storingen) per brug
# - Tijdsparsering met DST-veilige conversie naar Europe/Amsterdam
# - Overzicht & datakwaliteit
# - Interactieve selectie van signalen (tag1/tag2) en timeseries
# - Brugcyclus-detectie (open/dicht episodes) met duurdistributies
# - Beschikbaarheid (uptime %) per indicator
# - Daily opens & gemiddelde duur per dag (vergelijking bruggen)
# - Storingen: tijdlijn, top-berichten
# - Downloads van afgeleide tabellen
# - ...

import os
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import pytz
from datetime import timedelta
import plotly.express as px
import openpyxl
import plotly.graph_objects as go

# --------------------
# Page & Style
# --------------------
st.set_page_config(
    page_title="ProRail Bruggen Dashboard",
    layout="wide",
)
st.markdown(
    """
    <style>
    .small-muted { color:#6b7280; font-size:0.9rem; }
    .metric-card { padding:0.75rem 1rem; border-radius:1rem; background:#f8fafc; border:1px solid #e5e7eb; }
    .section { margin-top:1.5rem; }
    .pill { display:inline-block; padding:0.2rem 0.6rem; background:#eef2ff; border:1px solid #c7d2fe; color:#3730a3; border-radius:999px; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------
# Helpers
# --------------------
LOC_TZ = pytz.timezone("Europe/Amsterdam")

@st.cache_data(show_spinner=False)
def read_sensor_csv(path: str) -> pd.DataFrame:
    encodings_to_try = ["utf-8", "latin1", "windows-1252"]
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if "_time" in df.columns:
        df["_time"] = pd.to_datetime(df["_time"], errors="coerce", utc=True)
        df["time_local"] = df["_time"].dt.tz_convert(LOC_TZ)
    else:
        df["time_local"] = pd.NaT

    if "_value" in df.columns:
        if df["_value"].dtype == object:
            df["_value_norm"] = df["_value"].astype(str).str.lower().isin(["true", "1", "waar", "yes"])
        else:
            df["_value_norm"] = df["_value"].astype(float).fillna(0).astype(int).astype(bool)
    else:
        df["_value_norm"] = False

    for c in ["_measurement", "tag1", "tag2", "_field"]:
        if c not in df.columns:
            df[c] = ""

    return df.sort_values("time_local").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_storing_csv(path: str, sep: str = ";") -> pd.DataFrame:
    encodings_to_try = ["utf-8", "latin1", "windows-1252"]
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if "TimeString" in df.columns:
        ts = pd.to_datetime(df["TimeString"], errors="coerce", format="%d.%m.%Y %H:%M:%S")
        ts_naive = ts.dt.tz_localize(None)
        try:
            ts_local = ts_naive.dt.tz_localize(LOC_TZ, ambiguous="NaT", nonexistent="NaT")
        except Exception:
            ts_local = ts_naive.dt.tz_localize("UTC").dt.tz_convert(LOC_TZ)
        df["time_local"] = ts_local
    else:
        df["time_local"] = pd.NaT

    for c in ["MsgText", "PLC", "MsgClass", "MsgNumber"]:
        if c not in df.columns:
            df[c] = ""

    return df.sort_values("time_local").reset_index(drop=True)

# ===== ML helpers voor voorspelling volgende week (Top-5 kansen) =====
def _make_history_features_for_group(grp: pd.DataFrame) -> pd.DataFrame:
    s = grp['count'].astype(float).reset_index(drop=True)
    periods = grp['period'].reset_index(drop=True)
    n = len(s)

    cum_sum = s.expanding().sum()
    cum_mean = s.expanding().mean()
    cum_std = s.expanding().std(ddof=0).fillna(0)
    nonzero_cumsum = (s > 0).cumsum()

    last_nonzero = -1
    last_idx = []
    for i in range(n):
        if s.iloc[i] > 0:
            last_nonzero = i
        last_idx.append(last_nonzero)
    gap_since_last = np.array([i - last_idx[i] if last_idx[i] >= 0 else 999 for i in range(n)])

    roll_4  = s.rolling(window=4,  min_periods=1).sum()
    roll_12 = s.rolling(window=12, min_periods=1).sum()
    roll_52 = s.rolling(window=52, min_periods=1).sum()

    ewma_4  = s.ewm(span=4,  adjust=False).mean()
    ewma_12 = s.ewm(span=12, adjust=False).mean()
    ewma_26 = s.ewm(span=26, adjust=False).mean()

    slopes = []
    for i in range(n):
        if i < 2:
            slopes.append(0.0)
        else:
            y = s.iloc[:i+1].values
            x = np.arange(len(y))
            b = np.polyfit(x, y, 1)[0]
            slopes.append(b)
    slopes = np.array(slopes)

    freq_nonzero = (s > 0).expanding().sum() / np.maximum(1, np.arange(1, n+1))
    prop_recent  = roll_4 / np.maximum(1.0, cum_sum)

    return pd.DataFrame({
        'period': periods,
        'count': s,
        'cum_count': cum_sum,
        'cum_mean': cum_mean,
        'cum_std': cum_std,
        'nonzero_cumsum': nonzero_cumsum,
        'gap_since_last': gap_since_last,
        'roll_4': roll_4,
        'roll_12': roll_12,
        'roll_52': roll_52,
        'ewma_4': ewma_4,
        'ewma_12': ewma_12,
        'ewma_26': ewma_26,
        'trend_slope': slopes,
        'freq_nonzero': freq_nonzero,
        'prop_recent': prop_recent
    })

def _build_dataset_from_storingen(st_df: pd.DataFrame, agg: str = 'week') -> pd.DataFrame:
    """Maakt weekly (of monthly) counts per MsgText en history features."""
    df = st_df.copy()
    # bronvelden: time_local (tz-aware) & MsgText
    if 'time_local' not in df.columns or 'MsgText' not in df.columns:
        raise RuntimeError("Storingsdata mist 'time_local' of 'MsgText'.")

    # periode
    dt = pd.to_datetime(df['time_local'], errors='coerce')
    if agg == 'week':
        df['period'] = dt.dt.to_period('W').apply(lambda p: p.start_time)
        freq = 'W-MON'
    elif agg == 'month':
        df['period'] = dt.dt.to_period('M').apply(lambda p: p.start_time)
        freq = 'MS'
    else:
        raise ValueError("agg must be 'week' or 'month'")

    df['_fault'] = df['MsgText'].astype(str).str.strip()
    counts = df.groupby(['period','_fault']).size().reset_index(name='count')

    # full grid (zodat missende weken 0 krijgen)
    all_periods = pd.date_range(start=counts['period'].min(), end=counts['period'].max(), freq=freq)
    all_faults  = counts['_fault'].unique()
    grid = pd.MultiIndex.from_product([all_periods, all_faults], names=['period','_fault']).to_frame(index=False)
    df_pd = grid.merge(counts, on=['period','_fault'], how='left').fillna({'count':0.0})
    df_pd = df_pd.sort_values(['_fault','period']).reset_index(drop=True)

    frames = []
    for fname, grp in df_pd.groupby('_fault', sort=False):
        h = _make_history_features_for_group(grp)
        h['_fault'] = fname
        frames.append(h)
    hist = pd.concat(frames, ignore_index=True)

    df_full = df_pd.merge(hist, on=['_fault','period'], how='left')

    # target voor validatie (niet strikt nodig voor live voorspel)
    df_full['target_next'] = df_full.groupby('_fault')['count'].shift(-1)
    return df_full


# --------------------
# Laatste storingen (Excel)
# --------------------
@st.cache_data(show_spinner=False)
def read_latest_storingen_xlsx(path: str) -> dict[str, pd.DataFrame]:
    """Read the 'Laatste storingen.xlsx' file and return dict of DataFrames per bridge."""
    dfs: dict[str, pd.DataFrame] = {}

    # Try to open Excel file
    try:
        xl = pd.ExcelFile(path)
    except FileNotFoundError:
        st.warning(f"⚠ Bestand niet gevonden: {path}")
        return dfs
    except Exception as e:
        st.error(f"❌ Fout bij openen van Excel-bestand: {e}")
        return dfs

    # Map sheet names safely
    sheet_map = {}
    for name in xl.sheet_names:
        name_upper = name.upper()
        if "GWBR" in name_upper and "HG" not in name_upper:
            sheet_map["GWBR"] = name
        elif "HGWBRZ" in name_upper:
            sheet_map["HGWBRZ"] = name
        elif "HGWBRN" in name_upper:
            sheet_map["HGWBRN"] = name

    # Parse each sheet
    for key, sheet in sheet_map.items():
        try:
            df = xl.parse(sheet)
        except Exception as e:
            st.warning(f"⚠ Kon tabblad '{sheet}' niet lezen ({e})")
            continue

        if "Datumtijd" in df.columns:
            ts = pd.to_datetime(df["Datumtijd"], errors="coerce", format="%d.%m.%Y %H:%M:%S")
            ts_naive = ts.dt.tz_localize(None)
            try:
                ts_local = ts_naive.dt.tz_localize(LOC_TZ, ambiguous="NaT", nonexistent="NaT")
            except Exception:
                ts_local = ts_naive.dt.tz_localize("UTC").dt.tz_convert(LOC_TZ)
            df["time_local"] = ts_local
        else:
            df["time_local"] = pd.NaT

        for c in ["Bericht", "Bron", "Oorzaak", "Status"]:
            if c not in df.columns:
                df[c] = ""

        dfs[key] = df.sort_values("time_local", ascending=False).reset_index(drop=True)

    return dfs


def availability_from_boolean(df: pd.DataFrame, value_col: str = "_value_norm") -> float:
    if df.empty:
        return np.nan
    return float(df[value_col].mean())


def detect_episodes(df: pd.DataFrame, on_col: str = "_value_norm", time_col: str = "time_local"):
    if df.empty or df[time_col].isna().all():
        return pd.DataFrame(columns=["start", "end", "duration_s"])

    vals = df[on_col].fillna(False).to_numpy()
    times = df[time_col].to_numpy()

    starts, ends = [], []
    active = False
    start_t = None

    for t, v in zip(times, vals):
        if not active and v:
            active = True
            start_t = t
        elif active and not v:
            active = False
            if start_t is not None:
                starts.append(start_t)
                ends.append(t)

    if active and start_t is not None:
        starts.append(start_t)
        ends.append(times[-1])

    out = pd.DataFrame({"start": starts, "end": ends})
    out["duration_s"] = (out["end"] - out["start"]).dt.total_seconds()
    return out


def daily_counts_and_duration(epi: pd.DataFrame):
    if epi.empty:
        return pd.DataFrame(columns=["date", "count", "avg_duration_s"])
    g = (
        epi.assign(date=epi["start"].dt.date)
           .groupby("date")
           .agg(count=("start", "count"), avg_duration_s=("duration_s", "mean"))
           .reset_index()
    )
    return g


def nice_duration(seconds: float) -> str:
    if pd.isna(seconds):
        return "—"
    td = timedelta(seconds=float(seconds))
    mins = int(td.total_seconds() // 60)
    secs = int(td.total_seconds() % 60)
    return f"{mins}m {secs:02d}s"


# --------------------
# Data inladen
# --------------------
try:
    # Load sensor data
    df_HN = read_sensor_csv("data/HGWBRN.csv")
    df_HZ = read_sensor_csv("data/HGWBRZ.csv")
    df_GW = read_sensor_csv("data/GWBR_draaibrug.csv")

    # Load storings data with semicolon separator
    st_HN = read_storing_csv("data/Storingen_HGWBRN.csv", sep=';')
    st_HZ = read_storing_csv("data/Storingen_HGWBRZ.csv", sep=';')
    st_GW = read_storing_csv("data/Storingen_GWBR.csv", sep=';')

    # Load latest storings Excel file
    latest_storingen = read_latest_storingen_xlsx("data/Laatste storingen.xlsx")

    st.success("✅ Alle bestanden succesvol geladen!")
    
except FileNotFoundError as e:
    st.error(f"❌ Bestand niet gevonden: {e}")
    st.stop()
except Exception as e:
    st.error(f"⚠ Fout bij het inladen: {e}")
    st.stop()

# --------------------
# Overzicht
# --------------------
st.title("🚉 ProRail Bruggen Dashboard")

st.markdown("_____")

st.markdown("""
Hoge Gouwespoorbrug Noord/Zuid (hefbrug) & Lage Gouwespoorbrug (draaibrug).

**🎯 Interactie tips:** 
- **Zoomen:** Horizontaal of verticaal slepen in grafieken
- **Legenda:** Klik op legendalabels om series te tonen/verbergen
- **Details:** Hover over datapunten voor meer informatie

**📱 Mobiel gebruik:** Voor de beste ervaring op mobiel, activeer 'Desktopsite' in je browserinstellingen.
""")


# --------------------
# Sensor: selectie & time series
# --------------------
st.markdown("_____")

st.markdown("## 🔎 Sensorverkenning")
bridge_choice = st.selectbox("Kies een brug", options=["HGWBRN", "HGWBRZ", "GWBR"], index=0)

def pick_df(bridge):
    return {"HGWBRN": df_HN, "HGWBRZ": df_HZ, "GWBR": df_GW}[bridge]

df_sel = pick_df(bridge_choice)

# Keuzes voor tag1/tag2
tags1 = sorted([t for t in df_sel["tag1"].dropna().unique() if str(t).strip() != ""])
tags2 = sorted([t for t in df_sel["tag2"].dropna().unique() if str(t).strip() != ""])

col1, col2 = st.columns(2)

with col1:
    # Set default to Noostnop if available, otherwise first alphabetical
    default_tag1 = []
    preferred_tags = ["Noodstop", "Bedienbrugpost"]
    for pref_tag in preferred_tags:
        if pref_tag in tags1:
            default_tag1 = [pref_tag]
            break
    if not default_tag1 and tags1:
        default_tag1 = [tags1[0]]
    
    tag1_pick = st.multiselect(
        "tag1 (onderdeel)",
        tags1,
        default=default_tag1,
        key=f"tag1_{bridge_choice}"
    )

if tag1_pick:
    valid_tag2 = sorted(df_sel.loc[df_sel["tag1"].isin(tag1_pick), "tag2"].dropna().unique())
else:
    valid_tag2 = sorted(tags2)

key_tag2 = f"tag2_{bridge_choice}_{'_'.join(tag1_pick) if tag1_pick else 'all'}"

with col2:
    # Set default to emergency-related tags if available
    default_tag2 = []
    preferred_tags2 = ["bNSknop_OK", "bCmdOpenen"]
    
    # Check if any preferred tags are available
    available_preferred = [tag for tag in preferred_tags2 if tag in valid_tag2]
    if available_preferred:
        default_tag2 = [available_preferred[0]]  # Take the first available preferred tags
    elif valid_tag2:
        default_tag2 = [valid_tag2[0]]  # Fallback to first alphabetical
    
    tag2_pick = st.multiselect(
        "tag2 (status)",
        valid_tag2,
        default=default_tag2,
        key=key_tag2
    )

df_filt = df_sel.copy()
if tag1_pick:
    df_filt = df_filt[df_filt["tag1"].isin(tag1_pick)]
if tag2_pick:
    df_filt = df_filt[df_filt["tag2"].isin(tag2_pick)]

df_filt = df_filt.sort_values(
    ["time_local", "_value_norm"],
    ascending=[True, True]
).reset_index(drop=True)

df_filt["time_local"] = df_filt["time_local"] + pd.to_timedelta(
    df_filt.groupby("time_local").cumcount(), unit="us"
)

df_plot = df_filt.copy()
df_plot["time_next"] = df_plot["time_local"].shift(-1)

df_step = pd.concat([
    df_plot[["time_local", "_value_norm", "_measurement", "tag1", "tag2"]].rename(columns={"time_local": "time"}),
    df_plot[["time_next", "_value_norm", "_measurement", "tag1", "tag2"]].rename(columns={"time_next": "time"})
]).dropna(subset=["time"])

df_step = df_step.dropna(subset=["time"])
df_step = df_step.sort_values("time")

st.markdown("<div class='section'></div>", unsafe_allow_html=True)
st.subheader("📈 Tijdreeks (geselecteerde signalen)")

if df_filt.empty:
    st.info("Geen rijen voor deze selectie.")
else:
    ts = (
        df_filt
        .assign(second=df_filt["time_local"].dt.floor("s"))
        .groupby(["_measurement", "tag1", "tag2", "second"], as_index=False)["_value_norm"]
        .mean()
        .rename(columns={"second": "time"})
    )
    fig = px.line(
        df_step,
        x="time",
        y="_value_norm",
        color=df_step[["_measurement", "tag1", "tag2"]].astype(str).agg(" · ".join, axis=1),
        labels={"_value_norm": "Actief", "time": "Tijd"},
        title="Activiteit over tijd (op basis van statuswijzigingen)"
    )
    fig.update_traces(line_shape="hv")
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# Laatste storingen – dropdown overzicht
# --------------------
st.markdown("_____")

st.markdown("## 🗓️ Laatste week – Storingen overzicht")

if not latest_storingen:
    st.info("Geen Excel-bestand met laatste storingen gevonden.")
else:
    bridge_pick = st.selectbox("Kies brug", list(latest_storingen.keys()))
    df_latest = latest_storingen[bridge_pick].copy()

    if df_latest.empty:
        st.info("Geen Excel-bestand met laatste storingen voor deze brug.")
    if df_latest.empty or "Datumtijd" not in df_latest.columns:
        st.info("Geen geldige storingsdata voor deze brug.")
    else:
        if not pd.api.types.is_datetime64_any_dtype(df_latest["Datumtijd"]):
            df_latest["Datumtijd"] = pd.to_datetime(
                df_latest["Datumtijd"],
                format="%d.%m.%Y %H:%M:%S",
                errors="coerce"
            )

        df_latest = df_latest.sort_values("Datumtijd", ascending=False)
        df_latest["Datum"] = df_latest["Datumtijd"].dt.date

        unique_dates = sorted(df_latest["Datum"].unique(), reverse=True)
        date_pick = st.selectbox("📅 Kies datum", unique_dates)

        day_df = df_latest[df_latest["Datum"] == date_pick]
        st.markdown(f"*{len(day_df)} storingen op {date_pick}:*")

        for i, row in day_df.iterrows():
            with st.expander(f"🕒 {row['Datumtijd'].strftime('%H:%M:%S')} — {row['Bericht']}"):
                st.write(row)

# --------------------
# Brugcyclus-detectie (open/dicht episodes)
# --------------------
st.markdown("_____")

st.markdown("## 🔄 Brugcycli (open/dicht)")


candidate_words = ["Brug open", "bVrijgaveOpenen", "Heffen", "Opendraaien", "B_Openen", "F_bewegingswerk"]
def find_candidates(df):
    cols = df[["tag1","tag2"]].astype(str).fillna("")
    both = pd.unique(cols["tag1"].tolist() + cols["tag2"].tolist())
    hits = [c for c in both if any(w.lower() in c.lower() for w in candidate_words)]
    return hits

cands = find_candidates(df_sel)
default_sig = cands[0] if cands else (tags2[0] if tags2 else "")

open_signal = st.selectbox(
    "Open-indicator (kies uit tag2/tag1 die in je data zit)",
    options=sorted(set(tags1+tags2+[default_sig])),
    index=0 if not default_sig else sorted(set(tags1+tags2+[default_sig])).index(default_sig)
)

mask_open = (df_sel["tag2"].astype(str).str.contains(re.escape(open_signal), case=False, na=False)) | \
            (df_sel["tag1"].astype(str).str.contains(re.escape(open_signal), case=False, na=False))
open_df = df_sel.loc[mask_open, ["time_local", "_value_norm", "_measurement", "tag1", "tag2"]].copy()

episodes = detect_episodes(open_df, on_col="_value_norm", time_col="time_local")
daily = daily_counts_and_duration(episodes)

col3, col4 = st.columns(2)
with col3:
    if daily.empty:
        st.info("Geen episodes gedetecteerd voor dit signaal.")
    else:
        fig1 = px.bar(daily, x="date", y="count", labels={"date":"Datum","count":"Opens/dag"}, title="Brugopeningen per dag")
        st.plotly_chart(fig1, use_container_width=True)
with col4:
    if daily.empty:
        st.info("Geen episodes gedetecteerd.")
    else:
        fig2 = px.line(daily, x="date", y="avg_duration_s",
                       labels={"date":"Datum","avg_duration_s":"Gem. duur (sec.)"},
                       title="Gemiddelde duur brugopening per dag")
        st.plotly_chart(fig2, use_container_width=True)

if not episodes.empty:
    fig3 = px.histogram(episodes, x="duration_s", nbins=2500, title="Histogram tijdsduur brugopeningen (sec.)", labels={"duration_s": "Tijdsduur (sec.)", "count": "Frequentie"})
    st.plotly_chart(fig3, use_container_width=True)


    total_episodes = len(episodes)
    median_dur = episodes["duration_s"].median()
    p90_dur = episodes["duration_s"].quantile(0.9)
    st.markdown(
        f"- *Totaal gedetecteerde opens:* {total_episodes}  \n"
        f"- *Mediaan duur:* {nice_duration(median_dur)}  \n"
        f"- *90e percentiel duur:* {nice_duration(p90_dur)}"
    )

    csv = episodes.assign(start=episodes["start"].dt.tz_convert(LOC_TZ),
                          end=episodes["end"].dt.tz_convert(LOC_TZ)).to_csv(index=False)
    st.download_button("⬇ Download episodes (CSV)", data=csv, file_name=f"episodes_{bridge_choice}_{open_signal}.csv", mime="text/csv")

# --------------------
# Bruggen vergelijken (daily opens & duur)
# --------------------
st.markdown("## 🧭 Vergelijking bruggen")

def compute_daily_for(df_sensor, open_key_words=candidate_words):
    cands = find_candidates(df_sensor)
    if not cands:
        common2 = df_sensor["tag2"].value_counts().index.tolist()
        if not common2:
            return pd.DataFrame(columns=["date","count","avg_duration_s","bridge"])
        open_pick = common2[0]
    else:
        open_pick = cands[0]
    m = (df_sensor["tag2"].astype(str).str.contains(re.escape(open_pick), case=False, na=False)) | \
        (df_sensor["tag1"].astype(str).str.contains(re.escape(open_pick), case=False, na=False))
    ep = detect_episodes(df_sensor.loc[m, ["time_local", "_value_norm"]].rename(columns={"time_local": "time_local"}))
    d = daily_counts_and_duration(ep)
    return d

d_HN = compute_daily_for(df_HN).assign(bridge="HGWBRN")
d_HZ = compute_daily_for(df_HZ).assign(bridge="HGWBRZ")
d_GW = compute_daily_for(df_GW).assign(bridge="GWBR")
d_all = pd.concat([d_HN, d_HZ, d_GW], ignore_index=True)

if d_all.empty:
    st.info("Onvoldoende data voor vergelijking.")
else:
    col5, col6 = st.columns(2)
    with col5:
        figc1 = px.bar(d_all, x="date", y="count", color="bridge", barmode="group",
                       title="Opens per dag per brug", labels={"date":"Datum","count":"Frequentie"})
        st.plotly_chart(figc1, use_container_width=True)
    with col6:
        figc2 = px.line(d_all, x="date", y="avg_duration_s", color="bridge",
                        title="Gem. duur per dag per brug", labels={"date":"Datum","avg_duration_s":"Duur (sec.)"})
        st.plotly_chart(figc2, use_container_width=True)

# --------------------
# Storingen
# --------------------
st.markdown("_____")

st.markdown("## ⚠ Storingen")

# ✅ Alleen 2 tabs laten staan
tab1, tab2 = st.tabs(["Tijdlijn & volume", "Top meldingen"])

with tab1:
    st.markdown("### Volume door de tijd")
    st_choice = st.selectbox("Kies storingsbron", options=["HGWBRN","HGWBRZ","GWBR"], index=0, key="st_src")
    st_df = {"HGWBRN": st_HN, "HGWBRZ": st_HZ, "GWBR": st_GW}[st_choice]
    if st_df.empty or st_df["time_local"].isna().all():
        st.info("Geen storingsdata of geen parsebare tijd.")
    else:
        s_daily = st_df.assign(date=st_df["time_local"].dt.date).groupby("date").size().reset_index(name="count")
        fig_s = px.area(s_daily, x="date", y="count", title=f"Storingen per dag – {st_choice}",
                        labels={"date":"Datum", "count":"Aantal"})
        st.plotly_chart(fig_s, use_container_width=True)

with tab2:
    st.markdown("### Meest voorkomende meldingen")
    n_top = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)
    all_st = pd.concat([st_HN.assign(bridge="HGWBRN"),
                        st_HZ.assign(bridge="HGWBRZ"),
                        st_GW.assign(bridge="GWBR")], ignore_index=True)
    if all_st.empty:
        st.info("Geen storingsdata.")
    else:
        top_msgs = (
            all_st.assign(MsgText=all_st["MsgText"].fillna("").astype(str).str.strip())
                  .query("MsgText != ''")
                  .groupby(["bridge","MsgText"]).size()
                  .reset_index(name="count")
        )
        for b in ["HGWBRN","HGWBRZ","GWBR"]:
            sub = top_msgs[top_msgs["bridge"]==b].nlargest(n_top, "count")
            st.markdown(f"{b}")
            fig_top = px.bar(sub, x="count", y="MsgText", orientation="h", title=f"Top {n_top} storingsmeldingen – {b}",
                             labels={"count":"Aantal","MsgText":"Melding"})
            st.plotly_chart(fig_top, use_container_width=True)

    # (de uitgebreide follow-up analyse blijft, zoals in het origineel, binnen tab2)
    st.markdown("### Top 5 storingen + meest voorkomende follow-up storingen")

    analyse_scope = st.radio("Analyse scope", options=["Alle bruggen", "Per brug"], index=0)

    if analyse_scope == "Per brug":
        chosen_bridge_for_followup = st.selectbox("Kies brug", options=["HGWBRN", "HGWBRZ", "GWBR"], index=0)
    else:
        chosen_bridge_for_followup = None

    max_followup_minutes = st.slider("Max follow-up window (min) — zet lager om alleen korte opvolgingen te tellen",
                                     min_value=10, max_value=1440, value=1440, step=10)

    top_k = 5

    @st.cache_data(show_spinner=False)
    def compute_followups(all_st_df: pd.DataFrame, top_k: int = 5, bridge: str | None = None,
                          max_minutes: int | None = None):
        if all_st_df.empty:
            return pd.DataFrame(columns=["base_msg", "base_count", "followup_msg", "followup_count", "followup_share"])

        df = all_st_df.copy()
        df["MsgText"] = df["MsgText"].fillna("").astype(str).str.strip()
        df = df.query("MsgText != ''").copy()

        if bridge:
            df = df[df["bridge"] == bridge].copy()

        if df.empty:
            return pd.DataFrame(columns=["base_msg", "base_count", "followup_msg", "followup_count", "followup_share"])

        df = df.sort_values(["bridge", "time_local"]).reset_index(drop=True)

        if "StateAfter" in df.columns:
            df_bases = df[df["StateAfter"] == 1].copy()
        else:
            st.warning("Kolom 'StateAfter' niet gevonden — alle meldingen worden meegenomen.")
            df_bases = df.copy()

        df_bases["next_msg"] = df_bases.groupby("bridge")["MsgText"].shift(-1)
        df_bases["next_time"] = df_bases.groupby("bridge")["time_local"].shift(-1)
        df_bases["delta_min_next"] = (df_bases["next_time"] - df_bases["time_local"]).dt.total_seconds() / 60.0

        if max_minutes is not None:
            df_bases.loc[df_bases["delta_min_next"] > float(max_minutes), "next_msg"] = pd.NA

        top_bases = df_bases["MsgText"].value_counts().nlargest(top_k).index.tolist()

        rows = []
        for base in top_bases:
            sub = df_bases[df_bases["MsgText"] == base]
            base_count = int(len(sub))
            follow_counts = sub["next_msg"].dropna().value_counts()
            if not follow_counts.empty:
                follow_msg = follow_counts.index[0]
                follow_count = int(follow_counts.iloc[0])
                follow_share = follow_count / base_count if base_count > 0 else 0.0
            else:
                follow_msg = ""
                follow_count = 0
                follow_share = 0.0
            rows.append({
                "base_msg": base,
                "base_count": base_count,
                "followup_msg": follow_msg,
                "followup_count": follow_count,
                "followup_share": follow_share
            })

        out = pd.DataFrame(rows)
        out = out.sort_values("base_count", ascending=False).reset_index(drop=True)
        return out

    all_st = pd.concat([st_HN.assign(bridge="HGWBRN"),
                        st_HZ.assign(bridge="HGWBRZ"),
                        st_GW.assign(bridge="GWBR")], ignore_index=True)

    followup_df = compute_followups(all_st, top_k=top_k, bridge=chosen_bridge_for_followup,
                                    max_minutes=max_followup_minutes)

    if followup_df.empty:
        st.info("Geen storingsdata of geen matches binnen de gekozen instellingen.")
    else:
        followup_df_display = followup_df.copy()
        followup_df_display["followup_pct"] = (followup_df_display["followup_share"] * 100).round(1).astype(str) + "%"
        st.dataframe(
            followup_df_display[["base_msg", "base_count", "followup_msg", "followup_count", "followup_pct"]].rename(
                columns={
                    "base_msg": "Basis storing",
                    "base_count": "Aantal",
                    "followup_msg": "Meest voorkomende follow-up",
                    "followup_count": "Aantal follow-ups",
                    "followup_pct": "Share follow-up"
                }), use_container_width=True)

        long_rows = []
        for _, r in followup_df.iterrows():
            long_rows.append({"base_msg": r["base_msg"], "type": "Basis: count", "count": r["base_count"], "note": ""})
            long_rows.append({"base_msg": r["base_msg"], "type": "Follow-up: count", "count": r["followup_count"], "note": r["followup_msg"]})
        long_df = pd.DataFrame(long_rows)

        fig_follow = px.bar(long_df, x="count", y="base_msg", color="type", orientation="h",
                            title=f"Top {top_k} storingen en hun meest voorkomende follow-up (scope: {'alle' if not chosen_bridge_for_followup else chosen_bridge_for_followup})",
                            labels={"base_msg": "Storing", "count": "Aantal"},
                            text="count")
        fig_follow.update_traces(hovertemplate="<b>%{y}</b><br>%{x} items<br>%{customdata}",
                                 customdata=long_df["note"])
        st.plotly_chart(fig_follow, use_container_width=True)

        for _, r in followup_df.iterrows():
            follow_pct = f"{r['followup_share'] * 100:.1f}%"
            if r["followup_msg"]:
                st.markdown(
                    f"- **{r['base_msg']}** → meest voorkomende follow-up: **{r['followup_msg']}** ({r['followup_count']} keer, {follow_pct} van de gevallen)")
            else:
                st.markdown(f"- **{r['base_msg']}** → geen follow-up binnen {max_followup_minutes} min gevonden.")

# ❌ VERWIJDERD op verzoek: derde tab “Koppeling met sensor-events”


# --------------------
# Predictive Maintenance Analysis (Random Forest)
# --------------------
st.markdown("_____")

st.markdown("## 🤖 Predictive Maintenance Analyse")

# --------------------
# Configuration Section
# --------------------
st.markdown("### Model Configuratie")

# Initialize session state for configuration
if 'pm_config' not in st.session_state:
    st.session_state.pm_config = {
        'data_source': "Huidige brug",
        'selected_bridge': "HGWBRN"
    }

col_config1, col_config2 = st.columns(2)

with col_config1:
    data_source = st.radio(
        "Data bron:",
        ["Huidige brug", "Alle bruggen gecombineerd"],
        index=0 if st.session_state.pm_config['data_source'] == "Huidige brug" else 1,
        key="pm_data_source",
        help="Kies of je alleen de geselecteerde brug wilt gebruiken of data van alle bruggen combineert"
    )
    # Update session state
    st.session_state.pm_config['data_source'] = data_source

with col_config2:
    if data_source == "Huidige brug":
        predictive_bridge = st.selectbox(
            "Selecteer brug voor analyse:",
            options=["HGWBRN", "HGWBRZ", "GWBR"],
            index=["HGWBRN", "HGWBRZ", "GWBR"].index(st.session_state.pm_config['selected_bridge']),
            key="pm_bridge_select"
        )
        st.session_state.pm_config['selected_bridge'] = predictive_bridge
    else:
        st.info("🌉 Gebruikt data van alle bruggen")
        predictive_bridge = "Alle bruggen"


# --------------------
# Data Preparation for Predictive Modeling
# --------------------

@st.cache_data(show_spinner=False)
def prepare_predictive_data_combined():
    """Bereid gecombineerde data voor van alle bruggen"""
    try:
        all_bridges_data = []

        for bridge_name, storings_df in [("HGWBRN", st_HN), ("HGWBRZ", st_HZ), ("GWBR", st_GW)]:
            if storings_df is None or len(storings_df) == 0:
                continue

            # Controleer of we de vereiste kolommen hebben
            required_cols = ['time_local', 'MsgNumber', 'MsgProc', 'StateAfter']
            missing_cols = [col for col in required_cols if col not in storings_df.columns]
            if missing_cols:
                st.warning(f"Ontbrekende kolommen voor {bridge_name}: {missing_cols}")
                continue

            # Sorteer op tijd
            df_sorted = storings_df.sort_values('time_local').reset_index(drop=True)

            # Bereken tijdverschillen (interarrival times) in seconden
            df_sorted['time_diff'] = df_sorted['time_local'].diff().dt.total_seconds()
            df_sorted['interarrival_seconds'] = df_sorted['time_diff'].replace(0, 1e-6)

            # Maak lag features
            n_lags = 10

            def create_lags(series, n_lags, prefix):
                lags_df = pd.DataFrame()
                lags_df[prefix] = series
                for i in range(1, n_lags + 1):
                    lags_df[f'{prefix}_lag{i}'] = series.shift(i)
                return lags_df

            # Maak lag features voor alle variabelen
            msg_number_lags = create_lags(df_sorted['MsgNumber'], n_lags, 'MsgNumber')
            msg_proc_lags = create_lags(df_sorted['MsgProc'], n_lags, 'MsgProc')
            state_after_lags = create_lags(df_sorted['StateAfter'], n_lags, 'StateAfter')
            interarrival_lags = create_lags(df_sorted['interarrival_seconds'], n_lags - 1, 'interarrival_seconds')

            # Combineer alle features
            features_df = pd.concat([
                msg_number_lags,
                msg_proc_lags,
                state_after_lags,
                interarrival_lags
            ], axis=1)

            # Voeg target variabelen toe
            features_df['Y1_time_until_next'] = df_sorted['interarrival_seconds'].shift(-1)
            features_df['Y2_volgendeStoring'] = df_sorted['MsgNumber'].shift(-1)
            features_df['brug'] = bridge_name  # Voeg brug identificatie toe

            # Verwijder rijen met ontbrekende waarden
            final_df = features_df.dropna().reset_index(drop=True)
            all_bridges_data.append(final_df)

        if not all_bridges_data:
            return None

        # Combineer alle data
        combined_df = pd.concat(all_bridges_data, ignore_index=True)
        return combined_df

    except Exception as e:
        st.error(f"Fout bij voorbereiden gecombineerde data: {e}")
        return None


@st.cache_data(show_spinner=False)
def prepare_predictive_data_single(_storingen_df, bridge_name):
    """Bereid data voor predictive modeling voor één brug"""
    try:
        # Controleer of we de vereiste kolommen hebben
        required_cols = ['time_local', 'MsgNumber', 'MsgProc', 'StateAfter']
        missing_cols = [col for col in required_cols if col not in _storingen_df.columns]
        if missing_cols:
            st.warning(f"Ontbrekende kolommen voor predictive analyse: {missing_cols}")
            return None

        # Sorteer op tijd
        df_sorted = _storingen_df.sort_values('time_local').reset_index(drop=True)

        # Bereken tijdverschillen (interarrival times) in seconden
        df_sorted['time_diff'] = df_sorted['time_local'].diff().dt.total_seconds()
        df_sorted['interarrival_seconds'] = df_sorted['time_diff'].replace(0, 1e-6)

        # Maak lag features
        n_lags = 10

        def create_lags(series, n_lags, prefix):
            lags_df = pd.DataFrame()
            lags_df[prefix] = series
            for i in range(1, n_lags + 1):
                lags_df[f'{prefix}_lag{i}'] = series.shift(i)
            return lags_df

        # Maak lag features voor alle variabelen
        msg_number_lags = create_lags(df_sorted['MsgNumber'], n_lags, 'MsgNumber')
        msg_proc_lags = create_lags(df_sorted['MsgProc'], n_lags, 'MsgProc')
        state_after_lags = create_lags(df_sorted['StateAfter'], n_lags, 'StateAfter')
        interarrival_lags = create_lags(df_sorted['interarrival_seconds'], n_lags - 1, 'interarrival_seconds')

        # Combineer alle features
        features_df = pd.concat([
            msg_number_lags,
            msg_proc_lags,
            state_after_lags,
            interarrival_lags
        ], axis=1)

        # Voeg target variabelen toe
        features_df['Y1_time_until_next'] = df_sorted['interarrival_seconds'].shift(-1)
        features_df['Y2_volgendeStoring'] = df_sorted['MsgNumber'].shift(-1)
        features_df['brug'] = bridge_name  # Voeg brug identificatie toe

        # Verwijder rijen met ontbrekende waarden
        final_df = features_df.dropna().reset_index(drop=True)

        return final_df

    except Exception as e:
        st.error(f"Fout bij voorbereiden predictive data: {e}")
        return None


# --------------------
# Temporal Split Function
# --------------------

def safe_temporal_split_multiple(df, test_chunk_pct=0.05, buffer=10):
    """Temporale split met meerdere test chunks en buffers"""
    n = len(df)
    test_chunk_size = int(test_chunk_pct * n)

    # Definieer test chunk posities (25%, 50%, 75%, en final)
    test_positions_pct = [0.25, 0.50, 0.75, 0.95]
    test_positions = [int(pos * n) for pos in test_positions_pct]

    # Zorg dat posities geldig zijn en genoeg ruimte hebben voor buffers
    valid_positions = []
    for i, pos in enumerate(test_positions):
        test_end = pos + test_chunk_size - 1

        # Voor de laatste chunk hebben we geen buffer na nodig
        if i == len(test_positions) - 1:
            if (pos - buffer >= 0) and (test_end < n):
                valid_positions.append(pos)
        else:
            # Voor andere chunks, controleer buffer voor EN na
            if (pos - buffer >= 0) and (test_end + buffer < n):
                valid_positions.append(pos)

    test_positions = valid_positions

    # Verzamel alle indices
    all_test_indices = []
    all_buffer_indices = []

    for i, test_start in enumerate(test_positions):
        test_end = test_start + test_chunk_size - 1

        # Test indices
        test_indices = list(range(test_start, test_end + 1))
        all_test_indices.extend(test_indices)

        # Buffer voor test chunk
        buffer_before_start = max(0, test_start - buffer)
        buffer_before_end = test_start - 1
        if buffer_before_start <= buffer_before_end:
            buffer_before_indices = list(range(buffer_before_start, buffer_before_end + 1))
            all_buffer_indices.extend(buffer_before_indices)

        # Buffer na test chunk (behalve voor de laatste chunk)
        if i < len(test_positions) - 1:
            buffer_after_start = test_end + 1
            buffer_after_end = min(n - 1, test_end + buffer)
            if buffer_after_start <= buffer_after_end:
                buffer_after_indices = list(range(buffer_after_start, buffer_after_end + 1))
                all_buffer_indices.extend(buffer_after_indices)

    # Verwijder duplicaten
    all_test_indices = list(set(all_test_indices))
    all_buffer_indices = list(set(all_buffer_indices))

    # Training indices zijn al het andere
    all_indices = set(range(n))
    all_train_indices = list(all_indices - set(all_test_indices) - set(all_buffer_indices))

    # Maak datasets
    train_data = df.iloc[all_train_indices].reset_index(drop=True)

    # Maak test chunks
    test_chunks = []
    for test_start in test_positions:
        test_end = test_start + test_chunk_size - 1
        test_chunks.append(df.iloc[test_start:test_end + 1].reset_index(drop=True))

    # Bereken groottes
    total_train = len(all_train_indices)
    total_test = len(all_test_indices)
    total_buffer = len(all_buffer_indices)

    return {
        'main_train': train_data,
        'test_chunks': test_chunks,
        'test_positions': test_positions,
        'test_chunk_size': test_chunk_size,
        'buffer_used': buffer,
        'total_train': total_train,
        'total_test': total_test,
        'total_buffer': total_buffer,
        'total_original': n,
        'train_percentage': total_train / n,
        'test_percentage': total_test / n,
        'buffer_percentage': total_buffer / n
    }


# --------------------
# Model Training and Evaluation
# --------------------

def train_and_evaluate_models():
    """Train en evalueer Random Forest modellen voor predictive maintenance"""

    # Bereid data voor op basis van selectie
    with st.spinner("Data voorbereiden voor predictive modeling..."):
        if st.session_state.pm_config['data_source'] == "Alle bruggen gecombineerd":
            df_predictive = prepare_predictive_data_combined()
            data_source_info = "alle bruggen gecombineerd"
        else:
            storings_map = {
                "HGWBRN": st_HN,
                "HGWBRZ": st_HZ,
                "GWBR": st_GW
            }
            selected_bridge = st.session_state.pm_config['selected_bridge']
            storings_df = storings_map[selected_bridge]
            df_predictive = prepare_predictive_data_single(storings_df, selected_bridge)
            data_source_info = selected_bridge

    if df_predictive is None or len(df_predictive) < 100:
        st.warning(f"Niet genoeg data voor predictive modeling. Minimaal 100 complete observaties nodig.")
        return

    st.success(f"✅ {len(df_predictive)} observaties voorbereid voor modeling ({data_source_info})")

    # Pas temporale split toe
    with st.spinner("Temporale split toepassen..."):
        split_result = safe_temporal_split_multiple(df_predictive, test_chunk_pct=0.05, buffer=10)


    # --------------------
    # Data Overzicht with Last Observations Dropdown
    # --------------------
    with st.expander("📋 Data Overzicht"):
        # Dropdown to show last observations
        col1, col2 = st.columns([3, 1])

        with col2:
            show_last_observations = st.selectbox(
                "Toon laatste observaties:",
                options=["Laatste 10 observaties", "Volledige dataset"],
                index=0,
                key="last_obs_selector"
            )

        # Determine which original storings data to display
        if st.session_state.pm_config['data_source'] == "Alle bruggen gecombineerd":
            # For combined data, show last 10 observations from HGWBRN original data
            if st_HN is not None and len(st_HN) > 0:
                st.write("**Laatste 10 storingsobservaties - HGWBRN (originele data):**")
                if show_last_observations == "Laatste 10 observaties":
                    if len(st_HN) >= 14:
                        st.dataframe(st_HN.tail(14).head(10))  # Show rows 11-20 from the end
                    else:
                        st.dataframe(st_HN.head(10))  # Fallback if not enough data
                else:
                    st.dataframe(st_HN.head(10))
            else:
                st.warning("Geen HGWBRN storingsdata beschikbaar")
        else:
            # For single bridge, show last 10 observations of selected bridge's original data
            selected_bridge = st.session_state.pm_config['selected_bridge']
            storings_map = {
                "HGWBRN": st_HN,
                "HGWBRZ": st_HZ,
                "GWBR": st_GW
            }

            storings_df = storings_map[selected_bridge]
            if storings_df is not None and len(storings_df) > 0:
                st.write(f"**Laatste 10 storingsobservaties - {selected_bridge} (originele data):**")
                if show_last_observations == "Laatste 10 observaties":
                    if len(storings_df) >= 14:
                        st.dataframe(storings_df.tail(14).head(10))  # Show rows 11-20 from the end
                    else:
                        st.dataframe(storings_df.head(10))  # Fallback if not enough data
                else:
                    st.dataframe(storings_df.head(10))
            else:
                st.warning(f"Geen storingsdata beschikbaar voor {selected_bridge}")

        # Keep the existing statistics for the predictive dataset
        st.write("**Predictive Dataset Statistieken:**")
        st.write(f"Totaal observaties: {len(df_predictive)}")
        st.write(f"Training observaties: {split_result['total_train']}")
        st.write(f"Test observaties: {split_result['total_test']}")
        st.write(f"Buffer observaties: {split_result['total_buffer']}")
        st.write(f"Aantal test chunks: {len(split_result['test_chunks'])}")
        st.write(f"Features: {len(df_predictive.columns) - 3}")  # min 3 target kolommen (incl. brug)

        if st.session_state.pm_config['data_source'] == "Alle bruggen gecombineerd":
            st.write("**Verdeling over bruggen:**")
            bridge_dist = df_predictive['brug'].value_counts()
            st.write(bridge_dist)


    # Train modellen
    st.write("### Modellen Trainen")

    try:
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
        import numpy as np

        # Bereid training data voor
        train_data = split_result['main_train']

        # Features voor Y1 (exclude Y2 target en brug)
        exclude_cols = ['Y1_time_until_next', 'Y2_volgendeStoring', 'brug']
        X_train = train_data.drop(exclude_cols, axis=1)
        y1_train = train_data['Y1_time_until_next']
        y2_train = train_data['Y2_volgendeStoring'].astype('category')

        # Train Y1 model (regressie)
        with st.spinner("Tijd-tot-storing model (Y1) trainen..."):
            rf_Y1 = RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )
            rf_Y1.fit(X_train, y1_train)

        # Train Y2 model (classificatie)
        with st.spinner("Storingstype model (Y2) trainen..."):
            rf_Y2 = RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            )
            rf_Y2.fit(X_train, y2_train)

        st.success("✅ Modellen succesvol getraind!")

        # Sla modellen op in session state voor predictie visualisatie
        st.session_state.rf_Y1 = rf_Y1
        st.session_state.rf_Y2 = rf_Y2
        st.session_state.X_train_columns = X_train.columns.tolist()

        # Evalueer modellen (verborgen in expander)
        with st.expander("📊 Gedetailleerde Model Evaluatie", expanded=False):
            st.write("### Model Evaluatie Resultaten")

            evaluation_results = []
            all_y2_details = []

            # Maak tabs voor individuele chunks + gecombineerde analyse
            tab_names = [f"Test Chunk {i + 1}" for i in range(len(split_result['test_chunks']))]
            tab_names.extend(["Gecombineerde Analyse", "Feature Importance"])

            chunk_tabs = st.tabs(tab_names)

            # Tab 1-4: Individuele chunks
            for i, (test_chunk, tab) in enumerate(
                    zip(split_result['test_chunks'], chunk_tabs[:len(split_result['test_chunks'])])):
                with tab:
                    st.write(f"#### Test Chunk {i + 1}")

                    # Bereid test data voor
                    X_test = test_chunk.drop(exclude_cols, axis=1)
                    y1_test = test_chunk['Y1_time_until_next']
                    y2_test = test_chunk['Y2_volgendeStoring'].astype('category')

                    # Y1 voorspellingen en metrics
                    y1_pred = rf_Y1.predict(X_test)
                    y1_rmse = np.sqrt(mean_squared_error(y1_test, y1_pred))
                    y1_mae = mean_absolute_error(y1_test, y1_pred)

                    # Y2 voorspellingen met veilige afhandeling van onbekende classes
                    try:
                        y2_pred = rf_Y2.predict(X_test)
                        y2_proba = rf_Y2.predict_proba(X_test)
                        y2_accuracy = accuracy_score(y2_test, y2_pred)

                        # Bereken top-k accuracy met veilige class afhandeling
                        def safe_top_k_accuracy(proba, actual, classes, k):
                            correct = 0
                            total = len(actual)

                            for idx, true_label in enumerate(actual):
                                top_k_indices = np.argsort(proba[idx])[-k:]
                                top_k_labels = [classes[i] for i in top_k_indices]

                                if true_label in top_k_labels:
                                    correct += 1

                            return correct / total if total > 0 else 0

                        # Bereken top-k accuracies
                        y2_top2_accuracy = safe_top_k_accuracy(y2_proba, y2_test.values, rf_Y2.classes_, 2)
                        y2_top3_accuracy = safe_top_k_accuracy(y2_proba, y2_test.values, rf_Y2.classes_, 3)

                        # Gemiddelde confidence
                        y2_confidence = np.max(y2_proba, axis=1).mean()

                    except Exception as e:
                        st.warning(f"⚠️ Probleem met Y2 evaluatie in chunk {i + 1}: {str(e)}")
                        y2_pred = rf_Y2.predict(X_test)
                        y2_accuracy = accuracy_score(y2_test, y2_pred)
                        y2_top2_accuracy = np.nan
                        y2_top3_accuracy = np.nan
                        y2_confidence = np.nan
                        y2_proba = None

                    # Sla resultaten op
                    evaluation_results.append({
                        'chunk_id': i + 1,
                        'Y1_rmse': y1_rmse,
                        'Y1_mae': y1_mae,
                        'Y2_accuracy': y2_accuracy,
                        'Y2_top2_accuracy': y2_top2_accuracy,
                        'Y2_top3_accuracy': y2_top3_accuracy,
                        'avg_confidence': y2_confidence
                    })

                    # Toon chunk resultaten
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Y1 RMSE (seconden)", f"{y1_rmse:.2f}")
                        st.metric("Y1 MAE (seconden)", f"{y1_mae:.2f}")
                    with col2:
                        st.metric("Y2 Accuracy", f"{y2_accuracy:.4f}")
                        st.metric("Y2 Top-2 Accuracy",
                                  f"{y2_top2_accuracy:.4f}" if not np.isnan(y2_top2_accuracy) else "N/A")
                    with col3:
                        st.metric("Y2 Top-3 Accuracy",
                                  f"{y2_top3_accuracy:.4f}" if not np.isnan(y2_top3_accuracy) else "N/A")
                        st.metric("Gem. Confidence",
                                  f"{y2_confidence:.4f}" if not np.isnan(y2_confidence) else "N/A")

            # Tab 5: Gecombineerde Analyse
            with chunk_tabs[len(split_result['test_chunks'])]:
                st.write("#### Gecombineerde Analyse - Alle Test Data")

                try:
                    # Combineer alle test data voor uitgebreide evaluatie
                    all_test_data = pd.concat(split_result['test_chunks'])
                    X_test_combined = all_test_data.drop(exclude_cols, axis=1)
                    y1_test_combined = all_test_data['Y1_time_until_next']
                    y2_test_combined = all_test_data['Y2_volgendeStoring'].astype('category')

                    # Gecombineerde voorspellingen
                    y1_pred_combined = rf_Y1.predict(X_test_combined)
                    y2_pred_combined = rf_Y2.predict(X_test_combined)

                    # Y1 gecombineerde metrics
                    y1_rmse_combined = np.sqrt(mean_squared_error(y1_test_combined, y1_pred_combined))
                    y1_mae_combined = mean_absolute_error(y1_test_combined, y1_pred_combined)

                    # Y2 gecombineerde metrics met veilige afhandeling
                    try:
                        y2_proba_combined = rf_Y2.predict_proba(X_test_combined)
                        y2_accuracy_combined = accuracy_score(y2_test_combined, y2_pred_combined)

                        # Veilige top-k voor gecombineerde data
                        y2_top2_accuracy_combined = safe_top_k_accuracy(
                            y2_proba_combined, y2_test_combined.values, rf_Y2.classes_, 2
                        )
                        y2_top3_accuracy_combined = safe_top_k_accuracy(
                            y2_proba_combined, y2_test_combined.values, rf_Y2.classes_, 3
                        )

                    except Exception as e:
                        st.warning(f"⚠️ Probleem met gecombineerde Y2 evaluatie: {str(e)}")
                        y2_accuracy_combined = accuracy_score(y2_test_combined, y2_pred_combined)
                        y2_top2_accuracy_combined = np.nan
                        y2_top3_accuracy_combined = np.nan
                        y2_proba_combined = None

                    # Toon gecombineerde resultaten
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Y1 - Tijd tot Storing")
                        st.metric("RMSE (seconden)", f"{y1_rmse_combined:.2f}")
                        st.metric("MAE (seconden)", f"{y1_mae_combined:.2f}")

                        st.write("**Interpretatie:**")
                        st.write(f"- Gemiddelde voorspellingsfout: {y1_mae_combined:.2f} seconden")

                    with col2:
                        st.subheader("Y2 - Storingstype")
                        st.metric("Top-1 Accuracy", f"{y2_accuracy_combined:.4f}")
                        st.metric("Top-2 Accuracy",
                                  f"{y2_top2_accuracy_combined:.4f}" if not np.isnan(
                                      y2_top2_accuracy_combined) else "N/A")
                        st.metric("Top-3 Accuracy",
                                  f"{y2_top3_accuracy_combined:.4f}" if not np.isnan(
                                      y2_top3_accuracy_combined) else "N/A")

                        if y2_proba_combined is not None:
                            avg_confidence_combined = np.max(y2_proba_combined, axis=1).mean()
                            st.metric("Gem. Confidence", f"{avg_confidence_combined:.4f}")

                except Exception as e:
                    st.error(f"Fout in gecombineerde analyse: {e}")

            # Tab 6: Feature Importance
            with chunk_tabs[len(split_result['test_chunks']) + 1]:
                st.write("#### Feature Importance Analyse")

                col_imp1, col_imp2 = st.columns(2)

                with col_imp1:
                    # Y1 feature importance
                    y1_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': rf_Y1.feature_importances_
                    }).sort_values('importance', ascending=False)

                    st.write("**Y1 - Tijd tot Storing**")
                    fig_y1_imp = px.bar(
                        y1_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 15 Features - Tijd Voorspelling',
                        labels={'importance': 'Belangrijkheid', 'feature': 'Feature'}
                    )
                    st.plotly_chart(fig_y1_imp, use_container_width=True)


                with col_imp2:
                    # Y2 feature importance
                    y2_importance = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': rf_Y2.feature_importances_
                    }).sort_values('importance', ascending=False)

                    st.write("**Y2 - Storingstype**")
                    fig_y2_imp = px.bar(
                        y2_importance.head(15),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title='Top 15 Features - Storingstype Voorspelling',
                        labels={'importance': 'Belangrijkheid', 'feature': 'Feature'}
                    )
                    st.plotly_chart(fig_y2_imp, use_container_width=True)


        # --------------------
        # Actuele Voorspellingen
        # --------------------
        st.write("### 🔮 Actuele Voorspellingen")

        # Haal de laatste observatie op voor voorspelling
        final_observation = df_predictive.iloc[-1:].drop(exclude_cols, axis=1)

        # Zorg dat de laatste observatie dezelfde kolommen heeft als training data
        final_observation = final_observation.reindex(columns=X_train.columns, fill_value=0)

        # Haal individuele boom voorspellingen op voor Y1
        individual_tree_preds = []
        for tree in rf_Y1.estimators_:
            pred = tree.predict(final_observation)
            individual_tree_preds.append(pred[0])

        individual_tree_preds = np.array(individual_tree_preds)

        # Definieer tijdintervallen (in seconden)
        time_intervals = {
            '<10s': (0, 10),
            '10s-1m': (10, 60),
            '1m-60m': (60, 3600),
            '60m-12h': (3600, 43200),
            '12h-1week': (43200, 604800),
            '>1week': (604800, float('inf'))
        }

        # Tel voorspellingen in elk interval
        interval_counts = {}
        for interval_name, (low, high) in time_intervals.items():
            if high == float('inf'):
                count = np.sum(individual_tree_preds >= low)
            else:
                count = np.sum((individual_tree_preds >= low) & (individual_tree_preds < high))
            interval_counts[interval_name] = count

        # Storingstype voorspelling visualisatie
        failure_probs = rf_Y2.predict_proba(final_observation)[0]
        top_failure_indices = np.argsort(failure_probs)[-5:][::-1]  # Top 5

        # Maak een mapping van MsgNumber naar MsgText
        msg_mapping = {}
        for bridge_name, storings_df in [("HGWBRN", st_HN), ("HGWBRZ", st_HZ), ("GWBR", st_GW)]:
            if storings_df is not None and len(storings_df) > 0:
                bridge_mapping = storings_df[['MsgNumber', 'MsgText']].drop_duplicates().set_index('MsgNumber')[
                    'MsgText'].to_dict()
                msg_mapping.update(bridge_mapping)

        # Maak DataFrame met leesbare storingstypen
        failure_data = []
        for i in top_failure_indices:
            msg_number = rf_Y2.classes_[i]
            probability = failure_probs[i]
            # Zoek de bijbehorende MsgText, gebruik MsgNumber als fallback
            msg_text = msg_mapping.get(msg_number, f"Storing {msg_number}")
            failure_data.append({
                'Storingstype': msg_text,
                'Kans': probability,
                'Percentage': f"{probability * 100:.1f}%"
            })

        failure_type_df = pd.DataFrame(failure_data)

        # Toon beide grafieken naast elkaar
        col_pred1, col_pred2 = st.columns([1, 1.3])

        with col_pred1:
            # Tijd voorspelling - use inside labels
            interval_df = pd.DataFrame({
                'Tijd Interval': list(interval_counts.keys()),
                'Percentage': [(count / 500) * 100 for count in interval_counts.values()],
                'Percentage_tekst': [f"{(count / 500) * 100:.1f}%" for count in interval_counts.values()]
            })

            fig_time_intervals = px.bar(
                interval_df,
                x='Percentage',
                y='Tijd Interval',
                orientation='h',
                title='Voorspelde tijd tot volgende storing',
                labels={'Percentage': 'Kans (%)', 'Tijd Interval': 'Voorspelde tijd interval'},
                text='Percentage_tekst'
            )
            fig_time_intervals.update_traces(
                textposition='inside',  # Changed to inside to avoid cutoff
                texttemplate='%{text}',
                insidetextanchor='middle',
                marker_color='lightblue'
            )
            fig_time_intervals.update_layout(
                margin=dict(l=20, r=20, t=50, b=50),
                height=400
            )
            st.plotly_chart(fig_time_intervals, use_container_width=True)

            # Statistics...

        with col_pred2:
            # Storingstype voorspelling - use inside labels
            fig_failure_types = px.bar(
                failure_type_df,
                x='Kans',
                y='Storingstype',
                orientation='h',
                title='Voorspelde volgende storingstype',
                labels={'Kans': 'Voorspellingskans', 'Storingstype': 'Storingstype'},
                text='Percentage'
            )
            fig_failure_types.update_traces(
                textposition='inside',  # Changed to inside to avoid cutoff
                texttemplate='%{text}',
                insidetextanchor='middle',
                marker_color='lightcoral'
            )
            fig_failure_types.update_layout(
                margin=dict(l=20, r=20, t=50, b=50),
                height=400,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig_failure_types, use_container_width=True)


        # Download voorspellingen
        st.write("### Download Voorspellingen")

        # Maak voorspelling export
        prediction_export = X_test_combined.copy()
        prediction_export['Y1_werkelijk'] = y1_test_combined.values
        prediction_export['Y1_voorspeld'] = y1_pred_combined
        prediction_export['Y2_werkelijk'] = y2_test_combined.values
        prediction_export['Y2_voorspeld'] = y2_pred_combined

        # Voeg kansen toe voor elk storingstype indien beschikbaar
        if y2_proba_combined is not None:
            for i, failure_type in enumerate(rf_Y2.classes_):
                prediction_export[f'kans_{failure_type}'] = y2_proba_combined[:, i]

        csv_export = prediction_export.to_csv(index=False)

        st.download_button(
            label="⬇ Download Voorspellingsresultaten",
            data=csv_export,
            file_name=f"predictive_maintenance_voorspellingen_{data_source_info.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Fout in model training/evaluatie: {e}")
        st.info("Dit kan komen door onvoldoende data of ontbrekende vereiste kolommen in de storingsdata.")


# --------------------
# Run Predictive Analysis
# --------------------

# Voeg een knop toe om de predictive analyse uit te voeren
if st.button("🚀 Predictive Maintenance Analyse Uitvoeren"):
    train_and_evaluate_models()
else:
    st.info(
        "Klik op de knop hierboven om de predictive maintenance analyse uit te voeren. Dit traint Random Forest modellen om tijd tot volgende storing en storingstypes te voorspellen.")


# --------------------
# Data kwaliteit & exports
# --------------------
with st.expander("🧪 Datakwaliteit & Export"):
    st.write("Snelle checks en downloadmogelijkheden.")
    for name, df in [("HGWBRN", df_HN), ("HGWBRZ", df_HZ), ("GWBR", df_GW)]:
        st.markdown(f"{name} sensor** — NaT: {df['time_local'].isna().sum()} / {len(df)}")
    for name, df in [("HGWBRN storingen", st_HN), ("HGWBRZ storingen", st_HZ), ("GWBR storingen", st_GW)]:
        st.markdown(f"{name}** — NaT: {df['time_local'].isna().sum()} / {len(df)}")

    colx, coly, colz = st.columns(3)
    with colx:
        st.download_button("⬇ Download HGWBRN (sensor, sample 10k)",
                           data=df_HN.head(10_000).to_csv(index=False), mime="text/csv",
                           file_name="HGWBRN_sample.csv")
    with coly:
        st.download_button("⬇ Download HGWBRZ (sensor, sample 10k)",
                           data=df_HZ.head(10_000).to_csv(index=False), mime="text/csv",
                           file_name="HGWBRZ_sample.csv")
    with colz:
        st.download_button("⬇ Download GWBR (sensor, sample 10k)",
                           data=df_GW.head(10_000).to_csv(index=False), mime="text/csv",
                           file_name="GWBR_sample.csv")

import streamlit as st
import pandas as pd
import xarray as xr
import plotly.express as px
import os

# ==========================================
# 1. CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Dashboard Météo France (35 ans)",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Analyse Climatique France (1950-1985)")
st.markdown("Visualisation interactive des données météorologiques journalières.")

# ==========================================
# 2. FONCTION DE CHARGEMENT (Avec Cache)
# ==========================================
@st.cache_resource
def load_data():
    """
    Charge les données une seule fois pour optimiser les performances.
    Renvoie le DataFrame journalier et le DataArray mensuel pour la carte.
    """
    file_path = "donnees_carte_35ans_journalier.nc"

    if not os.path.exists(file_path):
        return None, None, None

    # Chargement NetCDF
    ds = xr.open_dataset(file_path)

    # --- Préparation A : Données Tabulaires (France entière) ---
    ds_mean = ds.mean(dim=['latitude', 'longitude'], skipna=True)
    df_france = ds_mean.to_dataframe().reset_index()

    # Nettoyage Colonnes
    if 'valid_time' in df_france.columns:
        df_france = df_france.rename(columns={'valid_time': 'time'})

    df_france['time'] = pd.to_datetime(df_france['time'])
    df_france['Year'] = df_france['time'].dt.year
    df_france['Month'] = df_france['time'].dt.month
    df_france['DayOfYear'] = df_france['time'].dt.dayofyear

    # Détection de la variable
    col_temp = 'Temperature_C' if 'Temperature_C' in df_france.columns else 't2m'

    # --- Préparation B : Données Carte (Ré-échantillonnage mensuel) ---
    # On prépare déjà la version mensuelle pour l'animation (pour éviter les lenteurs)
    da_temp = ds[col_temp]
    ds_map_viz = da_temp.resample(time='1MS').mean(skipna=True)

    return df_france, ds_map_viz, col_temp

# ==========================================
# 3. CHARGEMENT
# ==========================================
with st.spinner('Chargement des données NetCDF en cours...'):
    df_france, ds_map_viz, col_temp = load_data()

if df_france is None:
    st.error("❌ Fichier 'donnees_carte_35ans_journalier.nc' introuvable. Veuillez le placer dans le dossier.")
    st.stop()

# ==========================================
# 4. INTERFACE (ONGLETS)
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Anomalies", "🍂 Saisons", "🗺️ Carte Animée"])

# --- ONGLET 1 : Anomalie Annuelle ---
with tab1:
    st.header("Évolution de l'Anomalie Annuelle")
    st.markdown("Cet onglet montre l'écart de la température moyenne de chaque année par rapport à la moyenne globale des 35 ans.")

    # Calculs
    df_yearly = df_france.groupby('Year')[col_temp].mean().reset_index()
    ref_mean = df_yearly[col_temp].mean()
    df_yearly['Anomalie'] = df_yearly[col_temp] - ref_mean

    # Graphique
    fig1 = px.line(
        df_yearly,
        x='Year',
        y='Anomalie',
        markers=True,
        title="Anomalie de température annuelle (France)",
        labels={'Anomalie': 'Écart à la moyenne (°C)', 'Year': 'Année'},
        height=500
    )
    fig1.add_hline(y=0, line_dash="dash", line_color="grey")

    # Affichage Streamlit
    st.plotly_chart(fig1, use_container_width=True)

# --- ONGLET 2 : Saisons (Boxplot) ---
with tab2:
    st.header("Distribution Saisonnière")
    st.markdown("Utilisez le bouton 'Play' ci-dessous pour voir l'évolution de la dispersion des températures année après année.")

    # Mapping Saisons
    season_map = {
        1: 'Hiver', 2: 'Hiver', 12: 'Hiver',
        3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
        6: 'Été', 7: 'Été', 8: 'Été',
        9: 'Automne', 10: 'Automne', 11: 'Automne'
    }
    df_france['Saison'] = df_france['Month'].map(season_map)

    # Bornes Y fixes
    min_y = df_france[col_temp].min() - 5
    max_y = df_france[col_temp].max() + 5

    # Graphique
    fig2 = px.box(
        df_france.sort_values('Year'),
        x="Saison",
        y=col_temp,
        animation_frame="Year",
        color="Saison",
        category_orders={"Saison": ["Hiver", "Printemps", "Été", "Automne"]},
        range_y=[min_y, max_y],
        title="Distribution des températures par Saison",
        labels={col_temp: "Température journalière (°C)"},
        height=600
    )

    st.plotly_chart(fig2, use_container_width=True)

# --- ONGLET 3 : Carte Animée (Imshow) ---
with tab3:
    st.header("Animation Thermique de la France")
    st.markdown("Visualisation mensuelle des températures via le masque géographique.")

    # Graphique
    fig4 = px.imshow(
        ds_map_viz,
        animation_frame="time",
        origin='lower',
        aspect='geo',
        color_continuous_scale="RdBu_r",
        range_color=[-5, 25],
        title="Animation thermique (Mensuel)",
        labels={'color': 'Température (°C)', 'latitude': 'Latitude', 'longitude': 'Longitude'},
        height=650
    )

    # Formatage Slider
    fig4.layout.sliders[0].currentvalue.prefix = "Date : "
    for step in fig4.layout.sliders[0].steps:
        step["label"] = pd.to_datetime(step["label"]).strftime('%Y-%m')

    st.plotly_chart(fig4, use_container_width=True)
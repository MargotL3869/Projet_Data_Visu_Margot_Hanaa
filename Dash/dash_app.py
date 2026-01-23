import dash
from dash import dcc, html, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# =========================================================
# 1. CHARGEMENT
# =========================================================
print("⏳ Chargement des données...")

curr_dir = Path.cwd()
possibles = [curr_dir, curr_dir.parent / "TP", curr_dir / "TP"]

fichier_nc = None
fichier_poids = None
fichier_villes = None

# Recherche des fichiers
for p in possibles:
    if (p / "donnees_carte_70ans_journalier.nc").exists():
        fichier_nc = p / "donnees_carte_70ans_journalier.nc"
    elif (p / "meteo_france_70ans_final.nc").exists():
        fichier_nc = p / "meteo_france_70ans_final.nc"

    if (p / "poids_regions_finie.nc").exists():
        fichier_poids = p / "poids_regions_finie.nc"

    if (p / "villes_mapping.parquet").exists():
        fichier_villes = p / "villes_mapping.parquet"

    if fichier_nc and fichier_poids:
        break

if not fichier_nc:
    sys.exit("❌ Erreur : Fichier météo introuvable.")

# Chargement en mémoire (Optimisation)
try:
    print("⏳ Lecture RAM...")
    ds = xr.open_dataset(fichier_nc).load()
    ds_poids = xr.open_dataset(fichier_poids).load() if fichier_poids else None
except:
    print("⚠️ Lecture disque (fichier trop gros pour la RAM)")
    ds = xr.open_dataset(fichier_nc)
    ds_poids = xr.open_dataset(fichier_poids) if fichier_poids else None

# Renommage lat/lon
if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
if ds_poids and 'latitude' in ds_poids.coords: ds_poids = ds_poids.rename({'latitude': 'lat', 'longitude': 'lon'})

# Conversion Température
var_temp = 'Temperature_C' if 'Temperature_C' in ds else 't2m'
if ds[var_temp].mean() > 200:
    ds['temp_c'] = ds[var_temp] - 273.15
else:
    ds['temp_c'] = ds[var_temp]

# Liste Régions
liste_regions = sorted([str(r) for r in ds_poids.region.values]) if ds_poids else ["France"]

# --- CHARGEMENT DES VILLES (ET NETTOYAGE) ---
df_villes = pd.DataFrame()
if fichier_villes:
    df_villes = pd.read_parquet(fichier_villes)

    # Nettoyage et Label
    col_cp = "CodePostal" if "CodePostal" in df_villes.columns else ("cp" if "cp" in df_villes.columns else None)
    col_ville = "Ville" if "Ville" in df_villes.columns else df_villes.columns[0]

    # On enlève les villes sans coordonnées
    lat_col = next((c for c in df_villes.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df_villes.columns if 'lon' in c.lower()), None)

    if lat_col and lon_col:
        df_villes = df_villes.dropna(subset=[lat_col, lon_col])

        def clean_label(row):
            nom = str(row[col_ville])
            cp = str(row[col_cp]) if col_cp else ""
            if cp in ["0", "00000", "None", "nan"]: return nom
            return f"{nom} ({cp})"

        df_villes["label"] = df_villes.apply(clean_label, axis=1)
        # On garde une copie propre en mémoire globale
        df_villes = df_villes[[col_ville, "label", lat_col, lon_col]].rename(columns={lat_col: 'lat', lon_col: 'lon'})
        df_villes = df_villes.sort_values("label")
        print(f"✅ {len(df_villes)} villes prêtes.")
    else:
        df_villes = pd.DataFrame([{'label': 'Paris (Default)', 'lat': 48.85, 'lon': 2.35}])

# Dates
annee_min = int(pd.to_datetime(ds.time.values[0]).year)
annee_max = int(pd.to_datetime(ds.time.values[-1]).year)
liste_annees = list(range(annee_min, annee_max + 1))


# =========================================================
# 2. LAYOUT
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("🌤️ Dashboard Climatique : Filtrage Régional", className="text-primary mt-3"), width=12),
        dbc.Col(html.P(f"Données : {annee_min} - {annee_max}", className="text-muted"), width=12)
    ], className="mb-4 text-center"),

    dbc.Row([
        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🛠️ Paramètres", className="bg-primary text-white fw-bold"),
                dbc.CardBody([
                    html.Label("📍 1. Région :", className="fw-bold"),
                    dcc.Dropdown(
                        id='dd-region',
                        options=[{'label': r, 'value': r} for r in liste_regions],
                        value=liste_regions[0],
                        clearable=False, className="mb-3"
                    ),

                    html.Label("🏙️ 2. Ville (Filtrée par région) :", className="fw-bold"),
                    # Note : les options sont vides au départ, le callback va les remplir
                    dcc.Dropdown(
                        id='dd-ville',
                        options=[],
                        value=None,
                        placeholder="Chargement...",
                        clearable=False, searchable=True, className="mb-3"
                    ),
                    dbc.FormText("La liste des villes se met à jour selon la région choisie."),

                    html.Hr(),
                    html.Label("🔥 3. Seuil Canicule :", className="fw-bold text-danger"),
                    dcc.Slider(id='slider-seuil', min=25, max=40, step=1, value=30, marks={i: str(i) for i in range(25, 41, 5)}),

                    html.Hr(),
                    html.Label("📅 4. Année Zoom :", className="fw-bold"),
                    dcc.Dropdown(id='dd-annee', options=[{'label': str(a), 'value': a} for a in liste_annees], value=2003, clearable=False, className="mb-3"),
                ])
            ], className="shadow-sm sticky-top", style={"top": "20px"})
        ], width=12, lg=3),

        # Graphiques
        dbc.Col([
            dbc.Card([dbc.CardHeader("🆚 Comparaison : Ville vs Région"), dbc.CardBody(dcc.Graph(id='g-compare'))], className="mb-4 shadow-sm border-0"),
            dbc.Card([dbc.CardHeader("📊 Anomalies"), dbc.CardBody(dcc.Graph(id='g-master'))], className="mb-4 shadow-sm border-0"),
            dbc.Card([dbc.CardHeader(id='header-detail', className="bg-dark text-white"), dbc.CardBody(dcc.Graph(id='g-detail'))], className="mb-4 shadow-sm border-0"),
            dbc.Row([
                dbc.Col(dbc.Card([dbc.CardHeader("🌡️ Saisonnalité"), dbc.CardBody(dcc.Graph(id='g-heatmap'))], className="shadow-sm border-0"), width=12, lg=6),
                dbc.Col(dbc.Card([dbc.CardHeader("📈 Fréquence"), dbc.CardBody(dcc.Graph(id='g-simulateur'))], className="shadow-sm border-0"), width=12, lg=6),
            ])
        ], width=12, lg=9)
    ])
], fluid=True, className="bg-light pb-5")


# =========================================================
# 3. CALLBACKS
# =========================================================

# --- CALLBACK 1 : FILTRE VILLES (C'est lui qui fait la magie) ---
@app.callback(
    [Output('dd-ville', 'options'),
     Output('dd-ville', 'value')],
    Input('dd-region', 'value')
)
def update_cities_dropdown(selected_region):

    if ds_poids is None or df_villes.empty:
        options = [{'label': l, 'value': l} for l in df_villes['label']]
        return options, options[0]['value'] if options else None

    mask = ds_poids['weights'].sel(region=selected_region)

    villes_valides = []

    for _, row in df_villes.iterrows():
        try:
            val = mask.sel(
                lat=row['lat'],
                lon=row['lon'],
                method='nearest'
            ).values

            if not np.isnan(val) and val > 1e-6:
                villes_valides.append(row['label'])

        except Exception:
            continue

    # Sécurité : si jamais aucune ville trouvée
    if not villes_valides:
        print(f"⚠️ Aucune ville trouvée pour la région {selected_region}")

    options = [{'label': v, 'value': v} for v in villes_valides]
    default_val = options[0]['value'] if options else None

    return options, default_val




# --- CALLBACK 2 : GRAPHIQUES (Reste inchangé ou presque) ---
@app.callback(
    [Output('g-compare', 'figure'),
     Output('g-master', 'figure'),
     Output('g-detail', 'figure'),
     Output('g-heatmap', 'figure'),
     Output('g-simulateur', 'figure'),
     Output('header-detail', 'children'),
     Output('dd-annee', 'value')],
    [Input('dd-region', 'value'),
     Input('dd-ville', 'value'),
     Input('slider-seuil', 'value'),
     Input('dd-annee', 'value'),
     Input('g-master', 'clickData')]
)
def update_charts(region_name, ville_label, seuil, annee_dropdown, click_data):
    try:
        if not ville_label:
            return [go.Figure()] * 5 + ["Chargement...", annee_dropdown]

        # Trigger check
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
        annee_choisie = annee_dropdown
        if trigger_id == 'g-master' and click_data:
            annee_choisie = click_data['points'][0]['customdata']

        # Calcul Région
        if ds_poids and region_name in ds_poids.region.values:
            mask = ds_poids['weights'].sel(region=region_name)
            ts_region = (ds['temp_c'] * mask).sum(['lat', 'lon']) / mask.sum(['lat', 'lon'])
        else:
            ts_region = ds['temp_c'].mean(['lat', 'lon'])
        df_region = ts_region.to_dataframe(name='temp').reset_index()

        # Calcul Ville
        # On retrouve lat/lon dans le dataframe global filtré par le label
        ville_row = df_villes[df_villes['label'] == ville_label].iloc[0]
        ts_ville = ds['temp_c'].sel(lat=ville_row['lat'], lon=ville_row['lon'], method='nearest')
        df_ville = ts_ville.to_dataframe(name='temp').reset_index()

        # Agrégations
        df_reg_year = df_region.set_index('time').resample('YE')['temp'].mean()
        df_vil_year = df_ville.set_index('time').resample('YE')['temp'].mean()

        # G1 Compare
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=df_reg_year.index, y=df_reg_year, name=f"Région", line=dict(color='gray', dash='dot')))
        fig_comp.add_trace(go.Scatter(x=df_vil_year.index, y=df_vil_year, name="Ville", line=dict(color='#e74c3c', width=3)))
        fig_comp.update_layout(template="plotly_white", title="Comparaison Annuelle", hovermode="x unified")

        # G2 Master
        ref_mean = df_vil_year['1950':'1980'].mean()
        anomalies = df_vil_year - ref_mean
        colors = ['#e74c3c' if x > 0 else '#3498db' for x in anomalies]
        fig_master = go.Figure(data=[go.Bar(x=anomalies.index.year, y=anomalies, marker_color=colors, customdata=anomalies.index.year)])
        fig_master.update_layout(template="plotly_white", title="Anomalies (Ref 1950-80)", clickmode='event+select', showlegend=False)

        # G3 Detail
        df_detail = df_ville[df_ville['time'].dt.year == annee_choisie]
        fig_detail = px.line(df_detail, x='time', y='temp')
        fig_detail.add_hline(y=seuil, line_dash="dash", line_color="red")
        fig_detail.update_layout(template="plotly_white", margin=dict(t=10))

        # G4 Heatmap
        df_hm = df_ville.copy()
        df_hm['Year'] = df_hm['time'].dt.year
        df_hm['Month'] = df_hm['time'].dt.month
        hm_data = df_hm.groupby(['Year', 'Month'])['temp'].mean().unstack()
        fig_heat = px.imshow(hm_data, labels=dict(x="Mois", y="Année", color="°C"), color_continuous_scale="RdYlBu_r", origin='lower')
        fig_heat.update_layout(template="plotly_white")

        # G5 Simul
        days_over = df_ville[df_ville['temp'] > seuil].set_index('time').resample('YE')['temp'].count()
        days_over = days_over.reindex(df_vil_year.index, fill_value=0)
        fig_sim = px.bar(x=days_over.index.year, y=days_over.values, color=days_over.values, color_continuous_scale="OrRd")
        fig_sim.update_layout(template="plotly_white", title=f"Jours > {seuil}°C", showlegend=False)

        return fig_comp, fig_master, fig_detail, fig_heat, fig_sim, f"🔎 Zoom : {annee_choisie}", annee_choisie

    except Exception as e:
        print(f"Erreur Update: {e}")
        return [go.Figure()] * 5 + ["Erreur", annee_dropdown]

if __name__ == '__main__':
    app.run(debug=True)
import dash
from dash import dcc, html, Input, Output, ctx, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import pandas as pd
from pathlib import Path
import sys
import numpy as np

# =========================================================
# 1. CHARGEMENT ROBUSTE ET PRÉPARATION
# =========================================================
print("🚀 Démarrage de l'application...")

# Gestion des chemins (Code / Data)
base_script = Path(__file__).resolve().parent
dossier_data = base_script.parent / "TP"

print(f"📂 Dossier données visé : {dossier_data}")

# Recherche fichier Météo
fichier_nc = None
noms_possibles = ["meteo_france_1950_2025.nc", "donnees_carte_75ans_journalier.nc"]
for nom in noms_possibles:
    p = dossier_data / nom
    if p.exists():
        fichier_nc = p
        print(f"✅ Météo trouvée : {nom}")
        break

# Recherche fichiers annexes
chemin_villes = dossier_data / "villes_avec_regions.parquet"
chemin_poids = dossier_data / "weights_bool_precise.nc"
if not chemin_poids.exists():
    chemin_poids = dossier_data / "poids_regions_finie.nc"

# Vérification présence
if not fichier_nc or not chemin_villes.exists():
    print(f"❌ ERREUR : Fichiers manquants dans {dossier_data}")
    # On tente dans le dossier courant au cas où
    if Path("villes_avec_regions.parquet").exists():
        chemin_villes = Path("villes_avec_regions.parquet")
        fichier_nc = Path("meteo_france_1950_2025.nc")
        chemin_poids = Path("weights_bool_precise.nc")
        print("⚠️ Utilisation des fichiers du dossier courant (Plan B).")
    else:
        sys.exit("Arrêt : Impossible de trouver les fichiers.")

# Chargement Données
print("⏳ Chargement des données en mémoire...")
df_villes = pd.read_parquet(chemin_villes)

# Nettoyage des noms de régions (gestion des accents/espaces)
df_villes["Region_Assignee"] = df_villes["Region_Assignee"].fillna("Hors Région").astype(str).str.strip()

try:
    ds = xr.open_dataset(fichier_nc)
    # Chargement Poids (Lazy)
    ds_poids = xr.open_dataset(chemin_poids)
except Exception as e:
    sys.exit(f"❌ Erreur lecture NetCDF : {e}")

# Standardisation Noms Variables
if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
if 'latitude' in ds_poids.coords: ds_poids = ds_poids.rename({'latitude': 'lat', 'longitude': 'lon'})

# Gestion Température (Kelvin -> Celsius)
var_temp = 'Temperature_C' if 'Temperature_C' in ds else 't2m'
# Test sur une valeur pour savoir si Kelvin
val_test = ds[var_temp].isel(time=0, lat=int(ds.lat.size/2), lon=int(ds.lon.size/2)).values
if val_test > 200:
    ds['temp_c'] = ds[var_temp] - 273.15
else:
    ds['temp_c'] = ds[var_temp]

# Listes pour Dropdowns
liste_regions = sorted(df_villes["Region_Assignee"].unique())
# On ajoute une option "Toutes les villes" pour retrouver Marseille/Calais si la région est fausse
liste_regions.insert(0, "Toutes les régions")

liste_annees = sorted(list(set(pd.to_datetime(ds.time.values).year)))

print("✅ Application prête.")

# =========================================================
# 2. INTERFACE (LAYOUT)
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Observatoire du Climat Local", className="text-primary mt-4"), width=12),
        dbc.Col(dbc.Alert("Visualisez l'évolution climatique de votre ville (1950-2025)", color="info"), width=12)
    ]),

    dbc.Row([
        # COLONNE PARAMÈTRES
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Paramètres", className="bg-primary text-white fw-bold"),
                dbc.CardBody([
                    html.Label("1. Région :", className="fw-bold"),
                    dcc.Dropdown(id='dd-region', options=[{'label': r, 'value': r} for r in liste_regions],
                                 value="Toutes les régions", clearable=False, className="mb-3"),

                    html.Label("2. Ville :", className="fw-bold"),
                    dcc.Dropdown(id='dd-ville', options=[], value=None, placeholder="Cherchez votre ville...",
                                 clearable=False, searchable=True, className="mb-3"),

                    html.Hr(),
                    html.Label("3. Seuil Canicule :", className="fw-bold text-danger"),
                    dcc.Slider(id='slider-seuil', min=25, max=40, step=1, value=30, marks={i: str(i) for i in range(25, 41, 5)}),

                    html.Hr(),
                    html.Label("4. Année Zoom :", className="fw-bold"),
                    dcc.Dropdown(id='dd-annee', options=[{'label': str(a), 'value': a} for a in liste_annees], value=2003, clearable=False, className="mb-3"),
                ])
            ], className="shadow sticky-top", style={"top": "20px"})
        ], width=12, lg=3),

        # COLONNE VISUALISATIONS
        dbc.Col([
            # KPIs
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Moyenne Annuelle", className="text-muted small fw-bold"), html.H2(id="kpi-mean", className="text-primary fw-bold")])), width=12, md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Record Absolu", className="text-muted small fw-bold"), html.H2(id="kpi-max", className="text-danger fw-bold"), html.Small(id="kpi-max-date", className="text-muted")])), width=12, md=4),
                dbc.Col(dbc.Card(dbc.CardBody([html.H6("Réchauffement (+75 ans)", className="text-muted small fw-bold"), html.H2(id="kpi-delta", className="text-warning fw-bold"), html.Small("Différence 2020-2025 vs 1950-1955", className="text-muted small")])), width=12, md=4),
            ], className="mb-3"),

            # ONGLETS
            dbc.Tabs([
                dbc.Tab(label="Synthèse", tab_id="tab-synthese", children=[
                    dbc.Card([dbc.CardHeader("Anomalies (Warming Stripes)"), dbc.CardBody(dcc.Graph(id='g-master'))], className="mb-4 mt-3 shadow-sm border-0"),
                    dbc.Card([dbc.CardHeader("Comparatif : Ville vs Moyenne Régionale"), dbc.CardBody(dcc.Graph(id='g-compare'))], className="mb-4 shadow-sm border-0"),
                ]),
                dbc.Tab(label="Saisonnalité", tab_id="tab-saisons", children=[
                    dbc.Card([dbc.CardHeader("Évolution par Saison"), dbc.CardBody(dcc.Graph(id='g-saisons'))], className="mb-4 mt-3 shadow-sm border-0"),
                ]),
                dbc.Tab(label="Détails", tab_id="tab-details", children=[
                     dbc.Card([dbc.CardHeader("Heatmap Mensuelle"), dbc.CardBody(dcc.Graph(id='g-heatmap'))], className="shadow-sm border-0 mb-4 mt-3"),
                     dbc.Card([dbc.CardHeader("Zoom Journalier"), dbc.CardBody(dbc.Row([
                         dbc.Col(dcc.Graph(id='g-detail-ref'), width=12, lg=6),
                         dbc.Col(dcc.Graph(id='g-detail-main'), width=12, lg=6)
                     ]))], className="mb-4 shadow-sm border-0"),
                ]),
                dbc.Tab(label="Impacts", tab_id="tab-impacts", children=[
                    dbc.Card([dbc.CardHeader("Jours de Canicule"), dbc.CardBody(dcc.Graph(id='g-simulateur'))], className="shadow-sm border-0 mb-4 mt-3"),
                ]),
            ], id="tabs", active_tab="tab-synthese")
        ], width=12, lg=9)
    ])
], fluid=True, className="bg-light pb-5")


# =========================================================
# 3. CALLBACKS INTELLIGENTS
# =========================================================

# Mise à jour liste des villes (Avec option "Toutes")
@app.callback(
    [Output('dd-ville', 'options'), Output('dd-ville', 'value')],
    [Input('dd-region', 'value')],
    [State('dd-ville', 'value')]
)
def update_cities(region, current):
    if not region:
        return [], None

    if region == "Toutes les régions":
        df_f = df_villes # Pas de filtre
    else:
        df_f = df_villes[df_villes["Region_Assignee"] == region]

    # Tri alphabétique pour faciliter la recherche
    df_f = df_f.sort_values("label")

    opts = [{'label': r['label'], 'value': r['label']} for _, r in df_f.iterrows()]

    # On garde la ville sélectionnée si elle est dans la nouvelle liste, sinon on prend la première
    vals = [o['value'] for o in opts]
    if current in vals:
        val = current
    elif vals:
        val = vals[0]
    else:
        val = None

    return opts, val

# Mise à jour des Graphiques (Coeur du problème)
@app.callback(
   [Output('g-compare', 'figure'), Output('g-master', 'figure'),
    Output('g-detail-ref', 'figure'), Output('g-detail-main', 'figure'),
    Output('g-heatmap', 'figure'), Output('g-simulateur', 'figure'), Output('g-saisons', 'figure'),
    Output('kpi-mean', 'children'), Output('kpi-max', 'children'), Output('kpi-max-date', 'children'), Output('kpi-delta', 'children'),
    Output('dd-annee', 'value')],
   [Input('dd-region', 'value'), Input('dd-ville', 'value'),
    Input('slider-seuil', 'value'), Input('dd-annee', 'value'), Input('g-master', 'clickData')]
)
def update_charts(region, ville, seuil, annee_dd, click_data):
    if not ville: return [go.Figure()] * 7 + ["-", "-", "-", "-", annee_dd]

    annee = click_data['points'][0]['customdata'] if (ctx.triggered_id == 'g-master' and click_data) else annee_dd

    # ----------------------------------------------------
    # 1. EXTRACTION RÉGIONALE (Protection crash)
    # ----------------------------------------------------
    try:
        # On essaie de récupérer la moyenne régionale
        # Attention : le fichier de poids peut avoir 'mask', 'weights' ou 'region'
        if region != "Toutes les régions" and 'mask' in ds_poids and region in ds_poids.coords.get('region', []):
            mask_data = ds_poids['mask'].sel(region=region)
            df_reg = (ds['temp_c'] * mask_data).sum(['lat', 'lon']) / mask_data.sum(['lat', 'lon'])
        elif region != "Toutes les régions" and 'weights' in ds_poids and region in ds_poids.coords.get('region', []):
             mask_data = ds_poids['weights'].sel(region=region)
             df_reg = (ds['temp_c'] * mask_data).sum(['lat', 'lon']) / mask_data.sum(['lat', 'lon'])
        else:
            # Fallback : Si erreur ou "Toutes les régions", on fait une moyenne nationale simple
            # pour éviter que le graphe ne plante
            df_reg = ds['temp_c'].mean(['lat', 'lon'])

        df_reg = df_reg.to_dataframe(name='temp').resample('YE')['temp'].mean()

    except Exception as e:
        print(f"⚠️ Erreur Région ({e}). Passage en moyenne globale.")
        df_reg = ds['temp_c'].mean(['lat', 'lon']).to_dataframe(name='temp').resample('YE')['temp'].mean()

    # ----------------------------------------------------
    # 2. EXTRACTION VILLE (Correction Frontières / Mer)
    # ----------------------------------------------------
    try:
        row = df_villes[df_villes['label'] == ville].iloc[0]
        t_lat, t_lon = row['lat'], row['lon']

        # STRATÉGIE "LARGE": On ne prend pas le point exact (qui peut être NaN).
        # On prend un carré autour de la ville et on moyenne les points valides.

        # Rayon de recherche initial : 0.25 degrés (~25km)
        offset = 0.25
        subset = ds['temp_c'].sel(lat=slice(t_lat - offset, t_lat + offset),
                                  lon=slice(t_lon - offset, t_lon + offset))

        # Si vide (ex: ville côtière ou frontière stricte), on élargit à 100km
        if subset.isnull().all() or subset.mean().isnull():
            offset = 0.8
            subset = ds['temp_c'].sel(lat=slice(t_lat - offset, t_lat + offset),
                                      lon=slice(t_lon - offset, t_lon + offset))

        if subset.isnull().all():
             raise ValueError("Aucune donnée météo trouvée autour de cette coordonnée.")

        # Moyenne des points trouvés (ignore automatiquement les NaNs)
        ts_ville = subset.mean(['lat', 'lon']).to_dataframe(name='temp')
        df_vil_year = ts_ville.resample('YE')['temp'].mean()

    except Exception as e:
        print(f"❌ Erreur Ville {ville}: {e}")
        err_fig = go.Figure().add_annotation(text="Données indisponibles pour cette zone", showarrow=False)
        return [err_fig] * 7 + ["Erreur", "Erreur", "-", "Erreur", annee]

    # ----------------------------------------------------
    # 3. GÉNÉRATION DES GRAPHIQUES (Standard)
    # ----------------------------------------------------

    # G1 Compare
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_reg.index, y=df_reg, name=f"Moyenne Régionale", line=dict(color='gray', dash='dot')))
    fig_c.add_trace(go.Scatter(x=df_vil_year.index, y=df_vil_year, name=ville, line=dict(color='#2c3e50', width=3)))
    fig_c.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="°C", margin=dict(l=40, r=20, t=20, b=40))

    # G2 Warming Stripes
    ano = df_vil_year - df_vil_year['1950':'1980'].mean()
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in ano]
    fig_m = go.Figure(data=[go.Bar(x=ano.index.year, y=ano, marker_color=colors, customdata=ano.index.year)])
    if 2003 in ano.index.year: fig_m.add_annotation(x=2003, y=ano.loc[ano.index.year==2003].iloc[0], text="2003", showarrow=True, arrowhead=2, ay=-30)
    fig_m.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Écart", showlegend=False, margin=dict(l=40, r=20, t=20, b=40))

    # Zoom data
    df_ref = ts_ville[ts_ville.index.year == liste_annees[0]]
    df_choix = ts_ville[ts_ville.index.year == annee]
    min_y = min(df_ref['temp'].min(), df_choix['temp'].min()) - 2 if not df_ref.empty else 0
    max_y = max(df_ref['temp'].max(), df_choix['temp'].max()) + 2 if not df_ref.empty else 30

    # G3 Detail Ref
    fig_ref = px.line(df_ref, x=df_ref.index, y='temp', title=f"Année {liste_annees[0]}")
    fig_ref.update_layout(template="plotly_white", yaxis_range=[min_y, max_y], height=300, margin=dict(l=40, r=20, t=40, b=40))

    # G4 Detail Main
    fig_main = px.line(df_choix, x=df_choix.index, y='temp', title=f"Année {annee}")
    fig_main.add_hline(y=seuil, line_dash="dash", line_color="red")
    fig_main.update_layout(template="plotly_white", yaxis_range=[min_y, max_y], height=300, margin=dict(l=40, r=20, t=40, b=40))

    # G5 Heatmap
    hm = ts_ville.copy()
    hm['Y'], hm['M'] = hm.index.year, hm.index.month
    data_brute = hm.groupby(['Y', 'M'])['temp'].mean().unstack()
    ref_period = hm[hm['Y'].between(1950, 1980)]
    moyennes_mensuelles = ref_period.groupby('M')['temp'].mean()
    data_ecart = data_brute - moyennes_mensuelles.values

    fig_h = px.imshow(data_ecart, labels=dict(x="Mois", y="Année", color="Écart"), color_continuous_scale="RdBu_r", origin='lower', aspect="auto", zmin=-4, zmax=4)
    fig_h.update_layout(template="plotly_white", height=400, margin=dict(l=40, r=20, t=20, b=40))

    # G6 Canicule
    days = ts_ville[ts_ville['temp'] > seuil].resample('YE')['temp'].count().reindex(df_vil_year.index, fill_value=0)
    fig_s = px.bar(x=days.index.year, y=days.values, color=days.values, color_continuous_scale="OrRd")
    fig_s.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Jours", margin=dict(l=40, r=20, t=20, b=40))

    # G7 Saisons
    df_saison = ts_ville.copy()
    saison_map = {12:'Hiver', 1:'Hiver', 2:'Hiver', 3:'Printemps', 4:'Printemps', 5:'Printemps', 6:'Été', 7:'Été', 8:'Été', 9:'Automne', 10:'Automne', 11:'Automne'}
    df_saison['Saison'] = df_saison.index.month.map(saison_map)
    df_saison_yearly = df_saison.groupby([df_saison.index.year, 'Saison'])['temp'].mean().unstack()

    fig_saisons = go.Figure()
    colors_saisons = {'Hiver': '#3498db', 'Printemps': '#2ecc71', 'Été': '#e74c3c', 'Automne': '#e67e22'}
    for s in ['Hiver', 'Printemps', 'Été', 'Automne']:
        if s in df_saison_yearly.columns:
            fig_saisons.add_trace(go.Scatter(x=df_saison_yearly.index, y=df_saison_yearly[s], name=s, mode='lines'))
    fig_saisons.update_layout(template="plotly_white", xaxis_title="Année", margin=dict(l=40, r=20, t=20, b=40))

    # KPIs
    kpi_mean = f"{df_vil_year.mean():.1f}°C"
    kpi_max = f"{ts_ville['temp'].max():.1f}°C"
    kpi_max_date = f"Le {ts_ville['temp'].idxmax().strftime('%d/%m/%Y')}"
    delta = df_vil_year.iloc[-5:].mean() - df_vil_year.iloc[:5].mean()
    kpi_delta = f"+{delta:.1f}°C" if delta > 0 else f"{delta:.1f}°C"

    return fig_c, fig_m, fig_ref, fig_main, fig_h, fig_s, fig_saisons, kpi_mean, kpi_max, kpi_max_date, kpi_delta, annee

if __name__ == '__main__':
    app.run(debug=True)
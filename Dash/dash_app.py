import dash
from dash import dcc, html, Input, Output, ctx, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import pandas as pd
from pathlib import Path
import sys

# =========================================================
# 1. CHARGEMENT
# =========================================================
base = Path(".")
possibles = [base, base / "TP", base.parent / "TP", Path("..") / "TP"]

chemin_villes = None
chemin_poids = None
fichier_nc = None

for p in possibles:
    if (p / "villes_avec_regions.parquet").exists(): chemin_villes = p / "villes_avec_regions.parquet"
    if (p / "poids_regions_finie.nc").exists(): chemin_poids = p / "poids_regions_finie.nc"
    if (p / "donnees_carte_70ans_journalier.nc").exists(): fichier_nc = p / "donnees_carte_70ans_journalier.nc"
    elif (p / "meteo_france_1950_2025.nc").exists(): fichier_nc = p / "meteo_france_1950_2025.nc"

if not chemin_villes or not fichier_nc:
    sys.exit("ERREUR : Fichiers manquants. Lancez d'abord 'generer_mapping_villes.py'.")

# Chargement Données
df_villes = pd.read_parquet(chemin_villes)

try:
    ds = xr.open_dataset(fichier_nc).load()
    ds_poids = xr.open_dataset(chemin_poids).load()
except:
    ds = xr.open_dataset(fichier_nc)
    ds_poids = xr.open_dataset(chemin_poids)

# Renommage
if 'latitude' in ds.coords: ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
if 'latitude' in ds_poids.coords: ds_poids = ds_poids.rename({'latitude': 'lat', 'longitude': 'lon'})

# Température °C
var_temp = 'Temperature_C' if 'Temperature_C' in ds else 't2m'
if ds[var_temp].mean() > 200:
    ds['temp_c'] = ds[var_temp] - 273.15
else:
    ds['temp_c'] = ds[var_temp]

# Listes
liste_regions = sorted(df_villes["Region_Assignee"].dropna().unique().astype(str))
liste_annees = sorted(list(set(pd.to_datetime(ds.time.values).year)))
premiere_annee = liste_annees[0]

# =========================================================
# 2. LAYOUT (INTERFACE COMPLETE)
# =========================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

app.layout = dbc.Container([

    # --- EN-TÊTE ---
    dbc.Row([
        dbc.Col(html.H1("Observatoire du Climat Local (1950-2020)", className="text-primary mt-4"), width=12),
        dbc.Col(dbc.Alert("Comparaison : Température locale vs Moyenne régionale.", color="info"), width=12)
    ]),

    dbc.Row([
        # --- COLONNE DE GAUCHE : PARAMÈTRES ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎛️ Paramètres", className="bg-primary text-white"),
                dbc.CardBody([
                    html.Label("1. Région :", className="fw-bold"),
                    dcc.Dropdown(id='dd-region', options=[{'label': r, 'value': r} for r in liste_regions], value=liste_regions[0], clearable=False, className="mb-3"),

                    html.Label("2. Ville :", className="fw-bold"),
                    dcc.Dropdown(id='dd-ville', options=[], value=None, placeholder="Chargement...", clearable=False, searchable=True, className="mb-3"),

                    html.Hr(),
                    html.Label("3. Seuil Canicule :", className="fw-bold text-danger"),
                    dcc.Slider(id='slider-seuil', min=25, max=40, step=1, value=30, marks={i: str(i) for i in range(25, 41, 5)}),

                    html.Hr(),
                    html.Label("4. Année Zoom :", className="fw-bold"),
                    dcc.Dropdown(id='dd-annee', options=[{'label': str(a), 'value': a} for a in liste_annees], value=2003, clearable=False, className="mb-3"),
                ])
            ], className="shadow sticky-top", style={"top": "20px"})
        ], width=12, lg=3),

        # --- COLONNE DE DROITE : VISUALISATIONS ---
        dbc.Col([

            # 1. LIGNE DES 3 KPIS
            dbc.Row([
                # KPI 1 : Moyenne
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Moyenne (Ville)", className="text-muted small text-uppercase fw-bold"),
                        html.H2(id="kpi-mean", className="text-primary fw-bold"),
                    ])
                ], className="mb-3 text-center shadow-sm border-start border-primary border-4"), width=12, md=4),

                # KPI 2 : Record
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Record Absolu", className="text-muted small text-uppercase fw-bold"),
                        html.H2(id="kpi-max", className="text-danger fw-bold"),
                    ])
                ], className="mb-3 text-center shadow-sm border-start border-danger border-4"), width=12, md=4),

                # KPI 3 : Delta T (Réchauffement)
                dbc.Col(dbc.Card([
                    dbc.CardBody([
                        html.H6("Réchauffement", className="text-muted small text-uppercase fw-bold"),
                        html.H2(id="kpi-delta", className="text-warning fw-bold"),
                        html.Small("Différence 2020 vs 1950", className="text-muted small")
                    ])
                ], className="mb-3 text-center shadow-sm border-start border-warning border-4"), width=12, md=4),
            ]), # <-- Fin de la ligne KPI (N'oublie pas cette virgule !)

            # 2. LES GRAPHIQUES
            dbc.Card([
                dbc.CardHeader("Comparatif : Ville vs Région"),
                dbc.CardBody(dcc.Graph(id='g-compare'))
            ], className="mb-4 shadow-sm border-0"),

            dbc.Card([
    dbc.CardHeader("Comparaison Journalière : 1950 vs Année sélectionnée"),
    dbc.CardBody([
        dbc.Row([
            # Graphique de Gauche : 1950
            dbc.Col([
                html.H6("Année 1950 (Référence)", className="text-center text-muted fw-bold"),
                html.Div(  # <-- AJOUT D'UN DIV AVEC HAUTEUR FIXE
                    dcc.Graph(id='g-detail-ref', config={'displayModeBar': False}),
                    style={'height': '400px'}  # <-- HAUTEUR FIXE DU CONTENEUR
                )
            ], width=12, lg=6),

            # Graphique de Droite : Année Choisie
            dbc.Col([
                html.H6(id="titre-zoom-annee", className="text-center text-primary fw-bold"),
                html.Div(  # <-- MÊME CHOSE ICI
                    dcc.Graph(id='g-detail-main', config={'displayModeBar': False}),
                    style={'height': '400px'}
                )
            ], width=12, lg=6)
        ])
    ])
], className="mb-4 shadow-sm border-0"),
            dbc.Card([
                dbc.CardHeader("Anomalies (Warming Stripes)"),
                dbc.CardBody([
                    dbc.Alert("Rouge = Plus chaud que la normale | Bleu = Plus froid", color="light", style={"fontSize": "0.8rem", "padding": "5px"}),
                    dcc.Graph(id='g-master')
                ])
            ], className="mb-4 shadow-sm border-0"),

            dbc.Card([
                dbc.CardHeader("Heatmap : Évolution des Anomalies Mensuelles"),
                dbc.CardBody(dcc.Graph(id='g-heatmap'))
            ], className="shadow-sm border-0 mb-4"),

            dbc.Card([
                dbc.CardHeader("Fréquence des fortes chaleurs"),
                dbc.CardBody(dcc.Graph(id='g-simulateur'))
            ], className="shadow-sm border-0 mb-4"),

        ], width=12, lg=9) # Fin de la colonne de droite
    ])
], fluid=True, className="bg-light pb-5")

# =========================================================
# 3. CALLBACKS
# =========================================================

# Filtre Villes
@app.callback(
    [Output('dd-ville', 'options'), Output('dd-ville', 'value')],
    [Input('dd-region', 'value')],
    [State('dd-ville', 'value')]
)
def update_cities(region, current):
    if not region: return [], None

    # Filtre Pandas instantané grâce au fichier pré-calculé
    df_f = df_villes[df_villes["Region_Assignee"] == region]
    opts = [{'label': r['label'], 'value': r['label']} for _, r in df_f.iterrows()]

    vals = [o['value'] for o in opts]
    val = current if current in vals else (vals[0] if vals else None)
    return opts, val

# Graphiques
@app.callback(
   [Output('g-compare', 'figure'),
     Output('g-master', 'figure'),
     Output('g-detail-ref', 'figure'),  # Gauche
     Output('g-detail-main', 'figure'), # Droite
     Output('g-heatmap', 'figure'),
     Output('g-simulateur', 'figure'),
     Output('kpi-mean', 'children'),
     Output('kpi-max', 'children'),
     Output('kpi-delta', 'children'),
     Output('dd-annee', 'value'),
     Output('titre-zoom-annee', 'children')], # Le titre dynamique (11ème output)
    [Input('dd-region', 'value'), Input('dd-ville', 'value'),
     Input('slider-seuil', 'value'), Input('dd-annee', 'value'),
     Input('g-master', 'clickData')]
)
def update_charts(region, ville, seuil, annee_dd, click_data):
    if not ville: return [go.Figure()] * 5 + ["-", "-", annee_dd]

    trigger = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
    annee = click_data['points'][0]['customdata'] if (trigger == 'g-master' and click_data) else annee_dd

    # Données
    mask = ds_poids['weights'].sel(region=region)
    df_reg = (ds['temp_c'] * mask).sum(['lat', 'lon']) / mask.sum(['lat', 'lon'])
    df_reg = df_reg.to_dataframe(name='temp').resample('YE')['temp'].mean()

    row = df_villes[df_villes['label'] == ville].iloc[0]
    ts_ville = ds['temp_c'].sel(lat=row['lat'], lon=row['lon'], method='nearest').to_dataframe(name='temp')
    df_vil_year = ts_ville.resample('YE')['temp'].mean()

    kpi_mean = f"{df_vil_year.mean():.1f}°C"
    kpi_max = f"{ts_ville['temp'].max():.1f}°C"

    # On compare la moyenne des 5 premières années vs les 5 dernières pour être robuste
    start_temp = df_vil_year.iloc[:5].mean() # 1950-1955
    end_temp = df_vil_year.iloc[-5:].mean()  # 2016-2020 (ou fin dataset)
    delta = end_temp - start_temp

    # Formatage avec un "+" si positif
    kpi_delta = f"+{delta:.1f}°C" if delta > 0 else f"{delta:.1f}°C"

    # G1 Compare
    fig_c = go.Figure()
    fig_c.add_trace(go.Scatter(x=df_reg.index, y=df_reg, name=f"Moy {region}", line=dict(color='gray', dash='dot')))
    fig_c.add_trace(go.Scatter(x=df_vil_year.index, y=df_vil_year, name="Ville", line=dict(color='#2c3e50', width=3)))
    fig_c.add_annotation(x=df_vil_year.idxmax(), y=df_vil_year.max(), text="Record", showarrow=True, arrowhead=1)
    fig_c.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Temp (°C)", hovermode="x unified")

    # G2 Detail
   # Données 1950 (Référence Fixe)
    df_1950 = ts_ville[ts_ville.index.year == 1950]
    # Données Année Choisie
    df_choix = ts_ville[ts_ville.index.year == annee]

  # G3 : ZOOM COMPARATIF (GAUCHE / DROITE)
    # On récupère la vraie première année dispo (souvent 1950)
    df_ref = ts_ville[ts_ville.index.year == premiere_annee]
    df_choix = ts_ville[ts_ville.index.year == annee]

    # Sécurité pour éviter crash si données vides
    min_y, max_y = 0, 30
    if not df_ref.empty and not df_choix.empty:
        min_y = min(df_ref['temp'].min(), df_choix['temp'].min()) - 2
        max_y = max(df_ref['temp'].max(), df_choix['temp'].max()) + 2

    # Graph Gauche (Référence)
    fig_ref = px.line(df_ref, x=df_ref.index, y='temp')
    fig_ref.update_traces(line_color="#3498db")
    fig_ref.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Température (°C)",
        yaxis_range=[min_y, max_y],
        height=350,
        margin=dict(t=10, r=20, l=40, b=40)
    )
    # Graph Droite (Actuel)
    fig_main = px.line(df_choix, x=df_choix.index, y='temp')
    fig_main.add_hline(y=seuil, line_dash="dash", line_color="red")
    fig_main.update_traces(line_color="#e74c3c")
    fig_main.update_layout(
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Température (°C)",
        yaxis_range=[min_y, max_y],
        height=350,
        margin=dict(t=10, r=20, l=40, b=40)
    )

    ano = df_vil_year - df_vil_year['1950':'1980'].mean()
    colors = ['#e74c3c' if x > 0 else '#3498db' for x in ano]
    fig_m = go.Figure(data=[go.Bar(x=ano.index.year, y=ano, marker_color=colors, customdata=ano.index.year)])
    if 2003 in ano.index.year: fig_m.add_annotation(x=2003, y=ano.loc[ano.index.year==2003].iloc[0], text="2003", showarrow=True, arrowhead=2, ay=-30)
    fig_m.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Écart (°C)", showlegend=False)

   # G4 : HEATMAP (Avec chiffres parlants au survol)
    hm = ts_ville.copy()
    hm['Y'], hm['M'] = hm.index.year, hm.index.month

    # 1. Calculs (Année vs Moyenne 1950-1980)
    data_brute = hm.groupby(['Y', 'M'])['temp'].mean().unstack()
    ref_period = hm[hm['Y'].between(1950, 1980)]
    moyennes_mensuelles = ref_period.groupby('M')['temp'].mean()
    data_ecart = data_brute - moyennes_mensuelles.values

    # 2. Création Graphique
    fig_h = px.imshow(
        data_ecart,
        labels=dict(x="Mois", y="Année", color="Écart"),
        color_continuous_scale="RdBu_r", # Rouge=Chaud, Bleu=Froid
        origin='lower',
        aspect="auto",
        zmin=-4, zmax=4, # Borne l'échelle pour bien voir les contrastes
    )

    # 3. Personnalisation "Parlante"
    # On change le format des mois (1 -> Janvier) pour l'axe X
    fig_h.update_xaxes(
        tickmode='array',
        tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        ticktext=["Janv", "Févr", "Mars", "Avr", "Mai", "Juin", "Juil", "Août", "Sept", "Oct", "Nov", "Déc"]
    )

    fig_h.update_layout(
        template="plotly_white",
        height=500, # HAUTEUR FORCÉE
        coloraxis_colorbar=dict(title="Écart", tickvals=[-4, 0, 4], ticktext=["Froid", "Normal", "Chaud"])
    )

    # Infobulle parlante
    fig_h.update_traces(
        hovertemplate="<b>%{y} - %{x}</b><br>Écart : <b>%{z:+.1f}°C</b><extra></extra>"
    )


    # G5 Simul
    days = ts_ville[ts_ville['temp'] > seuil].resample('YE')['temp'].count().reindex(df_vil_year.index, fill_value=0)
    fig_s = px.bar(x=days.index.year, y=days.values, color=days.values, color_continuous_scale="OrRd")
    fig_s.update_layout(template="plotly_white", xaxis_title="Année", yaxis_title="Jours > Seuil", coloraxis_showscale=False)

    fig_s.update_layout(
        template="plotly_white",
        xaxis_title="Année",
        yaxis_title=f"Jours > {seuil}°C",
        coloraxis_colorbar=dict(title="Jours"),
        coloraxis_showscale=True
    )

    return fig_c, fig_m, fig_ref, fig_main, fig_h, fig_s, kpi_mean, kpi_max, kpi_delta, annee, f"Année {annee}"

if __name__ == '__main__':
    app.run(debug=True)
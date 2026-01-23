import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import pandas as pd
import numpy as np

# --- 1. Chargement et Préparation des Données ---
try:
    # On charge le fichier
    ds = xr.open_dataset("donnees_carte_35ans_journalier.nc")
    
    # Normalisation des noms de variables (lat, lon, time)
    rename_dict = {}
    if 'latitude' in ds.coords: rename_dict['latitude'] = 'lat'
    if 'longitude' in ds.coords: rename_dict['longitude'] = 'lon'
    if rename_dict: ds = ds.rename(rename_dict)

    # Détection de la variable de température
    var_temp = 't2m' if 't2m' in ds.data_vars else list(ds.data_vars)[0]
    
    # Conversion Kelvin -> Celsius si nécessaire
    if ds[var_temp].mean() > 200:
        ds['temp_c'] = ds[var_temp] - 273.15
    else:
        ds['temp_c'] = ds[var_temp]
except Exception as e:
    print(f"Erreur de chargement : {e}")
    # Création de données de secours si le fichier est manquant pour tester le code
    times = pd.date_range("1985-01-01", periods=35*365, freq="D")
    ds = xr.Dataset({"temp_c": (("time", "lat", "lon"), np.random.randn(len(times), 2, 2) + 15)},
                    coords={"time": times, "lat": [48.1, 48.2], "lon": [-1.7, -1.6]})

# --- 2. Mise en page (Layout) ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

app.layout = dbc.Container([
    # Header : Titre "Marc-Style"
    dbc.Row([
        dbc.Col([
            html.H1("📊 Diagnostic Climatique Local (1985-2020)", className="text-primary mt-4 fw-bold"),
            html.P("Outil d'aide à la décision pour le Plan Climat (PCAET)", className="text-muted")
        ], width=12)
    ], className="text-center mb-4"),

    dbc.Row([
        # Sidebar : Paramètres pour Marc
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📍 Configuration Locale", className="bg-primary text-white fw-bold"),
                dbc.CardBody([
                    html.Label("Sélectionner une zone :", className="fw-bold"),
                    dcc.Dropdown(
                        id='select-lat',
                        options=[{'label': f"Secteur {lat}°N", 'value': lat} for lat in ds.lat.values],
                        value=ds.lat.values[0],
                        clearable=False, className="mb-3"
                    ),
                    html.Label("Seuil de forte chaleur (°C) :", className="fw-bold"),
                    dcc.Slider(id='slider-hot', min=25, max=35, value=30, step=1, 
                               marks={i: f'{i}°' for i in range(25, 36, 5)}),
                    html.Div(id='desc-seuil', className="small text-muted mt-3")
                ])
            ], className="shadow border-0")
        ], width=12, lg=3),

        # Dashboard principal
        dbc.Col([
            # 1. Warming Stripes (Le Choc Visuel)
            dbc.Card([
                dbc.CardHeader("1. Écart à la normale : Le 'Mur de chaleur' local"),
                dbc.CardBody([
                    dcc.Graph(id='graph-stripes', config={'displayModeBar': False}),
                    html.P("Chaque barre représente une année. Le passage du bleu au rouge montre la rupture climatique.", className="small text-muted")
                ])
            ], className="mb-4 shadow border-0"),

            # 2 & 3 : Risques et Calendrier
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("2. Évolution des Risques (Gel vs Chaleur)"),
                    dbc.CardBody(dcc.Graph(id='graph-risques'))
                ], className="shadow border-0"), width=12, xl=7),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("3. État des Saisons (Mois/Année)"),
                    dbc.CardBody(dcc.Graph(id='graph-heatmap'))
                ], className="shadow border-0"), width=12, xl=5),
            ], className="mb-4"),

            # 4 & 5 : Distribution et Tendance
            dbc.Row([
                dbc.Col(dbc.Card([
                    dbc.CardHeader("4. Changement de la Normale (Distribution)"),
                    dbc.CardBody(dcc.Graph(id='graph-cloche'))
                ], className="shadow border-0"), width=12, xl=6),
                
                dbc.Col(dbc.Card([
                    dbc.CardHeader("5. Tendance Long Terme (Moyenne Mobile)"),
                    dbc.CardBody(dcc.Graph(id='graph-tendance'))
                ], className="shadow border-0"), width=12, xl=6),
            ])
        ], width=12, lg=9)
    ])
], fluid=True, className="bg-light pb-5")

# --- 3. Logique de Calcul (Callbacks) ---
@app.callback(
    [Output('graph-stripes', 'figure'),
     Output('graph-risques', 'figure'),
     Output('graph-heatmap', 'figure'),
     Output('graph-cloche', 'figure'),
     Output('graph-tendance', 'figure'),
     Output('desc-seuil', 'children')],
    [Input('select-lat', 'value'),
     Input('slider-hot', 'value')]
)
def update_dashboard(lat_val, seuil_hot):
    # Extraction des données pour le point choisi
    # On prend la première longitude disponible pour l'exemple
    df_point = ds.sel(lat=lat_val, lon=ds.lon.values[0], method='nearest').temp_c.to_series()
    
    # --- G1: Warming Stripes ---
    yearly_mean = df_point.groupby(df_point.index.year).mean()
    ref_mean = yearly_mean.iloc[:10].mean() # Référence sur les 10 premières années
    anomalies = yearly_mean - ref_mean
    
    fig_stripes = go.Figure(data=[go.Bar(
        x=anomalies.index, y=[1]*len(anomalies),
        marker=dict(color=anomalies, colorscale='RdBu_r', cmin=-1.5, cmax=1.5),
        showlegend=False
    )])
    fig_stripes.update_layout(height=120, yaxis_visible=False, margin=dict(t=5, b=5, l=0, r=0))

    # --- G2: Risques (Gel vs Chaleur) ---
    days_frost = df_point[df_point < 0].groupby(df_point[df_point < 0].index.year).count()
    days_hot = df_point[df_point > seuil_hot].groupby(df_point[df_point > seuil_hot].index.year).count()
    
    fig_risques = go.Figure()
    fig_risques.add_trace(go.Scatter(x=days_frost.index, y=days_frost, name="Jours de Gel", line=dict(color='#3498db', width=3)))
    fig_risques.add_trace(go.Scatter(x=days_hot.index, y=days_hot, name=f"Chaleur >{seuil_hot}°C", line=dict(color='#e74c3c', width=3)))
    fig_risques.update_layout(title="Indicateurs de gestion technique", hovermode="x unified")

    # --- G3: Heatmap (Saisonnalité) ---
    heatmap_df = df_point.groupby([df_point.index.year, df_point.index.month]).mean().unstack()
    fig_heat = px.imshow(heatmap_df.T, 
                         labels=dict(x="Année", y="Mois", color="Temp"),
                         color_continuous_scale='Turbo')

    # --- G4: Distribution (Cloche) ---
    mid_point = len(yearly_mean) // 2
    period1 = df_point[df_point.index.year <= yearly_mean.index[mid_point]]
    period2 = df_point[df_point.index.year > yearly_mean.index[mid_point]]
    
    fig_cloche = go.Figure()
    fig_cloche.add_trace(go.Histogram(x=period1, name='Début (1985-2000)', marker_color='#3498db', opacity=0.6))
    fig_cloche.add_trace(go.Histogram(x=period2, name='Fin (2001-2020)', marker_color='#e74c3c', opacity=0.6))
    fig_cloche.update_layout(barmode='overlay', title="Fréquence des températures")

    # --- G5: Tendance (Moyenne Mobile) ---
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=yearly_mean.index, y=yearly_mean, name="Brut", line=dict(color='lightgray')))
    fig_trend.add_trace(go.Scatter(x=yearly_mean.index, y=yearly_mean.rolling(5).mean(), name="Moyenne 5 ans", line=dict(color='#c0392b', width=4)))
    fig_trend.update_layout(title="Lissage de la hausse")

    desc = f"Ce seuil impacte le nombre de jours critiques affichés dans le graphique n°2."
    
    return fig_stripes, fig_risques, fig_heat, fig_cloche, fig_trend, desc

if __name__ == '__main__':
    app.run_server(debug=True)

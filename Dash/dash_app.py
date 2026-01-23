import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# --- Chargement des données ---

curr_dir = Path.cwd()
possibles_dirs = [curr_dir / "TP", curr_dir.parent / "TP", curr_dir]
dossier_data = None

noms_possibles = ["poids_regions_finie.nc"]

for p in possibles_dirs:
    for nom in noms_possibles:
        if (p / nom).exists():
            dossier_data = p
            file_poids = p / nom
            break
    if dossier_data:
        break

if not dossier_data:
    sys.exit("Erreur : Fichiers de poids introuvables.")

file_meteo = dossier_data / "donnees_carte_70ans_journalier.nc"
if not file_meteo.exists():
    file_meteo = dossier_data / "meteo_france_70ans_final.nc"

ds_poids = xr.open_dataset(file_poids)
ds_meteo = xr.open_dataset(file_meteo)

for ds in [ds_poids, ds_meteo]:
 rename_poids = {}
if 'latitude' in ds_poids.coords: rename_poids['latitude'] = 'lat'
if 'longitude' in ds_poids.coords: rename_poids['longitude'] = 'lon'
if rename_poids:
    ds_poids = ds_poids.rename(rename_poids)

# Pour ds_meteo
rename_meteo = {}
if 'latitude' in ds_meteo.coords: rename_meteo['latitude'] = 'lat'
if 'longitude' in ds_meteo.coords: rename_meteo['longitude'] = 'lon'
if rename_meteo:
    ds_meteo = ds_meteo.rename(rename_meteo)

var_temp = 'Temperature_C' if 'Temperature_C' in ds_meteo else 't2m'

if ds_meteo[var_temp].mean() > 200:
    ds_meteo['Temperature_C'] = ds_meteo[var_temp] - 273.15
    var_temp = 'Temperature_C'

liste_regions = sorted([str(r) for r in ds_poids.region.values])

# --- Mise en page (Layout) ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H1("Observatoire Climatique Regional", className="text-primary"), width=12),
        dbc.Col(html.P("Exploration interactive des anomalies thermiques (1950-2020)", className="text-muted"), width=12)
    ], className="my-4 text-center"),

    html.Hr(),

    dbc.Row([

        # Sidebar
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Parametres", className="bg-primary text-white"),
                dbc.CardBody([

                    html.Label("Territoire :", className="fw-bold"),
                    dcc.Dropdown(
                        id='dropdown-region',
                        options=[{'label': r, 'value': r} for r in liste_regions],
                        value=liste_regions[0],
                        clearable=False,
                        className="mb-4"
                    ),

                    html.Label("Seuil Forte Chaleur :", className="fw-bold"),
                    html.Div(id='seuil-output-container', className="text-primary fw-bold mb-2"),

                    dcc.Slider(
                        id='slider-seuil',
                        min=25, max=40, step=1,
                        value=30,
                        marks={i: f'{i}°' for i in range(25, 41, 5)},
                    ),
                ])
            ], className="shadow-sm border-0")
        ], width=12, lg=3),

        # Graphiques
        dbc.Col([

            dbc.Card([
                dbc.CardBody(dcc.Graph(id='graph-evolution', style={"height": "400px"}))
            ], className="mb-4 shadow-sm border-0"),

            dbc.Card([
                dbc.CardBody(dcc.Graph(id='graph-seuil', style={"height": "400px"}))
            ], className="shadow-sm border-0")

        ], width=12, lg=9)
    ])

], fluid=True, className="p-4 bg-light")

if __name__ == '__main__':
    # debug=True permet de voir les erreurs dans le navigateur et recharger auto
    app.run(debug=True)
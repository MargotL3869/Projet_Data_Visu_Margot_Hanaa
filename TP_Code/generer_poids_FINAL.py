import xarray as xr
import geopandas as gpd
import regionmask
import numpy as np
from pathlib import Path
import sys
import os

# 1. Config et Nettoyage
os.environ['SHAPE_RESTORE_SHX'] = 'YES'
print("🚀 Démarrage de la génération...")

curr_dir = Path.cwd()
possibles = [curr_dir.parent / "TP", curr_dir / "TP", curr_dir]

dossier_data = None
for p in possibles:
    if (p / "region.shp").exists():
        dossier_data = p
        break

if not dossier_data:
    print("❌ Erreur : region.shp introuvable.")
    sys.exit()

file_nc = dossier_data / "donnees_carte_70ans_journalier.nc"
file_shp = dossier_data / "region.shp"
output_file = dossier_data / "poids_regions_finie.nc"

# Si le fichier vide existe encore, on tente de le supprimer
if output_file.exists():
    try:
        output_file.unlink()
        print("🗑️ Ancien fichier supprimé.")
    except PermissionError:
        print("❌ IMPOSSIBLE de supprimer l'ancien fichier.")
        print("👉 Ferme VS Code et redémarre, ou supprime le fichier manuellement.")
        sys.exit()

if not file_nc.exists():
    file_nc = dossier_data / "meteo_france_70ans_final.nc"

# 2. Chargement
print("📦 Chargement des données...")
ds_grid = xr.open_dataset(file_nc)
rename_dict = {}
if 'latitude' in ds_grid.coords: rename_dict['latitude'] = 'lat'
if 'longitude' in ds_grid.coords: rename_dict['longitude'] = 'lon'
ds_grid = ds_grid.rename(rename_dict)

ds_template = xr.Dataset(coords={"lat": ds_grid.lat, "lon": ds_grid.lon})

gdf = gpd.read_file(file_shp)
if gdf.crs != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

# 3. Calculs
print("⚙️  Calcul des masques et poids...")
col_nom = "nom_offici" # On utilise la bonne colonne
regions_mask = regionmask.Regions(gdf.geometry, names=gdf[col_nom])

mask_data = regions_mask.mask_3D(ds_template.lon.values, ds_template.lat.values)

mask_3d = xr.DataArray(
    mask_data,
    coords={
        "region": regions_mask.names,
        "lat": ds_template.lat.values,
        "lon": ds_template.lon.values
    },
    dims=("region", "lat", "lon")
)

weights = mask_3d.astype(np.float32)
cos_lat = np.cos(np.deg2rad(ds_template.lat))
weights_weighted = weights * cos_lat

# 4. Sauvegarde
print(f"💾 Sauvegarde dans {output_file.name}...")
ds_out = xr.Dataset({
    "weights": weights_weighted,
    "mask_binary": weights
})

ds_out.to_netcdf(output_file)

# 5. Vérification finale
size = output_file.stat().st_size
print(f"✅ Terminé ! Taille du fichier : {size / 1024:.2f} Ko")

if size < 1000:
    print("⚠️  ATTENTION : Le fichier semble vide !")
else:
    print("🎉 Le fichier est valide.")
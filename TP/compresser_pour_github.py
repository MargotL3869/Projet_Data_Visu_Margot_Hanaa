import xarray as xr
import os
import numpy as np
from pathlib import Path

# CONFIGURATION
INPUT_FILE = Path("TP") / "donnees_carte_70ans_journalier.nc"
OUTPUT_FILE = Path("TP") / "meteo_70ans_opti.nc"

def compress_extreme():
    if not INPUT_FILE.exists():
        print(f"❌ Fichier introuvable : {INPUT_FILE}")
        return

    print(f"📦 Taille actuelle : {INPUT_FILE.stat().st_size / 1e6:.2f} Mo")
    print("🚀 Compression extrême (Int16 + Zlib)...")

    with xr.open_dataset(INPUT_FILE) as ds:
        # Détection du nom de la variable
        var_name = 'Temperature_C' if 'Temperature_C' in ds else 't2m'

        # PARAMÈTRES DE COMPRESSION
        # On stocke en entiers (int16) au lieu de flottants (float32)
        # scale_factor=0.01 permet une précision de 0.01°C (largement suffisant)
        encoding = {
            var_name: {
                'dtype': 'int16',       # Convertit en entier court (2 octets)
                'scale_factor': 0.01,   # Précision conservée
                'add_offset': 10.0,     # Centre les valeurs autour de 10°C
                '_FillValue': -32767,   # Valeur pour les données manquantes
                'zlib': True,           # Active la compression zip
                'complevel': 9          # Niveau max de compression
            }
        }

        # Sauvegarde
        ds.to_netcdf(OUTPUT_FILE, encoding=encoding)

    size_new = OUTPUT_FILE.stat().st_size / 1e6
    print(f"✅ Fichier optimisé généré : {OUTPUT_FILE}")
    print(f"📉 Nouvelle taille : {size_new:.2f} Mo")

    if size_new < 100:
        print("🎉 SUCCÈS : Le fichier fait moins de 100 Mo ! Tu peux le pousser sur GitHub.")
    else:
        print("⚠️ Toujours > 100 Mo. Essaie de réduire la période (ex: 50 ans).")

if __name__ == "__main__":
    compress_extreme()
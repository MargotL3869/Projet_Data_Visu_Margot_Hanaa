import xarray as xr
import os
from pathlib import Path

# =========================================================
# 1. CONFIGURATION
# =========================================================
# Tes fichiers actuels
FILE_DATA = Path("TP") / "donnees_carte_70ans_journalier.nc"
FILE_MASK = Path("TP") / "weights_bool_precise.nc"

# Le fichier de sortie
FILE_OUTPUT = Path("TP") / "donnees_carte_70ans_journalier_final.nc"

print("🚀 Démarrage du masquage...")

# Vérifications
if not FILE_DATA.exists():
    # Cas où on lance le script depuis l'intérieur du dossier TP
    FILE_DATA = Path("donnees_carte_70ans_journalier.nc")
    FILE_MASK = Path("weights_bool_precise.nc")
    FILE_OUTPUT = Path("donnees_carte_70ans_journalier_final.nc")

if not FILE_DATA.exists() or not FILE_MASK.exists():
    print(f"❌ Erreur : Fichiers introuvables.\nCherché : {FILE_DATA}\nEt : {FILE_MASK}")
    exit()

try:
    # 1. Chargement
    print("1. Chargement des données et du masque...")
    ds_data = xr.open_dataset(FILE_DATA)
    ds_mask = xr.open_dataset(FILE_MASK)

    # Identification des variables
    var_temp = 'Temperature_C' if 'Temperature_C' in ds_data else 't2m'
    print(f"   Variable météo identifiée : {var_temp}")

    # Le masque s'appelle souvent 'mask' ou 'weights_frac' dans tes fichiers
    # On vérifie ce qui est dispo
    var_mask = 'mask'
    if var_mask not in ds_mask:
        # Si pas trouvé, on prend la première variable du fichier masque
        var_mask = list(ds_mask.data_vars)[0]
    print(f"   Variable masque identifiée : {var_mask}")

    # 2. Application du masque
    print("2. Application du masque (Découpage France)...")
    # On aligne le masque sur les données (au cas où il y a un micro-décalage)
    mask_aligned = ds_mask[var_mask].reindex_like(ds_data, method='nearest')

    # On applique : on garde les valeurs là où le masque est > 0.5 (donc 1)
    # Les autres deviennent NaN (transparent sur la carte)
    ds_masked = ds_data.where(mask_aligned > 0.5)

    # 3. Sauvegarde Compressée
    print(f"3. Sauvegarde optimisée dans {FILE_OUTPUT}...")

    # On réutilise la compression INT16 pour que le fichier reste petit (<100Mo)
    encoding = {
        var_temp: {
            'dtype': 'int16',
            'scale_factor': 0.01,
            'add_offset': 10.0,
            '_FillValue': -32767,
            'zlib': True,
            'complevel': 9
        }
    }

    ds_masked.to_netcdf(FILE_OUTPUT, encoding=encoding)

    taille = os.path.getsize(FILE_OUTPUT) / (1024 * 1024)
    print(f"✅ TERMINÉ ! Fichier créé : {taille:.2f} Mo")

except Exception as e:
    print(f"❌ Erreur : {e}")
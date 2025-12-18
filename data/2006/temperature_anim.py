import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import numpy as np
import glob
import os
import gc
from matplotlib.colors import Normalize, LightSource

# ================= CONFIGURAZIONE =================
simulation_dir = "./"
output_dir = "png_temperature"
os.makedirs(output_dir, exist_ok=True)

morphology_path = "2006_000000000000_Morphology.asc" 
vents_path = "2006_000000000000_Vents.asc"

# lo metto solo per velocizzare lo script altrimenti ci vorrebbero ore
STEP_SKIP = 25 

print("--- Inizio Renderizzazione Semplificata ---")

# 1. Carico MORPHOLOGY (Sfondo)
try:
    with rasterio.open(morphology_path) as src_morph:
        elev_data = src_morph.read(1)
        elev_data = np.where(elev_data == src_morph.nodata, np.nan, elev_data)
        
        # Crea ombreggiatura
        ls = LightSource(azdeg=315, altdeg=45)
        rgb_hillshade = ls.shade(elev_data, cmap=plt.cm.gray, blend_mode='overlay', vert_exag=1.5)
        
        bounds = src_morph.bounds
        morph_transform = src_morph.transform
except Exception as e:
    print(f"ERRORE MORPHOLOGY: {e}")
    exit()

# 2. Carico VENTS (Bocche)
try:
    with rasterio.open(vents_path) as src_vents:
        vents_data = src_vents.read(1)
        vents_masked = np.ma.masked_where(vents_data <= 0, vents_data)
except Exception as e:
    print(f"ERRORE VENTS: {e}")
    exit()

# 3. Lista File
pattern = os.path.join(simulation_dir, "output_2006_*_Temperature.asc")
asc_files = sorted(glob.glob(pattern))
print(f"Trovati {len(asc_files)} file.")

# ================= CICLO =================
count = 0

for i, asc_file in enumerate(asc_files):
    
    if i % STEP_SKIP != 0:
        continue

    filename = os.path.basename(asc_file)
    try:
        step_num = filename.split('_')[2] 
    except:
        step_num = f"{i:06d}"
    
    # qua dico di prendere tutti i file nominati frame_XXXXXX.png
    output_png = os.path.join(output_dir, f"frame_{step_num}.png")

    fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
    
    ax.imshow(rgb_hillshade, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top), origin='upper')

    try:
        with rasterio.open(asc_file) as src_sim:
            sim_data = src_sim.read(1)
            
            sim_masked = np.ma.masked_where(sim_data <= 0, sim_data)

            if not sim_masked.mask.all():
                cmap = plt.get_cmap('autumn')
                norm = Normalize(vmin=1140, vmax=1360)
                
                show(sim_masked, transform=src_sim.transform, ax=ax, cmap=cmap, norm=norm, alpha=1.0)
    except Exception as e:
        print(f"Err {filename}: {e}")

    # VENTS
    show(vents_masked, transform=morph_transform, ax=ax, cmap='gray_r', alpha=1.0)

    # TITOLI
    ax.set_title(f"Simulazione Etna - Step {step_num}")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    
    plt.tight_layout()
    plt.savefig(output_png)
    
    plt.close('all') 
    fig.clf()
    del fig
    
    count += 1
    if count < 5 or count % 20 == 0:
        print(f"Generato frame {count} (File: {filename})")
        gc.collect()

print("Finito.")

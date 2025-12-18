#!/bin/bash
# Nota per il professore: lo script da per scontato che all'interno della cartella corrente siano presenti i file .asc per la lava thickness, per poterli avere è necessario andare
# nel file sciara_fv2.cu e de-commentare la funzione saveSnapshot nel ciclo while, questo
# genererà nella cartella data/2006 tanti file asc per la lava thickness alla fine di ogni iterazione
# successivamente avviare questo script e dovrebbe generarsi il video


source venv/bin/activate
python3 thickness_anim.py
ffmpeg -framerate 60 -pattern_type glob -i 'png_thickness/*.png' -c:v libx264 -pix_fmt yuv420p -crf 23 video_spessore.mp4

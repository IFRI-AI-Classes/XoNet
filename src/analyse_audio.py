import os
import soundfile as sf
import pandas as pd

# ============================================================
# CHEMINS DES DATASETS
# ============================================================
DATASET2 = r"C:\Users\DELL 7290\Desktop\Data\datasets_fongbe\ALFFA_PUBLIC\ASR\FONGBE\data"
DATASET4 = r"C:\Users\DELL 7290\Desktop\Data\datasets_fongbe\pyFongbe\data"

# ============================================================
# FONCTION : Lire le fichier text (transcriptions)
# ============================================================
def lire_transcriptions(chemin_text):
    transcriptions = {}
    with open(chemin_text, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:
                parties = ligne.split(" ", 1)
                if len(parties) == 2:
                    id_audio, texte = parties
                    transcriptions[id_audio] = texte
    return transcriptions

# ============================================================
# FONCTION : Analyser les fichiers audio d'un dossier wav
# ============================================================
def analyser_audio(dossier_wav):
    resultats = []
    for racine, dossiers, fichiers in os.walk(dossier_wav):
        for fichier in fichiers:
            if fichier.endswith(".wav"):
                chemin = os.path.join(racine, fichier)
                try:
                    info = sf.info(chemin)
                    resultats.append({
                        "fichier": fichier,
                        "duree_sec": round(info.duration, 2),
                        "sample_rate": info.samplerate,
                        "canaux": info.channels,
                        "chemin": chemin
                    })
                except Exception as e:
                    resultats.append({
                        "fichier": fichier,
                        "duree_sec": None,
                        "sample_rate": None,
                        "canaux": None,
                        "chemin": chemin
                    })
    return pd.DataFrame(resultats)

# ============================================================
# ANALYSE DATASET 2 (ALFFA)
# ============================================================
print("=" * 60)
print("DATASET 2 - ALFFA")
print("=" * 60)

# Transcriptions
trans2 = lire_transcriptions(os.path.join(DATASET2, "train", "text"))
print(f"\n Nombre de transcriptions (train) : {len(trans2)}")
print(f" Exemple : {list(trans2.items())[0]}")

# Audio
df2 = analyser_audio(os.path.join(DATASET2, "train", "wav"))
print(f"\n Nombre de fichiers audio (train) : {len(df2)}")
if len(df2) > 0:
    print(f" Durée totale : {round(df2['duree_sec'].sum() / 60, 2)} minutes")
    print(f" Durée moyenne : {round(df2['duree_sec'].mean(), 2)} secondes")
    print(f" Sample rate : {df2['sample_rate'].unique()}")
    print(f" Canaux : {df2['canaux'].unique()}")

# ============================================================
# ANALYSE DATASET 4 (pyFongbe)
# ============================================================
print("\n" + "=" * 60)
print("DATASET 4 - pyFongbe")
print("=" * 60)

# Transcriptions
trans4 = lire_transcriptions(os.path.join(DATASET4, "train", "text"))
print(f"\n Nombre de transcriptions (train) : {len(trans4)}")
print(f" Exemple : {list(trans4.items())[0]}")

# Audio
df4 = analyser_audio(os.path.join(DATASET4, "train", "wav"))
print(f"\n Nombre de fichiers audio (train) : {len(df4)}")
if len(df4) > 0:
    print(f" Durée totale : {round(df4['duree_sec'].sum() / 60, 2)} minutes")
    print(f" Durée moyenne : {round(df4['duree_sec'].mean(), 2)} secondes")
    print(f" Sample rate : {df4['sample_rate'].unique()}")
    print(f" Canaux : {df4['canaux'].unique()}")

print("\n Analyse terminée !")
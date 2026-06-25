import os
import soundfile as sf
import numpy as np
import pandas as pd

# Chemins des datasets
DATASET2 = r"C:\Users\DELL 7290\Desktop\Data\datasets_fongbe\ALFFA_PUBLIC\ASR\FONGBE\data"

# Dossier de sortie (dans XoNet)
SORTIE = r"C:\Users\DELL 7290\Desktop\Data\XoNet\data\audio\processed"
# Partie 2 : Supprimer les silences
def supprimer_silences(audio, sample_rate, seuil=0.01):
    """
    Supprime les silences au début et à la fin d'un audio.
    - audio : les données sonores (chiffres)
    - sample_rate : qualité du son (16000 Hz)
    - seuil : niveau minimum pour considérer qu'il y a du son
    """
    # On cherche où commence et où finit le vrai son
    indices = np.where(np.abs(audio) > seuil)[0]
    
    if len(indices) == 0:
        # Si le fichier est complètement silencieux
        return audio
    
    debut = indices[0]
    fin = indices[-1]
    
    # On garde seulement la partie avec du son
    return audio[debut:fin]
# Partie 3 : Lire les transcriptions
def lire_transcriptions(chemin_text):
    """
    Lit le fichier text et retourne un dictionnaire :
    { id_audio : transcription_fon }
    """
    transcriptions = {}
    with open(chemin_text, "r", encoding="utf-8") as f:
        for ligne in f:
            ligne = ligne.strip()
            if ligne:  # ignorer les lignes vides
                parties = ligne.split(" ", 1)
                if len(parties) == 2:
                    id_audio, texte = parties
                    transcriptions[id_audio] = texte
    return transcriptions
# Partie 4 : Nettoyer les audios et créer le CSV propre
def nettoyer_dataset(dossier_data, dossier_sortie):
    """
    Nettoie tous les audios et crée un CSV propre
    """
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(dossier_sortie, exist_ok=True)
    
    # Lire les transcriptions
    chemin_text = os.path.join(dossier_data, "train", "text")
    transcriptions = lire_transcriptions(chemin_text)
    
    resultats = []
    total = len(transcriptions)
    compteur = 0
    
    # Parcourir tous les fichiers audio
    dossier_wav = os.path.join(dossier_data, "train", "wav")
    for racine, dossiers, fichiers in os.walk(dossier_wav):
        for fichier in fichiers:
            if fichier.endswith(".wav"):
                compteur += 1
                id_audio = fichier.replace(".wav", "")
                chemin_audio = os.path.join(racine, fichier)
                
                try:
                    # Lire l'audio
                    audio, sample_rate = sf.read(chemin_audio)
                    
                    # Supprimer les silences
                    audio_propre = supprimer_silences(audio, sample_rate)
                    
                    # Sauvegarder l'audio nettoyé
                    chemin_sortie = os.path.join(dossier_sortie, fichier)
                    sf.write(chemin_sortie, audio_propre, sample_rate)
                    
                    # Récupérer la transcription
                    transcription = transcriptions.get(id_audio, "")
                    
                    # Ajouter au tableau
                    resultats.append({
                        "id": id_audio,
                        "chemin_audio": chemin_sortie,
                        "transcription_fon": transcription,
                        "duree_sec": round(len(audio_propre) / sample_rate, 2)
                    })
                    
                    # Afficher la progression
                    if compteur % 500 == 0:
                        print(f"Progression : {compteur}/{total} fichiers traités...")
                
                except Exception as e:
                    print(f"Erreur sur {fichier} : {e}")
    
    # Créer le CSV final
    df = pd.DataFrame(resultats)
    chemin_csv = os.path.join(dossier_sortie, "dataset_audio_propre.csv")
    df.to_csv(chemin_csv, index=False, encoding="utf-8")
    
    print(f"\nTerminé ! {len(resultats)} fichiers nettoyés.")
    print(f"CSV sauvegardé : {chemin_csv}")
    
    return df
# Partie 5 : Lancer le nettoyage
print("Démarrage du nettoyage...")
print("Cela peut prendre quelques minutes...\n")

df_propre = nettoyer_dataset(DATASET2, SORTIE)

print("\n Aperçu du CSV final :")
print(df_propre.head())
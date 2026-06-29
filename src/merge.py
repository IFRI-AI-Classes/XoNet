import csv

def merge_fongbe_datasets_pure_python():
    # Format : (Nom_du_fichier, index_colonne_fon, index_colonne_fr)
    datasets_config = [
        ("data/BibleFongbeFrench.csv", 0, 1),
        ("data/Fongbe_French_Parallel_Dataset1.csv", 0, 1),
        ("data/French_to_fongbe.csv", 0, 1)
    ]
    
    # Utilisation d'un set (ensemble) pour stocker les paires uniques.
    unique_pairs = set()
    total_lignes_ues = 0

    print("Lecture et fusion des fichiers...")

    for filename, fon_idx, fr_idx in datasets_config:
        try:
            with open(filename, "r", encoding="utf-8") as file:
                # csv.reader gère automatiquement les sauts de ligne et les virgules dans les textes
                reader = csv.reader(file)
                
                # Passer l'en-tête (première ligne)
                header = next(reader, None)
                if header is None:
                    continue
                
                lignes_fichier = 0
                for row in reader:
                    # Sécurité si la ligne est malformée ou vide
                    if len(row) <= max(fon_idx, fr_idx):
                        continue
                    
                    # Nettoyage des espaces blancs au début et à la fin
                    fon_text = row[fon_idx].strip()
                    fr_text = row[fr_idx].strip()
                    
                    # On ignore les lignes vides
                    if not fon_text or not fr_text:
                        continue
                    
                    # Ajouter sous forme de tuple
                    unique_pairs.add((fon_text, fr_text))
                    lignes_fichier += 1
                
                print(f"-> {filename} : {lignes_fichier} lignes lues.")
                total_lignes_ues += lignes_fichier

        except FileNotFoundError:
            print(f" Attention : Fichier '{filename}' introuvable. Ignoré.")

    output_filename = "fongbe_french_corpus_final.csv"
    
    with open(output_filename, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        # Écriture de l'en-tête unique
        writer.writerow(["fon", "fr"])
        
        # Écriture de toutes nos paires uniques
        writer.writerows(unique_pairs)

    print("Done :)")

if __name__ == "__main__":
    merge_fongbe_datasets_pure_python()
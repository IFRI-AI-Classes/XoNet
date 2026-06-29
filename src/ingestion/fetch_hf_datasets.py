import csv
from datasets import load_dataset#J'importe le dataset 3 depuis HuggingFace
dataset3 = load_dataset("Shads229/french-fongbe-corpus")#Je charge le dataset 3 depuis HuggingFace en utilisant la fonction load_dataset.
with open("French_Fongbe_dataset3.csv", "w", encoding = "utf-8", newline = "") as file :#J'ouvre mon fichier csv que je renomme file
    writer = csv.DictWriter(file, fieldnames =["french","fongbe"])#Je crée un objet writer qui va écrire chaque ligne de mon csv comme un dictionnaire avec les clés french et fongbe.
    writer.writeheader()#J'écris la première ligne de mon csv, celle qui contient les titres des colonnes.
    for objet in dataset3["train"] :#Je parcours chaque objet de mon dataset.
        messages = objet["messages"]
        texte_french = messages[0]["content"]#Je récupère chaque texte Français  de chaque objet du dataset.
        cleanFrench = texte_french.replace("Traduire en fon : ","")
        texte_fongbe = messages[1]["content"]#Je récupère chaque texte Fongbé  de chaque objet du dataset.
        clean_translation = {"french" : cleanFrench, "fongbe" : texte_fongbe}#Je crée un dictionnaire qui va contenir chaque texte Français et son texte Fongbé correspondant.
        writer.writerow(clean_translation)#J'ajoute le dictionnaire de traduction de chaque objet de mon dataset au fichier csv.
        
import json 
from datasets import load_dataset#J'importe le dataset 3 depuis HuggingFace
dataset3 = load_dataset("Shads229/french-fongbe-corpus")#Je charge le dataset 3 depuis HuggingFace en utilisant la fonction load_dataset.
with open("French_Fongbe.jsonl", "w", encoding = "utf-8") as file :#J'ouvre mon fichier jsonl que je renomme file
    for objet in dataset3["train"] :#Je parcours chaque objet de mon dataset.
        messages = objet["messages"]
        texte_french = messages[0]["content"]#Je récupère chaque texte Français  de chaque objet du dataset.
        texte_fongbe = messages[1]["content"]#Je récupère chaque texte Fongbé  de chaque objet du dataset.
        clean_translation = {"french" : texte_french, "fongbe" : texte_fongbe}#Je crée un dictionnaire qui va contenir chaque texte Français et son texte Fongbé correspondant.
        ligne_json = json.dumps(clean_translation, ensure_ascii = False)#Je convertis chaque dictionnaire de traduction en une chaîne de caractère ou objet json.
        file.write(ligne_json + "\n")#J'écris chaue objet json dans mon fichier conteneur.
"""
Script for extracting Fongbe and French Bible translations.

@url: https://fongbebenin.com/bible/bible-fongbe.html
@author: Victoria Tovihouande
"""

import csv
import re
import time

import requests
from bs4 import BeautifulSoup


csv_filename = "data/BibleFongbeFrench.csv"
N = 660  # Nombre total de pages à scraper
total_segments = 0

with open(csv_filename, "w", encoding="utf-8", newline="") as file:
     writer = csv.writer(file)
     writer.writerow(["Fongbe", "French"])

print(f"Scraping started...")

# 2. Boucle de scraping avec gestion des erreurs
for i in range(N): 
     url = f"https://fongbeBenin.com/bible/bible-fongbe-{i+1}.html"
     
     try:
          # Ajout d'un timeout de 10 secondes pour éviter que le script ne reste bloqué indéfiniment
          response = requests.get(url, timeout=10)
          response.encoding = "utf-8"
          
          if not response.ok:
               print(f" Page {i+1} ignorée (Code statut : {response.status_code})")
               continue

          print(f"Page {i+1}/{N} récupérée avec succès !")
          soup = BeautifulSoup(response.text, 'html.parser')
          ps = soup.find_all('p')
          
          page_translations = []
          
          for p in ps:
               strong = p.find('strong')
               if strong:
                    try:
                         # Extraction et nettoyage des segments textuels
                         strings_list = list(p.strings)
                         fon = strings_list[0].strip()
                         french = strings_list[1].strip()
                         
                         french_propre = re.sub(r"^\d+\.\d+\s", "", french)
                         
                         page_translations.append([fon, french_propre])
                    except IndexError:
                         # Sécurité au cas où la structure de la balise <p> dévie (ex: pas assez de chaînes de texte)
                         continue
          
          # 3. Sauvegarde immédiate des résultats de la page actuelle
          if page_translations:
               with open(csv_filename, "a", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerows(page_translations)
               total_segments += len(page_translations)
                    
          # Rate limiting
          time.sleep(2)

     except requests.exceptions.RequestException as e:
          # Capture les erreurs réseau (Timeout, Connexion interrompue, etc.)
          print(f" Erreur réseau sur la page {i+1} : {e}")
          time.sleep(2)
          continue
     except KeyboardInterrupt:
          # Permet d'arrêter le script proprement avec Ctrl+C sans corrompre le fichier
          print(f"\n Scraping interrompu à la page {i}.")
          break

print("Scraping terminé...")
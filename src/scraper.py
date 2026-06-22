import requests
import re
import time
import csv
from bs4 import BeautifulSoup
translation =[]#Elle contiendra toutes les traductions propres Fongbe French de toutes les balises p de chaque page html.
for i in range (660):#Il y a 660 pages html à scraper donc je vais faire une boucle for qui va de 0 à 660 pour scraper toutes les pages html. 660 car je compte mettre i + 1 dans l'url pour que ça commence à 1 et pas à 0.
     url = f"https://fongbeBenin.com/bible/bible-fongbe-{i+1}.html"
     response = requests.get(url)
     response.encoding = "utf-8"
     print(response)# Just to check if the request was successful(and it was cause it printed 200)
     if response.ok :
          print('Page html ' + str(i+1) + ' récupérée avec succès !')
          #print(response.text)# Pour voir le code source de la page html concernée
          soup = BeautifulSoup(response.text, 'html.parser')
          #print(soup)#J'aurai à nouveau le code html de la page sauf que maintenant il sera plus lisible et je vais pouvoir le parseret pouvoir chercher des éléments à l'intérieur.
          #p = soup.find('p')
          #print(p.text)# Je vais pouvoir voir le texte à l'intérieur de la balise p que j'ai trouvé.
          ps = soup.find_all('p')
          #print(len(ps))# Je vais pouvoir voir le nombre de balises p que j'ai trouvé dans la page html.
          #[print(str(p) + '\n\n') for p in ps]#
          for p in ps :
               strong = p.find('strong')#Il y a un seul strong à l'intérieur de chaque balise p donc je n'ai pas besoin de findAll pour chercher toutes les balises strong à l'intérieur de chaque balise p.
               if strong :
                    traduction_sale = list(p.strings)
                    #print(traduction_sale)#
                    french = list(p.strings)[1]#Je récupère les traductions en Français de chauie texte Fon qu'il y a dans ma balise p (versets inclus) et je les stocke dans une variable que j'appelle french.
                    french_clean = french.strip()#Je vais nettoyer les traductions en Français que j'ai récupéré en supprimant les espaces au début et à la fin de chaque traduction en Français.
                    french_propre = re.sub(r"^\d+\.\d+\s","",french_clean)#Je vais nettoyer les traductions en Français que j'ai récupéré en supprimant les numéros de versets qui sont au début de chaque traduction en Français.
                    #print(french_propre)#
                    traduction_propre = [list(p.strings)[0], french_propre]#Je vais créer une liste qui va contenir le texte Fon et sa traduction en Français propre (sans les versets et les espaces inutiles).
                    #print(traduction_propre)#
                    translation.append(traduction_propre)#J'ajoute au fur et à mseure que j'ai la traduction propre de chaque texte, cette liste dans la big liste.
          time.sleep(2)#Je vais mettre un temps d'attente de 2 secondes entre chaque requête pour ne pas surcharger le serveur du site web que je scrape.
print(len(translation))
with open("BibleFongbeFrench.csv","w", encoding = "utf-8",newline = "") as file :#Je crée et j'ouvre un fichier csv que j'appelle BibleFongbeFrench et que je renomme file.
     writer = csv.writer(file)#Je crée un objet writer qui va me permettre d'écrire dans mon fichier csv file.
     writer.writerow(["Fongbe", "French"])#J'écris la première ligne de mon fichier csv qui va contenir les titres de mes deux colonnes.(en-tête)
     for traduction_propre in translation :#Je parcours chaque traduction propre de ma big liste de traduction.
          writer.writerow(traduction_propre)#J'écris chaque traduction propre de ma big liste de traduction dans mon fichier csv file.
          

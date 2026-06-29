# Collecte de données

Le corpus utilisé dans ce projet a été constitué à partir de plusieurs sources afin d'obtenir un jeu de données plus riche.

Notre corpus est composé de :
* **Textes en fongbé** récupérés par web scraping sur le site *Fongbé Bénin*, principalement les textes de la Bible ;
* **Parallel text dataset for Neural Machine Translation (French → Fongbe)**, qui fournit des phrases parallèles français–fongbé ;
* **Fon French Daily Dialogues (FFR)**, qui contient des dialogues bilingues français–fongbé destinés aux travaux de traduction automatique.

Après la collecte, les différentes sources ont été fusionnées puis nettoyées (suppression des doublons, normalisation des textes et uniformisation du format) afin d'obtenir un corpus unique utilisable pour l'entraînement et l'évaluation de nos modèles.


## Remerciements

Nous remercions les auteurs et les personnes ayant rendu ces ressources disponibles publiquement.

## Références

```text
[1] Kevin Degila, Godson Kalipe, Jamiil Touré Ali and Momboladji Balogoun.
Parallel text dataset for Neural Machine Translation (French -> Fongbe, French -> Ewe).
Zenodo, 2020.
DOI : https://doi.org/10.5281/zenodo.4266935

[2] Bonaventure Dossou, Fabroni Yoclounon, Ricardo Ahounvlamè and Chris Emezue.
Fon French Daily Dialogues Parallel Data.
Zenodo, 2021.
DOI : https://doi.org/10.5281/zenodo.4432712

[3] Fongbé Bénin.
Bible en fongbé.
https://fongbebenin.com/bible/bible-fongbe.html
Consulté le 29 juin 2026.
```
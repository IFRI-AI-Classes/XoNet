# XoNet API

Backend FastAPI pour l'interface XoNet.  
Expose deux endpoints JSON consommables par `interface/index.html`.

---

## Endpoints

| Méthode | Route        | Description                                      |
|---------|--------------|--------------------------------------------------|
| GET     | `/`          | Health check                                     |
| POST    | `/translate` | Lookup traduction dans `data/corpus_final.csv`   |
| POST    | `/sentiment` | Classification de sentiment via HuggingFace      |

### POST `/translate`

```json
// Requête
{ "text": "a ɖò to ɔ mɛ à", "direction": "fon-fr" }

// Réponse
{ "translation": "comment vas-tu", "found": true }
```

`direction` : `"fon-fr"` (Fongbé → Français) ou `"fr-fon"` (Français → Fongbé)  
`found` : `true` si correspondance exacte dans le corpus, `false` sinon.

---

### POST `/sentiment`

```json
// Requête
{ "text": "Le soleil est très fort." }

// Réponse
{
  "label": "positif",
  "scores": { "positif": 0.78, "neutre": 0.22, "negatif": 0.0 },
  "raw_label": "POSITIVE",
  "raw_score": 0.78
}
```

Modèle : `philschmid/pt-tblard-tf-allocine`  
Seuil : score > 0.70 → positif / négatif ; sinon → neutre.

---

## Lancement

### 1. Installer les dépendances

```bash
pip install fastapi uvicorn[standard] transformers torch safetensors
```

Ou via le projet :

```bash
pip install -e .
```

### 2. Démarrer le serveur

Depuis la racine du projet `XoNet/` :

```bash
uvicorn api.main:app --reload
```

Le serveur écoute sur **http://127.0.0.1:8000**

### 3. Documentation interactive

Ouvrir **http://127.0.0.1:8000/docs** (Swagger UI automatique via FastAPI).

---

## Notes

- Le modèle HuggingFace est téléchargé automatiquement au premier appel `/sentiment` (~400 Mo).
- La traduction est un **lookup exact** dans le corpus (87 000+ paires). Un modèle NMT dédié pourra remplacer cette logique une fois entraîné.
- Le CORS est ouvert (`allow_origins=["*"]`) pour le développement local. Restreindre en production.

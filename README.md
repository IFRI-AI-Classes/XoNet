# XoNet

XoNet est un prototype d'analyse de sentiment pour des phrases en fon/fongbe.

## Entrainement ML v2

Le pipeline v2 conserve le dataset actuel et entraine une nouvelle generation de
modeles sans remplacer les artefacts existants.

Commande locale recommandee :

```bash
python -m src.training.train_sentiment_v2
```

Sorties creees dans `models/` :

- `sentiment_pipeline_v2.joblib` : pipeline complet TF-IDF + modele + seuils.
- `metrics_v2.json` : comparaison validation/test.
- `classification_report_v2.txt` : rapport lisible avec matrice de confusion.

La selection du modele se fait sur `F1 macro` en validation. Le test 15% est
utilise seulement pour l'evaluation finale.

Pour inclure LightGBM/XGBoost :

```bash
python -m src.training.train_sentiment_v2 --include-boosting
```

L'application Flask charge desormais `sentiment_pipeline_v3_tuned.joblib` par
defaut. Si le pipeline v3 est absent, elle retombe automatiquement sur les
anciens fichiers `tfidf.joblib` et `voting_model.joblib`.

Pour forcer explicitement le mode v2 :

```powershell
$env:XONET_PIPELINE_FILENAME="sentiment_pipeline_v3_tuned.joblib"
python app.py
```

Pour forcer explicitement le mode legacy (ancien couple tfidf+voting) :

```powershell
$env:XONET_PIPELINE_FILENAME=""
python app.py
```

Resultat local obtenu avec les modeles rapides :

```text
Best model: SVM lineaire calibre
Test accuracy: 0.7228
Test F1 macro: 0.6659
Existing voting F1 macro: 0.6520
```

## Benchmark ML v3

Une v3 experimentale teste la combinaison TF-IDF caracteres + mots avec un
SVM lineaire calibre, sans modifier le dataset ni remplacer la v2.

Commandes :

```bash
python -m src.training.train_sentiment_v3
python -m src.training.tune_sentiment_v3
```

Meilleur resultat obtenu :

```text
Modele : hybrid_char_word_svm_c0.5
Accuracy test : 0.7539
F1 macro test : 0.7051
```

Les artefacts v3 sont `sentiment_pipeline_v3_tuned.joblib`, `metrics_v3_tuned.json`
et `classification_report_v3_tuned.txt`. L'application reste sur la v2 par
defaut jusqu'a validation explicite de la v3.

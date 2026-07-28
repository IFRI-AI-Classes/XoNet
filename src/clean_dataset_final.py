# =============================================================================
# Script de nettoyage du corpus Fongbé-Français
# =============================================================================
# Ce script nettoie le dataset fongbe_french_corpus_final.csv en appliquant
# les bonnes pratiques de nettoyage pour les données textuelles fongbé.
#
# Opérations de nettoyage APPLIQUÉES :
#   - Uniformisation de l'encodage Unicode (normalisation NFC, UTF-8)
#   - Suppression des espaces inutiles (doublons, espaces en début/fin)
#   - Suppression des caractères parasites
#   - Correction des caractères Fongbé (đ → ɖ, Đ → Ɖ)
#   - Nettoyage de la ponctuation résiduelle (guillemets/parenthèses orphelins)
#   - Découpage des lignes de synonymes en paires individuelles
#   - Filtre sur le ratio de longueur Fon/FR (max/min ≤ 5)
#   - Filtre sur la longueur minimale (≥3 mots par langue)
#   - Suppression des artefacts numériques
#   - Suppression des définitions dictionnaire
#   - Suppression des doublons
#
# Opérations de nettoyage ÉVITÉES (spécifique au Fongbé) :
#   - PAS de suppression des accents
#   - PAS de remplacement de ɛ par e, ɔ par o, ɖ par d
#   - PAS de conversion systématique des lettres accentuées
# =============================================================================

import csv
import unicodedata
import re
import os

# --- Configuration des chemins ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # remonte à la racine
INPUT_FILE = os.path.join(PROJECT_DIR, "data/fongbe_french_corpus_final.csv")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "data/cleaned_corpus.csv")
REJECTED_FILE = os.path.join(PROJECT_DIR, "data/lignes_rejetées.csv")
# --- Paramètres de nettoyage ---
# Caractères Parasites :
#   - Guillemets simples/doubles résiduels inutiles (hors CSV quoting)
#   - Tirets cadratins/ondulants et other dash variants
#   - Caractères de contrôle invisibles (U+0000 à U+001F sauf tab)
#   - Zero-width characters (U+200B, U+200C, U+200D, U+FEFF)
#   - Autres symboles non-textuels courants (bullet, etc.)
PARASITE_PATTERN = re.compile(
    r'[\u200b\u200c\u200d\u2060\ufeff'   # caractères de largeur zéro
    r'\u00ad'                                # soft hyphen
    r'\u2022\u2023\u25e6\u2043'             # puces / bullets
    r'\u2010\u2011\u2012\u2013\u2014\u2015' # tires divers
    r']'
)


def uniformiser_encodage(texte: str) -> str:
    """
    Uniformise l'encodage Unicode du texte en normalisant en forme NFC.

    La normalisation NFC (Canonical Decomposition, followed by Canonical
    Composition) assure que les caractères accentués sont représentés de
    façon cohérente : par exemple, 'é' (U+00E9) reste en une seule unité
    au lieu d'être décomposé en 'e' + accent.

    Cela est essentiel pour le Fongbé où les caractères comme ɛ, ɔ, ɖ
    portent souvent des diacritiques qui ne doivent pas être altérés.
    """
    return unicodedata.normalize('NFC', texte)


def supprimer_caracteres_parasites(texte: str) -> str:
    """
    Supprime les caractères parasites du texte.

    Ceux-ci incluent :
      - Les caractères de contrôle invisibles (hors tabulation et saut de ligne)
      - Les caractères de largeur zéro (zero-width space, joiner, etc.)
      - Les tires non standard (remplacés par un tiret normal si utile)
      - Les puces/bullets résiduelles

    Attention : on conserve la ponctuation normale (virgules, points,
    deux-points, points d'interrogation, etc.) car elle fait partie
    intégrante du texte et de la structure linguistique.
    """
    # Suppression des caractères de contrôle (0x00-0x1F) sauf tab (0x09) et newline (0x0A, 0x0D)
    texte = ''.join(
        c for c in texte
        if (ord(c) > 0x1F or c in ('\t', '\n', '\r'))
    )
    # Suppression des caractères parasites identifiés par le pattern
    texte = PARASITE_PATTERN.sub('', texte)
    return texte


def normaliser_espaces(texte: str) -> str:
    """
    Normalise les espaces dans le texte.

    - Remplace tous les types d'espaces multiples (y compris insécables,
      typographiques) par un simple espace
    - Supprime les espaces en début et fin de texte
    - Réduit les suites de plusieurs espaces à un seul espace simple
    """
    # Remplacement de toutes les variantes d'espaces par un espace simple
    # Espaces Unicode : U+0020 (space), U+00A0 (nbsp), U+2000-U+200A,
    # U+202F (narrow nbsp), U+205F (med math space), U+3000 (ideographic space)
    texte = re.sub(r'[\u0020\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]+', ' ', texte)
    # Suppression des espaces en début et fin
    texte = texte.strip()
    # Réduction des doubles espaces multiples
    texte = re.sub(r' {2,}', ' ', texte)
    return texte


def corriger_caracteres_fongbe(texte: str) -> str:
    """
    Corrige les caractères incorrects utilisés à la place des vrais
    caractères Fongbé.

    Remplacement effectué :
      - đ (U+0111, Latin Small Letter D with Stroke) → ɖ (U+0256, Latin
        Small Letter D with Retroflex Hook)
      - Đ (U+0110, Latin Capital Letter D with Stroke) → Ɖ (U+0189,
        Latin Capital Letter D with Retroflex Hook)

    Pourquoi :
      Le caractère 'đ' (d barré) est couramment utilisé par erreur pour
      représenter le /ɖ/ fongbé, car il est plus accessible sur les
      claviers (disponible en vietnamien, croate, etc.). Or, le vrai
      caractère Fongbé est ɖ (U+0256), un D rétroflexe de l'alphabet
      phonétique international (API).
    """
    # Remplacement minuscule : đ → ɖ
    texte = texte.replace('đ', 'ɖ')
    # Remplacement majuscule : Đ → Ɖ
    texte = texte.replace('Đ', 'Ɖ')
    return texte


def nettoyer_ponctuation(texte: str) -> str:
    """
    Supprime les guillemets et parenthèses orphelins du texte.

    Cas traités :
      - Guillemets doubles non assortis (", ") → supprimés s'ils n'ont pas de pair
      - Guillemets courbes non assortis (", ") → supprimés s'ils n'ont pas de pair
      - Parenthèses/crochets/accolades non assortis → supprimés si orphelins
      - Guillemets français « » non assortis → supprimés si orphelins

    Attention : les apostrophes ' (U+2019) ne sont PAS touchées car elles
    servent de signe de contraction en français (j'ai, l'homme, qu'ils, etc.)
    et ne sont pas des guillemets.
    """
    # Paires de ponctuation à vérifier : (ouvrant, fermant)
    # NOTE : on n'inclut PAS les apostrophes \u2018/\u2019 car en français
    # le ' (U+2019) est une apostrophe de contraction, pas un guillemet.
    paires = [
        ('"', '"'),               # guillemets doubles droits
        ('\u201c', '\u201d'),     # guillemets courbes "" 
        ('(', ')'),               # parenthèses
        ('[', ']'),               # crochets
        ('{', '}'),               # accolades
        ('\u00ab', '\u00bb'),     # guillemets français « »
    ]

    for ouvrant, fermant in paires:
        nb_ouvrant = texte.count(ouvrant)
        nb_fermant = texte.count(fermant)

        if nb_ouvrant > nb_fermant:
            # Plus d'ouvrants que de fermants → supprimer les excédents
            # On retire depuis la fin pour ne pas casser les paires valides
            texte = texte[::-1].replace(ouvrant[::-1], '', nb_ouvrant - nb_fermant)[::-1]
        elif nb_fermant > nb_ouvrant:
            # Plus de fermants que d'ouvrants → supprimer les excédents
            texte = texte.replace(fermant, '', nb_fermant - nb_ouvrant)

    return texte


def nettoyer_ligne(texte: str) -> str:
    """
    Applique le pipeline complet de nettoyage à un texte unique.

    Ordre d'opération :
      1. Normalisation Unicode (NFC)
      2. Suppression des caractères parasites
      3. Correction des caractères Fongbé (đ → ɖ, Đ → Ɖ)
      4. Nettoyage de la ponctuation résiduelle
      5. Normalisation des espaces
    """
    texte = uniformiser_encodage(texte)
    texte = supprimer_caracteres_parasites(texte)
    texte = corriger_caracteres_fongbe(texte)
    texte = nettoyer_ponctuation(texte)
    texte = normaliser_espaces(texte)
    return texte


def decouper_synonymes(lignes: list[dict]) -> list[dict]:
    """
    Découpe les lignes contenant des synonymes Fongbé en paires individuelles.

    Certaines lignes du corpus sont de type dictionnaire : le champ Fongbé
    contient plusieurs synonymes séparés par ' - ' (tiret entouré d'espaces),
    associés à un seul mot ou expression en français.

    Exemple avant découpage :
      Fon : "bε̆ adăn - bε̆ azɔ̀n - j'azɔ̀n"
      Fr  : "tomber malade"

    Après découpage, cela devient 3 paires individuelles :
      ("bε̆ adăn",     "tomber malade")
      ("bε̆ azɔ̀n",     "tomber malade")
      ("j'azɔ̀n",      "tomber malade")

    Cela permet au modèle d'apprendre chaque variante lexicale
    individuellement, tout en laissant la déduplication éliminer
    les éventuels chevauchements avec le reste du corpus.
    """
    resultats = []
    nb_decoupees = 0

    for ligne in lignes:
        fon = ligne['fon']

        # Séparation uniquement sur ' - ' (tiret avec espace des deux côtés)
        # pour ne pas casser les mots Fongbé contenant un tiret simple
        parties = [p.strip() for p in fon.split(' - ') if p.strip()]

        if len(parties) > 1:
            # La ligne contient des synonymes → on découpe
            nb_decoupees += 1
            for synonime in parties:
                resultats.append({
                    'fon': synonime,
                    'fr': ligne['fr'],
                    'ligne_brute': ligne['ligne_brute'],
                })
        else:
            # Pas de tiret ou un seul terme → on garde tel quel
            resultats.append(ligne)

    return resultats, nb_decoupees


def valider_ligne(ligne: dict) -> str:
    """
    Vérifie qu'une ligne contient du contenu valide après nettoyage.

    Retourne None si la ligne est valide, sinon une chaîne décrivant
    la raison du rejet :
      - "vide"        : un des champs est vide après nettoyage
      - "artefact"    : le texte français ne contient que des chiffres/
                        ponctuation (artefact d'indexation, ex: '28.18')
      - "trop_court"  : l'un des textes contient moins de 3 mots
      - "ratio"       : le ratio de longueur max/min entre Fon et FR
                        dépasse 5 (paire mal alignée)
      - "definition"  : le texte français est une définition ou explication
                        longue (≥10 mots) pour un terme Fongbé court (≤4 mots)
    """
    fon = ligne['fon']
    fr = ligne['fr']

    # Rejet si un des champs est vide après nettoyage
    if not fon or not fr:
        return "vide"

    # Rejet si le texte français ne contient que des chiffres/points/virgules
    if re.match(r'^[\d\s.,;:]+$', fr):
        return "artefact"

    # Calcul du nombre de mots de chaque côté
    nb_fon = len(fon.split())
    nb_fr = len(fr.split())

    # Rejet si l'un des textes est trop court (< 3 mots)
    # Un texte trop court n'apporte pas assez de contexte pour l'apprentissage
    if nb_fon < 3 or nb_fr < 3:
        return "trop_court"

    # Rejet si le ratio de longueur est trop élevé (> 5)
    # Cela détecte les paires mal alignées où une langue est beaucoup plus
    # longue que l'autre (ex: une phrase complète vs un seul mot)
    ratio = max(nb_fon, nb_fr) / min(nb_fon, nb_fr)
    if ratio > 5:
        return "ratio"

    # Rejet si le texte est une définition lexicale plutôt qu'une traduction :
    # Fon court (≤4 mots) associé à un FR très long (≥10 mots)
    if nb_fon <= 4 and nb_fr >= 10:
        return "definition"

    return None


def main():
    """
    Fonction principale : orchestre le nettoyage complet du corpus.

    Étapes :
      1. Lecture du fichier CSV d'entrée
      2. Nettoyage de chaque ligne (encodage, caractères, espaces)
      3. Validation des lignes (suppression des lignes invalides)
      4. Suppression des doublons
      5. Écriture du fichier nettoyé
      6. Écriture du fichier des lignes rejetées
    """
    print(f"Lecture du fichier d'entrée : {INPUT_FILE}")

    lignes = []

    # --- Étape 1 : Lecture du CSV ---
    # On utilise csv.reader (pas DictReader) pour accéder à reader.line_num
    # qui donne le vrai numéro de ligne brute du fichier, y compris pour
    # les enregistrements multi-lignes (champs guillemets contenant \n).
    with open(INPUT_FILE, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # saute l'en-tête (ligne 1)
        for row in reader:
            if len(row) < 2:
              continue
            ligne = {'fon': row[0], 'fr': row[1]}
            ligne['ligne_brute'] = reader.line_num
            lignes.append(ligne)

    print(f"  -> {len(lignes)} lignes lues (avant nettoyage)")

    # --- Étape 2 : Nettoyage de chaque ligne ---
    for ligne in lignes:
        ligne['fon'] = nettoyer_ligne(ligne['fon'])
        ligne['fr'] = nettoyer_ligne(ligne['fr'])

    # --- Étape 3 : Découpage des lignes de synonymes ---
    # Avant validation, on découpe les lignes de type dictionnaire
    # (plusieurs termes Fon séparés par ' - ' → une paire individuelle par terme)
    lignes, nb_synonymes = decouper_synonymes(lignes)
    print(f"  -> {nb_synonymes} lignes de synonymes découpées en paires individuelles")

    # --- Étape 4 : Validation (suppression des lignes invalides) ---
    lignes_valides = []
    lignes_rejetees = []

    for ligne in lignes:
        raison = valider_ligne(ligne)
        if raison is None:
            lignes_valides.append(ligne)
        else:
            ligne['raison'] = raison
            lignes_rejetees.append(ligne)

    nb_invalides = len(lignes_rejetees)
    print(f"  -> {nb_invalides} lignes invalides supprimées (champs vides ou artefacts)")

    # --- Étape 5 : Suppression des doublons ---
    vus = set()
    lignes_finales = []
    for ligne in lignes_valides:
        cle = (ligne['fon'], ligne['fr'])
        if cle not in vus:
            vus.add(cle)
            lignes_finales.append(ligne)
        else:
            ligne['raison'] = "doublon"
            lignes_rejetees.append(ligne)

    nb_doublons = len(lignes_valides) - len(lignes_finales)
    print(f"  -> {nb_doublons} doublons supprimés")

    # --- Étape 6 : Écriture du fichier nettoyé ---
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fon', 'fr'])
        writer.writeheader()
        for ligne in lignes_finales:
            ligne.pop('ligne_brute', None)
            writer.writerow(ligne)

    print(f"  -> {len(lignes_finales)} lignes finales écrites dans : {OUTPUT_FILE}")

    # --- Étape 7 : Écriture du fichier des lignes rejetées ---
    with open(REJECTED_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ligne_brute', 'fon', 'fr', 'raison'])
        writer.writeheader()
        writer.writerows(lignes_rejetees)

    print(f"  -> {len(lignes_rejetees)} lignes rejetées écrites dans : {REJECTED_FILE}")
    print("Nettoyage terminé avec succès !")


if __name__ == '__main__':
    main()

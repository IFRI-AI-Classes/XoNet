with open("data/cleaned_corpus.csv", "r", encoding="utf-8") as f:
    for i, ligne in enumerate(f, start=1):
        mots = ligne.split()
        for mot in mots:
            if "đ" in mot:
                print(i, mot)
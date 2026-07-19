import pandas as pd

# Chargement du corpus original
df = pd.read_csv(r'd:\XoNet\data\corpus_final.csv', sep='|')

# Garder uniquement fon, fr et sentiment_final converti en int
df_clean = df[['fon', 'fr', 'sentiment_final']].copy()
df_clean['sentiment_final'] = df_clean['sentiment_final'].astype(int)

# Sauvegarder le dataset propre
df_clean.to_csv(r'd:\XoNet\data\corpus_clean.csv', sep='|', index=False)

print('Dataset propre sauvegardé !')
print('Shape:', df_clean.shape)
print('Types:', df_clean.dtypes.to_dict())
print('Valeurs uniques sentiment_final:', sorted(df_clean['sentiment_final'].unique()))

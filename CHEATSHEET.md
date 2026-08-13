# Data Science Piscine — Cheatsheet (Module 3 : The Present)

Notes de théorie + patterns Python, mis à jour au fur et à mesure des exos.

---

## Stats descriptives

### Moyenne (mean)
Somme des valeurs / nombre de valeurs.

```python
mean = df["col"].sum() / len(df["col"])
# ou directement : df["col"].mean()
```

### Variance
Moyenne des écarts au carré par rapport à la moyenne. Mesure la dispersion des données.

- **Population** (tu as TOUTES les données) : diviser par `n`
- **Échantillon** (tu as un sous-ensemble, tu estimes la variance de la population) : diviser par `n - 1` (correction de Bessel, compense le biais d'un échantillon)

```python
# Population
variance = sum((x - mean) ** 2 for x in df["col"]) / len(df["col"])

# Échantillon
variance_sample = sum((x - mean) ** 2 for x in sample) / (len(sample) - 1)
```

`pandas.std()` / `.var()` utilisent `n-1` par défaut (`ddof=1`).

### Écart-type (std)
Racine carrée de la variance. Remet l'unité de mesure d'origine (la variance est en unité², le std est en unité).

```python
import math
std = math.sqrt(variance)
```

### Loi normale (distribution gaussienne)
Formule de densité de probabilité :

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \, e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

- μ = mean, σ = std

```python
def normal_pdf(x, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-((x - mean) ** 2) / (2 * std ** 2))
```

**Concept clé** : plus un échantillon est petit, plus sa courbe estimée s'écarte de la vraie distribution (théorème central limite / loi des grands nombres). Vérifiable en faisant varier la taille d'échantillon et en comparant les courbes.

### Corrélation de Pearson
Mesure la force et le sens d'une relation linéaire entre deux variables, entre -1 et 1.
- proche de 1 : corrélation positive forte
- proche de -1 : corrélation négative forte
- proche de 0 : pas de relation linéaire

```python
df.corr()["target_col"].sort_values(ascending=False)
```

### Standardisation vs Normalisation
- **Standardisation (Z-score)** : centre autour de 0, écart-type 1. Ne borne pas les valeurs. Utile quand les données suivent (à peu près) une loi normale, ou pour les modèles sensibles à l'échelle (SVM, régression logistique, PCA).
  ```python
  z = (x - mean) / std
  ```
- **Normalisation (Min-Max)** : ramène toutes les valeurs entre 0 et 1. Sensible aux outliers (un seul point extrême écrase l'échelle). Utile pour les réseaux de neurones, ou quand tu veux une borne stricte.
  ```python
  x_norm = (x - min) / (max - min)
  ```

### Train / Validation / Test split
- **Train** : entraîne le modèle
- **Validation** : ajuste les hyperparamètres, évite l'overfitting sur le test
- **Test** : évaluation finale, jamais vu par le modèle avant

Split classique : 70-80% train, 20-30% test (puis re-split train en train/validation si besoin).

```python
from sklearn.model_selection import train_test_split
train, val = train_test_split(df, test_size=0.2, random_state=42)
```

---

## Python / Pandas / Matplotlib patterns

### Charger un CSV relatif au script (peu importe d'où on lance)
```python
from pathlib import Path
p = Path(__file__).parents[1]
df = pd.read_csv(p / "fichier.csv")
```

### Vérifier les types de colonnes (repérer la target catégorielle)
```python
print(df.dtypes)
# object = string (catégoriel), float64/int64 = numérique
```

### Sélectionner uniquement les colonnes numériques (exclure la target)
```python
features = df.select_dtypes(include="number").columns
```

### Grille de subplots (Figure vs Axes)
- **Figure** = la fenêtre entière
- **Axes** = chaque sous-graphe individuel dans la grille

```python
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(20, 24))
axes = axes.flatten()  # tableau 2D -> 1D pour boucler simplement

for i, col in enumerate(features):
    sns.histplot(df[col], ax=axes[i])
    axes[i].set_title(col)

plt.tight_layout()  # évite le chevauchement des titres/labels
plt.show()
```

### Superposer deux distributions (comparer Train vs Test)
```python
sns.histplot(data=df_test[col], ax=ax, label="Test", stat="density")
sns.histplot(data=df_train[col], ax=ax, label="Train", stat="density")
ax.legend()
```
`stat="density"` obligatoire si les deux datasets ont des tailles différentes (sinon comparaison en comptage brut faussée).

### Tracer une fonction mathématique (ex: loi normale) sur un histogramme
```python
import numpy as np
x_values = np.linspace(df["col"].min(), df["col"].max(), 200)  # axe X lisse, indépendant des données réelles
y_values = [normal_pdf(x, mean, std) for x in x_values]
plt.plot(x_values, y_values, color="red", label="Loi normale")
```

### sns vs plt
Seaborn est construit sur matplotlib, pas un remplaçant. sns = contenu statistique du plot (histplot, hue, etc), plt = structure (figure, grille, show, tight_layout).

---

## Pièges rencontrés
- `import matplotlib as plt` ≠ `import matplotlib.pyplot as plt` — le premier n'a pas `.hist()`/`.show()`
- `enumerate(liste)` renvoie `(index, valeur)` — pas 2 valeurs indépendantes si on unpack sur une seule liste
- Attention à la priorité des opérateurs dans les formules mathématiques traduites en Python (parenthéser explicitement chaque terme du numérateur/dénominateur)
- Passer l'élément de boucle (`x1`) et pas la liste entière (`x1_values`) à une fonction censée recevoir un scalaire

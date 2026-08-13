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

### Corrélation de Pearson, réimplémentée à la main
Formule :

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \cdot \sum (y_i - \bar{y})^2}}$$

```python
def correlation_factor(x_val, y_val):
    x_mean = x_val.mean()
    y_mean = y_val.mean()
    covariance = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_val, y_val))
    sum_sq_x = sum((x - x_mean) ** 2 for x in x_val)
    sum_sq_y = sum((y - y_mean) ** 2 for y in y_val)
    return covariance / math.sqrt(sum_sq_x * sum_sq_y)
```

Le `/ len(...)` de la covariance s'annule avec celui du dénominateur — pas la peine de diviser.

### Encoder une colonne catégorielle en numérique (binaire)
Nécessaire pour corréler une target textuelle (ex: "Jedi"/"Sith") avec des features numériques.

```python
y_val = (df["knight"] == "Jedi").astype(int)  # Jedi -> 1, Sith -> 0
```

Piège : le sens de l'encodage (quelle classe = 1) inverse le **signe** de toute la corrélation, pas la magnitude. Si les valeurs matchent en grandeur mais sont inversées en signe par rapport à une référence, c'est l'encodage à vérifier en premier.

Note : pour des corrélations proches de 0 (features peu discriminantes), le signe peut varier légèrement même avec le bon encodage — c'est du bruit statistique, pas un bug. La magnitude compte, pas le signe à ce niveau-là.

---

## Python — lecture "de l'extérieur vers l'intérieur"

Contrairement au C (lecture séquentielle gauche→droite), beaucoup de fonctions Python embarquent d'autres fonctions. Lire en 3 couches : quelle fonction englobante ? sur quoi elle agit ? avec quelles règles ?

### Lambda — fonction anonyme en une ligne
```python
lambda x: x * 2
# équivaut à :
def double(x):
    return x * 2
```
Limite : une seule expression, pas plusieurs instructions (pas de `def` multi-lignes en lambda).

### `sorted(iterable, key=..., reverse=...)`
`key` reçoit une fonction qui extrait, pour chaque élément, la valeur à utiliser pour comparer (au lieu de comparer l'élément brut).

```python
results = {"Empowered": 0.79, "Push": -0.02}
for col, r in sorted(results.items(), key=lambda item: item[1], reverse=True):
    print(f"{col}: {r}")
```
`results.items()` → séquence de tuples `(clé, valeur)`. `key=lambda item: item[1]` dit "trie en te basant sur le 2e élément du tuple (la valeur), pas le tuple entier".

### Higher-order function
Une fonction qui prend une autre fonction en paramètre (ex: `plot_func` passée à `plot_grid`, ou `key=` dans `sorted`). Permet de factoriser une structure commune (boucle, grille) en injectant seulement ce qui varie.

### Closure
Une fonction définie à l'intérieur d'une autre a accès aux variables de la fonction englobante, sans qu'on les lui passe en paramètre explicitement.

```python
def main():
    df_train = ...
    def plot_overlay(col, ax):  # utilise df_train sans le recevoir en paramètre
        sns.histplot(data=df_train[col], ax=ax)
```

### Generator expression dans `sum`/`max`/`sorted`
```python
sum(x for x in liste if x > 3)      # pas de [] : pas de liste intermédiaire construite en mémoire
sum([x for x in liste if x > 3])    # fonctionne aussi mais moins économe
```

---

## Pièges rencontrés
- `import matplotlib as plt` ≠ `import matplotlib.pyplot as plt` — le premier n'a pas `.hist()`/`.show()`
- `enumerate(liste)` renvoie `(index, valeur)` — pas 2 valeurs indépendantes si on unpack sur une seule liste
- Attention à la priorité des opérateurs dans les formules mathématiques traduites en Python (parenthéser explicitement chaque terme du numérateur/dénominateur)
- Passer l'élément de boucle (`x1`) et pas la liste entière (`x1_values`) à une fonction censée recevoir un scalaire
- `sys.path.append(...)` doit être exécuté **avant** tout `import` qui en dépend (l'ordre des lignes compte)
- `Path` vs `str` : une fonction qui appelle `.exists()` sur son argument attend un `Path`, pas une string brute — convertir avec `Path(arg)` en entrée si la fonction doit accepter les deux
- `\t` (tabulation) a une largeur variable selon le mot précédent — pour aligner des colonnes, utiliser le formatage `{var:<15}` (largeur fixe) plutôt que `\t`

---

## Mémo commandes

```bash
# Setup env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Jupyter
pip install jupyter                # si pas déjà dans requirements.txt
jupyter notebook                   # ouvre le file browser
jupyter notebook training.ipynb    # ouvre directement un fichier

```
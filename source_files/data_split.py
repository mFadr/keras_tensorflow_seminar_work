import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Načtení dat ze souboru
# Soubor obsahuje větu a label oddělené tabulátorem
data = pd.read_csv(
    "amazon_cells_labelled.txt",
    sep="\t",
    header=None,
    names=["veta", "label"]
)

# Oddělení vstupních textů a cílových tříd
X = data["veta"].values
y = data["label"].values

# Pevné nastavení generátoru náhodných čísel pro reprodukovatelnost
RANDOM_STATE = 42

# První rozdělení: trénovací množina (70 %) versus dočasná množina (30 %)
# Parametr stratify=y zajistí zachování poměru tříd v obou množinách
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

# Druhé rozdělení: dočasná množina se dělí napůl
# Vznikne validační množina (15 % původku) a testovací množina (15 % původku)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

# Kontrolní výpis velikostí a rozložení tříd
print(f"Trénovací množina: {len(X_train)} vět, "
      f"pozitivních: {y_train.sum()}, negativních: {len(y_train) - y_train.sum()}")
print(f"Validační množina: {len(X_val)} vět, "
      f"pozitivních: {y_val.sum()}, negativních: {len(y_val) - y_val.sum()}")
print(f"Testovací množina: {len(X_test)} vět, "
      f"pozitivních: {y_test.sum()}, negativních: {len(y_test) - y_test.sum()}")
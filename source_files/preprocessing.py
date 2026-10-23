"""
Předzpracování textových dat pro klasifikaci sentiment
Dataset: amazon_cells_labelled.txt
Cíl: Příprava dat pro trénování neuronové sítě
"""

import numpy as np
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================================
# 1. NAČTENÍ DAT
# ==============================================================================

print("="*70)
print("KROK 1: Načítání datasetu")
print("="*70)

# Načtení souboru s oddělovačem tabulátor
data = pd.read_csv(
    "amazon_cells_labelled.txt",
    sep="\t",
    header=None,
    names=["veta", "label"]
)

print(f"Celkem vět v datasetu: {len(data)}")
print(f"Pozitivních vět (label=1): {(data['label'] == 1).sum()}")
print(f"Negativních vět (label=0): {(data['label'] == 0).sum()}")
print(f"\nPříklady vět:")
print(data.head(3))

# Oddělení textu a cílové proměnné
X = data["veta"].values
y = data["label"].values

# ==============================================================================
# 2. ROZDĚLENÍ DAT NA TRÉNOVACÍ, VALIDAČNÍ A TESTOVACÍ MNOŽINU
# ==============================================================================

print("\n" + "="*70)
print("KROK 2: Rozdělení dat (70/15/15)")
print("="*70)

RANDOM_STATE = 42

# První rozdělení: trénovací (70 %) vs dočasná (30 %)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

# Druhé rozdělení: validační a testovací (po 15 %)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Trénovací: {len(X_train)} vět (pozitivních: {y_train.sum()}, "
      f"negativních: {len(y_train) - y_train.sum()})")
print(f"Validační: {len(X_val)} vět (pozitivních: {y_val.sum()}, "
      f"negativních: {len(y_val) - y_val.sum()})")
print(f"Testovací:  {len(X_test)} vět (pozitivních: {y_test.sum()}, "
      f"negativních: {len(y_test) - y_test.sum()})")

# ==============================================================================
# 3. PŘEDZPRACOVÁNÍ TEXTU - ČIŠTĚNÍ
# ==============================================================================

print("\n" + "="*70)
print("KROK 3: Čištění textu")
print("="*70)

def cistit_text(text):
    """
    Provádí základní čištění textu:
    - Převod na malá písmena
    - Odebrání diakritiky (ponechání základních znaků)
    - Odebrání speciálních znaků a číslic (ponechání slov)
    - Odebrání více mezer za sebou
    """
    # Převod na malá písmena
    text = text.lower()

    # Odebrání URL adresy
    text = re.sub(r'http\S+|www\S+|<.*?>', '', text)

    # Odebrání speciálních znaků, ponechání pouze písmen, číslic a mezer
    # České znaky jsou zachovány (zahrnují se v \w se správným kódováním)
    text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)

    # Odebrání číslic
    text = re.sub(r'\d+', '', text)

    # Odebrání více mezer za sebou
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# Aplikace čištění na všechny věty
print("Čištění trénovací množiny...")
X_train_cleaned = np.array([cistit_text(veta) for veta in X_train])

print("Čištění validační množiny...")
X_val_cleaned = np.array([cistit_text(veta) for veta in X_val])

print("Čištění testovací množiny...")
X_test_cleaned = np.array([cistit_text(veta) for veta in X_test])

print("\nPříklad čištění:")
print(f"Původní:  {X_train[0]}")
print(f"Vyčištěno: {X_train_cleaned[0]}")

# Kontrola odebraných prázdných řetězců
print(f"\nPrázdné věty po čištění:")
print(f"Trénovací: {(X_train_cleaned == '').sum()}")
print(f"Validační: {(X_val_cleaned == '').sum()}")
print(f"Testovací: {(X_test_cleaned == '').sum()}")

# ==============================================================================
# 4. TOKENIZACE - VYTVOŘENÍ SLOVNÍKU
# ==============================================================================

print("\n" + "="*70)
print("KROK 4: Tokenizace a vytvoření slovníku")
print("="*70)

# Parametry tokenizéru
# num_words: maximální počet nejčastějších slov v slovníku
# oov_token: token pro neznámá slova (mimo slovník)
VOCAB_SIZE = 5000
OOV_TOKEN = "<OOV>"

# Vytvoření tokenizéru pouze na základě trénovací množiny
# Tímto se zamezí úniku informace z validační/testovací množiny
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train_cleaned)

# Získání skutečného počtu unikátních slov (může být menší než VOCAB_SIZE)
slovnik_velikost = len(tokenizer.word_index) + 1
print(f"Velikost slovníku: {slovnik_velikost} slov")
print(f"(počet unikátních slov + 1 pro OOV token)")

# Převod textu na sekvence čísel (word indices)
print("\nPřevod textu na sekvence indexů...")
X_train_seq = tokenizer.texts_to_sequences(X_train_cleaned)
X_val_seq = tokenizer.texts_to_sequences(X_val_cleaned)
X_test_seq = tokenizer.texts_to_sequences(X_test_cleaned)

print(f"Příklad sekvence (první věta trénovací množiny):")
print(f"Text: {X_train_cleaned[0]}")
print(f"Sekvence: {X_train_seq[0]}")

# Analýza délek sekvencí
def analyze_sequence_lengths(sequences, name):
    """Analýza délek sekvencí"""
    lengths = [len(seq) for seq in sequences]
    print(f"\n{name}:")
    print(f"  Minimální délka: {min(lengths)}")
    print(f"  Maximální délka: {max(lengths)}")
    print(f"  Průměrná délka: {np.mean(lengths):.2f}")
    print(f"  Medián: {np.median(lengths):.1f}")
    return lengths

train_lengths = analyze_sequence_lengths(X_train_seq, "Trénovací množina")
val_lengths = analyze_sequence_lengths(X_val_seq, "Validační množina")
test_lengths = analyze_sequence_lengths(X_test_seq, "Testovací množina")

# Výpočet optimální délky pro pad_sequences
# Vybereme délku tak, aby pokryla cca 95 % nejdelších vět bez příliš velkého paddingu
percentil_95 = np.percentile(train_lengths, 95)
MAX_LENGTH = int(np.ceil(percentil_95))
print(f"\nZvolená maximální délka sekvence: {MAX_LENGTH} slov")
print(f"(odpovídá 95. percentilu trénovací množiny)")

# ==============================================================================
# 5. PADDING - VYROVNÁNÍ DÉLEK SEKVENCÍ
# ==============================================================================

print("\n" + "="*70)
print("KROK 5: Padding sekvencí na jednotnou délku")
print("="*70)

# pad_sequences zarovnává sekvence na stejnou délku
# pre: padding na začátek (doporučeno pro RNN/LSTM)
# post: padding na konec (doporučeno pro CNN)
# Volíme "pre" pro textovou klasifikaci s neuronovou sítí
X_train_padded = pad_sequences(
    X_train_seq,
    maxlen=MAX_LENGTH,
    padding='pre',
    truncating='pre'
)

X_val_padded = pad_sequences(
    X_val_seq,
    maxlen=MAX_LENGTH,
    padding='pre',
    truncating='pre'
)

X_test_padded = pad_sequences(
    X_test_seq,
    maxlen=MAX_LENGTH,
    padding='pre',
    truncating='pre'
)

print(f"Tvar po paddingu:")
print(f"Trénovací: {X_train_padded.shape}")
print(f"Validační: {X_val_padded.shape}")
print(f"Testovací: {X_test_padded.shape}")

print(f"\nPříklad padované sekvence:")
print(f"Původní délka: {len(X_train_seq[0])}")
print(f"Padovaná délka: {len(X_train_padded[0])}")
print(f"Padovaná sekvence: {X_train_padded[0]}")

# ==============================================================================
# 6. KONTROLA DAT A ULOŽENÍ PŘEDZPRACOVANÝCH DAT
# ==============================================================================

print("\n" + "="*70)
print("KROK 6: Uložení předzpracovaných dat")
print("="*70)

# Vytvoření slovníku s veškerými informacemi
data_dict = {
    # Předzpracovaná data - padované sekvence
    'X_train': X_train_padded,
    'X_val': X_val_padded,
    'X_test': X_test_padded,

    # Cílové proměnné
    'y_train': y_train,
    'y_val': y_val,
    'y_test': y_test,

    # Originální vyčištěné texty (pro reference)
    'X_train_text': X_train_cleaned,
    'X_val_text': X_val_cleaned,
    'X_test_text': X_test_cleaned,

    # Parametry předzpracování
    'vocab_size': slovnik_velikost,
    'max_length': MAX_LENGTH,
    'oov_token': OOV_TOKEN,
}

# Uložení jako NumPy soubor (efektivní, obsahuje všechny numpy arraye)
np.savez_compressed(
    'preprocessed_data.npz',
    **{k: v for k, v in data_dict.items() if isinstance(v, np.ndarray)}
)

# Uložení tokenizéru a parametrů jako pickle
preprocessing_params = {
    'tokenizer': tokenizer,
    'vocab_size': slovnik_velikost,
    'max_length': MAX_LENGTH,
    'oov_token': OOV_TOKEN,
}

with open('preprocessing_params.pkl', 'wb') as f:
    pickle.dump(preprocessing_params, f)

print("Uloženo:")
print("  - preprocessed_data.npz (padované sekvence a cílové proměnné)")
print("  - preprocessing_params.pkl (tokenizér a parametry)")

# ==============================================================================
# 7. STATISTIKA PŘEDZPRACOVÁNÍ
# ==============================================================================

print("\n" + "="*70)
print("KROK 7: Finální statistika předzpracování")
print("="*70)

print(f"\nVelikost slovníku: {slovnik_velikost} slov")
print(f"Maximální délka sekvence: {MAX_LENGTH} indexů")

print(f"\nRozdělení dat:")
print(f"  Trénovací: {len(X_train_padded)} vět → tvar {X_train_padded.shape}")
print(f"  Validační: {len(X_val_padded)} vět → tvar {X_val_padded.shape}")
print(f"  Testovací: {len(X_test_padded)} vět → tvar {X_test_padded.shape}")

print(f"\nRozdělení tříd v trénovací množině:")
print(f"  Negativní (0): {(y_train == 0).sum()} ({100*(y_train == 0).sum()/len(y_train):.1f}%)")
print(f"  Pozitivní (1): {(y_train == 1).sum()} ({100*(y_train == 1).sum()/len(y_train):.1f}%)")

print(f"\nProcentuelní pokrytí délek sekvencí v trénovací množině:")
for perc in [50, 75, 90, 95, 99]:
    p_val = np.percentile(train_lengths, perc)
    covered = (np.array(train_lengths) <= MAX_LENGTH).sum() / len(train_lengths) * 100
    print(f"  {perc}. percentil: {p_val:.0f} slov")
print(f"  Pokryto zvolenou délkou {MAX_LENGTH}: {covered:.1f}%")

print("\n" + "="*70)
print("Předzpracování dat hotovo!")
print("="*70)

# ==============================================================================
# 8. FUNKCE PRO NABÍJENÍ DAT V TRÉNOVACÍCH SKRIPTECH
# ==============================================================================

def load_preprocessed_data():
    """
    Funkce pro nabíjení předzpracovaných dat v trénovacích skriptech

    Returns:
        dict: Slovník s klíči 'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'
    """
    # Nabíjení padovaných dat
    loaded = np.load('preprocessed_data.npz')

    # Nabíjení tokenizéru a parametrů
    with open('preprocessing_params.pkl', 'rb') as f:
        params = pickle.load(f)

    return {
        'X_train': loaded['X_train'],
        'X_val': loaded['X_val'],
        'X_test': loaded['X_test'],
        'y_train': loaded['y_train'],
        'y_val': loaded['y_val'],
        'y_test': loaded['y_test'],
        'tokenizer': params['tokenizer'],
        'vocab_size': params['vocab_size'],
        'max_length': params['max_length'],
    }

print("\nFunkce load_preprocessed_data() je připravena k importu.")
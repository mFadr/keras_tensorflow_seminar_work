"""
Demonstrace načítání a inspekce předzpracovaných dat
"""

import numpy as np
import pickle

# ==============================================================================
# NAČÍTÁNÍ PŘEDZPRACOVANÝCH DAT
# ==============================================================================

print("="*70)
print("NAČÍTÁNÍ PŘEDZPRACOVANÝCH DAT")
print("="*70)

# Nabíjení padovaných sekvencí a cílových proměnných
loaded = np.load('preprocessed_data.npz')

X_train = loaded['X_train']
X_val = loaded['X_val']
X_test = loaded['X_test']
y_train = loaded['y_train']
y_val = loaded['y_val']
y_test = loaded['y_test']

print(f"\nNabíjeno z preprocessed_data.npz:")
print(f"  X_train: {X_train.shape} (dtype: {X_train.dtype})")
print(f"  X_val:   {X_val.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  y_val:   {y_val.shape}")
print(f"  y_test:  {y_test.shape}")

# Nabíjení parametrů předzpracování
with open('preprocessing_params.pkl', 'rb') as f:
    params = pickle.load(f)

tokenizer = params['tokenizer']
vocab_size = params['vocab_size']
max_length = params['max_length']
oov_token = params['oov_token']

print(f"\nNabíjeno z preprocessing_params.pkl:")
print(f"  Velikost slovníku: {vocab_size}")
print(f"  Maximální délka: {max_length}")
print(f"  OOV token: {oov_token}")

# ==============================================================================
# INSPEKCE DAT
# ==============================================================================

print("\n" + "="*70)
print("INSPEKCE DAT")
print("="*70)

# Inspekce prvního příkladu
print(f"\nPrvní příklad z trénovací množiny:")
print(f"  Padovaná sekvence: {X_train[0]}")
print(f"  Cílová třída: {y_train[0]} (0=negativní, 1=pozitivní)")
print(f"  Nenulové indexy: {X_train[0][X_train[0] > 0]}")

# Dekódování sekvence zpět na slova
word_index = tokenizer.word_index
index_word = {v: k for k, v in word_index.items()}
index_word[0] = '<pad>'
index_word[1] = oov_token

decoded_text = ' '.join([index_word.get(idx, '?') for idx in X_train[0]])
print(f"  Dekódovaný text: {decoded_text}")

# ==============================================================================
# STATISTIKA CÍLOVÝCH TŘÍD
# ==============================================================================

print(f"\n" + "="*70)
print("STATISTIKA CÍLOVÝCH TŘÍD")
print("="*70)

for name, y in [('Trénovací', y_train), ('Validační', y_val), ('Testovací', y_test)]:
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    pct_pos = 100 * n_pos / len(y)
    print(f"\n{name}:")
    print(f"  Negativní (0): {n_neg} ({100*n_neg/len(y):.1f}%)")
    print(f"  Pozitivní (1): {n_pos} ({pct_pos:.1f}%)")

# ==============================================================================
# OVĚŘENÍ INTEGRITY DAT
# ==============================================================================

print(f"\n" + "="*70)
print("OVĚŘENÍ INTEGRITY DAT")
print("="*70)

# Kontrola, že všechny sekvence mají stejnou délku
assert X_train.shape[1] == max_length, "Chyba: trénovací data nemají správnou délku!"
assert X_val.shape[1] == max_length, "Chyba: validační data nemají správnou délku!"
assert X_test.shape[1] == max_length, "Chyba: testovací data nemají správnou délku!"

# Kontrola, že všechny prvky jsou v rozsahu [0, vocab_size)
assert X_train.max() < vocab_size, "Chyba: trénovací data obsahují vysoké indexy!"
assert X_val.max() < vocab_size, "Chyba: validační data obsahují vysoké indexy!"
assert X_test.max() < vocab_size, "Chyba: testovací data obsahují vysoké indexy!"

# Kontrola, že cílové proměnné jsou binární
assert set(y_train) <= {0, 1}, "Chyba: trénovací cíle nejsou binární!"
assert set(y_val) <= {0, 1}, "Chyba: validační cíle nejsou binární!"
assert set(y_test) <= {0, 1}, "Chyba: testovací cíle nejsou binární!"

print("\n✓ Všechny ověřovací testy prošly!")
print("  Data jsou připravena pro trénování neuronové sítě.")

print("\n" + "="*70)
print("INSTRUKCE PRO POUŽITÍ V TRÉNOVACÍM SKRIPTU")
print("="*70)

print("""
V trénovacím skriptu použijte:

    import numpy as np
    import pickle
    
    # Nabíjení dat
    loaded = np.load('preprocessed_data.npz')
    X_train = loaded['X_train']
    X_val = loaded['X_val']
    X_test = loaded['X_test']
    y_train = loaded['y_train']
    y_val = loaded['y_val']
    y_test = loaded['y_test']
    
    # Nabíjení parametrů
    with open('preprocessing_params.pkl', 'rb') as f:
        params = pickle.load(f)
    
    vocab_size = params['vocab_size']
    max_length = params['max_length']
    
    # Použití v modelu
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        # ... dalších vrstev ...
    ])
    
    # Trénování
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
""")

print("="*70)
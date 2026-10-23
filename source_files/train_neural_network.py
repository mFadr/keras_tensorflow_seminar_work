"""
Klasifikace sentimentu v textu pomocí neuronových sítí
Datová sada: Amazon recenze (amazon_cells_labelled.txt)
Cíl: Trénování a vyhodnocení různých architektur neuronových sítí
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# TensorFlow a Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Pevné nastavení seedu (Python, NumPy, TensorFlow) pro reprodukovatelnost výsledků
RANDOM_STATE = 42
keras.utils.set_random_seed(RANDOM_STATE)

# Nastavení vzhledu grafů
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# ==============================================================================
# 1. NAČÍTÁNÍ PŘEDZPRACOVANÝCH DAT
# ==============================================================================

print("="*80)
print("KLASIFIKACE SENTIMENTU POMOCÍ NEURONOVÝCH SÍTÍ")
print("="*80)

print("\n[1/5] Načítání předzpracovaných dat...")

# Načtení padovaných sekvencí
loaded = np.load('preprocessed_data.npz')
X_train = loaded['X_train']
X_val = loaded['X_val']
X_test = loaded['X_test']
y_train = loaded['y_train']
y_val = loaded['y_val']
y_test = loaded['y_test']

# Načtení parametrů
with open('preprocessing_params.pkl', 'rb') as f:
    params = pickle.load(f)

vocab_size = params['vocab_size']
max_length = params['max_length']

print(f"✓ Data načtena: X_train {X_train.shape}, X_val {X_val.shape}, X_test {X_test.shape}")
print(f"✓ Parametry: vocab_size={vocab_size}, max_length={max_length}")

# ==============================================================================
# 2. DEFINICE MODELŮ
# ==============================================================================

print("\n[2/5] Příprava modelů...")

# Pomocná funkce pro vytváření modelů
def build_dense_model(layers_config, embedding_dim=64, dropout_rate=0.3, optimizer='adam'):
    """
    Vytvoří vícevrstvou husitou neuronovou síť
    
    Args:
        layers_config: seznam počtů neuronů v jednotlivých vrstvách
        embedding_dim: rozměr embedding vektoru
        dropout_rate: rate pro dropout regularizaci
        optimizer: optimalizátor trénování
    """
    model = Sequential()
    
    # Embedding vrstva
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    
    # Flattening pro přechod na husité vrstvy
    model.add(Flatten())
    
    # Skryté vrstvy
    for i, units in enumerate(layers_config[:-1]):
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(dropout_rate))
    
    # Výstupní vrstva
    model.add(Dense(layers_config[-1], activation='sigmoid'))
    
    # Kompilace
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_cnn_model(conv_filters, embedding_dim=64, dropout_rate=0.3, optimizer='adam'):
    """
    Vytvoří konvoluční neuronovou síť
    
    Args:
        conv_filters: seznam počtů filtrů v konvolučních vrstvách
        embedding_dim: rozměr embedding vektoru
        dropout_rate: rate pro dropout regularizaci
        optimizer: optimalizátor trénování
    """
    model = Sequential()
    
    # Embedding vrstva
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    
    # Konvoluční vrstvy
    for filters in conv_filters:
        model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
    
    # Flattening
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    
    # Husité vrstvy
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # Kompilace
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def build_lstm_model(lstm_units=64, embedding_dim=64, dropout_rate=0.3, optimizer='adam'):
    """
    Vytvoří LSTM neuronovou síť
    
    Args:
        lstm_units: počet jednotek v LSTM vrstvě
        embedding_dim: rozměr embedding vektoru
        dropout_rate: rate pro dropout regularizaci
        optimizer: optimalizátor trénování
    """
    model = Sequential()
    
    # Embedding vrstva
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    
    # LSTM vrstva
    model.add(LSTM(lstm_units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    
    # Husité vrstvy
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # Kompilace
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ==============================================================================
# 3. DEFINICE EXPERIMENTÁLNÍCH KONFIGURACÍ
# ==============================================================================

print("Definice experimentů...")

experiments = {
    # Experiment 1: Jednoduchá dvouvrstvá síť
    "Dense_2L_32_16": {
        "type": "dense",
        "config": [32, 16, 1],
        "embedding_dim": 32,
        "dropout": 0.2,
        "optimizer": "adam",
        "epochs": 15,
        "batch_size": 32
    },
    
    # Experiment 2: Hlubší čtyřvrstvá síť
    "Dense_4L_128_64_32": {
        "type": "dense",
        "config": [128, 64, 32, 1],
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "adam",
        "epochs": 20,
        "batch_size": 32
    },
    
    # Experiment 3: Síť s vyšší regularizací
    "Dense_3L_64_high_dropout": {
        "type": "dense",
        "config": [64, 32, 1],
        "embedding_dim": 64,
        "dropout": 0.5,
        "optimizer": "adam",
        "epochs": 20,
        "batch_size": 32
    },
    
    # Experiment 4: Síť se SGD optimalizátorem
    "Dense_2L_SGD": {
        "type": "dense",
        "config": [64, 1],
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "sgd",
        "epochs": 25,
        "batch_size": 16
    },
    
    # Experiment 5: Jednoduchá CNN
    "CNN_simple": {
        "type": "cnn",
        "conv_filters": [32, 64],
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "adam",
        "epochs": 15,
        "batch_size": 32
    },
    
    # Experiment 6: Hlubší CNN
    "CNN_deep": {
        "type": "cnn",
        "conv_filters": [64, 64, 32],
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "adam",
        "epochs": 20,
        "batch_size": 32
    },
    
    # Experiment 7: LSTM síť
    "LSTM_64": {
        "type": "lstm",
        "lstm_units": 64,
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "adam",
        "epochs": 15,
        "batch_size": 32
    },
    
    # Experiment 8: LSTM s RMSprop
    "LSTM_RMSprop": {
        "type": "lstm",
        "lstm_units": 64,
        "embedding_dim": 64,
        "dropout": 0.3,
        "optimizer": "rmsprop",
        "epochs": 15,
        "batch_size": 32
    }
}

print(f"✓ Definováno {len(experiments)} experimentů")

# ==============================================================================
# 4. TRÉNOVÁNÍ MODELŮ
# ==============================================================================

print("\n[3/5] Trénování modelů...")

results = {}
histories = {}

for exp_name, config in experiments.items():
    print(f"\n  Experiment: {exp_name}")
    
    # Vytvoření modelu
    if config["type"] == "dense":
        model = build_dense_model(
            layers_config=config["config"],
            embedding_dim=config["embedding_dim"],
            dropout_rate=config["dropout"],
            optimizer=config["optimizer"]
        )
    elif config["type"] == "cnn":
        model = build_cnn_model(
            conv_filters=config["conv_filters"],
            embedding_dim=config["embedding_dim"],
            dropout_rate=config["dropout"],
            optimizer=config["optimizer"]
        )
    else:  # lstm
        model = build_lstm_model(
            lstm_units=config["lstm_units"],
            embedding_dim=config["embedding_dim"],
            dropout_rate=config["dropout"],
            optimizer=config["optimizer"]
        )
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Trénování
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stop],
        verbose=0
    )
    
    # Vyhodnocení na testovací sadě
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metriky
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    final_val_loss = history.history['val_loss'][-1]
    final_train_loss = history.history['loss'][-1]
    
    results[exp_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "val_loss": final_val_loss,
        "train_loss": final_train_loss,
        "epochs_trained": len(history.history['loss']),
        "model": model,
        "y_pred": y_pred
    }
    
    histories[exp_name] = history
    
    print(f"    ✓ Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"    ✓ Val Loss: {final_val_loss:.4f}, Train Loss: {final_train_loss:.4f}")

# ==============================================================================
# 5. VYHODNOCENÍ A VIZUALIZACE
# ==============================================================================

print("\n[4/5] Vyhodnocení výsledků...")

# Seřazení experimentů podle F1 skóre
sorted_results = sorted(results.items(), key=lambda x: x[1]["f1"], reverse=True)

print("\nPořadí modelů podle F1 skóre:")
for rank, (name, metrics) in enumerate(sorted_results, 1):
    print(f"  {rank}. {name:25} - F1: {metrics['f1']:.4f}, Acc: {metrics['accuracy']:.4f}")

# Uložení výsledků
# Ukládají se pouze metriky a predikce (bez objektu modelu), aby byl soubor malý
# a šel načíst i bez TensorFlow (např. v generate_report.py)
with open('training_results.pkl', 'wb') as f:
    pickle.dump({name: {k: v for k, v in m.items() if k != 'model'}
                 for name, m in results.items()}, f)

# Ukládá se pouze slovník history.history (průběh loss/accuracy po epochách)
with open('training_histories.pkl', 'wb') as f:
    pickle.dump({name: h.history for name, h in histories.items()}, f)

# ==============================================================================
# 6. GENEROVÁNÍ GRAFŮ
# ==============================================================================

print("\n[5/5] Generování grafů...")

# Graf 1: Porovnání metrik všech modelů
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Porovnání metrik všech modelů na testovací sadě', fontsize=14, fontweight='bold')

metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
titles = ['Přesnost', 'F1 skóre', 'Preciznost', 'Úplnost']

for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
    ax = axes[idx // 2, idx % 2]
    
    names = [name for name, _ in sorted_results]
    values = [metrics[metric] for _, metrics in sorted_results]
    
    colors = ['green' if v == max(values) else 'steelblue' for v in values]
    ax.barh(names, values, color=colors)
    ax.set_xlabel(title)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(values):
        ax.text(v + 0.02, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('metric_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Uloženo: metric_comparison.png")
plt.close()

# Graf 2: Křivky trénování pro vybrané modely
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Křivky trénování vybraných modelů', fontsize=14, fontweight='bold')

top_models = [name for name, _ in sorted_results[:4]]

for idx, model_name in enumerate(top_models):
    ax = axes[idx // 2, idx % 2]
    history = histories[model_name]
    
    ax.plot(history.history['loss'], label='Trénovací ztráta', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validační ztráta', linewidth=2)
    ax.set_title(model_name)
    ax.set_xlabel('Epocha')
    ax.set_ylabel('Ztráta')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
print("  ✓ Uloženo: training_curves.png")
plt.close()

# Graf 3: Porovnání ztrátových funkcí
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Srovnění ztrátových funkcí na konci trénování', fontsize=14, fontweight='bold')

names = [name for name, _ in sorted_results]
train_losses = [metrics['train_loss'] for _, metrics in sorted_results]
val_losses = [metrics['val_loss'] for _, metrics in sorted_results]

x = np.arange(len(names))
width = 0.35

ax1.bar(x - width/2, train_losses, width, label='Trénovací ztráta', color='skyblue')
ax1.bar(x + width/2, val_losses, width, label='Validační ztráta', color='coral')
ax1.set_ylabel('Ztráta')
ax1.set_title('Trénovací vs validační ztráta')
ax1.set_xticks(x)
ax1.set_xticklabels(names, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Graf porovnání AUC
auc_values = [metrics['auc'] for _, metrics in sorted_results]
colors = ['green' if v == max(auc_values) else 'steelblue' for v in auc_values]
ax2.barh(names, auc_values, color=colors)
ax2.set_xlabel('AUC skóre')
ax2.set_xlim(0.5, 1)
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(auc_values):
    ax2.text(v + 0.01, i, f'{v:.3f}', va='center')

plt.tight_layout()
plt.savefig('loss_and_auc_comparison.png', dpi=150, bbox_inches='tight')
print("  ✓ Uloženo: loss_and_auc_comparison.png")
plt.close()

# Graf 4: Matice zmatení pro nejlepší model
best_model_name = sorted_results[0][0]
best_model_y_pred = results[best_model_name]['y_pred']
cm = confusion_matrix(y_test, best_model_y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Počet'})
ax.set_title(f'Matice zmatení - Nejlepší model: {best_model_name}', fontsize=12, fontweight='bold')
ax.set_xlabel('Předpokládaná třída')
ax.set_ylabel('Skutečná třída')
ax.set_xticklabels(['Negativní', 'Pozitivní'])
ax.set_yticklabels(['Negativní', 'Pozitivní'])

plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=150, bbox_inches='tight')
print("  ✓ Uloženo: confusion_matrix_best.png")
plt.close()

# ==============================================================================
# 7. FINÁLNÍ SHRNUTÍ
# ==============================================================================

print("\n" + "="*80)
print("TRÉNOVÁNÍ DOKONČENO")
print("="*80)

best_name, best_metrics = sorted_results[0]
print(f"\nNejlepší model: {best_name}")
print(f"  Accuracy: {best_metrics['accuracy']:.4f}")
print(f"  F1 skóre: {best_metrics['f1']:.4f}")
print(f"  Precision: {best_metrics['precision']:.4f}")
print(f"  Recall: {best_metrics['recall']:.4f}")
print(f"  AUC: {best_metrics['auc']:.4f}")
print(f"  Époch trénování: {best_metrics['epochs_trained']}")

print("\nVygenerované grafy:")
print("  - metric_comparison.png")
print("  - training_curves.png")
print("  - loss_and_auc_comparison.png")
print("  - confusion_matrix_best.png")

print("\nUložené soubory:")
print("  - training_results.pkl")
print("  - training_histories.pkl")

print("\n" + "="*80)

# Sentiment classification with neural networks

Seminar project for **4IZ232 – Umělá inteligence a její aplikace** (VŠE).
The goal is binary sentiment classification (positive / negative) of short English
product reviews with several neural network architectures (dense, CNN, LSTM),
plus an automatically generated seminar report in IMRaD format (`.docx`).

All code comments, console output and the generated report are in Czech.

## Dataset

[Sentiment Labelled Sentences](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences)
by Kotzias et al., *From Group to Individual Labels using Deep Features*, KDD 2015
(see `readme.txt`).

| File | Content | Used |
|---|---|---|
| `amazon_cells_labelled.txt` | 1,000 Amazon reviews of mobile phones and accessories (500 positive, 500 negative) | yes |
| `imdb_labelled.txt` | 1,000 IMDb movie review sentences | no (available for further experiments) |

Format: one sentence per line, `sentence<TAB>label`, where `label` is `1` (positive) or `0` (negative).

## Project structure

```
SEM_04/
├── amazon_cells_labelled.txt      # dataset used
├── imdb_labelled.txt              # additional dataset (unused)
├── readme.txt                     # original dataset description
├── README.md
└── source_files/
    ├── data_split.py              # standalone demo of the 70/15/15 split
    ├── preprocessing.py           # step 1: cleaning, tokenization, padding
    ├── inspect_preprocessing.py   # step 2 (optional): checks preprocessed data
    ├── train_neural_network.py    # step 3: trains and evaluates 8 models, plots charts
    └── generate_report.py         # step 4: builds the seminar report (.docx)
```

## Requirements

- Python 3.10+ (tested with Python 3.13)
- TensorFlow 2.x (tested with TensorFlow 2.21 / Keras 3.15)

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn python-docx
```

## How to run

The scripts use relative paths, so **run them from the project root**
(the folder with the `.txt` data files), not from `source_files/`:

```bash
python source_files/preprocessing.py
python source_files/inspect_preprocessing.py   # optional
python source_files/train_neural_network.py
python source_files/generate_report.py
```

On Windows, if the console shows errors with Czech characters, set
`PYTHONIOENCODING=utf-8` first. Training all 8 models on CPU takes about one minute.

## Pipeline

### 1. Data split (`preprocessing.py`, `data_split.py`)

Stratified split with `random_state=42` into train / validation / test:
700 / 150 / 150 sentences, each with exactly 50 % positive sentences.
`data_split.py` only shows this split; `preprocessing.py` contains the same split
as part of the full pipeline.

### 2. Preprocessing (`preprocessing.py`)

1. Text cleaning: lowercasing, removal of URLs/HTML tags, punctuation and digits, collapsing whitespace.
2. Tokenization with the Keras `Tokenizer` (max. 5,000 words, `<OOV>` token for unknown words),
   fitted **only on the training set** to avoid data leakage.
   The resulting vocabulary has 1,464 entries.
3. Padding / truncation (`pre`) to a fixed length equal to the 95th percentile of training
   sentence lengths, i.e. **23 tokens** (covers 95.7 % of training sentences).

Outputs:
- `preprocessed_data.npz` – padded sequences `X_*` and labels `y_*` for all three sets
- `preprocessing_params.pkl` – fitted tokenizer, `vocab_size`, `max_length`, `oov_token`

### 3. Data check (`inspect_preprocessing.py`)

Loads the saved files, decodes an example sequence back to words, prints the class
balance and asserts that sequence lengths, token indices and labels are valid.

### 4. Training (`train_neural_network.py`)

Every model starts with an `Embedding` layer and ends with one sigmoid neuron;
loss is binary cross-entropy. Early stopping monitors `val_loss` (patience 3,
best weights restored). Seed 42 is fixed for Python, NumPy and TensorFlow.

| Model | Architecture | Embedding | Dropout | Optimizer | Max epochs | Batch |
|---|---|---|---|---|---|---|
| `Dense_2L_32_16` | Flatten → Dense 32 → Dense 16 | 32 | 0.2 | Adam | 15 | 32 |
| `Dense_4L_128_64_32` | Flatten → Dense 128 → 64 → 32 | 64 | 0.3 | Adam | 20 | 32 |
| `Dense_3L_64_high_dropout` | Flatten → Dense 64 → 32 | 64 | 0.5 | Adam | 20 | 32 |
| `Dense_2L_SGD` | Flatten → Dense 64 | 64 | 0.3 | SGD | 25 | 16 |
| `CNN_simple` | 2× (Conv1D 32/64 + MaxPool) → Dense 64 | 64 | 0.3 | Adam | 15 | 32 |
| `CNN_deep` | 3× (Conv1D 64/64/32 + MaxPool) → Dense 64 | 64 | 0.3 | Adam | 20 | 32 |
| `LSTM_64` | LSTM 64 → Dense 32 | 64 | 0.3 | Adam | 15 | 32 |
| `LSTM_RMSprop` | LSTM 64 → Dense 32 | 64 | 0.3 | RMSprop | 15 | 32 |

Each model is evaluated on the test set (accuracy, precision, recall, F1, ROC AUC).

Outputs:
- `training_results.pkl` – metrics and test predictions for each model
- `training_histories.pkl` – loss/accuracy per epoch for each model
- `metric_comparison.png` – accuracy, F1, precision, recall of all models
- `training_curves.png` – train/validation loss of the 4 best models
- `loss_and_auc_comparison.png` – final losses and AUC of all models
- `confusion_matrix_best.png` – confusion matrix of the best model (by F1)

### 5. Report (`generate_report.py`)

Generates `Klasifikace_sentimentu_seminární_práce.docx` (title page, abstract,
introduction, methods, results, discussion, conclusion, references). The results
table is filled in from `training_results.pkl`; **the rest of the text is static**.

## Results

Test set results from a verification run (TensorFlow 2.21, CPU, seed 42), sorted by F1:

| # | Model | Accuracy | F1 |
|---|---|---|---|
| 1 | LSTM_RMSprop | 0.7733 | 0.7901 |
| 2 | LSTM_64 | 0.7933 | 0.7862 |
| 3 | Dense_2L_32_16 | 0.7467 | 0.7654 |
| 4 | CNN_simple | 0.7267 | 0.7485 |
| 5 | CNN_deep | 0.7067 | 0.7284 |
| 6 | Dense_3L_64_high_dropout | 0.6933 | 0.7051 |
| 7 | Dense_4L_128_64_32 | 0.6533 | 0.6905 |
| 8 | Dense_2L_SGD | 0.5400 | 0.4733 |

The LSTM models do best. Plain SGD with default learning rate barely learns within
25 epochs. With only 700 training sentences, all models overfit quickly (training
loss close to 0 while validation loss rises), so early stopping matters.

With the seed fixed, repeated runs on the same machine give identical results. Other
TensorFlow versions or hardware (GPU, oneDNN on/off) can produce slightly different
numbers; set `TF_ENABLE_ONEDNN_OPTS=0` for maximum determinism on CPU.

## Known limitations

- The text of the report (`generate_report.py`) contains **hardcoded numbers and
  conclusions** (e.g. "best model `Dense_2L_32_16` with accuracy 0.8400", "sequence length 50")
  that do not match the actual output of the code (best model is an LSTM with F1 ≈ 0.79,
  sequence length 23). Update the text to the results of your run before submitting.
- The dataset is very small (1,000 sentences), so differences between models of a few
  percent are within the noise of a single split.

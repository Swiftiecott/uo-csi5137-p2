# The code can only require Python 3.11 for running

import os
import json
import random
from pathlib import Path
from pprint import pprint
import numpy as np
from tqdm import tqdm

# -------------------------
# Importing library
# -------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import (AutoTokenizer, AutoModelForMaskedLM, AutoConfig,
                          DataCollatorForLanguageModeling, TrainingArguments, Trainer)
from datasets import load_dataset, Dataset
import torch

print("Setting up environments...\n")
# -------------------------
# Set base parameters
# -------------------------
THRESHOLD = 0.3
RANDOM_SEED = 42
MINI_MLM_EPOCHS = 8
SENT_EMB_EPOCHS = 1
MLM_BATCH_SIZE = 8
SENT_BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../datasets"
BASE_DIR = Path("../repro_table1")
BASE_DIR.mkdir(exist_ok=True)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# -------------------------
# Read datasets
# -------------------------
RAW_DATASETS_FILE = os.path.join(DATA_DIR, "raw_datasets.json")
MODEL_SCHEMA_FILE = os.path.join(DATA_DIR, "model_schema.json")
GOLD_FILE = os.path.join(DATA_DIR, "gold.json")

# Load the JSON datasets
def load_datasets():
    with open(RAW_DATASETS_FILE, "r") as f: raw = json.load(f)
    with open(MODEL_SCHEMA_FILE, "r") as f: schema = json.load(f)
    with open(GOLD_FILE, "r") as f: gold = json.load(f)
    return raw, schema, gold

# Create synthetic variables from read files
raw_datasets, model_schema, gold = load_datasets()

# -------------------------
# Create small synthetic corpus text for domain-adaptation
# -------------------------
def build_synthetic_sentences(raw_datasets, model_schema):
    sents = []
    # create templated sentences that mention energy domain vocabulary
    templates = [
        "At {time} the {device} reported {value} for {measure}.",
        "{device} measurement: {measure} = {value} at {time}.",
        "The {device} sensor recorded {measure} of {value}.",
        "{measure} is measured by {device} in the building at {time}."
    ]
    measures = [
        ("energy_kwh","kWh"),("energy_consumption","kWh"),("solar_watts","W"),("solar_power","W"),
        ("temperature","Celsius"),("room_temperature","Celsius"),("humidity","%"),("co2","ppm"),
        ("voltage","V"),("current","A"),("power_usage","W")
    ]
    times = ["2023-04-01 12:00","2023-04-02 08:00","2023-04-03 18:30","2023-04-04 07:45"]
    devices = ["meter","panel","thermostat","sensor","smartmeter","boiler","light","station"]
    for _ in range(200):  # produce around 200 sentences
        t = random.choice(times)
        dev = random.choice(devices)
        m, val_unit = random.choice(measures)
        val = str(round(random.uniform(0.1,5000),2)) + " " + val_unit
        templ = random.choice(templates)
        sents.append(templ.format(time=t, device=dev, measure=m, value=val))

    # Also include attribute names and model attribute names as sentences to bias tokenizer
    for ds in raw_datasets:
        for a in ds["attributes"]:
            sents.append(f"Attribute {a} in dataset {ds['name']}.")

    for e in model_schema:
        for a in e["attributes"]:
            sents.append(f"Model attribute {a} of entity {e['entity']}.")

    return sents

synthetic_sentences = build_synthetic_sentences(raw_datasets, model_schema)
corpus_path = BASE_DIR/"synthetic_energy_corpus.txt"
with open(corpus_path, "w", encoding="utf-8") as f:
    for s in synthetic_sentences:
        f.write(s.strip()+"\n")

print("Synthetic corpus is writen.\n")

# -------------------------
# Mini Energy BERT
# -------------------------
mini_mlm_dir = BASE_DIR/"mini_energy_bert"
if not mini_mlm_dir.exists():
    mini_mlm_dir.mkdir(parents=True)

print("Training Mini Energy BERT")

base_mlm = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(base_mlm, use_fast=True)
config = AutoConfig.from_pretrained(base_mlm)
model_mlm = AutoModelForMaskedLM.from_pretrained(base_mlm, config=config)

# Prepare dataset for HuggingFace datasets
text_lines = [line.strip() for line in open(corpus_path, "r", encoding="utf-8") if line.strip()]
hf_ds = Dataset.from_dict({"text": text_lines})

def tokenize_function(ex):
    return tokenizer(ex["text"], truncation=True, max_length=128)

tokenized = hf_ds.map(tokenize_function, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir=str(mini_mlm_dir/"training"),
    per_device_train_batch_size=MLM_BATCH_SIZE,
    num_train_epochs=MINI_MLM_EPOCHS,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_total_limit=1,
    fp16=False,
    no_cuda=(DEVICE=="cpu")
)

trainer = Trainer(
    model=model_mlm,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator
)

# Training Mini Energy BERT
trainer.train()
trainer.save_model(str(mini_mlm_dir))
tokenizer.save_pretrained(str(mini_mlm_dir))

print("Done.\n")

# -------------------------
# Energy Sentence BERT
# -------------------------

print("Training small Energy Sentence-BERT...")

sent_model_base = "all-MiniLM-L6-v2"
sent_out = BASE_DIR/"energy_sentence_bert"
sent_model = SentenceTransformer(sent_model_base)

# prepare denoising pairs: original sentence + corrupted version
def corrupt_text(s, drop_prob=0.12):
    toks = s.split()
    out = [t for t in toks if random.random() > drop_prob]
    if len(out) == 0:
        out = toks[:max(1, len(toks)//2)]
    return " ".join(out)

examples = []
for s in synthetic_sentences:
    corrupted = corrupt_text(s)
    examples.append(InputExample(texts=[s, corrupted]))

random.shuffle(examples)
train_dataloader = torch.utils.data.DataLoader(examples, batch_size=SENT_BATCH_SIZE, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(sent_model)

sent_model.fit(train_objectives=[(train_dataloader, train_loss)],
               epochs=SENT_EMB_EPOCHS,
               show_progress_bar=True,
               output_path=str(sent_out))

print("Done.\n")

# -------------------------
# Embedding helpers
# -------------------------
# Prepare COMA-like functons
tfidf_vectorizer = TfidfVectorizer().fit(
    [" ".join(ds["attributes"]) for ds in raw_datasets] +
    [" ".join(e["attributes"]) for e in model_schema]
)

def coma_like_match(raw_attr, target_attrs):
    # Compare raw_attr and each target_attr
    raw_txt = raw_attr
    cand_texts = target_attrs

    # Set tfidf vectors
    vecs = tfidf_vectorizer.transform([raw_txt] + cand_texts)
    raw_v = vecs[0].toarray()
    cand_v = vecs[1:].toarray()
    tfidf_sims = cosine_similarity(raw_v, cand_v)[0]

    # small Jaro-Winkler approximation: use normalized prefix and length overlap (cheap proxy)
    def jw_proxy(a,b):
        a_l = a.lower(); b_l = b.lower()
        common = sum(1 for c in a_l if c in b_l)
        return common / max(len(a_l), len(b_l), 1)
    
    jw_scores = np.array([jw_proxy(raw_txt, c) for c in cand_texts])
    combined = 0.5 * tfidf_sims + 0.5 * jw_scores
    best_idx = int(np.argmax(combined))
    return cand_texts[best_idx], float(combined[best_idx])

# Set BERT baseline and Mini Energy-BERT
def mean_pooling_transformers(model, tokenizer, texts, device=DEVICE, batch_size=32):
    model.to(device)
    model.eval()
    all_emb = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            out = model(**encoded, return_dict=True)
            token_emb = out.last_hidden_state  # (B, L, H)
            attention_mask = encoded['attention_mask'].unsqueeze(-1)
            summed = (token_emb * attention_mask).sum(1)
            counts = attention_mask.sum(1)
            mean_pooled = (summed / counts).cpu().numpy()
            all_emb.extend(mean_pooled)
    return np.vstack(all_emb)

# SentenceTransformer embeddings
sbert_base = SentenceTransformer("all-MiniLM-L6-v2")
sbert_base.max_seq_length = 256

# Load mini-mlm model for embeddings
mini_tokenizer = AutoTokenizer.from_pretrained(str(mini_mlm_dir))
mini_mlm_model = AutoModelForMaskedLM.from_pretrained(str(mini_mlm_dir))
encoder_model = mini_mlm_model.bert

# BERT baseline encoder
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
bert_encoder = bert_model.bert

# -------------------------
# Set matching function
# -------------------------
def embed_with_sbert(model, texts):
    return model.encode(texts, convert_to_tensor=False, show_progress_bar=False)

def dataset_level_match_and_attr_map(embedding_fn, raw_ds, schema_entities, threshold=THRESHOLD, is_sentence_transformer=False):
    results = {}

    # prepare dataset-level texts as joined attributes
    raw_texts = [" ".join(r["attributes"]) for r in raw_ds]
    entity_texts = [" ".join(e["attributes"]) for e in schema_entities]
    raw_emb = embedding_fn(raw_texts)
    ent_emb = embedding_fn(entity_texts)

    # compute dataset-level similarities and pick best entity for each raw
    sims = cosine_similarity(raw_emb, ent_emb)
    top_entity_indices = np.argmax(sims, axis=1)
    for i, raw in enumerate(raw_ds):
        ent_idx = int(top_entity_indices[i])
        ent = schema_entities[ent_idx]

        # attribute-level mapping
        raw_attrs = raw["attributes"]
        target_attrs = ent["attributes"]

        # embed attributes
        raw_attr_emb = embedding_fn(raw_attrs)
        target_attr_emb = embedding_fn(target_attrs)
        attr_sim = cosine_similarity(raw_attr_emb, target_attr_emb)  # (n_raw_attr, n_target_attr)
        mappings = []
        for ri, a in enumerate(raw_attrs):
            best_j = int(np.argmax(attr_sim[ri]))
            best_score = float(attr_sim[ri][best_j])
            mapped = target_attrs[best_j] if best_score >= threshold else None
            mappings.append({"raw_attr": a, "mapped_attr": mapped, "score": best_score, "entity": ent["entity"]})
        results[(raw["id"], ent["entity"])] = mappings
    return results

# Wrappers for each model
def run_coma_like(raw_ds, schema):
    # create embedding function for TF-IDF single-strings using vectorizer and naive combos
    # def embed_texts(texts):
    #     # represent each text by tfidf vector for the text join (vectorizer handles)
    #     return tfidf_vectorizer.transform(texts).toarray()
    
    # But we need mapping at attribute-level with coma_like_match function
    results = {}
    for raw in raw_ds:
        # dataset-level match: find entity whose combined-attribute text has highest tfidf cosine
        raw_text = " ".join(raw["attributes"])
        ent_texts = [" ".join(e["attributes"]) for e in schema]
        vecs = tfidf_vectorizer.transform([raw_text] + ent_texts)
        raw_v = vecs[0].toarray()
        cand_v = vecs[1:].toarray()
        ds_sims = cosine_similarity(raw_v, cand_v)[0]
        best_ent_idx = int(np.argmax(ds_sims))
        ent = schema[best_ent_idx]
        mappings = []

        for a in raw["attributes"]:
            mapped_attr, score = coma_like_match(a, ent["attributes"])
            mapped_attr = mapped_attr if score >= THRESHOLD else None
            mappings.append({"raw_attr": a, "mapped_attr": mapped_attr, "score": score, "entity": ent["entity"]})

        results[(raw["id"], ent["entity"])] = mappings
    return results

def run_bert_baseline(raw_ds, schema):
    def emb_fn(texts):
        return mean_pooling_transformers(bert_encoder, bert_tokenizer, texts, device=DEVICE, batch_size=32)
    return dataset_level_match_and_attr_map(emb_fn, raw_ds, schema, threshold=THRESHOLD)

def run_mini_energy_bert(raw_ds, schema):
    def emb_fn(texts):
        return mean_pooling_transformers(encoder_model, mini_tokenizer, texts, device=DEVICE, batch_size=32)
    return dataset_level_match_and_attr_map(emb_fn, raw_ds, schema, threshold=THRESHOLD)

def run_energy_sentence_bert(raw_ds, schema):
    sent_trained = SentenceTransformer(str(sent_out))
    def emb_fn(texts):
        return send_encode_numpy(sent_trained, texts)
    return dataset_level_match_and_attr_map(emb_fn, raw_ds, schema, threshold=THRESHOLD)

# Helper to get numpy embeddings from SentenceTransformer safely
def send_encode_numpy(model, texts, batch_size=64):
    embs = model.encode(texts, convert_to_tensor=False, batch_size=batch_size, show_progress_bar=False)
    return np.array(embs)

# -------------------------
# STEP 6: Run all models and evaluate
# -------------------------
print("\nRunning COMA...")
coma_results = run_coma_like(raw_datasets, model_schema)

print("\nRunning BERT...")
bert_results = run_bert_baseline(raw_datasets, model_schema)

print("\nRunning Mini Energy BERT...")
mini_results = run_mini_energy_bert(raw_datasets, model_schema)

print("\nRunning Energy Sentence BERT...")
energy_sent_results = run_energy_sentence_bert(raw_datasets, model_schema)

all_res = {
    "COMA-like": coma_results,
    "BERT": bert_results,
    "MiniEnergyBERT": mini_results,
    "EnergySentenceBERT": energy_sent_results
}

# -------------------------
# EVALUATION
# -------------------------
def evaluate_result(pred_dict, gold):
    TP = 0
    FP = 0
    FN = 0

    # Set cases for gold.json
    def process_one(dsid, raw_attr, pred_attr):
        nonlocal TP, FP, FN

        # If dataset is missing OR gold dataset is None, treat as empty
        gold_ds = gold.get(dsid)
        if gold_ds is None:
            gold_attr = None
        else:
            gold_attr = gold_ds.get(raw_attr, None)

        # Case: gold says "no match"
        if gold_attr is None:
            if pred_attr is not None:
                FP += 1
            return

        # Case: gold gives a real match
        if pred_attr == gold_attr:
            TP += 1
        else:
            FP += 1
            FN += 1

    # Detect shapes
    try:
        sample_key = next(iter(pred_dict))
    except StopIteration:
        return 0.0, 0.0, 0.0

    val0 = pred_dict[sample_key]

    if isinstance(val0, list):
        for (dsid, entity), mappings in pred_dict.items():
            for m in mappings:
                raw_attr = m.get("raw_attr")
                pred_attr = m.get("mapped_attr")
                process_one(dsid, raw_attr, pred_attr)
    else:
        for dsid, mapping in pred_dict.items():
            if mapping is None:
                continue
            for raw_attr, pred_attr in mapping.items():
                process_one(dsid, raw_attr, pred_attr)

    # Calculate evaluation results
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    f = 2 * p * r / (p + r) if p + r > 0 else 0
    return p, r, f

table = []
for model_name, res in all_res.items():
    p, r, f = evaluate_result(res, gold)
    table.append((model_name, p, r, f))

# Print table
print("\nEvaluation results")
print("---------------------------------------------------------------")
print(f"{'Model':25s} {'Precision':9s} {'Recall':7s} {'F1':7s}")
print("---------------------------------------------------------------")
for row in table:
    print(f"{row[0]:25s} {row[1]:9.3f} {row[2]:7.3f} {row[3]:7.3f}")

print("\nEvaluation done. Code executive complete.")

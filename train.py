import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from huggingface_hub import login
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import encode_data, load_all_genres, seed_everything, split_data
from utils import (
    CACHED_MODEL_DIR,
    HF_REPO_ID,
    MAX_LENGTH,
    MODEL_NAME,
    build_label_maps,
    compute_metrics,
    get_hf_token,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"


def main():
    seed_everything(42)
    hf_token = get_hf_token()
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")

    print("\n═══ Loading data ═══")
    genre_reviews = load_all_genres()
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    label2id, id2label = build_label_maps(train_labels)
    print(f"Labels: {list(label2id.keys())}")

    print("\n═══ Tokenizing ═══")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_dataset = encode_data(tokenizer, train_texts, train_labels, label2id, MAX_LENGTH)
    test_dataset = encode_data(tokenizer, test_texts, test_labels, label2id, MAX_LENGTH)

    print("\n═══ Loading pre-trained model ═══")
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=16,
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        output_dir="./results",
        logging_dir="./logs",
        logging_steps=100,
        eval_strategy="steps",
        report_to=[],
    )

    print("\n═══ Training ═══")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    print("\n═══ Local Evaluation ═══")
    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")

    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1).flatten().tolist()
    pred_label_names = [id2label[l] for l in pred_labels]
    report = classification_report(test_labels, pred_label_names, output_dict=True)
    print(classification_report(test_labels, pred_label_names))

    # ── Confusion Matrix Heatmap (all classifications) ──
    print("\n═══ Generating Plots ═══")
    genre_classifications = defaultdict(int)
    for _true, _pred in zip(test_labels, pred_label_names):
        genre_classifications[(_true, _pred)] += 1

    dicts_to_plot = []
    for (_true_genre, _pred_genre), _count in genre_classifications.items():
        dicts_to_plot.append({
            "True Genre": _true_genre,
            "Predicted Genre": _pred_genre,
            "Number of Classifications": _count,
        })

    df_to_plot = pd.DataFrame(dicts_to_plot)
    df_wide = df_to_plot.pivot_table(
        index="True Genre", columns="Predicted Genre", values="Number of Classifications"
    )

    plt.figure(figsize=(9, 7))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("✓ Saved confusion_matrix.png")

    # ── Misclassification Heatmap (diagonal removed) ──
    misclass = defaultdict(int)
    for _true, _pred in zip(test_labels, pred_label_names):
        if _true != _pred:
            misclass[(_true, _pred)] += 1

    dicts_misclass = []
    for (_true_genre, _pred_genre), _count in misclass.items():
        dicts_misclass.append({
            "True Genre": _true_genre,
            "Predicted Genre": _pred_genre,
            "Number of Classifications": _count,
        })

    df_misclass = pd.DataFrame(dicts_misclass)
    df_wide_misclass = df_misclass.pivot_table(
        index="True Genre", columns="Predicted Genre", values="Number of Classifications"
    )

    plt.figure(figsize=(9, 7))
    sns.set(style="ticks", font_scale=1.2)
    sns.heatmap(df_wide_misclass, linewidths=1, cmap="Purples", annot=True, fmt=".0f")
    plt.xticks(rotation=45, ha="right")
    plt.title("Misclassification Heatmap (Diagonal Removed)")
    plt.tight_layout()
    plt.savefig("misclassification_heatmap.png", dpi=150)
    plt.close()
    print("✓ Saved misclassification_heatmap.png")

    eval_output = {
        "eval_loss": eval_results.get("eval_loss"),
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "classification_report": report,
    }
    with open("eval_results.json", "w") as f:
        json.dump(eval_output, f, indent=2)
    print("✓ Saved eval_results.json")

    print(f"\n═══ Saving model to {CACHED_MODEL_DIR} ═══")
    trainer.save_model(CACHED_MODEL_DIR)
    tokenizer.save_pretrained(CACHED_MODEL_DIR)
    print("✓ Model saved locally")

    print(f"\n═══ Pushing to HuggingFace: {HF_REPO_ID} ═══")
    model.push_to_hub(HF_REPO_ID, token=hf_token)
    tokenizer.push_to_hub(HF_REPO_ID, token=hf_token)
    print(f"✓ Model & tokenizer pushed to {HF_REPO_ID}")

    label_maps = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open("label_maps.json", "w") as f:
        json.dump(label_maps, f, indent=2)
    print("✓ Saved label_maps.json")

    print("\n✅ Training pipeline complete!")


if __name__ == "__main__":
    main()

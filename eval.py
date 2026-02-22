import json
import os

from huggingface_hub import login
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from data import encode_data, load_all_genres, split_data
from utils import (
    HF_REPO_ID,
    MAX_LENGTH,
    build_label_maps,
    compute_metrics,
    get_hf_token,
)

os.environ["WANDB_DISABLED"] = "true"


def main():
    hf_token = get_hf_token()
    login(token=hf_token)
    print("✓ Logged in to HuggingFace")

    print("\n═══ Loading test data ═══")
    genre_reviews = load_all_genres()
    _, _, test_texts, test_labels = split_data(genre_reviews)
    print(f"Test samples: {len(test_texts)}")

    if os.path.exists("label_maps.json"):
        with open("label_maps.json") as f:
            maps = json.load(f)
        label2id = maps["label2id"]
        id2label = {int(k): v for k, v in maps["id2label"].items()}
        print("✓ Loaded label maps from label_maps.json")
    else:
        label2id, id2label = build_label_maps(test_labels)
        print("⚠ Built label maps from test data (label_maps.json not found)")

    print(f"\n═══ Loading model from HuggingFace: {HF_REPO_ID} ═══")
    tokenizer = DistilBertTokenizerFast.from_pretrained(HF_REPO_ID, token=hf_token)
    model = DistilBertForSequenceClassification.from_pretrained(HF_REPO_ID, token=hf_token)
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"✓ Model loaded on {device}")

    print("\n═══ Tokenizing test data ═══")
    test_dataset = encode_data(tokenizer, test_texts, test_labels, label2id, MAX_LENGTH)

    print("\n═══ Evaluation from HuggingFace Model ═══")
    eval_args = TrainingArguments(
        output_dir="./eval_results_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print(f"Eval results: {eval_results}")

    predictions = trainer.predict(test_dataset)
    pred_labels = predictions.predictions.argmax(-1).flatten().tolist()
    pred_label_names = [id2label[l] for l in pred_labels]
    report = classification_report(test_labels, pred_label_names, output_dict=True)
    print(classification_report(test_labels, pred_label_names))

    hf_eval_output = {
        "eval_loss": eval_results.get("eval_loss"),
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "classification_report": report,
    }
    with open("eval_results_from_hf.json", "w") as f:
        json.dump(hf_eval_output, f, indent=2)
    print("✓ Saved eval_results_from_hf.json")

    print("\n═══ Comparison: Local vs HuggingFace ═══")
    if os.path.exists("eval_results.json"):
        with open("eval_results.json") as f:
            local_eval = json.load(f)
        print(f"  Local  Accuracy: {local_eval.get('eval_accuracy', 'N/A')}")
        print(f"  HF     Accuracy: {hf_eval_output['eval_accuracy']}")
        print(f"  Local  Loss:     {local_eval.get('eval_loss', 'N/A')}")
        print(f"  HF     Loss:     {hf_eval_output['eval_loss']}")
    else:
        print("  ⚠ eval_results.json not found — run train.py first for comparison")

    print("\n✅ Evaluation from HuggingFace repo complete!")


if __name__ == "__main__":
    main()

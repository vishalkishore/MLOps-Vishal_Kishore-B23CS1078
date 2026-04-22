from pathlib import Path
from typing import List

import sacrebleu
from striprtf.striprtf import rtf_to_text
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "Helsinki-NLP/opus-mt-bn-en"
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "references" / "input.rtf"
REFERENCE_FILE = BASE_DIR / "references" / "output.rtf"
OUTPUT_FILE = BASE_DIR / "output.txt"


def read_rtf_lines(path: Path) -> List[str]:
    text = rtf_to_text(path.read_text(encoding="utf-8", errors="ignore"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith("#"):
        lines = lines[1:]
    return lines


def translate_lines(lines: List[str]) -> List[str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    outputs = []
    for line in lines:
        tokens = tokenizer(line, return_tensors="pt", truncation=True, padding=True)
        generated = model.generate(**tokens, max_length=256)
        outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True).strip())
    return outputs


def main() -> None:
    input_lines = read_rtf_lines(INPUT_FILE)
    reference_lines = read_rtf_lines(REFERENCE_FILE)
    output_lines = translate_lines(input_lines)
    OUTPUT_FILE.write_text("\n".join(output_lines), encoding="utf-8")
    bleu = sacrebleu.corpus_bleu(output_lines, [reference_lines])
    print("First statement:", output_lines[0])
    print("BLEU score:", round(bleu.score, 2))


if __name__ == "__main__":
    main()

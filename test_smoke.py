"""
Quick functional smoke test for container build.
Verifies that model and tokenizer from Hugging Face work correctly.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

MODEL_NAME = "distilbert-base-uncased"


def main() -> None:
    print("🚀 Running smoke test...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    inputs = tokenizer("hello world", return_tensors="pt")
    outputs = model(**inputs)

    print("✅ Model forward pass OK. Logits shape:", outputs.logits.shape)

    loss = outputs.logits.sum()
    loss.backward()
    print("✅ Backward pass OK.")

    print("🎉 Smoke test passed successfully.")


if __name__ == "__main__":
    main()

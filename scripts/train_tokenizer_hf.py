from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

def main():
    files = ["data/corpus_1gb_clean.txt"]
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=files,
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "<sep>", "<cls>"],
    )
    save_dir = Path("artifacts/tokenizer")
    save_dir.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(save_dir))
    print("Saved HuggingFace tokenizer to", save_dir)

if __name__ == "__main__":
    main()

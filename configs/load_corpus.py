from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)
outfile = "data/corpus.txt"

wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)

with open(outfile, "w", encoding="utf-8") as f:
    for i, row in enumerate(wiki):
        f.write(row["text"].replace("\n", " ") + "\n")
        if i % 100000 == 0:
            print(f"Written {i} wiki articles")

    for i, row in enumerate(c4):
        f.write(row["text"].replace("\n", " ") + "\n")
        if i % 100000 == 0:
            print(f"Written {i} C4 samples")

print(f"Saved combined corpus to {outfile}")

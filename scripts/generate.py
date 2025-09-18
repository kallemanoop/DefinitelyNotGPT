import torch, yaml, json
from slm.model import TransformerLM, ModelConfig
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F

#load config
with open("configs/train_small.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f)

#inject vocab_size
with open("artifacts/tokenizer/vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)
cfg_dict["vocab_size"] = len(vocab)

cfg = ModelConfig(**cfg_dict)

#build model
device = "cuda"
model = TransformerLM(cfg).to(device)
ckpt = torch.load("artifacts/checkpoints/slm-small/step_005000.pt", map_location=device)
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

#load tokenizer
tok = ByteLevelBPETokenizer(
    "artifacts/tokenizer/vocab.json",
    "artifacts/tokenizer/merges.txt"
)

#generation utils
def sample_next_token(logits, temperature=1.0, top_k=50):
    logits = logits / temperature
    values, indices = torch.topk(logits, k=top_k)
    probs = F.softmax(values, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return indices.gather(-1, idx)

#prompt
prompt = "Once upon a time"
ids = tok.encode(prompt).ids
x = torch.tensor([ids], device=device).long()  # (1, T)

#generate
max_new_tokens = 50
temperature = 0.8
top_k = 50

with torch.no_grad():
    for _ in range(max_new_tokens):
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k)
        x = torch.cat([x, next_id], dim=1)

#decode
print("Prompt:", prompt)
print("Generated:", tok.decode(x[0].tolist()))

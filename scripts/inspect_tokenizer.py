from __future__ import annotations
import argparse
from slm.tokenizer import ByteLevelBPE
from slm.utils.logging import ok, info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--text", default="Hello, world!")
    args = ap.parse_args()

    tok = ByteLevelBPE(model_dir=args.model_dir)
    tok.load()
    ids = tok.encode(args.text, add_bos=True, add_eos=True)
    rec = tok.decode(ids)
    info(f"input:  {args.text}")
    info(f"ids:    {ids}")
    ok(  f"decode: {rec}")

if __name__ == "__main__":
    main()

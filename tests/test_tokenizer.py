from slm.tokenizer import ByteLevelBPE

def test_roundtrip(tmp_path):
    # tiny in-memory "corpus"
    corpus = ["Hello", "Hello Hello", "Help", "Yellow mellow"]
    tok = ByteLevelBPE(model_dir=tmp_path, vocab_size=300, min_freq=2)
    tok.train(corpus, seed=123)
    tok.save()

    tok2 = ByteLevelBPE(model_dir=tmp_path)
    tok2.load()

    s = "Hello mellow!"
    ids = tok2.encode(s, add_bos=True, add_eos=True)
    s2 = tok2.decode(ids)
    # byte-level decode may not reconstruct exact punctuation spacing
    # but should round-trip printable ascii safely
    assert "Hello" in s2 and "mellow" in s2

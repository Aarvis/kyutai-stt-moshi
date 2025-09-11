import sentencepiece as spm, json
sp = spm.SentencePieceProcessor(model_file="hi_sp.model")
print("pad", sp.pad_id(), "unk", sp.unk_id(), "bos", sp.bos_id(), "eos", sp.eos_id())
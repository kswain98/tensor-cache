
## wikitext-2 dataset

after running `prepare.py` (preprocess) we get:

- train.bin and val.bin (numpy memmap, uint16 GPT-2 BPE tokens)
- ~2M training tokens, ~250K validation tokens

prepare wikitext-2:

```sh
python data/wikitext2/prepare.py --out_dir=data/wikitext2
```

references:

- [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843), Merity et al. 2016 — original WikiText paper
- HuggingFace dataset: `wikitext` config `wikitext-2-raw-v1`

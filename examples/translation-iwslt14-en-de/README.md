
# Data
The used translation dataset IWSLT14 is from fairseq: https://github.com/facebookresearch/fairseq/blob/main/examples/translation/README.md#iwslt14-german-to-english-transformer
```bash 
# install dependencies
bash scripts/install_libs.sh
# download data
mkdir -p local/data/exp
bash examples/translation-iwslt14-en-de local/data/iwslt14
```


# Training
```bash
# Ours de -> en, args are: data_dir exp_dir cuda_device src_lang tgt_lang
bash examples/translation-iwslt14-en-de/train.sh local/data/iwslt14/data-converted-en-de-raw local/data/exp/iwslt14-en-de 0 de en 
# Ours de - en, eval 
bash examples/translation-iwslt14-en-de/predict.sh local/data/iwslt14/data-converted-en-de-raw local/data/exp/iwslt14-en-de 0 en de
# Ours en -> de, train 
bash examples/translation-iwslt14-en-de/train.sh local/data/iwslt14/data-converted-en-de-raw local/data/exp/iwslt14-en-de 0 en de
# Ours en -> de, eval 
bash examples/translation-iwslt14-en-de/predict.sh local/data/iwslt14/data-converted-en-de-raw local/data/exp/iwslt14-en-de 0 en de
# fairseq de -> en, args are: data_dir cuda_device src_lang tgt_lang
bash examples/translation-iwslt14-en-de/fairseq_run.sh local/data/iwslt14 0 de en
# fairseq en -> de
bash examples/translation-iwslt14-en-de/fairseq_run.sh local/data/iwslt14 0 en de
```


# BLEU Results
| Model | de -> en | en -> de |
| :--:  |  :--:    |   :--:   |
| Transformer | 33.27 | 27.72 | 
| Tied Transformers | 35.10 | 29.07 | 
| fairseq | 34.54 | 28.61 | 
| Ours | 34.36 | 28.33 | 


# Reference 
- The BLEU results are from: 
[Tied Transformers: Neural Machine Translation with Shared Encoder and Decoder](https://taoqin.github.io/papers/tiedT.AAAI2019.pdf)
- Dataset IWSLT 14
iwslt 14 en-de: https://wit3.fbk.eu/2014-01 </br>
training and dev set: https://drive.google.com/file/d/1GnBarJIbNgEIIDvUyKDtLmv35Qcxg6Ed/view </br>
test: https://drive.google.com/file/d/1JyKJZbbf2hIIXe-xVwW1uqMy8LDY6qgN/view </br>
processors: https://drive.google.com/file/d/1YNfSBwyyzaYZqWol2PYf5sTUcH1Z9f2h/view </br>

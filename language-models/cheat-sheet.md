# Transformers Catalog

## Pre-training Tasks
1. **Language Modeling (LM):** Predict the next token (unidirectional models), or,  previous and next token (bidirectional models).
2. **Masked Language Modeling (MLM):** Randomly mask out some tokens from the input sentences and then train the model to predict the masked tokens by using the rest of the tokens.
3. **Permuted Language Modeling (PLM):** same as LM but on a random permutation of input sequences. A permutation is randomly sampled from all possible permutations. Then some of the tokens are chosen as the target, and the model is trained to predict these targets.
4. **Denoising Autoencoder (DAE):** take a partially corrupted input (e.g. Randomly sampling tokens from the input and replacing them with [MASK] elements. randomly deleting tokens from the input, or shuffling sentences in random order) and aim to recover the original undistorted input.
5. **Contrastive Learning (CTL):** <To-do>

## Masked Language Modelling
### BERT [1810](https://arxiv.org/abs/1810.04805)
1. **Architecture:** Encoder
2. **Pretraining Objective:** Masked Language Modelling, Next-Sentence Prediction
3. **Number of Parameters:** Base 110M, Large 340M
4. **Dataset:** 3.3B Tokens from Toronto BookCorpus and Wikipedia 

### RoBERTa [1907](https://arxiv.org/abs/1907.11692)
1. **Architecture:** Encoder
2. **Pretraining Objective:** Masked Language Modelling
3. **Number of Parameters:** 356M
4. **Dataset:** 160GB and 33B Tokens from Toronto BookCorpus, Wikipedia, CC News, OpenWebText, and Stories

### ALBERT [1909](https://arxiv.org/abs/1909.11942)
1. **Architecture:** Encoder
2. **Pretraining Objective:** Masked Language Modelling, Next-Sentence Prediction
3. **Number of Parameters:** Base 110M, Large 340M
4. **Dataset:** 3.3B Tokens from Toronto BookCorpus and Wikipedia 
5. **Contribution:** Compressed version of BERT by using parameter sharing in each embedding layer to more efficient training.

### DistilBERT [1910](https://arxiv.org/abs/1910.01108)
1. **Architecture:** Encoder
2. **Pretraining Objective:** Masked Language Modelling, Next-Sentence Prediction
3. **Number of Parameters:** 66M
4. **Dataset:** 3.3B Tokens from Toronto BookCorpus and Wikipedia 
5. **Contribution:** Compressed version of BERT using distillation.

### ELECTRA [2003](https://arxiv.org/abs/2003.10555)
1. **Architecture:** Encoder
2. **Pretraining Objective:** Masked Language Modelling, Next-Sentence Prediction
3. **Number of Parameters:** Base 110M, Large 340M
4. **Dataset:** 3.3B Tokens from Toronto BookCorpus and Wikipedia 
5. **Dataset:** Trains two transformer models: the generator and the discriminator. Generator replaces tokens in a sequence, and trained as a masked language model. The discriminator tries to identify which tokens were replaced by the generator in the sequence.


## Casual Language Modeling
### GPT [1806](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 117M
4. **Dataset:** BookCorpus

### Transformer XL [1901](https://arxiv.org/abs/1901.02860)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 151M
4. **Dataset:** Wikitext-103
5. **Contribution:** Uses Relative Positional Embeddings which enable longer-context attention.

### GPT-2 [1902](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 1.5B
4. **Dataset:** WebText

### XLNet [1905](https://arxiv.org/abs/1906.08237)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 360M
4. **Dataset:** Toronto Book Corpus, Wikipedia (3.3B Tokens), Giga5 (16GB text), ClueWeb 2012-B (19GB), and Common Crawl (110 GB).


### GPT-3 [2005](https://arxiv.org/abs/2005.14165)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 175B
4. **Dataset:** CommonCrawl (410B), WebText2 (19B), Books1 (12B), Books2 (55B), and Wikipedia (3B)

### GLAM [2112](https://arxiv.org/abs/2112.06905)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 1.2T across 64 experts
4. **Dataset:** 1.6T tokens including web pages filtered by Wikipedia and books for quality

### Gopher [2112](https://arxiv.org/abs/2112.11446)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 1.2T across 64 experts
4. **Dataset:** 1.6T tokens including web pages filtered by Wikipedia and books for quality
5. **Contribution:** GPT-2 based. Uses RSNorm and Absoulte Positional Encoding rather than LayerNorm and Relative Positional Encoding.

### LAMBDA [2201](https://arxiv.org/abs/2201.08239)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 137B
4. **Dataset:** 1.56T tokens from public dialog data and web documents


### Chinchilla [2203](https://arxiv.org/abs/2203.15556)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 70B
4. **Dataset:** Massive Text

### PalM [2204](https://arxiv.org/abs/2204.02311)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 540B
4. **Dataset:** 780B tokens from webpages, books, Wikipedia, news articles, source code, social media conversations, and 24 programming languages.
5. **Contribution:** SwiGLU activations, parallel layers, multi-query attention, RoPE embeddings, Shared Input-Output Embeddings, and without biases.

### OPT [2205](https://arxiv.org/abs/2205.01068)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 540B
4. **Dataset:** 180B tokens from RoBERTa, the Pile, and PushShift.io Reddit.
5. **Contribution:** GPT-3 with training improvements introduced in Megatron-LM.

### GPT-3.5 [2210](https://openai.com/blog/chatgpt/)
1. **Architecture:** Decoder
2. **Pretraining Objective:** Next-token prediction
3. **Number of Parameters:** 175B
4. **Dataset:** CommonCrawl (410B), WebText2 (19B), Books1 (12B), Books2 (55B), and Wikipedia (3B).
5. **Contribution:** Built up on a number of models like Davinci-003 which are basically versions of the InstructGPT. 

## Encoder and Decoder-based Models
### T5 [1910](https://arxiv.org/abs/1910.10683)
1. **Architecture:** Encder+Decoder
2. **Pretraining Objective:** Denoising Autoencoder
3. **Number of Parameters:** 11B
4. **Dataset:** 750 GB of data from Colossal Clean Crawled Corpus (C4) and the Common Crawl dataset.
5. **Contribution:** Built on vanilla Transformer but uses Relative Positional Embeddings. 

### BART [1910](https://arxiv.org/abs/1910.13461)
1. **Architecture:** Encder+Decoder
2. **Pretraining Objective:** Denoising Autoencoder
3. **Number of Parameters:** 345M
4. **Dataset:** 160 GB with 33B Tokens from Toronto BookCorpus, Wikipedia, CC News, OpenWebText, and Stories
5. **Contribution:** Combination of BERT and GPT.

### BigBird [2007](https://arxiv.org/abs/2007.14062)
1. **Architecture:** Encder+Decoder
2. **Pretraining Objective:** Masked Language Modelling
3. **Number of Parameters:** 175B
4. **Dataset:** Books, CC-News, Stories, and Wikipedia.
5. **Contribution:** Uses Sparse Attention Mechanism which overcome the issue of quadratic dependency and leads training with longer sequences.

### DQ-BART [2203](https://arxiv.org/abs/2203.11239)
1. **Architecture:** Encder+Decoder
2. **Pretraining Objective:** Denoising Autoencoder
3. **Number of Parameters:** ~103M
4. **Dataset:** 1M tokens; CNN/DM, XSUM, ELI5, and  WMT16 En-Ro.
5. **Contribution:** Decrease the model size with the increase in performance by adding  quantization and distillation to the BART.

## Multilingual Language Modelling
### XLM-RoBERTa
1910
### mBART
2001
### BLOOM
2207

## Dialog
### DialoGPT
1910
### Anthropic Assistant 
2112
### GPTInstruct
2201
### GopherCite
2203
### BlenderBot3
2208
### Sparrow
2209
### ChatGPT
2210

## Summarization
### Pegasus
1912

## Text to Image
### DALL-E
2101
### GLIDE
2112
### Stable Diffusion
2112
### Flamingo
2204
### DALL-E 2
2204

## Image Classification
### ViT
2010
### Swin Transformer
2103
### Global Context ViT
2206

## Image-Text Pair
### CLIP
2102
### ALBEF
2107
### FLAVA
2112
### BLIP
2201
### Imagen
2206
### BLIP-2
2301

## Reinforcement Learning
### Decision Transformers
2106
### Trajectory Transformers
2106
### Gato
2205

## Others
### AlphaFold
2107
### Minerva
2206

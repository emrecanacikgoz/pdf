# Transformers Catalog

## RLHF
<To-do>

## Pre-training Tasks
1. **Language Modeling (LM):** Predict the next token (unidirectional models), or,  previous and next token (bidirectional models).
2. **Masked Language Modeling (MLM):** Randomly mask out some tokens from the input sentences and then train the model to predict the masked tokens by using the rest of the tokens.
3. **Permuted Language Modeling (PLM):** same as LM but on a random permutation of input sequences. A permutation is randomly sampled from all possible permutations. Then some of the tokens are chosen as the target, and the model is trained to predict these targets.
4. **Denoising Autoencoder (DAE):** take a partially corrupted input (e.g. Randomly sampling tokens from the input and replacing them with [MASK] elements. randomly deleting tokens from the input, or shuffling sentences in random order) and aim to recover the original undistorted input.
5. **Contrastive Learning (CTL):** <To-do>

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


## Masked Language Modelling
### BERT
1810
### RoBERTa
1907
### Albert
1909
### DistilBERT
1910
### ELECTRA
2003


## Casual Language Modeling
### GPT
1806
### Transformer XL
1901
### GPT-2
1902
### XLNet
1905
### CTRL
1909
### GPT-3
2005
### GPT-Neo
2103
### Jurrasic-1
2109
### GLAM
2112
### Gopher
2112
### LAMBDA
2201
### Chinchilla
2203
### PalM
2204
### OPT
2205
### GPT-3.5
2210

## Encoder and Decoder-based Models
### T5
1910
### BART
1910
### BigBird
2007
### Switch
2101
### HTML
2107
### DQ-BART
2203

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


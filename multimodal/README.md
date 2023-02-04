# Multimodality Survey

## VL-Bert
[r221008:1908:Microsoft:940cite:VL-BERT:Pre-training of Generic Visual-Linguistic Representations.pdf](https://arxiv.org/pdf/1908.08530.pdf). 

VL-BERT takes both visual and linguistic embedded features as input. Each input element is either of a word from the input sentence, or a region-of-interest (RoI) from the input image, together with certain special elements to disambiguate different input formats. For each input element, its embedding feature is the summation of four types of embedding: token embedding, visual feature embedding, segment embedding, and sequence position embedding. The visual feature embedding is newly introduced for capturing visual clues, while the other three embeddings follow the design in the original BERT paper. VL-BERT is pre-trained on two different tasks: Masked Language Modeling with Visual Clues and Masked RoI Classification with Linguistic Clues. In MLM with Visual Clues task, each word in the input sentence are masked randomly with the probability of 15% and the model is trained to predict masked words by looking on the unmasked words in the sentence together with the visual features. Masked RoI Classification with Linguistic Clues: To avoid any visual clue leakage from the visual feature embedding of other elements, the pixels laid in the masked RoI are set as zeros before applying Fast R-CNN (What?). During pre-training, the final output feature corresponding to the masked RoI is fed into a classifier with Softmax cross-entropy loss for object category classification. The category label predicted by pre-trained Faster R-CNN is set as the ground-truth. In VL-BERT, the parameters of Fast R-CNN are also updated. To avoid visual clue leakage in the pre-training task of Masked RoI Classification with Linguistic Clues, the masking operation is applied to the input raw pixels, other than the feature maps produced by layers of convolution. VL-BERT is pre-trained on Cenceptual Captions Dataset (visual-linguistic), Books Corpus (text), English Wikipedia (text). The captions of CC is clauses that are too simple and short which leads overfitting. So, VL-BERT is also pretrained on text-only Books Corpus and the English Wikipedia datasets. 

### Supplements
```
1. Token Embedding: a special token which is assigned for each of the inputs (i.e. [CLS], [WORD], [MASK], [SEP], [IMG], [END]).
2. Visual Feature Embedding: Visual feature embeddings are combined form of visual appearance features and visual geometry embeddings. The visual appearance feature is extracted by applying a Fast R-CNN detector as the visual feature embedding (of 2048-d in paper). The visual geometry embedding is designed to inform VL-BERT the geometry location of each input visual element in image. Each RoI is characterized by a 4-d vector, as ( xLT , yLT , xRB , hRB ). Following the practice in Relation Networks (Hu et al., 2018), the 4-d vector is embedded into a high-dimensional representation (of 2048-d in paper) by computing sine and cosine functions of different wavelengths. The visual feature embedding is added to all the inputs, which is an output of a fully connected layer taking the concatenation of visual appearance feature and visual geometry embedding as input.
3. Segment Embedding:  A learned segment embedding is added to every input element for indicating which segment it belongs to. There are three types of segments in VL-BERT: A, B, C. They are defined to separate input elements from different sources. A and B are defined for the words from the first and second input sentence, and C for the RoIs from the input image. For QA the format is <Question, Answer, Image> where A denotes Question, B denotes Answer, and C denotes Image. For Image-Caption task, the input format is <Caption, Image> where A denotes Caption, and C denotes Image.
4. Sequence Position Embedding: Same in BERT. Each input element indicated its order in the input sequence and the learnable sequence position embedding is added to every input element indicating its order in the input sequence. The sequence position embedding for all visual elements are the same since there is no natural order among input visual elements.
```
![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/vlbert.png)



## Visual-Bert
[r221031:1908:UCLA:816cite:VisualBERT:A Simple and Performant Baseline for Vision and Language.pdf](https://arxiv.org/pdf/1908.03557.pdf)

VisualBERT is a pre-trained model build on stack of Transformer layers image and text inputs are jointly processed with self-attention. It consist of seperate embeddings as language (text) embeddings and visual (image) embeddings. Language embeddings follows the Bert's embedding structure which is the sum of: token embeddings, segment embeddings, and position embeddings. In addition to Bert, they introduced visual embeddings by summing: a visual feature representation of the bounding region, a segment embedding indicating it is an image embedding, a position embedding which is used when alignments between words and bounding regions are provided as part of the input, and set to the sum of the position embeddings corresponding to the aligned words (what?). Object proposals are extracted by using Faster-RCNN. VisualBert is trained on COCO image caption dataset that contains 5 captions per image. which It is pre-trained on two visually grounded tasks: masked language modeling and sentence image alignments. In MLM, part of the text is masked randomly and model tries to predict the masked words by using remaining words and the visual context coming from the corresponding image. In sentence-image prediction objective, the model is trained to decide wheter the given text matches the image or not. VisualBERT is evaluated on four different datasets: VQA v2.0 (question answering), VCR (visual reasoning), NLVR (visual reasoning), and Flickr30k (region-to-phrase grounding.)

![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/visualbert.png)



## Lxmert
[r221114:1908:UNC:1130cite:LXMERT:Learning Cross-Modality Encoder Representations from Transformers.pdf](https://arxiv.org/pdf/1908.07490.pdf)

Lxmert built on two single-modal network architectures that is used for source sentences and images respectively and a follow-up cross-modal Encoder combines these two modality. Lxmert consist of three encoders: object relationship encoder, language encoder, and cross-modality encoder. It is pre-trained on 5 different tasks: masked language modeling, masked object prediction via RoI-feature regressin, masked objectd predicition via detected-label classification, cross-modallity matching, and image question answering. On the other hand, is tis pre-trained on MS COCO(captioning), Visual Genome (captioning), VQA v2.0 (question answering), GQA (question answering), and VG-QA (question answering) datasets. They pre-train all the encoders and embedding layers from scratch, they didn't use any LLM embeddings for initialization. They used WordPiece tokenizer as in Bert for language side before giving them to the language encoder and they used 101-layer Faster R-CNN (pre-trained on Visual Genome) as feature extractor for image side before feeding them to object-relationship encoder. They accepted these detected labels output as ground truths. The outputs of the object-relationship encoder and language encoder are given to cross-modality encoder to align the features between these two modality to learn the joint representations. At the end, model produces three outputs as vision output (RoI Feature Regression + Detected-Label Classification), cross-modality output (Cross-Modality Matching + Q/A), and language output (Masked Cross-Modality LM). Pros: pre-trained on a massive visual question answering data. Cons: Single-stream architecture that contaions two single modal Transformer for vision and language, followed by one cross-modal Transformer for joint representation.

![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/lxmert.png)



## ViLBert
VilBert is a two-stream multimodal model that contains seperate networks to process image and text inputs. It fuses two different modality by using attention-based interactions, i.e. transformer layers. It is pre-trained on two tasks: Masked Multi-modal Modelling and Multi-modal Alignment Prediction. In Masked Multi-modal Modelling, %15 of the words and image regions in the input are masked tandomly. Model tries to reconstruct the masked feature from remaining words and regions. On the other hand, in Multi-modal Alignment Prediction, model tries to predict the correspondance of an image and text segment, i.e. whether the image content is described by the caption or not. They initilized the text-side of the VilBert model with BERT-based embeddings. The region features for image side is exctracted by using Faster R-CNN model that is pretrain on VG dataset. Then, pre-training is done on Conceptual Captions dataset by following these two proxy tasks. VilBert is evaluated on five different downstream task: Vusial Question Answering (on VQA 2.0 dataset), Visual Commensense Reasoning (on VCR dataset), Grounding Referring Expressions (RefCOCO+ dataset), Caption-Based Image Retrieval (Flickr30k dataset), ‘Zero-shot’ Caption-Based Image Retrieval (Flicker30k without fine-tunning). ViLBERT and LXMERT introduced the two-stream architecture, where two Transformers are applied to images and text independently, which is fused by a third Transformer in a later stage. 

![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/vilbert-model-1.png)
![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/vilbert-model-2.png)
![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/vilbert-model-3.png)



## Uniter
Uniter is a single model image-text representation model where the multimodal inputs are processed jointly for grounded understanding. Uniter is pre-trained on 4 different tasks: Masked Language Modeling (conditioned on image), Masked Region Modeling (conditioned on text), Image-Text Matching (ITM), and Word-Region Alignment (WRA). For more details in MRM, there are 3 different MRM variants are introduced: Masked Region Classification (MRC), Masked Region Feature Regression (MRFR), and Masked Region Classification with KL-divergence (MRC-kl). First, for a given a pair of image and text, an Image Embedder and a Text Embedder used to encode the image regions (visual features and bbox features) and text words (tokens, positions, and segments) into shared common embedding space. Inside the Image Embedder, a Faster-RCNN algorithm is used to extract the visual features (ROIs). Also, the 7-dimensional location features are encoded seperatly. Then these two visual and location features are given to two seperate Fully-connected layer. Then output of these FCs are summed to get the final visual embedding. On the other hand, inside the Text Embedder, sentences are first tokenied by using WordPiece tokenizer. Textual embedding is the sum of word embeddings, positional embeddings and segment embeddings. Finally, the summed embeddings are given to a Layer Normalization layer. Then, during pre-training, a Transformer model is used to learn the cross-modality contextualized embeddings for each region and each word. Uniter is pretrained on 4 datasets: COCO, Visual Genome (VG), Conceptual Captions (CC), and SBU Captions. Uniter achieves SoTA results on six VL tasks across nine datasets: VQA, VCR, NLVR, Visual Entailment, Image-Text Retrieval, and Referring Expression Comprehension. It beats all the previous task specific (MCAN, MaxEnt, B2T2, SCAN, EVE-image, MAttNet) and pre-trained models (ViLBERT, LXMERT, Unicoder-VL, VisualBERT, VL-BERT) and achieves SoTA on a wide range of VL benchmarks, outperforming existing multimodal pre-training methods by a large margin.

### Supplements
```
1. Its mentioned that: "training on additional CC and SBU data (containing unseen images/text in downstream tasks) further boosts model performance over training on COCO and VG only.".
2. It is also worth mentioning that both VilBERT and LXMERT observed two-stream two-stream model outperforms single-stream model, while our results show empirically that with our pre-training setting, single-stream model can achieve new state-of-the-art results, with much fewer parameters (UNITER-base: 86M, LXMERT: 183M, VilBERT: 221M).
3. Compared with previous work on multimodal pre-training: (i) our masked language/region modeling is conditioned on full observation of image/text, rather than applying joint random masking to both modalities; (ii) they introduce a novel WRA pre-training task via the use of Optimal Transport (OT) to explicitly encourage fine-grained alignment between words and image regions. 
```

![alt text](https://github.com/emrecanacikgoz/papers/blob/main/multimodal/figs/uniter.png)

## BLIP
[2201:salesforces:BLIP:Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation.pdf](https://arxiv.org/abs/2201.12086)

Pre-training Dataset: 
1. COCO 
2. Visual Genome (VG)
3. Conceptual Captions (CC3M)
4. Conceptual Captions 12M (CC12M)
5. SBU
6. LAION

Downstream Evaluations:
1. Image-Text Retrieval: They evaluated BLIP both for image-to-text and text-to-image retrieval tasks on COCO and Flickr30K. They used ITC and ITM losses during fine-tuning.
2. Image Captioning: They have used two datatsets during fine-tuning: COCO and No-Caps with LM loss. They also add "a picture of" prompt at the begining of each caption.
3. Visual Question Answering (VQA): 
4. Natural Language Visual Reasoning (NLVR2)
5. Visual Dialog (VisDial)

Model
1. Image Encoder:
2. Text Encoder:
3. Image Grounded Text Encoder:
4. Image Grounded Text Decoder:

Pre-training Objectives (Loss)
1. Image-Text Contrastive Loss
2. Image-Text Matching
3. Image-conditioned LM

Dataset
1. CapFlit
    1. Captioner:
    2. Filter:


Downstream

## Unicoder-VL
To-Do

## CLIP
To-Do

## Vokenizer
To-Do

# Readings Left
- [ ] MCB, 2017
- [ ] DFAF, 2019
- [ ] BAN, 2018
- [ ] B2T2, 2019
- [ ] Villa, 2020
- [ ] SCAN, 2018
- [ ] MattNet, 2018
- [ ] VLP, 2020
- [ ] VinVL, 2021
- [ ] Florence, 2021
- [ ] BLIP, 2022
- [ ] Glide, 2022
- [ ] SimVLM, 2022
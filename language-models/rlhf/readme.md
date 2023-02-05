# RLHF

## InstructGPT [2201]()
Making bigger models does not always results with better and more meaningful results; LMs can generate outputs that are untruthful, toxic, or simply not helpful to the user which are not aligned with their users. It would be great if we use human feedback for generated text as a measure of performance or go even one step further and use that feedback as a loss to optimize the model. InstructGPT fine-tunes the language models to align it with the human prompts by using reinforcement learning from human feedback (RLHF). In this way, it uses human preferences as a reward signal to fine-tune the language models.
### How it works?
1. Collect a dataset of human-written demonstrations of the desired output behavior on prompts submitted to the OpenAI API and some labeler-written prompts.
2. Use this data to train the supervised learning baselines.
3. Collect a dataset of human-labeled comparisons between outputs from the models.
4. Train a reward model (RM) on this dataset to predict which model output the labelers would prefer.
5. Use this RM as a reward function and fine-tune the supervised learning baseline to maximize this reward using the PPO algorithm.
OR
1. **Step 1:** Collect demonstration data, and train a supervised policy. Our labelers provide demon- strations of the desired behavior on the input prompt distribution. Then fine-tune a pretrained GPT-3 model on this data using supervised learning.
2. **Step 2:** Collect comparison data, and train a reward model. They collect a dataset of comparisons between model outputs, where labelers indicate which output they prefer for a given input. Then train a reward model to predict the human-preferred output.
3. **Step 3:** Optimize a policy against the reward model using PPO. Use the output of the RM as a scalar reward and fine-tune the supervised policy to optimize this reward using the PPO algorithm.
### Datasets
- **SFT Datasets:** Labeler demonstrations used to train our SFT models, 13k training prompts (from the API and labeler-written).
- **RM Datasets:** Labeler rankings of model outputs used to train our RMs, has 33k training prompts (from the API and labeler-written).
- **PPO Datasets:** Without any human labels, which are used as inputs for RLHF fine-tuning, has 31k training prompts (only from the API).


## Readings
### Fine-Tuning Language Models from Human Preferences [1909](https://arxiv.org/pdf/1909.08593.pdf)
### Learning to summarize from human feedback [2009](https://arxiv.org/pdf/2009.01325.pdf)

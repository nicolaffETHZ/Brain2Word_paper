# nicolaff_NLPBrain

This repository contains the code to reproduce the results from the paper **[Brain2Word:Decoding Brain Activity for Language Generation](https://arxiv.org/abs/2009.04765)**. 

## Setup

Install the required libraries from the requirements.txt file.

Download the fMRI data for all the subjects from the dataset webpage of the paper **[Toward a universal decoder of linguistic meaning from brain activation](https://evlab.mit.edu/sites/default/files/documents/index2.html)** and store them in the subjects subfolder.
Also download the GloVe embedding vectors from the same webpage if they aren't available yet in the glove_data subfolder.

If the look_ups_gpt-2 folder does not contain five files, you also need to download the glove.42B.300d.txt file from the webpage **[GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)** and store it in the data folder.
Additionally if the words_fMRI.txt does not exist you need to download some preprocessed text (and store under the file name words_fMRI.txt), from which the GPT-2 model will use snippets as the context for further text generation. (In our case the harry potter books were used and the cleaned text can be found in this  **[github](https://github.com/khushmeeet/potter-nlp/tree/master/final_data)**)
Finally if there is no word_sets.txt file you need to generate word sets first. This can be done by printing the top 5 words predicted by our classification model into a text file named word_sets.txt. 

## Classification and GloVe prediction models

Except for the PCA&XGBoost model all models have two set ups. One version is the classification model and the other is the GloVe prediction model. This can be selected in the `run.sh` file with the parameter `-class_model` (0 for GloVe prediction model and 1 for the classification model).

Another parameter is the `-subject` parameter, which selectes for which subject the model is evaluated on. 

### PCA and XGBoost

This model uses in a first step PCA to reduce the amount of features of the fMRI vectors. With the reduced dataset a standard XGBClassifier is trained to predict the right word for the given fMRI input sample. 

### Small Model

This is a small MLP model and it consists only of two layers. The first layer reduces the fMRI input vector to an intermediate size. The ouput of this first layer is then either used in an additional layer to classify the matching word or predict the matching  GLoVe embedding for the given fMRI. 

### Big Model

This is a slightly bigger MLP model and uses in a first layer small individual layers for all the ROI regions of the fMRI. The outputs are then concatenated and used in an additional layer to classify the matching word or predict the matching  GLoVe embedding for the given fMRI. 

### VQ-VAE Model

This model uses the approach of the **[Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937)** paper. 

### Our Model

This model is our main model and combines multiple ideas like small dense layers for ROI regions and uses an autoencoder approach. The detailed model can be found in our paper.

## GPT-2 anchored generation

The only available parameter for this model is the `-constant` value in the `run_gpt.sh` file. This parameter decides, how strong the anchoring effect based on the given top 5 word set is on the text generation. A constant value of zero represents no anchoring. The higher the value is set, the stronger the anchoring effect becomes.

This model uses the regular GPT-2 model from the paper **[Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)**. The model is adapted that the word generation can be affected during inference time. This is done by a set of five words, which are used as anchores during the word sampling procedure.


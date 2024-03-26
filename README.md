# GestaltMML
The GestaltMML is a cutting-edge multimodal machine learning model integrating frontal facial images, demographic information of patients and clinical text data for the diagnosis of rare genetic disorders. It leverages the power of fine-tuning the Vision-and-Language Transformer (ViLT) (see [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)) using data from the [GestaltMatcher Database (GMDB)](https://db.gestaltmatcher.org). To gain access to the GMDB, interested individuals need to submit an application [here](https://db.gestaltmatcher.org/documents).

GestaltMML is distributed under 

## Package Installation
We need to first install the following required packages for model training and inference.
```
!conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
!pip install transformers dataset
```
## Training
Once the package has been installed and access to the GMDB has been granted, the following notebook can be executed to train the GestaltMML model. Note that...

## Inference
After completing the training process, proceed to the "Inference" section of the script to carry out inference. It is strongly advised to adhere to the guidelines provided [here](https://github.com/igsb/GestaltMatcher-Arc/tree/service?tab=readme-ov-file#crop-and-align-faces) or cropping and aligning frontal facial images prior to conducting the inference.

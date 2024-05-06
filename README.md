# GestaltMML
The GestaltMML is a cutting-edge multimodal machine learning model integrating frontal facial images, demographic information of patients and clinical text data for the diagnosis of rare genetic disorders. It leverages the power of fine-tuning the Vision-and-Language Transformer (ViLT) (see [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)) using data from the [GestaltMatcher Database (GMDB)](https://db.gestaltmatcher.org). To gain access to the GMDB, interested individuals need to submit an application [here](https://db.gestaltmatcher.org/documents).

GestaltMML is distributed under [MIT License by Wang Genmoics Lab](https://wglab.mit-license.org).

## Package Installation
We need to first install the following required packages for model training and inference.
```
conda create -n gestaltmml python=3.11
conda activate gestaltmml
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install transformers dataset
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=gestaltmml
```
## Training
Once the package has been installed and access to the GMDB has been granted, the following [GestaltMML_Training.ipynb](https://github.com/WGLab/GestaltMML/blob/main/GestaltMML_Training.ipynb) can be executed to train the GestaltMML model. Please be aware that it is necessary to download the hp.json (HPO dictionary) from [https://hpo.jax.org/app/data/ontology](https://hpo.jax.org/app/data/ontology), and the omim_summarized_1.0.9.json (summarized OMIM texts for data augmentation) is available for public access in this GitHub repository. The model weights will be made available upon request to those who have gained access to GMDB. Additionally, the notebook can be modified to conduct your own experiments. Please be aware that the current version of GestaltMML has been developed using GMDB (v1.0.9). The training script is designed for easy adaptation to subsequent versions of the GMDB.

## Inference
For inference using GestaltMML, please see the notebook [GestaltMML_Inference.ipynb](https://github.com/WGLab/GestaltMML/blob/main/GestaltMML_Inference.ipynb) for detailed instruction. 
If you want to simply load GestaltMML on your local machine for inference, the model weights GestaltMML_model.pt are saved in [GestaltMML_model_weights.zip](https://github.com/WGLab/GestaltMML/releases/download/v1.0.9/GestaltMML_model_weights.zip). In addition, disease_dict.json, three sample test images and sample texts (in GestaltMML_input.csv) are provided in the folder [inference](inference) in this Github page. Please use the following command:
```
python inference.py --csv_file path/to/GestaltMML_input.csv --base_image_path path/to/test_images --model_path path/to/GestaltMML_model.pt --disease_dict_path path/to/disease_dict.json --top_n 5
```
It is strongly advised to adhere to the guidelines provided [here](https://github.com/igsb/GestaltMatcher-Arc/tree/service?tab=readme-ov-file#crop-and-align-faces) for cropping and aligning frontal facial images prior to conducting the inference.
## Citation
Wu, D., Yang, J., Liu, C., Hsieh, T.C., Marchi, E., Krawitz, P., Weng, C., Chung, W., Lyon, G.J., Krantz, I.D., Kalish, J.M. and Wang, K., 2024. GestaltMML: Enhancing Rare Genetic Disease Diagnosis through Multimodal Machine Learning Combining Facial Images and Clinical Texts. arXiv preprint [arXiv:2312.15320](https://arxiv.org/pdf/2312.15320).

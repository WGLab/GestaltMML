import argparse
import pandas as pd
import torch
import json
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

def load_model(model_path, device):
    num_list = [int(i) for i in range(528)]
    label2id = dict(zip(num_list,num_list))
    id2label = dict(zip(num_list,num_list))
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=528,id2label=id2label,label2id=label2id)
    if device == "cpu":
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def load_disease_dict(dictionary_path):
    with open(dictionary_path) as json_file:
        disease_dict = json.load(json_file)
    return disease_dict

def prepare_data(csv_file, base_image_path):
    data = pd.read_csv(csv_file)
    data['filename'] = data['filename'].apply(lambda x: f"{base_image_path}/{x}")
    data['label'] = 0  #label is not important for inference
    test_questions = []
    for i in range(len(data)):
        temp_dic = {'image_id':data.loc[i,'image_id'],'question':data.loc[i,'texts']}
        test_questions.append(temp_dic)
    test_annotations = []
    for i in range(len(data)):
        temp_dic = {'labels':[0],'scores':[1]}
        test_annotations.append(temp_dic)
    filename_to_id = {data.loc[i,'filename']: data.loc[i,'image_id'] for i in range(len(data))}
    id_to_filename = {v:k for k,v in filename_to_id.items()}
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    num_list = [int(i) for i in range(528)]
    label2id = dict(zip(num_list,num_list))
    id2label = dict(zip(num_list,num_list))
    class VQADataset(torch.utils.data.Dataset):
        def __init__(self, questions, annotations, processor):
            self.questions = questions
            self.annotations = annotations
            self.processor = processor
        def __len__(self):
            return len(self.annotations)
        def __getitem__(self, idx):
            annotation = self.annotations[idx]
            questions = self.questions[idx]
            image = Image.open(id_to_filename[questions['image_id']])
            text = questions['question']
            encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
            for k,v in encoding.items():
                encoding[k] = v.squeeze()
            labels = annotation['labels']
            scores = annotation['scores']
            targets = torch.zeros(len(id2label))
            for label, score in zip(labels, scores):
                targets[label] = score
            encoding["labels"] = targets
            return encoding
    test_dataset = VQADataset(questions = test_questions,
                     annotations = test_annotations,
                     processor=processor)
    return test_dataset

def run_inference(model, test_dataset, device, disease_dict, top_n):
    predicted_diseases = []
    for i in range(len(test_dataset)):
        example = test_dataset[i]
        example = {k: v.unsqueeze(0).to(device) for k,v in example.items()}
        outputs = model(**example)
        logits = outputs.logits
        if top_n == 1:
            predicted_classes = [logits.argmax(-1).item()]
        else:
            predicted_classes = torch.topk(logits, top_n).indices.flatten().tolist()
        diseases = [disease_dict[str(i)] for i in predicted_classes]
        predicted_diseases.append(diseases)
    return predicted_diseases

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = load_model(args.model_path, device)
    disease_dict = load_disease_dict(args.disease_dict_path)
    data = prepare_data(args.csv_file, args.base_image_path)
    predictions = run_inference(model, data, device, disease_dict, args.top_n)
    for i, pred in enumerate(predictions):
        print(f"Data {i + 1}: Top-{args.top_n} predicted diseases: {pred}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on GestaltMML for diagnosing rare genetic diseases.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file with image ids, image file names and corresponding texts')
    parser.add_argument('--base_image_path', type=str, required=True, help='Base path to the folder of images')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the GestaltMML model weights')
    parser.add_argument('--disease_dict_path', type=str, required=True, help='Path to the disease dictionary JSON file')
    parser.add_argument('--top_n', type=int, default=1, help='Number of top predicted diseases to return')
    args = parser.parse_args()

    main(args)

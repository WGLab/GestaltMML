import argparse
import pandas as pd
import torch
import json
from transformers import ViltProcessor, ViltForQuestionAnswering

def load_model(model_path, device):
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm", num_labels=528)
    if device == "cpu":
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def load_disease_dict(dictionary_path):
    with open(dictionary_path, 'r') as file:
        disease_dict = json.load(file)
    return disease_dict

def prepare_data(csv_file, base_image_path):
    data = pd.read_csv(csv_file)
    data['filename'] = data['filename'].apply(lambda x: f"{base_image_path}/{x}")
    data['label'] = 0  #label is not important for inference
    return data

def run_inference(model, data, device, disease_dict, top_n):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")
    predicted_diseases = []
    for index, row in data.iterrows():
        test_dataset = processor(row['filename'], row['texts'], return_tensors="pt").to(device)
        outputs = model(**test_dataset)
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
    parser = argparse.ArgumentParser(description='Run inference on GestaltMML for diagnosing genetic diseases.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file with image file names and corresponding texts')
    parser.add_argument('--base_image_path', type=str, required=True, help='Base path to the image files')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model file')
    parser.add_argument('--disease_dict_path', type=str, required=True, help='Path to the disease dictionary JSON file')
    parser.add_argument('--top_n', type=int, default=1, help='Number of top predicted diseases to return')
    args = parser.parse_args()

    main(args)

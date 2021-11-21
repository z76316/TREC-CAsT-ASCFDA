import json
import argparse
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from dataset import t5_generation_dataset, collate_base
from functools import partial
from torch import nn, optim
import os 

parser = argparse.ArgumentParser(description='Process and train rerank data.')
parser.add_argument('--data_path')
parser.add_argument('--config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as fp:
    config = json.load(fp)
    
# model
model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(config["device"])
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
optimizer = optim.Adam(model.parameters(), lr=config["lr"])
collate_fn = partial(
    collate_base, tokenizer=tokenizer, max_input_len=config["max_input_len"], \
    max_label_len=config["max_label_len"], device=config["device"]
)# assign tokenizer to collate_fn

def get_dataset_and_dataloader(merged_inputs, labels, config):
    # dataset, dataloader 
    dataset = t5_generation_dataset(merged_inputs, labels)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,num_workers=0, collate_fn=collate_fn)
    return dataset, dataloader

def train(dataloader, model, optimizer=optimizer):
    model.train()
    for i_batched, (inputs, labels) in enumerate(dataloader):
        loss = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels.input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    for i in tqdm(range(config["train_epoch"])):
        count = 0
        merged_inputs = []
        labels = []
        with open(args.data_path, 'r') as fp_in:
            for line in tqdm(fp_in):
                line = line.strip("\n")
                query, pos_doc, neg_doc = line.split("\t")
                pos_pair = f"Query: {query} Document: {pos_doc} Relevant:"
                pos_label = "true"
                neg_pair = f"Query: {query} Document: {neg_doc} Relevant:"
                neg_label = "false"
                merged_inputs += [pos_pair, neg_pair]
                labels += [pos_label, neg_label]
            
                count += 1
                if count % config["data_split"] == 0:
                    dataset, dataloader = get_dataset_and_dataloader(merged_inputs, labels, config)
                    train(dataloader, model, optimizer)
                    merged_inputs = []
                    labels = []
        if merged_inputs and labels: # not null
            dataset, dataloader = get_dataset_and_dataloader(merged_inputs, labels, config)
            train(dataloader, model, optimizer)
            merged_inputs = []
            labels = []
    
    model_dir = os.path.dirname(config["model_output_path"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(config["model_output_path"])

    
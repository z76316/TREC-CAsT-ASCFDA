import json
import argparse
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.utils.data import Dataset, DataLoader
from dataset import t5_generation_dataset, collate_base
from functools import partial
from torch import nn, optim
from tqdm import tqdm
import os 

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path')
parser.add_argument('--config_path')
args = parser.parse_args()

with open(args.config_path, 'r') as fp:
    config = json.load(fp)

with open(args.data_path, 'r') as fp:
    data = json.load(fp)
    
merged_inputs = []
labels = []
for each_query in tqdm(data):
    # order: metadata ||| ... ||| query to rewrite
    merged_inputs.append("|||".join(each_query["History"] + [each_query["Question"]]))
    labels.append("What group disbanded?")

    
# model
model = T5ForConditionalGeneration.from_pretrained(config["model_name"]).to(config["device"])
tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
optimizer = optim.Adam(model.parameters(), lr=config["lr"])

# dataset, dataloader 
dataset = t5_generation_dataset(merged_inputs, labels)
collate_fn = partial(
    collate_base, tokenizer=tokenizer, max_input_len=config["max_input_len"], \
    max_label_len=config["max_label_len"], device=config["device"]
)# assign tokenizer to collate_fn
dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True,num_workers=0, collate_fn=collate_fn)


def train(dataloader, model, optimizer=optimizer):
    model.train()
    for i_batched, (inputs, labels) in enumerate(dataloader):
        loss = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels.input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    for i in tqdm(range(config["train_epoch"])):
        train(dataloader, model, optimizer)
    
    model_dir = os.path.dirname(config["model_output_path"])
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model.save_pretrained(config["model_output_path"])

    
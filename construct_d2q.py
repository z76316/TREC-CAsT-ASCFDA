import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Generate potential queries with t5.')
parser.add_argument('--collections', help='the path to the collections')
parser.add_argument('--output', help='the path to the rewritten collections')
args = parser.parse_args()


# doc2-t5-query
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')
model.to(device)

delimeters = [",", ".", "?", "!"]
with open(args.collections, "r") as fp_in:
    with open(args.output, "w") as fp_out:
        for line in tqdm(fp_in):
            line = line.strip("\n")
            p_id, text = line.split("\t")
            # split text into sentences
            for d in delimeters:
                text.replace(d, ",")
            sentences = text.split(",")
            for sentence in sentences:
                input_ids = tokenizer.encode(sentence, return_tensors='pt').to(device)
                output = model.generate(
                    input_ids=input_ids,
                    max_length=64,
                    do_sample=True,
                    top_k=10,
                    num_return_sequences=1)[0]
                potential_query = tokenizer.decode(output, skip_special_tokens=True)
                fp_out.write(f"{p_id}\t{sentence}\t{potential_query}\n")
    
        
    

            
        

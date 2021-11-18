import argparse
from tqdm import tqdm
import hnswlib
import numpy as np

parser = argparse.ArgumentParser(description='Build dense index')
parser.add_argument('--potential_q', help='the path to the collections')
parser.add_argument('--index', help='the path of the index')
parser.add_argument('--dim', help='the dimension of transformer', default=384)
args = parser.parse_args()

potential_queies = []
with open(potential_q) as fp:
    for index, line in enumerate(fp):
        line = line.strip("\n")
        p_id, sentence, p_q = line.split("\t")
        potential_queies.append(p_q)
        sentence_list.append(sentence)
query_embeddings = model.encode(potential_queies)
num_elements = len(potential_queies)

p = hnswlib.Index(space='cosine', dim=args.dim)
p.init_index(max_elements=num_elements, ef_construction=100, M=16)
p.set_ef(10)
p.set_num_threads(4)
p.add_items(query_embeddings)
p.save_index(args.index)
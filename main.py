import argparse
from pyserini.search import SimpleSearcher
from chatty_goose.cqr import Ntr
from chatty_goose.settings import NtrSettings
from chatty_goose.pipeline import RetrievalPipeline
from pygaggle.rerank.base import Query, Text, hits_to_texts
from pygaggle.rerank.transformer import MonoT5
from sentence_transformers import SentenceTransformer
import hnswlib
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Retrieve documents with CQR querues.')
parser.add_argument('--index', help='the path to the Lucene index')
parser.add_argument('--queries', help='the path to the queries')
parser.add_argument('--r1', help='the number to retrieve in first stage', default=20)
parser.add_argument('--r2', help='the number to retrieve in second stage', default=10)
parser.add_argument('--potential_q', help='the path to the potentail queries')
parser.add_argument('--ser_index', help='the path of ser index')
parser.add_argument('--ser_dim', help='the dimension of ser index', default=384)
parser.add_argument('--output', help='the path to the rewritten queries')
parser.add_argument('--k1', default=0.82)
parser.add_argument('--b', default=0.68)
parser.add_argument('--ser_high', default=0.9)
parser.add_argument('--ser_low', default=0.8)
args = parser.parse_args()

searcher = SimpleSearcher(args.index)
# set BM25 (k1, b)
searcher.set_bm25(args.k1, args.b)

# set Ntr method to rewrite queries
ntr = Ntr(NtrSettings())

# Create a new RetrievalPipeline
# We don't rerank here for it's only retrieval
# rp = RetrievalPipeline(searcher, dense_searcher=None, reformulators=[ntr], searcher_num_hits=args.r1, reranker=None)


# load potential queries and sentences
ser_sentences = []
with open(args.potential_q) as fp:
    for index, line in enumerate(fp):
        line = line.strip("\n")
        p_id, sentence, p_q = line.split("\t")
        ser_sentences.append((p_id, sentence))
        # sentence_list.append(sentence)
        
def cos_sim(a, b): # 1D, 1D
    return np.sum(a*b)/max(np.sum(a*a)*np.sum(b*b), 0.000000001) 

# ser index
p = hnswlib.Index(space='cosine', dim=args.ser_dim)
p.load_index(args.ser_index, max_elements=len(ser_sentences))
p.set_ef(10)
p.set_num_threads(4)

# models
ser_model = SentenceTransformer('all-MiniLM-L6-v2')
reranker =  MonoT5() 

current_turn = -1
historical_queries = []
with open(args.queries, "r") as fp_in:
    with open(args.output, "w") as fp_out:
        for line in tqdm(fp_in):
            line = line.strip("\n")
            q_id, text = line.split("\t")
            turn, count = q_id.split("_")
            if turn != current_turn:
                # reset historical queries
                historical_queries = []
            historical_queries.append(text)
            merged_text = "|||".join(historical_queries)
            rewritten_query = ntr.rewrite(merged_text)
            
            # stage 1 retrieve
            print("stage 1 retrieve")
            cqr_query = Query(rewritten_query)
            stage_1_hits = searcher.search(cqr_query.text, k=args.r1)
            stage_1_doc_ids = [h.docid for h in stage_1_hits]
            
            # stage 2 retrieve
            print("stage 2 retrieve")
            cqr_query_embed = ser_model.encode(cqr_query.text)
            h_query_match = []
            if len(historical_queries) > 0:
                historical_query_embed = ser_model.encode(historical_queries)
                for index, embed in enumerate(historical_query_embed):
                    # print("cos sim: ", cos_sim(embed, cqr_query_embed))
                    if args.ser_high > cos_sim(embed, cqr_query_embed) > args.ser_low:
                        h_query_match.append(historical_queries[index])
            labels, distances = p.knn_query(cqr_query_embed, k=100)
            labels = labels[0]
            distances = distances[0]
            labels = [label for label, distance in zip(labels, distances) if args.ser_high >= distance >= args.ser_low ]
            topic_sentences_match = [ser_sentences[label][1] for label in labels if ser_sentences[label][0] in stage_1_doc_ids]
            merged_query = " ".join([cqr_query.text] + h_query_match+ topic_sentences_match)
            ser_query = Query(merged_query)
            stage_2_hits = searcher.search(ser_query.text, k=args.r2)
            stage_2_doc_ids = [h.docid for h in stage_2_hits]
            
            # rerank
            print("rerank")
            texts = hits_to_texts(stage_2_hits)
            reranked = reranker.rerank(cqr_query, texts)
            for i in range(args.r2):
                fp_out.write(f"{q_id}\t{reranked[i].metadata['docid']}\t{reranked[i].text}")
                # print(f'{i+1:2} {reranked[i].metadata["docid"]} {reranked[i].score:.5f} {reranked[i].text}')
            
        

import random
import argparse
import os
import json
import csv
import pdb
import sys

from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager
from flexneuart.retrieval.fwd_index import get_forward_index

COLLECTION = os.getenv('COLLECTION')
COLLECTION_ROOT = os.environ.get('COLLECTION_ROOT')

if COLLECTION=='':
    print("Please export COLLECTION. Example -> export COLLECTION=msmarco_pass")
    sys.exit(1)

if COLLECTION_ROOT=='':
    print("Please export COLLECTION_ROOT. Example -> export COLLECTION_ROOT=/home/ubuntu/efs/capstone/data")
    sys.exit(1)


configure_classpath()

from flexneuart.retrieval.cand_provider import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_data', type=str, required=True)
    parser.add_argument('--output_train_pairs', type=str, required=True)
    parser.add_argument('--output_qrels', type=str, required=True)
    # parser.add_argument('--output_ids', type=str, required=True)
    # parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--index', type=str, default='')
    # parser.add_argument('--max_hits', type=int, default=1000)
    # parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--max_docs_retrieved', type=int, default=100)
    parser.add_argument('--new_query_id', type=int, default=670000)
    args = parser.parse_args()

    # create a resource manager
    resource_manager=create_featextr_resource_manager(resource_root_dir=f'{COLLECTION_ROOT}/{COLLECTION}/', fwd_index_dir='forward_index', model1_root_dir=f'derived_data/giza', embed_root_dir=f'derived_data/embeddings')
                                                  
    # create a candidate provider/generator
    cand_prov = create_cand_provider(resource_manager, PROVIDER_TYPE_LUCENE, args.index)
    generated_query_id = args.new_query_id

    with open(args.input) as new_queries_file, open(args.output_data, 'w') as output_data, open(args.output_train_pairs, 'w') as output_train_pairs, open(args.output_qrels, 'w') as output_qrels:
        #json_array = json.load(new_queries_file
        tsv_writer = csv.writer(output_data, delimiter='\t')
        tsv_writer_pairs = csv.writer(output_train_pairs, delimiter='\t')
        tsv_writer_qrels = csv.writer(output_qrels, delimiter='\t')

        for line in new_queries_file:
            query_line = json.loads(line)
            new_query = query_line['question']
            positive_doc_id = query_line['doc_id']
            positive_doc_text = query_line['doc_text']
            #log_probs = query_line['log_probs']

            query_response = run_text_query(cand_prov, args.max_docs_retrieved, new_query) # response = (1329, [CandidateEntry(doc_id='639661', score=18.328275680541992),...])
            no_of_responses =  query_response[0]
            if no_of_responses == 0:
                continue
            #pdb.set_trace()
            print("No of responses: "+str(no_of_responses))
            random_number = random.randint(0, min(no_of_responses, args.max_docs_retrieved))
            print("Random number: "+ str(random_number))
            negative_doc = query_response[1][random_number]
            negative_doc_id = negative_doc.doc_id

            # TODO: get negative_doc_text from code using doc_id
            #fwd_index = resource_manager.getFwdIndex('text_raw')
            raw_index = get_forward_index(resource_manager, 'text_raw')
            negative_doc_text = raw_index.get_doc_text_raw(negative_doc_id)
            #negative_doc_text = fwd_index.getDocEntryTextRaw(negative_doc_id)

            # set query id to a random large number
            generated_query_id = generated_query_id + 1
            query_id = random.randint(generated_query_id, generated_query_id)
            
            query_row = ['query']
            query_row.append(query_id)
            query_row.append(new_query)
            tsv_writer.writerow(query_row)

            doc_row_neg = ['doc']
            doc_row_neg.append(negative_doc_id)
            doc_row_neg.append(negative_doc_text)
            tsv_writer.writerow(doc_row_neg)

            doc_row_pos = ['doc']
            doc_row_pos.append(positive_doc_id)
            doc_row_pos.append(positive_doc_text)
            tsv_writer.writerow(doc_row_pos)

            tsv_writer_pairs.writerow([query_id, positive_doc_id])
            tsv_writer_pairs.writerow([query_id, negative_doc_id])

            tsv_writer_qrels.writerow([query_id, 0, positive_doc_id, 1])
            tsv_writer_qrels.writerow([query_id, 0, negative_doc_id, -1])
            
        
        

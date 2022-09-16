import random
import argparse
import os
import json
import csv
from flexneuart.retrieval.cand_provider import *
from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_data', type=str, required=True)
    parser.add_argument('--output_train_pairs', type=str, required=True)
    parser.add_argument('--output_qrels', type=str, required=True)
    # parser.add_argument('--output_ids', type=str, required=True)
    # parser.add_argument('--corpus', type=str, required=True)
    parser.add_argument('--index', type=str, default='msmarco-passage')
    # parser.add_argument('--max_hits', type=int, default=1000)
    # parser.add_argument('--n_samples', type=int, default=100)
    args = parser.parse_args()

    # Get environment variables
    COLLECTION = os.getenv('COLLECTION')
    COLLECTION_ROOT = os.environ.get('COLLECTION_ROOT')

    # add Java JAR to the class path --> for setting java classpath
    configure_classpath()

    # create a resource manager
    resource_manager=create_featextr_resource_manager(resource_root_dir=f'{COLLECTION_ROOT}/{COLLECTION}/') # optional params -> fwd_index_dir='forward_index', model1_root_dir=f'derived_data/giza', embed_root_dir=f'derived_data/embeddings')
                                                  
    # create a candidate provider/generator
    cand_prov = create_cand_provider(resource_manager, PROVIDER_TYPE_LUCENE, args.index)


    with open(args.input) as new_queries_file, open(args.output_data, 'w') as output_data, open(args.output_train_pairs, 'w') as output_train_pairs, open(args.output_qrels, 'w') as output_qrels:
        json_array = json.load(new_queries_file)
        tsv_writer = csv.writer(output_data, delimiter='\t')
        tsv_writer_pairs = csv.writer(output_train_pairs, delimiter='\t')
        tsv_writer_qrels = csv.writer(output_qrels, delimiter='\t')

        for query_line in json_array:
            new_query = query_line['question']
            positive_doc_id = query_line['doc_id']
            positive_doc_text = query_line['doc_text']
            log_probs = query_line['log_probs']

            query_response = run_text_query(cand_prov, 20, new_query) # response = (1329, [CandidateEntry(doc_id='639661', score=18.328275680541992),...])
            no_of_responses =  query_response[0]
            random_number = random.randint(0, no_of_responses)
            negative_doc = query_response[0][random_number]
            negative_doc_id = negative_doc.doc_id

            # TODO: get it from code
            negative_doc_text = "get_negative_doc_text"

            # set query id to a random large number
            query_id = random.randint(670000, 6700000)
            
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
            
        
        

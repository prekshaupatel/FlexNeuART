from flexneuart.retrieval.cand_provider import *
import random
import argparse
import os
import json
from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
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


    with open(args.input) as new_queries_file, open(args.output, 'w') as new_triples_file, open(args.output_ids, 'w') as fout_ids:
        json_array = json.load(new_queries_file)
        
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

            # TSV -> query/doc, id, string of query/doc
            new_triples_file.write(f'{new_query}\t{positive_doc_id}\t{negative_doc_id}\n')

        
        

from flexneuart.retrieval.cand_provider import *
import random
import argparse
import os
from flexneuart import configure_classpath
from flexneuart.retrieval import create_featextr_resource_manager

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    # parser.add_argument('--output_ids', type=str, required=True)
    # parser.add_argument('--corpus', type=str, required=True)
    # parser.add_argument('--index', type=str, default='msmarco-passage')
    # parser.add_argument('--max_hits', type=int, default=1000)
    # parser.add_argument('--n_samples', type=int, default=100)

    # Get environment variables
    COLLECTION = os.getenv('COLLECTION')
    COLLECTION_ROOT = os.environ.get('COLLECTION_ROOT')

    # add Java JAR to the class path --> dont know why this is used?
    configure_classpath()

    # create a resource manager
    resource_manager=create_featextr_resource_manager(resource_root_dir=f'{COLLECTION_ROOT}/{COLLECTION}/',
                                                  fwd_index_dir='forward_index',
                                                  model1_root_dir=f'derived_data/giza',
                                                  embed_root_dir=f'derived_data/embeddings')
    
    # create a candidate provider/generator
    cand_prov = create_cand_provider(resource_manager, PROVIDER_TYPE_LUCENE, f'lucene_index')

    args = parser.parse_args()

    with open(args.input) as new_queries_file, open(args.output, 'w') as new_triples_file, open(args.output_ids, 'w') as fout_ids:
        for query_line in new_queries_file.readlines():
            query = query_line.strip()
            positive_doc_id = "Find from input"
            query_response = run_text_query(cand_prov, 20, query) # response = (1329, [CandidateEntry(doc_id='639661', score=18.328275680541992),...])
            no_of_responses =  query_response[0]
            random_number = random.randint(0, no_of_responses)
            negative_doc = query_response[0][random_number]
            negative_doc_id = negative_doc.doc_id

            new_triples_file.write(f'{query}\t{positive_doc_id}\t{negative_doc_id}\n')

        
        

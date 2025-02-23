#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""
    This script converts raw query files from DPR repository into the FlexNeuArt internal format.

    In addition, it can generate BITEXT data to train Model 1.
"""

import os
import json
import argparse
import tqdm

from flexneuart.io import FileWrapper
from flexneuart.io.qrels import write_qrels, add_qrel_entry
from flexneuart.io.stopwords import read_stop_words, STOPWORD_FILE
from flexneuart.text_proc.parse import SpacyTextParser, Sentencizer, get_retokenized, add_retokenized_field
from flexneuart.data_convert import add_bert_tok_args, create_bert_tokenizer_if_needed, \
                    OUT_BITEXT_PATH_OPT, OUT_BITEXT_PATH_OPT_META, OUT_BITEXT_PATH_OPT_HELP

from flexneuart.config import TEXT_BERT_TOKENIZED_NAME, \
    TEXT_FIELD_NAME, DOCID_FIELD, \
    TEXT_RAW_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME, \
    SPACY_MODEL, \
    BITEXT_QUESTION_PREFIX, BITEXT_ANSWER_PREFIX,\
    ANSWER_LIST_FIELD_NAME
from flexneuart.data_convert.wikipedia_dpr import dpr_json_reader, get_passage_id

def parse_args():
    parser = argparse.ArgumentParser(description='Script converts raw query files from the DPR repository '
                                                 'into the FlexNeuArt internal format.')
    parser.add_argument('--input', metavar='input file',
                        help='input file',
                        type=str, required=True)
    parser.add_argument('--part_type', metavar='partion type (unique)',
                        type=str, required=True,
                        help='A unique partition type, which will be used as a prefix for all query IDs, must be unique!')
    parser.add_argument('--output_queries', metavar='output queries file',
                        help='output queries file',
                        type=str, required=True)
    parser.add_argument('--output_qrels', metavar='output qrels file',
                        help='output qrels file',
                        type=str, required=True)
    parser.add_argument('--use_precomputed_negatives',
                        type=bool, default=False,
                        help='Use negative_ctxs field as a source for negative examples')
    parser.add_argument('--min_query_token_qty', type=int, default=0,
                        metavar='min # of query tokens', help='ignore queries that have smaller # of tokens')
    parser.add_argument('--' + OUT_BITEXT_PATH_OPT, metavar=OUT_BITEXT_PATH_OPT_META,
                        help=OUT_BITEXT_PATH_OPT_HELP,
                        type=str, default=None)
    add_bert_tok_args(parser)

    args = parser.parse_args()

    return args


args = parse_args()
arg_vars=vars(args)
inp_file = FileWrapper(args.input)
out_queries = FileWrapper(args.output_queries, 'w')
min_query_tok_qty = args.min_query_token_qty
use_precomputed_negatives = args.use_precomputed_negatives
stop_words = read_stop_words(STOPWORD_FILE, lower_case=True)
out_bitext_dir = arg_vars[OUT_BITEXT_PATH_OPT]
nlp = SpacyTextParser(SPACY_MODEL, stop_words, keep_only_alpha_num=True, lower_case=True)
sent_split = Sentencizer(SPACY_MODEL)

bitext_fields = [TEXT_FIELD_NAME, TEXT_UNLEMM_FIELD_NAME, TITLE_UNLEMM_FIELD_NAME]

bert_tokenizer = create_bert_tokenizer_if_needed(args)

bi_quest_files = {}
bi_answ_files = {}

glob_qrel_dict = {}

if out_bitext_dir:
    if not os.path.exists(out_bitext_dir):
        os.makedirs(out_bitext_dir)

    for fn in bitext_fields:
        bi_quest_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_QUESTION_PREFIX + fn), 'w')
        bi_answ_files[fn] = open(os.path.join(out_bitext_dir, BITEXT_ANSWER_PREFIX + fn), 'w')


seen_qrels = set()

for qid, json_str in tqdm.tqdm(enumerate(dpr_json_reader(inp_file))):
    query_idx = f'{args.part_type}_{qid}'
    fields = json.loads(json_str)
    query_orig = fields["question"]
    answer_list = list(fields["answers"])
    answer_list_lc = [s.lower() for s in answer_list]
    query_lemmas, query_unlemm = nlp.proc_text(query_orig)
    query_bert_tok = None

    query_toks = query_lemmas.split()
    if len(query_toks) >= min_query_tok_qty:
        doc = {
            DOCID_FIELD: query_idx,
            TEXT_FIELD_NAME: query_lemmas,
            TEXT_UNLEMM_FIELD_NAME: query_unlemm,
            TEXT_RAW_FIELD_NAME: query_orig,
            ANSWER_LIST_FIELD_NAME: answer_list
        }
        add_retokenized_field(doc, TEXT_RAW_FIELD_NAME, TEXT_BERT_TOKENIZED_NAME, bert_tokenizer)
        if TEXT_BERT_TOKENIZED_NAME in doc:
            query_bert_tok = doc[TEXT_BERT_TOKENIZED_NAME]

        doc_str = json.dumps(doc) + '\n'
        out_queries.write(doc_str)

        for entry in fields["positive_ctxs"]:
            psg_id = get_passage_id(entry)
            add_qrel_entry(qrel_dict=glob_qrel_dict, qid=query_idx, did=psg_id, grade=1)
            if bi_quest_files and bi_answ_files:
                title_text = entry["title"]
                if title_text:
                    _, title_unlemm = nlp.proc_text(title_text)
                    bi_quest_files[TITLE_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')
                    bi_answ_files[TITLE_UNLEMM_FIELD_NAME].write(title_unlemm + '\n')

                for ctx_sent in sent_split(entry["text"]):
                    ctx_sent = str(ctx_sent)
                    ctx_sent_lc = ctx_sent.lower()
                    # False positives are possible, b/c this check doesn't
                    # take sentence boundaries into account. However,
                    # we know that a positive context already contains an answer,
                    # so in the worst case we would pick up a somewhat less relevant
                    # sentence from the overall relevant context.
                    # We think such positives would be rare and shouldn't affect performance much.
                    has_answ = False
                    for answ in answer_list_lc:
                        if ctx_sent_lc.find(answ) >= 0:
                            has_answ = True
                            break

                    if has_answ:
                        sent_lemmas, sent_unlemm = nlp.proc_text(ctx_sent)

                        bi_quest_files[TEXT_FIELD_NAME].write(query_lemmas + '\n')
                        bi_quest_files[TEXT_UNLEMM_FIELD_NAME].write(query_unlemm + '\n')

                        bi_answ_files[TEXT_FIELD_NAME].write(sent_lemmas + '\n')
                        bi_answ_files[TEXT_UNLEMM_FIELD_NAME].write(sent_unlemm + '\n')

                        if bert_tokenizer is not None:
                            answ_bert_tok = get_retokenized(bert_tokenizer, ctx_sent_lc)
                            bi_quest_files[TEXT_BERT_TOKENIZED_NAME].write(query_bert_tok + '\n')
                            bi_answ_files[TEXT_BERT_TOKENIZED_NAME].write(answ_bert_tok + '\n')


        if use_precomputed_negatives:
            for entry in fields["negative_ctxs"]:
                psg_id = get_passage_id(entry)
                add_qrel_entry(qrel_dict=glob_qrel_dict, qid=query_idx, did=psg_id, grade=0)

inp_file.close()
out_queries.close()

write_qrels([qrel_entry for qrel_key, qrel_entry in glob_qrel_dict.items()], args.output_qrels)

for _, f in bi_quest_files.items():
    f.close()
for _, f in bi_answ_files.items():
    f.close()


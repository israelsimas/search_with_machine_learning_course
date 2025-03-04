# A simple client for querying driven by user input on the command line.  Has hooks for the various
# weeks (e.g. query understanding).  See the main section at the bottom of the file
from opensearchpy import OpenSearch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import json
import os
import sys
from getpass import getpass
from urllib.parse import urljoin
import pandas as pd
import fileinput
import logging
import fasttext
import re
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s:%(message)s')

embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

MODEL_DEFALT_FASTTEXT = "/workspace/datasets/fasttext/model.bin"
fasttext_model = fasttext.load_model(MODEL_DEFALT_FASTTEXT)
MODEL_THRESHOLD = 0.5

def normalize_query(query: str):
    norm_query = query.lower()
    norm_query = re.sub(r'[^a-zA-Z0-9]', ' ', norm_query)
    norm_query = re.sub(r'\s+', ' ', norm_query).strip()
    norm_query = ' '.join([stemmer.stem(token) for token in norm_query.split()])
    return norm_query    

# expects clicks and impressions to be in the row
def create_prior_queries_from_group(
        click_group):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    if click_group is not None:
        for item in click_group.itertuples():
            try:
                click_prior_query += "%s^%.3f  " % (item.doc_id, item.clicks / item.num_impressions)

            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query

def create_vector_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", source=None):
    if click_prior_query is not None or filters is not None or sort != "_score" or sortDir != "desc":
        raise NotImplementedError
    query_vector = embedding_model.encode(user_query).tolist()
    query_obj = {
        "size": 10,
        "_source": True if source is None else source,
        "query": {
            "knn": {
                "name_embedding": {
                    "vector": query_vector,
                    "k": 100
                }
            }
        }
    }
    return query_obj

def create_exact_vector_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", source=None):
    if click_prior_query is not None or sort != "_score" or sortDir != "desc":
        raise NotImplementedError
    query_vector = embedding_model.encode(user_query).tolist()
    query_obj = {
        "size": 10,
        "_source": True if source is None else source,
        "query": {
            "script_score": {
                "query": {
                    "bool": {
                        "filter": {
                            "bool": {
                                "must": filters
                            }
                        }
                    }
                },
                "script": {
                    "source": "knn_score",
                    "lang": "knn",
                    "params": {
                        "field": "name_embedding",
                        "query_value": query_vector,
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }
    return query_obj

# expects clicks from the raw click logs, so value_counts() are being passed in
def create_prior_queries(doc_ids, doc_id_weights,
                         query_times_seen):  # total impressions isn't currently used, but it mayb worthwhile at some point
    click_prior_query = ""
    # Create a string that looks like:  "query": "1065813^100 OR 8371111^89", where the left side is the doc id and the right side is the weight.  In our case, the number of clicks a document received in the training set
    click_prior_map = ""  # looks like: '1065813':100, '8371111':809
    if doc_ids is not None and doc_id_weights is not None:
        for idx, doc in enumerate(doc_ids):
            try:
                wgt = doc_id_weights[doc]  # This should be the number of clicks or whatever
                click_prior_query += "%s^%.3f  " % (doc, wgt / query_times_seen)
            except KeyError as ke:
                pass  # nothing to do in this case, it just means we can't find priors for this doc
    return click_prior_query


# Hardcoded query here.  Better to use search templates or other query config.
def create_query(user_query, click_prior_query, filters, sort="_score", sortDir="desc", size=10, source=None):
    name = source[0]
    query_obj = {
        'size': size,
        "sort": [
            {sort: {"order": sortDir}}
        ],
        "query": {
            "function_score": {
                "query": {
                    "bool": {
                        "must": [

                        ],
                        "should": [  #
                            {
                                "match": {
                                    name: {
                                        "query": user_query,
                                        "fuzziness": "1",
                                        "prefix_length": 2,
                                        # short words are often acronyms or usually not misspelled, so don't edit
                                        "boost": 0.01
                                    }
                                }
                            },
                            {
                                "match_phrase": {  # near exact phrase match
                                    "name.hyphens": {
                                        "query": user_query,
                                        "slop": 1,
                                        "boost": 50
                                    }
                                }
                            },
                            {
                                "multi_match": {
                                    "query": user_query,
                                    "type": "phrase",
                                    "slop": "6",
                                    "minimum_should_match": "2<75%",
                                    "fields": ["{}^10".format(name), "name.hyphens^10", "shortDescription^5",
                                               "longDescription^5", "department^0.5", "sku", "manufacturer", "features",
                                               "categoryPath"]
                                }
                            },
                            {
                                "terms": {
                                    # Lots of SKUs in the query logs, boost by it, split on whitespace so we get a list
                                    "sku": user_query.split(),
                                    "boost": 50.0
                                }
                            },
                            {  # lots of products have hyphens in them or other weird casing things like iPad
                                "match": {
                                    "name.hyphens": {
                                        "query": user_query,
                                        "operator": "OR",
                                        "minimum_should_match": "2<75%"
                                    }
                                }
                            }
                        ],
                        "minimum_should_match": 1,
                        "filter": filters  #
                    }
                },
                "boost_mode": "multiply",  # how _score and functions are combined
                "score_mode": "sum",  # how functions are combined
                "functions": [
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankShortTerm"
                            }
                        },
                        "gauss": {
                            "salesRankShortTerm": {
                                "origin": "1.0",
                                "scale": "100"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankMediumTerm"
                            }
                        },
                        "gauss": {
                            "salesRankMediumTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "filter": {
                            "exists": {
                                "field": "salesRankLongTerm"
                            }
                        },
                        "gauss": {
                            "salesRankLongTerm": {
                                "origin": "1.0",
                                "scale": "1000"
                            }
                        }
                    },
                    {
                        "script_score": {
                            "script": "0.0001"
                        }
                    }
                ]

            }
        }
    }
    if click_prior_query is not None and click_prior_query != "":
        query_obj["query"]["function_score"]["query"]["bool"]["should"].append({
            "query_string": {
                # This may feel like cheating, but it's really not, esp. in ecommerce where you have all this prior data,  You just can't let the test clicks leak in, which is why we split on date
                "query": click_prior_query,
                "fields": ["_id"]
            }
        })
    if user_query == "*" or user_query == "#":
        # replace the bool
        try:
            query_obj["query"] = {"match_all": {}}
        except:
            print("Couldn't replace query for *")
    if source is not None:  # otherwise use the default and retrieve all source
        query_obj["_source"] = source
    return query_obj


def search(client, user_query, create_query_func, index="bbuy_products", sort="_score", sortDir="desc", synonyms=False, query_filter=False):
    #### W3: classify the query
    #### W3: create filters and boosts
    filters = None
    if query_filter is not None and len(query_filter) > 0:
        filters = [
            {
                "terms": {
                    "categoryPathIds.keyword": query_filter
                }
            }
        ]

    name_field = "name.synonyms" if synonyms else "name"
    query_obj = create_query_func(user_query, click_prior_query=None, filters=filters, sort=sort, sortDir=sortDir, source=["name", "shortDescription"], name_field=name_field)
    logging.info(query_obj)
    response = client.search(query_obj, index=index)
    if response and response['hits']['hits'] and len(response['hits']['hits']) > 0:
        hits = response['hits']['hits']
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    host = 'localhost'
    port = 9200
    auth = ('admin', 'admin')  # For testing only. Don't store credentials in code.
    parser = argparse.ArgumentParser(description='Build LTR.')
    general = parser.add_argument_group("general")
    general.add_argument("-i", '--index', default="bbuy_products",
                         help='The name of the main index to search')
    general.add_argument("-s", '--host', default="localhost",
                         help='The OpenSearch host name')
    general.add_argument("-p", '--port', type=int, default=9200,
                         help='The OpenSearch port')
    general.add_argument('--user',
                         help='The OpenSearch admin.  If this is set, the program will prompt for password too. If not set, use default of admin/admin')
    general.add_argument('--synonyms', default=0, type=int,
                         help='Use this parameter to search with synonyms.  If not set, use default of false')
    general.add_argument('--enable_filters', action='store_true', help='Enable category filters on the query')
    general.add_argument('--vector', default=None, help='Vector search method to use (ann|exact)')

    args = parser.parse_args()

    if len(vars(args)) == 0:
        parser.print_usage()
        exit()

    synonyms = True if args.synonyms == 1 else False
    host = args.host
    port = args.port
    enable_filters = args.enable_filters
    if args.vector == "ann":
        create_query_func = create_vector_query
    elif args.vector == "exact":
        create_query_func = create_exact_vector_query
    else:
        create_query_func = create_query
    if args.user:
        password = getpass()
        auth = (args.user, password)

    base_url = "https://{}:{}/".format(host, port)
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,  # enables gzip compression for request bodies
        http_auth=auth,
        # client_cert = client_cert_path,
        # client_key = client_key_path,
        use_ssl=True,
        verify_certs=False,  # set to true if you have certs
        ssl_assert_hostname=False,
        ssl_show_warn=False,

    )
    index_name = args.index
    name_search_field = "name.synonym" if args.synonyms else "name"

    stemmer = PorterStemmer()
    query_prompt = "\nEnter your query (type 'Exit' to exit or hit ctrl-c):"
    print(query_prompt)
    for line in fileinput.input():
        query = line.rstrip()
        if query == "Exit":
            break
        query = normalize_query(query, stemmer)
        predicted_categories, scores = fasttext_model.predict(query, 5)
        print(predicted_categories, scores)

        categories = None
        if enable_filters:
            sum_scores = 0
            categories = []
            for i in range(len(scores)):
                sum_scores += scores[i]
                categories.append(predicted_categories[i].removeprefix('__label__'))
                if sum_scores >= MODEL_THRESHOLD:
                    break

            if sum_scores < MODEL_THRESHOLD or not args.enable_filters:
                categories = None

        search(client=opensearch, user_query=query, create_query_func=create_query_func, index=index_name)

        print(query_prompt)

    
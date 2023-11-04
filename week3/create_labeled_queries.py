import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv
import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1000, type=int, help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output
min_queries = args.min_queries

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
queries_df = pd.read_csv(queries_file_name)[['category', 'query']]
queries_df = queries_df[queries_df['category'].isin(categories)]

def normalize_query(query: str):
    norm_query = query.lower()
    norm_query = re.sub(r'[^a-zA-Z0-9]', ' ', norm_query)
    norm_query = re.sub(r'\s+', ' ', norm_query).strip()
    norm_query = ' '.join([stemmer.stem(token) for token in norm_query.split()])
    return norm_query

queries_df['query'] = queries_df['query'].apply(normalize_query)

# Roll up categories to ancestors to satisfy the minimum number of queries per category.
counts_cat = queries_df.groupby('category')['query'].count().reset_index() # https://stackoverflow.com/questions/39778686/pandas-reset-index-after-groupby-value-counts
while True:
    threshold_cat = counts_cat[counts_cat['query'] < min_queries]
    if threshold_cat.shape[0] == 0:
        break

    valid_cat = threshold_cat.loc[threshold_cat['query'].idxmin()]['category']
    parent = parents_df[parents_df['category'] == valid_cat]['parent'].values[0]

    counts_cat.loc[counts_cat['category'] == parent, 'query'] += counts_cat.loc[
    counts_cat['category'] == valid_cat, 'query'].values[0]

    queries_df.loc[queries_df['category'] == valid_cat, 'category'] = parent
    queries_df = queries_df[queries_df['category'] != valid_cat]
    counts_cat = counts_cat[counts_cat['category'] != valid_cat]

# Create labels in fastText format.
queries_df['label'] = '__label__' + queries_df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
queries_df = queries_df[queries_df['category'].isin(categories)]
queries_df['output'] = queries_df['label'] + ' ' + queries_df['query']
queries_df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)

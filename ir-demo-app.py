from sentence_transformers import SentenceTransformer, CrossEncoder, util
import os
import torch
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
import numpy as np
import pandas as pd
import warnings
import streamlit as st

from beir.datasets.data_loader_hf import HFDataLoader
from beir.reranking.models.mono_t5 import MonoT5



warnings.filterwarnings("ignore")

NUM_DOC_TO_LOAD = 100 # If you don't have GPU, encoding all documents from dataset might take a long time
bi_enc_options = ["sentence-transformers/distiluse-base-multilingual-cased-v1", 'intfloat/multilingual-e5-base', 'nthakur/mcontriever-base-msmarco']
cross_enc_options = [ 'clarin-knext/plt5-base-msmarco', 'clarin-knext/herbert-base-reranker-msmarco', 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1']
datasets_options = ["nfcorpus-pl", "scifact-pl", "fiqa-pl"]



@st.cache_data()
def load_data(dataset_type):

    corpus, queries, qrels = HFDataLoader(hf_repo="clarin-knext/"+dataset_type, streaming=False, keep_in_memory=False).load(split="test")
    corpus = [ doc['text']for doc in corpus][:NUM_DOC_TO_LOAD]
    queries = [ query['text']for query in queries]
    return queries, corpus

@st.cache_data()
def bi_encode(bi_encoder_name,passages, dataset_name='scifact-pl'):
    
    global bi_encoder
    #We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
    bi_encoder = SentenceTransformer(bi_encoder_name)
    
    # Thos code would be used if we would embed the passages, but here to make it fast we will load already embedded tensors:
    with st.spinner('Encoding passages into a vector space...'):

        if bi_encoder_name == 'intfloat/multilingual-e5-base':
            
            corpus_embeddings = bi_encoder.encode(['passage: ' + sentence for sentence in passages], convert_to_tensor=True)

        else:
            corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True)


    st.success(f"Embeddings computed. Shape: {corpus_embeddings.shape}")
    
    return bi_encoder, corpus_embeddings
    
@st.cache_resource()
def cross_encode(cross_encoder_name):
    
    global cross_encoder

    if cross_encoder_name == "clarin-knext/plt5-base-msmarco":
        cross_encoder = MonoT5(cross_encoder_name, use_amp=False, token_true='▁prawda', token_false='▁fałsz')
    else:
        cross_encoder = CrossEncoder(cross_encoder_name)#('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    
    return cross_encoder
    
@st.cache_data()
def bm25_tokenizer(text):
    
# We also compare the results to lexical search (keyword search). Here, we use 
# the BM25 algorithm which is implemented in the rank_bm25 package.
# We lower case our text and remove stop-words from indexing
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc

@st.cache_resource()
def bm25_api(passages):

    tokenized_corpus = []
    
    for passage in passages:
        tokenized_corpus.append(bm25_tokenizer(passage))

    bm25 = BM25Okapi(tokenized_corpus)
    
    return bm25




def display_df_as_table(model,top_k,score='score'):
    # Display the df with text and scores as a table
    df = pd.DataFrame([(hit[score], passages[hit['corpus_id']]) for hit in model[0:top_k]],columns=['Score','Text'])
    df['Score'] = round(df['Score'],2)
    
    return df
        
#Streamlit App
    
st.title("Retrieval Demo")

"""
Example of retrieval over BEIR datasets.
"""


st.sidebar.title("Menu")

dataset_type = st.sidebar.selectbox("Dataset", options=datasets_options, key='dataset_select')

bi_encoder_type = st.sidebar.selectbox("Bi-Encoder", options=bi_enc_options, key='bi_select')

cross_encoder_type = st.sidebar.selectbox("Cross-Encoder", options=cross_enc_options, key='cross_select')

top_k = st.sidebar.slider("Number of Top Hits Generated",min_value=1,max_value=5,value=2)

hide_bm25 = st.sidebar.checkbox("Hide BM25 results?")
hide_biencoder = st.sidebar.checkbox("Hide Bi-Encoder results?")
hide_crossencoder = st.sidebar.checkbox("Hide Cross-Encoder results?")

# This function will search all wikipedia articles for passages that
# answer the query
def search_func(query, bi_encoder_type, top_k=top_k):
    
    global bi_encoder, cross_encoder
    
    st.subheader(f"Search Query:\n_{query}_")

    ##### BM25 search (lexical search) #####
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -5)[-5:]
    bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
    bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)
    
    if not hide_bm25:
        st.subheader(f"Top-{top_k} lexical search (BM25) hits")
        
        bm25_df = display_df_as_table(bm25_hits,top_k)
        st.write(bm25_df.to_html(index=False), unsafe_allow_html=True)

    ##### Sematic Search #####
    # Encode the query using the bi-encoder and find potentially relevant passages
    if bi_encoder_type == 'intfloat/multilingual-e5-base':
        question_embedding = bi_encoder.encode("query: " + query, convert_to_tensor=True)
    else:
         question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    question_embedding = question_embedding.cpu()
    hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k,score_function=util.dot_score)
    hits = hits[0]  # Get the hits for the first query

    ##### Re-Ranking #####
    # Now, score all retrieved passages with the cross_encoder
    cross_inp = [[query, passages[hit['corpus_id']]] for hit in hits]
    cross_scores = cross_encoder.predict(cross_inp)

    # Sort results by the cross-encoder scores
    for idx in range(len(cross_scores)):
        hits[idx]['cross-score'] = cross_scores[idx]

    if not hide_biencoder:
        # Output of top-k hits from bi-encoder
        st.markdown("\n-------------------------\n")
        st.subheader(f"Top-{top_k} Bi-Encoder Retrieval hits")
        hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    
        biencoder_df = display_df_as_table(hits,top_k)
        st.write(biencoder_df.to_html(index=False), unsafe_allow_html=True)

    if not hide_crossencoder:
        # Output of top-3 hits from re-ranker
        st.markdown("\n-------------------------\n")
        st.subheader(f"Top-{top_k} Cross-Encoder Re-ranker hits")
        hits = sorted(hits, key=lambda x: x['cross-score'], reverse=True)
        
        rerank_df = display_df_as_table(hits,top_k,'cross-score')
        st.write(rerank_df.to_html(index=False), unsafe_allow_html=True)

st.markdown("---")

def clear_text():
    st.session_state["text_input"]= ""


question, passages = load_data(dataset_type)

st.write(pd.DataFrame(question[:5], columns=["Example queries from dataset"]).to_html(index=False, justify='center'), unsafe_allow_html=True)

search_query = st.text_input("Ask your question:",
                             value=question[0],
                             key="text_input")


col1, col2 = st.columns(2)

with col1:
  search = st.button("Search",key='search_but', help='Click to Search!')
  
with col2:
  clear = st.button("Clear Text Input", on_click=clear_text,key='clear',help='Click to clear the search query')

if search:
    if bi_encoder_type:

        with st.spinner(
            text=f"Loading {bi_encoder_type} bi-encoder and embedding document into vector space. This might take a few seconds depending on the length of your document..."
        ):
            bi_encoder, corpus_embeddings = bi_encode(bi_encoder_type,passages)
            cross_encoder = cross_encode(cross_encoder_type)
            bm25 = bm25_api(passages)
            
        with st.spinner(
            text="Embedding completed, searching for relevant text for given query and hits..."):
            
            search_func(search_query,bi_encoder_type,top_k)

st.markdown("""
            """)
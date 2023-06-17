# Introduction to Modern Information Retrieval
## AI TECH 2023

Contact: konrad.wojtasik@pwr.edu.pl

Examples how to use sentence transformer and beir libraries are in the jupyter notebook.



To start streamlit retrieval demo app localy, after installing requirements.txt run the command:
```
streamlit run ir-demo-app.py
```

If you want to load your own dataset, modify:
```
def load_data(dataset_type)
```


You can also try streamlit demo on huggingface: https://huggingface.co/spaces/clarin-knext/IR-Demo


## How to run elasticsearch on your machine with docker:

```
docker run -d --name elasticsearch_ir -p 9200:9200 -p 9300:9300 -e "ES_JAVA_OPTS=-Xms512m -Xmx512m" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch-oss:7.9.2
```

Useful links:
- https://www.sbert.net/
- https://github.com/beir-cellar/beir
- https://huggingface.co/sentence-transformers
- https://huggingface.co/clarin-knext


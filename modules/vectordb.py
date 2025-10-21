# from transformers import AutoTokenizer
# from torch import cuda
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from modules.utils import getconfig

def get_local_qdrant(): 
    """once the local qdrant server is created this is used to make the connection to exisitng server"""
    config = getconfig("./model_params.cfg")
    qdrant_collections = {}

    client = QdrantClient(path="./data/local_qdrant_BAAI_ip") 
    collections = client.get_collections()



    embeddings = HuggingFaceEmbeddings(
                                    model_kwargs = {
                                        'device': 'cuda:0',
                                        'trust_remote_code': True,
                                    },
                                    encode_kwargs = {
                                        'normalize_embeddings': True
                                    },
                                    model_name="BAAI/bge-en-icl")

    # Use the correct collection name 
    collection_name = 'ip'
    if collection_name in [c.name for c in collections.collections]:
        qdrant_collections['ip'] = Qdrant(client=client, collection_name=collection_name, embeddings=embeddings)


    return qdrant_collections
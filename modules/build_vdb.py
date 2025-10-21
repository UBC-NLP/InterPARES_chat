import logging
from typing import List, Tuple
import os
import glob
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from torch import cuda
# Use the updated import to fix deprecation warning
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain.docstore.document import Document
import re
import torch

# Set memory management for better GPU utilization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

device = 'cuda' if cuda.is_available() else 'cpu'
base_path = "/mnt/home/afakhrad/iTrustAI/demo"


def extract_metadata(path: str) -> Tuple[str, int, str]:
    # regex to capture phase and page
    match = re.search(r"pdf_data/(ip\d+)/.+?_page_(\d+)\.md", path)

    if match:
        phase = match.group(1)
        page = int(match.group(2))
        # construct filename
        filename = re.sub(r"/markdowns/.+?_page_\d+\.md", f"/{match.group(0).split('/')[-2]}.pdf", path)
        return phase, page, filename
    return None, None, None

def load_markdown_files(folder_path: str, base_path: str = ".") -> List[Document]:
    docs = []
    # Ensure folder_path is relative (strip leading "/")
    folder_path = folder_path.lstrip(os.sep)

    # Full path
    path = os.path.join(base_path, folder_path)

    print(f"Base path is {base_path}")
    print(f"Loading markdown files from {path}")

    markdown_dir = os.path.join(path, "markdowns")
    markdown_files = glob.glob(os.path.join(markdown_dir, "*.md"))
    print(f"Found {len(markdown_files)} markdown files in {markdown_dir}")

    filename_org = glob.glob(os.path.join(path, "*.p*"))[0]

    for md_file in markdown_files:
        phase, page, filename = extract_metadata(md_file)

        with open(md_file, 'r', encoding='utf-8') as file:
            content = file.read()
            page_data = {
                "content": content,
                "metadata": {
                    "source": os.path.relpath(md_file, base_path),
                    "filename_org": os.path.relpath(filename_org, base_path),
                    "phase": phase,
                    "category": phase,
                    "filename": os.path.relpath(md_file, base_path),  # <-- fix filename too
                    "page": page,
                    "imagepdf": False,
                    "year": "2018",
                    "document_processor": "docling",
                }
            }

        doc = Document(page_content=page_data['content'], metadata=page_data['metadata'])
        docs.append(doc)

    return docs






class DoclingMDLoader(BaseLoader):

    def __init__(self, paths: str | list[str]) -> None:
        self._paths = paths if isinstance(paths, list) else [paths]


    def lazy_load(self) -> Iterator[Document]:
        for source in self._paths:
            docs = load_markdown_files(source)
            for doc in docs:
                yield doc
    # A non-lazy, EAGER loader
    def load(self) -> list[Document]:
        all_docs = []
        for source in self._paths:
            docs = load_markdown_files(source,base_path)
            all_docs.extend(docs)
        return all_docs

def chunk_documents(documents, chunk_size=128, chunk_overlap=20):  # Reduced chunk size significantly
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5", trust_remote_code=True),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n"],
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def embed_documents(split_docs, collection_name, qdrant_path, device):
    # Clear GPU cache before starting
    torch.cuda.empty_cache()
    
    embeddings = HuggingFaceEmbeddings(
        model_kwargs = {
            'device': device,
            'trust_remote_code': True,
        },
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 2,  # Reduced from 8 to 4
        },
        model_name="BAAI/bge-en-icl")

    # Process in smaller batches to avoid OOM
    batch_size = 24  # Reduced from 128 to 32
    total_docs = len(split_docs)
    
    # Check if collection already exists and remove it if needed
    if os.path.exists(qdrant_path):
        print(f"Removing existing Qdrant collection at {qdrant_path}")
        import shutil
        shutil.rmtree(qdrant_path)
    
    # Initialize once and reuse
    qdrant_collections = None
    
    # Process documents in batches
    for i in range(0, total_docs, batch_size):
        batch_docs = split_docs[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size} with {len(batch_docs)} documents")
        
        # Clear cache before each batch
        torch.cuda.empty_cache()
        
        if i == 0:
            # Create collection with first batch
            qdrant_collections = Qdrant.from_documents(
                batch_docs,
                embeddings,
                path=qdrant_path,
                collection_name=collection_name,
            )
        else:
            # Add subsequent batches to existing collection
            qdrant_collections.add_documents(batch_docs)
        
        # Clear GPU cache after each batch
        torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
    
    return qdrant_collections

def get_phases():
    phases = {
        "ip1":"data/pdf_data/ip1",
         "ip2":"data/pdf_data/ip2",
         "ip3":"data/pdf_data/ip3",
         "ip4":"data/pdf_data/ip4"
            }
    return phases

def main():
    parser = argparse.ArgumentParser(description='Build vector database for a specific phase')
    parser.add_argument('--phase', type=str, required=True, choices=['ip1', 'ip2', 'ip3', 'ip4'],
                        help='Phase to process (ip1, ip2, ip3, or ip4)')
    parser.add_argument('--gpu_index', type=int, required=True,
                        help='GPU index to use (0, 1, 2, 3, etc.)')
    
    args = parser.parse_args()
    
    # Set device based on GPU index
    if cuda.is_available() and args.gpu_index < cuda.device_count():
        device = f'cuda:{args.gpu_index}'
        torch.cuda.set_device(args.gpu_index)
    else:
        device = 'cpu'
        print(f"Warning: GPU {args.gpu_index} not available, using CPU")
    
    print(f"Using device: {device} for phase: {args.phase}")
    
    # Get phases and process only the specified one
    phases = get_phases()
    
    if args.phase not in phases:
        raise ValueError(f"Phase {args.phase} not found in available phases")
    
    folder_path = phases[args.phase]
    print(f"Processing phase: {args.phase}")
    
    docs_paths = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    loader = DoclingMDLoader(docs_paths)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    split_docs = chunk_documents(documents, chunk_size=256, chunk_overlap=20)
    print(f"Chunked into {len(split_docs)} documents.")
    
    # Create phase-specific Qdrant path
    qdrant_path = os.path.join(base_path, f"data/local_qdrant_BAAI_{args.phase}")
    
    qdrant_collections = embed_documents(split_docs, collection_name=args.phase, qdrant_path=qdrant_path, device=device)
    
    print(f"Completed processing phase {args.phase} on {device}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
# Terminal 1 (GPU 0)
# python build_vdb.py --phase ip1 --gpu_index 0

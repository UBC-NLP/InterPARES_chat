from qdrant_client.http import models as rest
from qdrant_client.http.models import Filter  # Add this import
from modules.utils import getconfig
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing import List, Optional, Union

model_config = getconfig("model_params.cfg")

# re-ranking the retrieved results
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
# model = HuggingFaceCrossEncoder(model_name="Qwen/Qwen3-Reranker-0.6B")
compressor = CrossEncoderReranker(model=model, top_n=5)

def create_metadata_filter(
    language: Optional[Union[str, List[str]]] = None,
    categories: Optional[Union[str, List[str]]] = None,
    phase: Optional[Union[str, List[str]]] = None,
    filename_org: Optional[Union[str, List[str]]] = None
) -> Optional[Filter]:
    """Create metadata filter for Qdrant search"""
    conditions = []
    
    if language:
        # Handle both single string and list of strings
        lang_list = [language] if isinstance(language, str) else language
        conditions.append(
            rest.FieldCondition(
                key="language",
                match=rest.MatchAny(any=lang_list)
            )
        )
    
    if categories:
        # Handle both single string and list of strings
        cat_list = [categories] if isinstance(categories, str) else categories
        conditions.append(
            rest.FieldCondition(
                key="categories",
                match=rest.MatchAny(any=cat_list)
            )
        )
    
    if phase:
        # Handle both single string and list of strings
        phase_list = [phase] if isinstance(phase, str) else phase
        conditions.append(
            rest.FieldCondition(
                key="phase",
                match=rest.MatchAny(any=phase_list)
            )
        )
    
    if filename_org:
        # Handle both single string and list of strings
        file_list = [filename_org] if isinstance(filename_org, str) else filename_org
        conditions.append(
            rest.FieldCondition(
                key="filename_org",
                match=rest.MatchAny(any=file_list)
            )
        )
    
    if not conditions:
        return None
    
    return Filter(must=conditions)

def get_context(vectorstore, query, language=None, categories=None, phase=None, filename_org=None):
    """
    Retrieve and rerank context based on query and optional metadata filters.
    
    Args:
        vectorstore: Vector database instance
        query: Search query string
        language: Optional language filter
        categories: Optional category filter (single value, will match against list in metadata)
        phase: Optional phase filter
        filename_org: Optional filename filter
    
    Returns:
        List of retrieved and reranked documents
    """
    # Create metadata filter
    metadata_filter = create_metadata_filter(
        language=language,
        categories=categories,
        phase=phase,
        filename_org=filename_org
    )
    
    # Configure search kwargs
    search_kwargs = {
        "score_threshold": 0.4,
        "k": 20
    }
    
    # Add filter if any metadata filters were specified
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
        print(f"Applying filters - language: {language}, categories: {categories}, phase: {phase}, filename: {filename_org}")
    
    # getting context
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs=search_kwargs
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    
    context_retrieved = compression_retriever.invoke(query)
    print(f"retrieved paragraphs: {len(context_retrieved)}")

    return context_retrieved
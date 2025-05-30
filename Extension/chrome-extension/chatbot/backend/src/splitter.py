from llama_index.core import Document
from llama_index.core.node_parser import TokenTextSplitter

def split_document(text, metadata=None):
    doc = Document(
        text=text,
        metadata=metadata
    )

    splitter = TokenTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separator="."
    )
    nodes = splitter.get_nodes_from_documents([doc])
    return nodes

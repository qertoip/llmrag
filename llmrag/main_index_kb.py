"""
Run this after knowledge base (KB) gets updated with new documents.
"""
import logging as log

import numpy as np

from chunkers.markdown_headings_chunker import MarkdownHeadingsChunker
from embedders.aws_universal_sentence_encoder_cmlm import AwsUniversalSentenceEncoderCMLM
from embedders.local_all_mpnet_base_v2 import LocalAllMpnetBaseV2
#from embedders.local_universal_sentence_encoder import LocalUniversalSentenceEncoder
from vectordbs.chroma_vector_db import ChromaVectorDB

from tools import kb_path, setup_logging, root_path, chunk_id


def main():
    setup_logging('index_kb.log')

    chunker = MarkdownHeadingsChunker()
    #embedder = AwsUniversalSentenceEncoderCMLM()
    #embedder = LocalUniversalSentenceEncoder()
    embedder = LocalAllMpnetBaseV2()
    vectordb = ChromaVectorDB()
    
    documents_filepaths = list(kb_path().rglob('*.md'))

    for doc_i, doc_path in enumerate(documents_filepaths):
        doc = doc_path.read_text()
        chunks = chunker.chunk(doc)

        log.info(f'Doc {doc_i+1:4}/{len(documents_filepaths):4}  -  {doc_path.name}')
        log.info(f'    -> {len(chunks)} chunks')

        for chunk_i, chunk in enumerate(chunks):
            embedding = embedder.create_embedding(chunk)
            if not chunk in vectordb:
                embedding = embedder.create_embedding(chunk)
                vectordb.insert(
                    id=chunk_id(chunk),
                    embedding=embedding,
                    text=chunk,
                    metadata={'filename': doc_path.name, 'heading': chunk_i}
                )
                log.info(f'    -> added chunk {chunk_id(chunk)}')
            else:
                log.info(f'    -> present chunk {chunk_id(chunk)}')


if __name__ == '__main__':
    main()

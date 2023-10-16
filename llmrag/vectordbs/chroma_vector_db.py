import numpy as np
import chromadb
from chromadb import Settings

from tools import root_path, chunk_id, UNBOLD, BOLD
from vectordbs.vector_db import VectorDB, Item


class ChromaVectorDB(VectorDB):
    client: chromadb.API

    def __init__(self, collection_name='main7'):
        chroma_dir = (root_path() / 'db' / 'chroma')
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def insert(self, id: str, embedding: list | np.ndarray, text: str, metadata: dict):
        self.collection.add(
            ids=id,
            embeddings=self.ndarray_to_list(embedding),
            documents=text,
            metadatas=metadata
        )

    def query(self, query_embedding: list | np.ndarray, top: int = 1) -> list[Item]:
        # res = self.collection.query(
        #     query_texts=query_text,
        #     n_results=top
        # )
        res = self.collection.query(
            query_embeddings=self.ndarray_to_list(query_embedding),
            n_results=top
        )
        items = []
        for text, metadata, distance in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            item = Item(text=text, filename=metadata['filename'], distance=distance)
            items.append(item)
        return items

    def __contains__(self, chunk: str) -> bool:
        res = self.collection.get(ids=chunk_id(chunk))
        return len(res['ids']) > 0

    def report(self):
        print(f'\n========[ ChromaDB Report ]========')
        print(f'Collection: {BOLD}{self.collection.name}{UNBOLD}')
        print(f'Number or items: {BOLD}{self.collection.count()}{UNBOLD}')
        print(f'Metadata: {BOLD}{self.collection.metadata}{UNBOLD}')

        items = self.collection.peek(10)
        if len(items['ids']) > 0:
            print(f'Example items:')
            for i in range(len(items['ids'])):
                print('\tItem:')
                id = items['ids'][i]
                embedding = items['embeddings'][i]
                metadata = items['metadatas'][i]
                text = items['documents'][i]
                print(f'\t\tid={id}')
                print(f'\t\tembedding[0:5]={embedding[0:5]}')
                vector_length = (np.array(embedding) ** 2).sum() ** 0.5
                print(f'\t\tembedding_vector_length={vector_length}')
                print(f'\t\tmetadata={metadata}')
                print(f'\t\ttext[0:200]={text[0:500]}[CUT]')

    def ndarray_to_list(self, embedding):
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()
        return embedding


if __name__ == '__main__':
    ChromaVectorDB().report()

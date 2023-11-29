# LLM RAG

PoC for assistant for company-internal knowledge-base.

## Featuring

* Retrieval Augmented Generation (RAG) pattern / approach.
 
* AWS ready:
  * Deployed on **EC2**
  * Uses **SageMaker Endpoints** for:
    * Text embeddings model
    * Generative LLM model
  * Architecture open to integrate a production-grade vector DB (AWS RDS/pg_vector, AWS OpenSearch k-NN, etc)

* ...yet **rapid and free localhost development** still possible:
  * Integrated local embeddings model
  * Integrated local vector database (ChromaDB)
  * Integrated free LLM API by replicate.com (only for non-proprietary data)

* Loosely coupled and easily swappable components:
  * Chunker
  * Embedder
  * VectorDB
  * LLM
  * Assistant
  
* Tentative PoC choices:
  * **LLAMA-2 7B** as generative LLM
  * **All-mpnet-base-v2** for semantic text embeddings (paragraph / section large)

* Multiple assistants for easier testing:
  * LLM+RAG assitant
  * RAG only assistant (to assess retrieval quality)
  * LLM only assistant (to have a baseline for LLM+RAG assesment)
  
## Installation

Tested on Python 3.10.12.

After cloning the repository, run:

    cd llmrag
    bin/reset-venv  # creates .venv and installs dependencies

This will take a while because we pull in the heavyweight sentence_transformers dependency **:{**

## Running

Vectorized public documents are conveniently embedded in the project (db/chroma),
so you can go straight to running the webapp.

To run webserver locally:

    bin/webserver

Then go to: http://127.0.0.1:8080/

You will also need to create:
config/replicate_api_key.txt  # paste API key from https://replicate.com/

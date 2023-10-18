# LLM+RAG assistant for company-internal knowledge-base

## Demo

See it in action at: https://llmrag.qertoip.com/

## Featuring

* AWS ready:
  * Deployed on **EC2**
  * Uses **SageMaker Endpoints** for embeddings model and LLM model
  * Ready to integrate production-grade vector DB (AWS RDS/pg_vector, AWS OpenSearch k-NN, etc)

* ...yet **rapid and free local development** still possible:
  * Integrated local version of embeddings model
  * Integrated local version of vector database (ChromaDB)
  * Integrated free LLM API (only for non-proprietary data!)

* Decoupled and easily swappable components:
  * Chunker
  * Embedder
  * VectorDB
  * LLM
  * Assistant
  
* Tentative PoC choices:
  * **LLAMA-2 7B** as LLM
  * **All-mpnet-base-v2** for semantic text embeddings

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

Running
-------

Vectorized public documents are conveniently embedded in the project (db/chroma),
so you can go straight to running the webapp.

To run webserver locally:

    bin/webserver

Then go to: http://127.0.0.1:8080/

You will also need to create:
config/replicate_api_key.txt  # paste API key from https://replicate.com/

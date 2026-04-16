# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Build a RAG (Retrieval-Augmented Generation) pipeline for AWS S3 documentation
using Amazon Bedrock for embeddings/LLM and Amazon S3 Vectors for vector storage.

Pipeline:
  1. Crawl   - Fetch S3 docs from AWS sitemap
  2. Build   - Split docs → generate embeddings → store in S3 Vectors
  3. Query   - Embed question → retrieve similar docs → generate answer
  4. Cleanup - Delete S3 Vectors resources

Usage:
  python rag_s3vectors.py crawl              # Step 1: Crawl S3 docs
  python rag_s3vectors.py build              # Step 2: Build vector index
  python rag_s3vectors.py query "question"   # Step 3: Ask a question
  python rag_s3vectors.py cleanup            # Step 4: Delete resources
"""

import pickle
import json
import hashlib

import boto3
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REGION = "us-east-1"
DOCS_PKL = "s3_docs.pkl"                          # Local cache of crawled docs
VECTOR_BUCKET = "s3-doc-helper-vectors"            # S3 Vectors bucket name
VECTOR_INDEX = "s3-docs"                           # S3 Vectors index name
EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"   # Bedrock embedding model
LLM_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"  # Bedrock LLM for answering

# AWS clients
s3v = boto3.client("s3vectors", region_name=REGION)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


# ===========================================================================
# Step 1: Crawl - Fetch documentation from AWS
# ===========================================================================
def crawl():
    """Crawl all S3 User Guide pages via sitemap and cache locally as pkl."""
    from langchain_community.document_loaders import SitemapLoader

    print("Crawling S3 documentation from AWS sitemap...")
    loader = SitemapLoader(
        web_path="https://docs.aws.amazon.com/AmazonS3/latest/userguide/sitemap.xml",
        requests_per_second=2,
    )
    docs = loader.load()

    with open(DOCS_PKL, "wb") as f:
        pickle.dump(docs, f)
    print(f"Done. Crawled {len(docs)} pages → saved to {DOCS_PKL}")


# ===========================================================================
# Step 2: Build - Split, embed, and store vectors
# ===========================================================================
def create_vector_store():
    """Create S3 Vectors bucket and index. Safe to call multiple times."""
    try:
        s3v.create_vector_bucket(vectorBucketName=VECTOR_BUCKET)
        print(f"Created vector bucket: {VECTOR_BUCKET}")
    except s3v.exceptions.ConflictException:
        print(f"Vector bucket already exists: {VECTOR_BUCKET}")

    try:
        s3v.create_index(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            dataType="float32",
            dimension=1024,          # Must match embedding model output dimension
            distanceMetric="cosine",
        )
        print(f"Created vector index: {VECTOR_INDEX}")
    except s3v.exceptions.ConflictException:
        print(f"Vector index already exists: {VECTOR_INDEX}")


def generate_embedding(text):
    """Generate a single embedding vector via Bedrock Titan Embed v2."""
    resp = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL,
        body=json.dumps({"inputText": text}),
        contentType="application/json",
        accept="application/json",
    )
    return json.loads(resp["body"].read())["embedding"]


def build():
    """Full build pipeline: load docs → split into chunks → embed → store."""
    create_vector_store()

    with open(DOCS_PKL, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs)} documents from {DOCS_PKL}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]

        vectors = []
        for chunk in batch:
            embedding = generate_embedding(chunk.page_content)
            vectors.append({
                "key": hashlib.md5(chunk.page_content.encode()).hexdigest(),
                "data": {"float32": embedding},
                "metadata": {
                    "source": chunk.metadata.get("source", ""),
                    "text": chunk.page_content[:1000],
                },
            })

        s3v.put_vectors(
            vectorBucketName=VECTOR_BUCKET,
            indexName=VECTOR_INDEX,
            vectors=vectors,
        )
        print(f"  Stored {i + len(batch)}/{len(chunks)} vectors")

    print("Build complete!")


# ===========================================================================
# Step 3: Query - Retrieve relevant docs and generate answer
# ===========================================================================
def query(question):
    """RAG query: embed question → retrieve top-3 chunks → LLM generates answer."""
    question_embedding = generate_embedding(question)

    results = s3v.query_vectors(
        vectorBucketName=VECTOR_BUCKET,
        indexName=VECTOR_INDEX,
        queryVector={"float32": question_embedding},
        topK=3,
        returnMetadata=True,
    )
    context = "\n\n".join(v["metadata"]["text"] for v in results["vectors"])

    resp = bedrock.invoke_model(
        modelId=LLM_MODEL,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": (
                    "You are a helpful AWS documentation assistant. "
                    "Answer the question based on the information below.\n\n"
                    f"{context}\n\n"
                    f"Question: {question}"
                ),
            }],
        }),
        contentType="application/json",
        accept="application/json",
    )
    answer = json.loads(resp["body"].read())["content"][0]["text"]

    print(f"\nQuestion: {question}")
    print(f"Answer:   {answer}")


# ===========================================================================
# Step 4: Cleanup - Remove AWS resources
# ===========================================================================
def cleanup():
    """Delete the vector index and bucket to avoid ongoing charges."""
    s3v.delete_index(vectorBucketName=VECTOR_BUCKET, indexName=VECTOR_INDEX)
    s3v.delete_vector_bucket(vectorBucketName=VECTOR_BUCKET)
    print("Cleaned up all S3 Vectors resources")


# ===========================================================================
# CLI entry point
# ===========================================================================
if __name__ == "__main__":
    import sys

    commands = {
        "crawl":   crawl,
        "build":   build,
        "cleanup": cleanup,
    }

    if len(sys.argv) < 2 or sys.argv[1] not in {*commands, "query"}:
        print(__doc__)
    elif sys.argv[1] == "query":
        query(" ".join(sys.argv[2:]))
    else:
        commands[sys.argv[1]]()

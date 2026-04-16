# RAG with Amazon S3 Vectors and Amazon Bedrock

A complete Retrieval-Augmented Generation (RAG) pipeline that answers questions about AWS S3 documentation using Amazon S3 Vectors for vector storage and Amazon Bedrock for embeddings and text generation.

![Architecture](./RAG-S3Vectors.png)

## How it works

- **Build Pipeline** – Crawl AWS documentation, split into chunks, generate embeddings with Bedrock Titan Embed v2, and store in S3 Vectors
- **Query Pipeline** – Embed a question, retrieve similar chunks from S3 Vectors, and generate an answer with Bedrock Claude

## Prerequisites

- AWS account with access to Amazon Bedrock models (Titan Embed Text v2 and Claude 3 Haiku) in `us-east-1`
- Python 3.10+
- AWS credentials configured (via IAM role, environment variables, or AWS CLI)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Crawl S3 documentation (only needed once)
python rag_s3vectors.py crawl

# 2. Build the vector index
python rag_s3vectors.py build

# 3. Ask questions
python rag_s3vectors.py query "What is S3 Vectors?"
python rag_s3vectors.py query "How does S3 versioning work?"

# 4. Clean up AWS resources when done
python rag_s3vectors.py cleanup
```

## Configuration

Edit the constants at the top of `rag_s3vectors.py` to match your environment:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REGION` | `us-east-1` | AWS region |
| `VECTOR_BUCKET` | `s3-doc-helper-vectors` | S3 Vectors bucket name |
| `VECTOR_INDEX` | `s3-docs` | S3 Vectors index name |
| `EMBEDDING_MODEL` | `amazon.titan-embed-text-v2:0` | Bedrock embedding model |
| `LLM_MODEL` | `anthropic.claude-3-haiku-20240307-v1:0` | Bedrock LLM for answering |

## Cost

This project uses AWS services that may incur charges:

- **Amazon Bedrock** – Per-token pricing for embedding and LLM inference
- **Amazon S3 Vectors** – Storage and request pricing

See [Amazon S3 pricing](https://aws.amazon.com/s3/pricing/) and [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) for details. Remember to run `cleanup` when done to avoid ongoing charges.

## Blog post

See [blog-rag-s3-vectors.md](./blog-rag-s3-vectors.md) for a detailed walkthrough of this project.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

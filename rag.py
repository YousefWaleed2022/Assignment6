import os
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from sklearn.metrics import precision_score, recall_score, f1_score
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain.chains import RetrievalQA
import shutil
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_system")

# Initialize console
console = Console()

class TestCases:
    @staticmethod
    def load_test_cases(file_path="test_cases.json"):
        """Load test cases from JSON file or return default if not found"""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading test cases from {file_path}: {str(e)}")
        
        # Default test cases
        return {
            "retrieval_tests": [
                {
                "query": "What is the Transformer architecture and how does it use attention mechanisms?",
                "relevant_doc_ids": ["attention_is_all_you_need.pdf"],
                "expected_snippets": [
                    "The Transformer model is based on self-attention mechanisms...",
                    "Positional encoding is used to maintain sequence order..."
                ],
                "metadata_filters": {
                    "year": "2017",
                    "domain": "Natural Language Processing"
                },
                "query_type": "factual"
                },
            ],
            "generation_tests": [
                {
                "query": "Summarize the evolution of diffusion models from 2020 to 2023.",
                "required_sources": ["denoising_diffusion_probabilistic_models.pdf"],
                "expected_answer_keywords": ["DDPM", "score-based models", "image generation", "stability"],
                "evaluation_criteria": {
                    "accuracy": True,
                    "coherence": True,
                    "source_integration": True
                }
                },
            ],
            "edge_cases": [
                {
                "query": "What is the capital of Mars?",
                "expected_response": "The question contains false premises. Mars does not have a capital city.",
                "type": "invalid_query"
                },
            ],
            "metadata_test": {
                "filters": {
                "year_range": [2015, 2023],
                "domain": "Reinforcement Learning",
                "methodology": "empirical study"
                },
                "expected_results_count": 2
            }
        }

class DocumentLoader:
    """Loads and processes documents from a directory"""
    def __init__(self, data_dir="documents"):
        self.data_dir = data_dir
        self.supported_types = {
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".txt": TextLoader
        }

    def load_documents(self):
        """Load documents from the specified directory"""
        documents = []
        try:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
                self._create_sample_docs()
            
            files = [f for f in os.listdir(self.data_dir) 
                    if any(f.endswith(ext) for ext in self.supported_types.keys())]
            
            with Progress() as progress:
                task = progress.add_task("[green]Loading documents...", total=len(files))
                
                # Process files in parallel
                with ThreadPoolExecutor(max_workers=min(10, os.cpu_count() or 1)) as executor:
                    future_to_file = {
                        executor.submit(self._load_file, os.path.join(self.data_dir, filename)): filename
                        for filename in files
                    }
                    
                    for future in as_completed(future_to_file):
                        filename = future_to_file[future]
                        try:
                            result = future.result()
                            if result:
                                documents.extend(result)
                            progress.advance(task)
                        except Exception as e:
                            logger.error(f"Error loading {filename}: {str(e)}")
                            progress.advance(task)

        except Exception as e:
            logger.error(f"Error accessing directory: {str(e)}")
        
        return documents

    def _load_file(self, file_path):
        """Load a single file and extract documents"""
        try:
            filename = os.path.basename(file_path)
            ext = os.path.splitext(filename)[1]
            
            if ext in self.supported_types:
                loader_class = self.supported_types[ext]
                loader = loader_class(file_path)
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = filename
                    # Add extraction timestamp
                    doc.metadata["extraction_time"] = time.time()
                    # Add file size
                    doc.metadata["file_size"] = os.path.getsize(file_path)
                    # Extract extension and file type
                    doc.metadata["file_type"] = ext[1:]  # Remove the dot
                
                return docs
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
        
        return []


class TextChunker:
    """Splits documents into chunks"""
    def __init__(self, chunk_size=512, chunk_overlap=128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, documents):
        """Split documents into chunks with progress tracking"""
        console.print(f"[blue]Splitting {len(documents)} documents into chunks...[/blue]")
        
        total_chunks = []
        with Progress() as progress:
            task = progress.add_task("[green]Chunking documents...", total=len(documents))
            
            for doc in documents:
                try:
                    chunks = self.splitter.split_documents([doc])
                    # Preserve original metadata and add chunk info
                    for i, chunk in enumerate(chunks):
                        chunk.metadata.update({
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap
                        })
                    total_chunks.extend(chunks)
                    progress.advance(task)
                except Exception as e:
                    logger.error(f"Error chunking document {doc.metadata.get('source', 'unknown')}: {str(e)}")
                    progress.advance(task)
        
        logger.info(f"Split {len(documents)} documents into {len(total_chunks)} chunks")
        return total_chunks


class Embedder:
    """Handles document embedding generation"""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.model = HuggingFaceEmbeddings(model_name=self.model_name)
            self.dimension = 384  # Default for all-MiniLM-L6-v2
            self.index = None
            self.index_path = f"faiss_index_{self.model_name.replace('/', '_')}"
        except Exception as e:
            logger.error(f"Error initializing embedder: {str(e)}")
            raise

    def add_documents(self, documents):
        """Add documents to FAISS index with batching"""
        try:
            texts = [doc.page_content for doc in documents]
            
            # Initialize index if needed
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            
            # Process in batches to avoid memory issues
            batch_size = 100
            with Progress() as progress:
                task = progress.add_task("[green]Embedding documents...", total=len(texts))
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.model.embed_documents(batch_texts)
                    self.index.add(np.array(batch_embeddings))
                    progress.update(task, advance=len(batch_texts))
                    
            logger.info(f"Added {len(texts)} documents to the index")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to index: {str(e)}")
            return False

    def save_index(self):
        """Save FAISS index to disk"""
        try:
            if self.index:
                faiss.write_index(self.index, f"{self.index_path}.bin")
                logger.info(f"Index saved to {self.index_path}.bin")
                return True
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
        return False

    def load_index(self):
        """Load FAISS index from disk"""
        try:
            index_file = f"{self.index_path}.bin"
            if os.path.exists(index_file):
                self.index = faiss.read_index(index_file)
                logger.info(f"Index loaded from {index_file}")
                return True
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
        return False


class VectorRetriever:
    """Implements retrieval strategies"""
    def __init__(self, embedder, strategy="hybrid"):
        self.embedder = embedder
        self.strategy = strategy  # "similarity", "mmr", "hybrid"
        self.k = 5
        self.lambda_param = 0.5
        self.vector_store = None

    def create_vector_store(self, documents):
        """Create vector store from documents"""
        console.print("[blue]Creating vector store...[/blue]")
        try:
            # Preserve all metadata
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            start_time = time.time()
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embedder.model,
                metadatas=metadatas
            )
            duration = time.time() - start_time
            
            # Save the vector store
            self.vector_store.save_local(self.embedder.index_path)
            logger.info(f"Vector store created with {len(documents)} documents in {duration:.2f} seconds")
            return self.vector_store
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None

    def load_vector_store(self):
        """Load vector store from disk"""
        try:
            if os.path.exists(f"{self.embedder.index_path}.faiss"):
                self.vector_store = FAISS.load_local(
                    self.embedder.index_path,
                    self.embedder.model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Vector store loaded successfully")
                return self.vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
        return None

    def similarity_search(self, query, k=None):
        """Basic similarity search"""
        if k is None:
            k = self.k
            
        if not self.vector_store:
            self.load_vector_store()
            
        if not self.vector_store:
            logger.error("Vector store not available for search")
            return []
            
        try:
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def mmr_search(self, query, k=None):
        """MMR search for diverse results"""
        if k is None:
            k = self.k
            
        if not self.vector_store:
            self.load_vector_store()
            
        if not self.vector_store:
            logger.error("Vector store not available for MMR search")
            return []
            
        try:
            return self.vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=k*2, lambda_mult=self.lambda_param
            )
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            return []
    
    def hybrid_search(self, query, k=None):
        """Hybrid search combining similarity and MMR results"""
        if k is None:
            k = self.k
            
        try:
            # Get both sets of results
            similarity_results = self.similarity_search(query, k=k)
            mmr_results = self.mmr_search(query, k=k)
            
            # Combine results with deduplication
            seen_content = set()
            combined_results = []
            
            # Alternate between result sets for diversity
            for i in range(max(len(similarity_results), len(mmr_results))):
                if i < len(similarity_results):
                    doc = similarity_results[i]
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        combined_results.append(doc)
                
                if i < len(mmr_results):
                    doc = mmr_results[i]
                    if doc.page_content not in seen_content:
                        seen_content.add(doc.page_content)
                        combined_results.append(doc)
                        
                if len(combined_results) >= k:
                    break
                    
            return combined_results[:k]
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            return []
    
    def retrieve(self, query, k=None, strategy=None):
        """Main retrieval method that uses the selected strategy"""
        if strategy is None:
            strategy = self.strategy
            
        if strategy == "similarity":
            return self.similarity_search(query, k)
        elif strategy == "mmr":
            return self.mmr_search(query, k)
        elif strategy == "hybrid":
            return self.hybrid_search(query, k)
        else:
            logger.warning(f"Unknown retrieval strategy '{strategy}', falling back to similarity search")
            return self.similarity_search(query, k)


class LLMProvider:
    """Base class for LLM integration"""
    def __init__(self, model_name, temperature=0.7, max_tokens=512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
    

class GroqProvider(LLMProvider):
    """Groq LLM integration"""
    def __init__(self, model_name="qwen-qwq-32b", temperature=0.7, max_tokens=1024):
        super().__init__(model_name, temperature, max_tokens)
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not found in environment")

    def get_llm(self):
        """Initialize and return the Groq LLM"""
        if not self.llm:
            try:
                self.llm = ChatGroq(
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    groq_api_key=self.api_key
                )
            except Exception as e:
                logger.error(f"Error initializing Groq LLM: {str(e)}")
        return self.llm

    def create_qa_chain(self, vector_store):
        """Create QA chain with enhanced prompt"""
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an AI research assistant. Answer the question based ONLY on the context provided.
            
            Context:
            {context}
            
            Question: {question}
            
            Instructions:
            1. Answer based solely on the provided context
            2. If the context doesn't contain relevant information, admit you don't know
            3. Provide specific references to the source documents when possible
            4. Be concise and precise in your response
            
            Answer:"""
        )

        llm = self.get_llm()
        if not llm:
            logger.error("Failed to initialize LLM")
            return None

        try:
            return RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt}
            )
        except Exception as e:
            logger.error(f"Error creating QA chain: {str(e)}")
            return None


class RAGEvaluator:
    """Evaluator for RAG system performance"""
    def __init__(self, rag_system):
        self.rag = rag_system

    def evaluate_retrieval(self, test_cases):
        """Evaluate retrieval performance"""
        results = {
            "precision": [],
            "recall": [],
            "f1": [],
            "by_query_type": {}
        }

        console.print("[blue]Evaluating retrieval performance...[/blue]")
        
        with Progress() as progress:
            task = progress.add_task("[green]Running retrieval tests...", total=len(test_cases.get("retrieval_tests", [])))
            
            for test in test_cases.get("retrieval_tests", []):
                query_type = test.get("query_type", "unknown")
                query = test["query"]
                
                # Initialize query type metrics if needed
                if query_type not in results["by_query_type"]:
                    results["by_query_type"][query_type] = {
                        "precision": [], "recall": [], "f1": [], "count": 0
                    }
                
                try:
                    retrieved_docs = self.rag.retrieve(query)
                    retrieved_ids = [doc.metadata.get("source", "") for doc in retrieved_docs]
                    
                    # Calculate true positives, false positives, etc.
                    true_positives = [doc_id for doc_id in retrieved_ids if doc_id in test["relevant_doc_ids"]]
                    
                    # Calculate metrics
                    precision = len(true_positives) / len(retrieved_ids) if retrieved_ids else 0
                    recall = len(true_positives) / len(test["relevant_doc_ids"]) if test["relevant_doc_ids"] else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    # Add to results
                    results["precision"].append(precision)
                    results["recall"].append(recall)
                    results["f1"].append(f1)
                    
                    # Add to query type results
                    results["by_query_type"][query_type]["precision"].append(precision)
                    results["by_query_type"][query_type]["recall"].append(recall)
                    results["by_query_type"][query_type]["f1"].append(f1)
                    results["by_query_type"][query_type]["count"] += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating retrieval for query '{query}': {str(e)}")
                
                progress.advance(task)
        
        # Calculate averages
        if results["precision"]:
            results["avg_precision"] = sum(results["precision"]) / len(results["precision"])
            results["avg_recall"] = sum(results["recall"]) / len(results["recall"])
            results["avg_f1"] = sum(results["f1"]) / len(results["f1"])
            
            # Calculate query type averages
            for query_type, metrics in results["by_query_type"].items():
                if metrics["count"] > 0:
                    metrics["avg_precision"] = sum(metrics["precision"]) / metrics["count"]
                    metrics["avg_recall"] = sum(metrics["recall"]) / metrics["count"]
                    metrics["avg_f1"] = sum(metrics["f1"]) / metrics["count"]
        
        return results

    def evaluate_generation(self, test_cases):
        """Evaluate generation quality"""
        results = {
            "accuracy": 0,
            "keyword_coverage": [],
            "by_query_type": {}
        }
        
        console.print("[blue]Evaluating answer generation...[/blue]")
        
        test_cases_list = test_cases.get("generation_tests", [])
        if not test_cases_list:
            return {"accuracy": 0, "error": "No generation test cases found"}
        
        with Progress() as progress:
            task = progress.add_task("[green]Running generation tests...", total=len(test_cases_list))
            
            for test in test_cases_list:
                query_type = test.get("query_type", "unknown")
                query = test["query"]
                
                # Initialize query type metrics if needed
                if query_type not in results["by_query_type"]:
                    results["by_query_type"][query_type] = {
                        "correct": 0, "total": 0, "coverage": []
                    }
                
                try:
                    response = self.rag.query(query)
                    expected_keywords = test.get("expected_answer_keywords", [])
                    
                    # Calculate keyword coverage
                    if expected_keywords:
                        matched_keywords = [k for k in expected_keywords if k.lower() in response.lower()]
                        coverage = len(matched_keywords) / len(expected_keywords)
                        results["keyword_coverage"].append(coverage)
                        
                        if query_type in results["by_query_type"]:
                            results["by_query_type"][query_type]["coverage"].append(coverage)
                        
                        # Consider test passed if at least half the keywords are found
                        if coverage >= 0.5:
                            results["by_query_type"][query_type]["correct"] += 1
                            
                    results["by_query_type"][query_type]["total"] += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating generation for query '{query}': {str(e)}")
                
                progress.advance(task)
        
        # Calculate overall accuracy and averages
        total_correct = sum(metrics["correct"] for metrics in results["by_query_type"].values())
        total_tests = sum(metrics["total"] for metrics in results["by_query_type"].values())
        
        results["accuracy"] = total_correct / total_tests if total_tests > 0 else 0
        
        if results["keyword_coverage"]:
            results["avg_keyword_coverage"] = sum(results["keyword_coverage"]) / len(results["keyword_coverage"])
        
        # Calculate query type accuracies
        for query_type, metrics in results["by_query_type"].items():
            if metrics["total"] > 0:
                metrics["accuracy"] = metrics["correct"] / metrics["total"]
                if metrics["coverage"]:
                    metrics["avg_coverage"] = sum(metrics["coverage"]) / len(metrics["coverage"])
        
        return results

    def evaluate_edge_cases(self, test_cases):
        """Evaluate edge case handling"""
        results = {
            "correct": 0,
            "total": 0,
            "by_type": {}
        }
        
        console.print("[blue]Evaluating edge case handling...[/blue]")
        
        edge_cases = test_cases.get("edge_cases", [])
        if not edge_cases:
            return {"accuracy": 0, "error": "No edge cases found"}
        
        with Progress() as progress:
            task = progress.add_task("[green]Testing edge cases...", total=len(edge_cases))
            
            for test in edge_cases:
                case_type = test.get("type", "unknown")
                query = test["query"]
                
                # Initialize type metrics if needed
                if case_type not in results["by_type"]:
                    results["by_type"][case_type] = {"correct": 0, "total": 0}
                
                try:
                    response = self.rag.query(query)
                    # Check if response contains key phrases from expected response
                    expected = test.get("expected_response", "").lower()
                    
                    # Simple keyword matching for now - could be improved with semantic similarity
                    keywords = expected.split()
                    matches = sum(1 for k in keywords if len(k) > 4 and k in response.lower())
                    
                    # Consider correct if at least half the keywords are found
                    if matches >= len(keywords) / 2:
                        results["correct"] += 1
                        results["by_type"][case_type]["correct"] += 1
                    
                    results["total"] += 1
                    results["by_type"][case_type]["total"] += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating edge case '{query}': {str(e)}")
                    results["total"] += 1
                    results["by_type"][case_type]["total"] += 1
                
                progress.advance(task)
        
        # Calculate accuracies
        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0
        
        for case_type, metrics in results["by_type"].items():
            if metrics["total"] > 0:
                metrics["accuracy"] = metrics["correct"] / metrics["total"]
        
        return results

    def run_full_evaluation(self, test_cases):
        """Run comprehensive evaluation suite"""
        console.print(Panel("[bold blue]Running Full RAG Evaluation[/bold blue]"))
        
        eval_results = {
            "retrieval": self.evaluate_retrieval(test_cases),
            "generation": self.evaluate_generation(test_cases),
            "edge_cases": self.evaluate_edge_cases(test_cases)
        }
        
        # Save results
        try:
            with open("evaluation_results.json", "w") as f:
                json.dump(eval_results, f, indent=2)
            logger.info("Evaluation results saved to evaluation_results.json")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {str(e)}")
        
        return eval_results


class RAGPipeline:
    """Main RAG pipeline with improved caching and diagnostics"""
    def __init__(self, data_dir="documents", force_reload=False):
        # Configuration
        self.data_dir = data_dir
        self.force_reload = force_reload
        self.chunk_size = 512
        self.chunk_overlap = 128
        self.k = 5
        self.index_path = "faiss_index"
        
        # Initialize components
        self.loader = DocumentLoader(data_dir)
        self.chunker = TextChunker(self.chunk_size, self.chunk_overlap)
        self.embedder = Embedder()
        self.retriever = VectorRetriever(self.embedder, strategy="hybrid")
        self.llm = GroqProvider()
        self.vector_store = None
        self.qa_chain = None
        
        # Cache for query results
        self.query_cache = {}
        
        # Load vector store if available and not forcing reload
        if not self.force_reload:
            self._load_vector_store()

    def _load_vector_store(self):
        """Load vector store from disk"""
        try:
            if os.path.exists(f"{self.embedder.index_path}.faiss"):
                self.vector_store = FAISS.load_local(
                    self.embedder.index_path,
                    self.embedder.model,
                    allow_dangerous_deserialization=True
                )
                self.retriever.vector_store = self.vector_store
                logger.info("Loaded existing vector store")
                return True
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
        return False

    def ingest_documents(self):
        """Process documents through the pipeline"""
        console.print(Panel("[bold blue]Starting document ingestion...[/bold blue]"))
        
        start_time = time.time()
        
        # Step 1: Load documents
        raw_docs = self.loader.load_documents()
        if not raw_docs:
            console.print("[red]No documents found![/red]")
            return False
        
        console.print(f"✅ Loaded {len(raw_docs)} documents")
        
        # Step 2: Split into chunks
        chunked_docs = self.chunker.split_documents(raw_docs)
        if not chunked_docs:
            console.print("[red]Error: No chunks created![/red]")
            return False
            
        console.print(f"✅ Split into {len(chunked_docs)} chunks")
        
        # Step 3: Create vector store
        vector_store = self.retriever.create_vector_store(chunked_docs)
        if not vector_store:
            console.print("[red]Error: Failed to create vector store![/red]")
            return False
            
        self.vector_store = vector_store
        self.retriever.vector_store = vector_store
        
        # Step 4: Save embedding index
        if not self.embedder.save_index():
            console.print("[yellow]Warning: Failed to save embedding index[/yellow]")
        
        duration = time.time() - start_time
        console.print(f"✅ Document ingestion complete in {duration:.2f} seconds")
        
        return True

    def retrieve(self, query, strategy=None):
        """Retrieve relevant documents for a query"""
        if not self.vector_store:
            if not self._load_vector_store():
                console.print("[red]Error: Vector store not available![/red]")
                return []
        
        # Use cached results if available
        cache_key = f"{query}_{strategy or self.retriever.strategy}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Perform retrieval
        results = self.retriever.retrieve(query, strategy=strategy)
        
        # Cache results
        self.query_cache[cache_key] = results
        
        return results

    def query(self, question, strategy=None):
        """Generate answer to a question using RAG"""
        start_time = time.time()
        
        try:
            # Check if vector store is available
            if not self.vector_store:
                if not self._load_vector_store():
                    return "Error: Vector store not available. Please ingest documents first."
            
            # Create QA chain if not already created
            if not self.qa_chain:
                self.qa_chain = self.llm.create_qa_chain(self.vector_store)
                if not self.qa_chain:
                    return "Error: Failed to create QA chain."
            
            # Query the model
            result = self.qa_chain.invoke({"query": question})
            
            # Format the response
            response = result.get("result", "")
            
            # Get the source documents used
            source_docs = result.get("source_documents", [])
            sources = []
            if source_docs:
                for i, doc in enumerate(source_docs[:3]):  # Limit to top 3 sources
                    source = doc.metadata.get("source", f"Source {i+1}")
                    sources.append(source)
            
            # Add source attribution if available
            if sources:
                unique_sources = list(set(sources))
                response += f"\n\nSources: {', '.join(unique_sources)}"
            
            duration = time.time() - start_time
            logger.info(f"Query processed in {duration:.2f} seconds")
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            logger.error(error_msg)
            return f"An error occurred while processing your question: {str(e)}"

    def get_system_info(self):
        """Return system information and status"""
        info = {
            "status": "operational" if self.vector_store else "needs_ingestion",
            "documents_path": self.data_dir,
            "document_count": 0,
            "chunk_count": 0,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embedder.model_name,
            "llm_model": self.llm.model_name,
            "retrieval_strategy": self.retriever.strategy,
            "top_k": self.k
        }
        
        # Try to get document and chunk counts if vector store exists
        if self.vector_store:
            try:
                # Extract unique document sources from metadata
                sources = set()
                chunks = 0
                
                for docstore_id in self.vector_store.docstore._dict.keys():
                    doc = self.vector_store.docstore.search(docstore_id)
                    if doc and hasattr(doc, "metadata"):
                        source = doc.metadata.get("source")
                        if source:
                            sources.add(source)
                        chunks += 1
                
                info["document_count"] = len(sources)
                info["chunk_count"] = chunks
            except Exception as e:
                logger.error(f"Error getting system info: {str(e)}")
        
        return info


def force_cleanup(index_path="faiss_index"):
    """Force cleanup of existing FAISS index"""
    try:
        # Remove index files
        for ext in [".bin", ".faiss", ".pkl"]:
            if os.path.exists(f"{index_path}{ext}"):
                os.remove(f"{index_path}{ext}")
                logger.info(f"Removed {index_path}{ext}")
        
        # Remove directory if it exists
        if os.path.isdir(index_path):
            shutil.rmtree(index_path)
            logger.info(f"Removed directory {index_path}")
            
        return True
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        return False


def run_interactive_mode(rag):
    """Run an interactive query session"""
    console.print(Panel("[bold blue]Interactive Query Mode[/bold blue]"))
    console.print("Type 'exit' to quit, 'help' for commands.")
    
    while True:
        query = console.input("[bold green]Query>[/bold green] ")
        
        if query.lower() in ["exit", "quit", "q"]:
            break
        elif query.lower() == "help":
            console.print(Panel("""
            [bold]Available commands:[/bold]
            - help: Show this help message
            - info: Show system information
            - ingest: Re-ingest documents
            - clear: Clear the screen
            - reload: Force reload of vector store
            - exit/quit: Exit the program
            """))
        elif query.lower() == "info":
            info = rag.get_system_info()
            console.print(Panel(f"""
            [bold]System Information:[/bold]
            Status: {info['status']}
            Documents: {info['document_count']}
            Chunks: {info['chunk_count']}
            Embedding model: {info['embedding_model']}
            LLM model: {info['llm_model']}
            Retrieval strategy: {info['retrieval_strategy']}
            """))
        elif query.lower() == "ingest":
            console.print("Re-ingesting documents...")
            rag.ingest_documents()
        elif query.lower() == "clear":
            os.system('cls' if os.name == 'nt' else 'clear')
        elif query.lower() == "reload":
            console.print("Forcing reload of vector store...")
            force_cleanup()
            rag = RAGPipeline(force_reload=True)
            rag.ingest_documents()
        else:
            # Process regular query
            console.print("[blue]Processing query...[/blue]")
            
            # Get retrieved documents
            retrieved_docs = rag.retrieve(query)
            console.print(f"[blue]Found {len(retrieved_docs)} relevant documents.[/blue]")
            
            # Generate answer
            console.print("[blue]Generating answer...[/blue]")
            answer = rag.query(query)
            
            # Display answer
            console.print(Panel(answer, title="Answer"))


def main():
    """Main entry point with improved error handling and options"""
    console.print(Panel("[bold blue]RAG System[/bold blue]", subtitle="Retrieval-Augmented Generation"))
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]Warning: GROQ_API_KEY not found in environment![/red]")
        console.print("You can set it using 'export GROQ_API_KEY=your_key_here' or in a .env file.")
    
    try:
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description="RAG System")
        parser.add_argument("--ingest", action="store_true", help="Force document ingestion")
        parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
        parser.add_argument("--clean", action="store_true", help="Clean existing indices")
        parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
        parser.add_argument("--query", type=str, help="Run a single query")
        parser.add_argument("--data-dir", type=str, default="documents", help="Directory containing documents")
        args = parser.parse_args()
        
        # Clean if requested
        if args.clean:
            console.print("[yellow]Cleaning existing indices...[/yellow]")
            force_cleanup()
        
        # Initialize RAG system
        rag = RAGPipeline(data_dir=args.data_dir, force_reload=args.ingest)
        
        # Ingest documents if requested or if vector store doesn't exist
        if args.ingest or not rag.vector_store:
            rag.ingest_documents()
        
        # Evaluate if requested
        if args.evaluate:
            test_cases = TestCases.load_test_cases()
            evaluator = RAGEvaluator(rag)
            results = evaluator.run_full_evaluation(test_cases)
            
            # Print summary
            console.print(Panel("[bold blue]Evaluation Summary[/bold blue]"))
            console.print(f"Retrieval Precision: {results['retrieval'].get('avg_precision', 0):.2f}")
            console.print(f"Retrieval Recall: {results['retrieval'].get('avg_recall', 0):.2f}")
            console.print(f"Retrieval F1: {results['retrieval'].get('avg_f1', 0):.2f}")
            console.print(f"Generation Accuracy: {results['generation'].get('accuracy', 0):.2f}")
            console.print(f"Edge Case Handling: {results['edge_cases'].get('accuracy', 0):.2f}")
        
        # Run query if provided
        if args.query:
            console.print(f"Query: {args.query}")
            answer = rag.query(args.query)
            console.print(Panel(answer, title="Answer"))
        
        # Run in interactive mode if requested or if no other action was specified
        if args.interactive or (not args.evaluate and not args.query):
            run_interactive_mode(rag)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        logger.exception("Unhandled exception in main")


if __name__ == "__main__":
    main()
# Training System in Episodic Knowledge Mesh (EKM)

## Overview

The training system in the Episodic Knowledge Mesh (EKM) is responsible for ingesting new knowledge and converting raw text into structured knowledge representations. The system follows a single-step extraction approach that simultaneously extracts summaries and Atomic Knowledge Units (AKUs) from input text, creating a rich knowledge graph structure.

## Architecture

### Core Components

#### 1. Training Service
The `TrainingService` orchestrates the entire training process:

```python
class TrainingService:
    def __init__(self, storage, llm, embeddings, scalable_backend, ...):
        self.storage = storage
        self.llm = llm
        self.embeddings = embeddings
        self.scalable_backend = scalable_backend
        self.text_processor = SemanticTextProcessor(...)
        self.knowledge_extractor = KnowledgeExtractor(...)
```

**Responsibilities:**
- Text preprocessing and chunking
- Knowledge extraction orchestration
- Storage of extracted knowledge
- Relationship generation
- Vector index management

#### 2. Text Processor
The `SemanticTextProcessor` handles text chunking:

```python
def chunk_text(self, text: str) -> List[str]:
    # Processes text into semantic chunks
    # Respects document structure and semantic boundaries
    pass
```

**Features:**
- Configurable chunk sizes (default: 512-2048 tokens)
- Semantic boundary detection
- Document structure preservation

#### 3. Knowledge Extractor
The `KnowledgeExtractor` performs the core extraction:

```python
class KnowledgeExtractor:
    async def single_step_extraction(self, text: str) -> Dict[str, Any]:
        # Performs simultaneous summary and AKU extraction
        pass
    
    async def extract_akus_with_reasoning(self, text: str, max_akus: int = 10) -> Dict[str, Any]:
        # Extracts AKUs with explicit reasoning
        pass
```

**Capabilities:**
- Single-step extraction of summaries and AKUs
- Chain-of-thought prompting for accuracy
- Detailed extraction with reasoning steps
- JSON parsing and fallback mechanisms

## Training Process

### 1. Text Ingestion and Preprocessing

The training process begins with text ingestion:

```python
async def train(self, workspace_id: str, text: str, title: Optional[str] = None):
    # 1. Chunk text according to paper specifications
    chunks = await self.text_processor.chunk_text(text)
    
    for chunk in chunks:
        # Process each chunk individually
        pass
```

**Chunking Strategy:**
- Maintains semantic coherence within chunks
- Respects document structure
- Configurable size parameters

### 2. Single-Step Knowledge Extraction

For each chunk, the system performs simultaneous extraction:

```python
# 2. Single-step extraction: Extract both summary and AKUs in one LLM call
extraction_result = await self.knowledge_extractor.single_step_extraction(chunk)

# 3. Process the extraction result
summary = extraction_result.get('summary', chunk[:200] + "...")
aku_contents = extraction_result.get('facts', [])
```

**Extraction Process:**
- Uses Chain-of-Thought prompting
- Generates JSON-formatted output
- Includes both summary and atomic facts
- Has fallback parsing mechanisms

### 3. Knowledge Storage

The extracted knowledge is stored in multiple forms:

#### Episode Storage
```python
# 5. Save Episode
episode_id = await self.storage.save_episode(
    workspace_id, chunk, summary, ep_embedding, {"title": title}
)
```

#### AKU Storage
```python
# 6. Save AKUs
akus_to_save = []
for content, emb in zip(aku_contents, aku_embeddings):
    akus_to_save.append({"content": content, "embedding": emb})

aku_ids = await self.storage.save_akus(workspace_id, episode_id, akus_to_save)
```

### 4. Vector Index Integration

AKUs are added to scalable vector indexes:

```python
# 7. Add AKUs to scalable vector index
await self.scalable_backend.add_akus_batch(
    workspace_id,
    [{"content": content, "id": id} for content, id in zip(aku_contents, aku_ids)]
)
```

### 5. Relationship Generation

The system generates relationships between AKUs:

#### Global Vector Search Relationships
```python
# Search for global connections for each new AKU
for i, (current_id, current_emb) in enumerate(zip(aku_ids, aku_embeddings)):
    distances, neighbor_ids = self.scalable_backend.vector_index.search(
        np.array(current_emb), k=15
    )
    
    for dist, neighbor_id in zip(distances, neighbor_ids):
        if neighbor_id is None or neighbor_id == current_id:
            continue
        
        sim = 1.0 / (1.0 + float(dist))
        
        if sim > 0.7:  # Threshold for relationship creation
            relationships.append({
                'source_aku_id': current_id,
                'target_aku_id': neighbor_id,
                'semantic_similarity': sim,
                'temporal_proximity': 0.0
            })
```

#### Local Temporal Relationships
```python
# Ensure local temporal connections (fallback/reinforcement)
for i in range(len(aku_ids)):
    for j in range(i + 1, len(aku_ids)):
        sim = cosine_similarity(aku_embeddings[i], aku_embeddings[j])
        if sim > 0.7:
            relationships.append({
                'source_aku_id': aku_ids[i],
                'target_aku_id': aku_ids[j],
                'semantic_similarity': sim,
                'temporal_proximity': 1.0  # Same episode
            })
```

## Prompts and Extraction Logic

### Knowledge Extraction Prompt
The system uses sophisticated prompts for extraction:

```python
KNOWLEDGE_EXTRACTION_PROMPT = """Perform knowledge extraction step-by-step:

STEP 1: Identify the main topic and key concepts in the text.
STEP 2: Determine the most important factual statements.
STEP 3: Extract atomic knowledge units (AKUs) - these should be standalone, factual statements that capture distinct pieces of knowledge.
STEP 4: Create a summary that captures the overall meaning of the text.

TEXT: {text}

REASONING: Let me think through this step by step...
- Main topic: [identify main topic]
- Key concepts: [list key concepts]
- Important facts: [list important facts]
- AKUs: [formulate atomic knowledge units]
- Summary: [create summary]

FINAL OUTPUT: Return in JSON format:
{{
    "summary": "...",
    "facts": ["...", "..."]
}}"""
```

### AKU Extraction with Reasoning
For more detailed extraction:

```python
DETAILED_AKU_EXTRACTION_PROMPT = """Perform detailed knowledge extraction with explicit reasoning:

INPUT TEXT: {text}

STEP-BY-STEP REASONING:
1. TEXT ANALYSIS:
   - Main subject(s): [identify main topics]
   - Key entities: [list important entities]
   - Important relationships: [describe key relationships]

2. ATOMIC DECOMPOSITION:
   - Break down complex statements into simpler facts
   - Identify implicit knowledge that can be made explicit
   - Ensure each fact is self-contained

3. AKU FORMULATION:
   - Create atomic knowledge units following these rules:
     * Each AKU contains exactly one fact or relationship
     * Each AKU can stand alone without context
     * Each AKU is factually accurate
     * Each AKU is concise but complete
   - Limit to approximately {max_akus} AKUs

4. QUALITY CHECK:
   - Verify each AKU meets atomicity criteria
   - Ensure no redundancy between AKUs
   - Validate factual accuracy

OUTPUT FORMAT:
{{
  "akus": ["AKU 1", "AKU 2", ...],
  "reasoning": "Brief explanation of extraction process",
  "confidence_scores": [0.8, 0.9, ...]
}}"""
```

## Scalability Features

### 1. Batch Processing
The system supports batch ingestion:

```python
async def add_akus_batch(self, workspace_id: str, aku_data: List[Dict[str, Any]]) -> List[str]:
    # Process embeddings in batch
    contents = [aku['content'] for aku in aku_data]
    embeddings_list = await self.embeddings.embed_documents(contents)
    
    # Store in database
    aku_ids = await self.storage.save_akus(workspace_id, None, akus_to_store)
    
    # Add to vector index
    embedding_array = np.array(embeddings_list).astype('float32')
    self.vector_index.add_vectors(embedding_array, aku_ids)
    
    return aku_ids
```

### 2. Vector Indexing
Integration with FAISS for efficient similarity search:

- Multiple index types (Flat, IVF, HNSW)
- Automatic dimension handling
- Training for IVF indexes
- Fallback to in-memory operations

### 3. Caching
Caching mechanisms for improved performance:

- Embedding caching
- Result caching
- TTL-based expiration
- Redis support for distributed caching

## API Integration

### Server Endpoint
The training system is exposed via REST API:

```python
@app.post("/train/{workspace_id}")
async def train(
    workspace_id: str, 
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...)
):
    # Process uploaded files
    contents_to_train = []
    for file in files:
        # Extract text content from various formats
        text_content = await extract_text_from_file(file)
        contents_to_train.append(text_content)
    
    # Run training in background
    async def run_training():
        for content in contents_to_train:
            await ekm.train(workspace_id, content)
    
    background_tasks.add_task(run_training)
    return {"status": "Processing", "file_count": len(contents_to_train)}
```

## Configuration Options

### Training Parameters
- `min_chunk_size`: Minimum chunk size (default: 512)
- `max_chunk_size`: Maximum chunk size (default: 2048)
- `semantic_threshold`: Similarity threshold for relationships (default: 0.82)
- `max_akus`: Maximum number of AKUs per extraction (default: 10)

### Performance Tuning
- Vector index type selection
- Batch processing sizes
- Caching policies
- Memory management settings

## Quality Assurance

### Extraction Validation
- JSON parsing with fallback mechanisms
- Semantic similarity thresholds
- Duplicate detection and prevention
- Confidence scoring for extractions

### Relationship Validation
- Semantic similarity filtering
- Temporal context preservation
- Cross-document linking
- Graph connectivity maintenance

## Best Practices

### For Developers
1. Use appropriate chunk sizes based on document type
2. Monitor vector index performance
3. Implement proper error handling
4. Regularly validate extraction quality
5. Optimize for your specific domain

### For Users
1. Provide clean, well-structured text
2. Use meaningful titles for documents
3. Monitor training progress
4. Review extracted knowledge periodically
5. Provide feedback for quality improvement

## Limitations and Considerations

### Known Limitations
- Fixed extraction templates may not suit all domains
- Potential for information fragmentation
- Resource-intensive processing
- No real-time training support

### Performance Considerations
- LLM costs for extraction
- Memory usage for large knowledge graphs
- Processing time for large documents
- Storage requirements for embeddings

## Future Enhancements

### Planned Improvements
- Active learning integration
- Domain-specific extraction templates
- Incremental knowledge refinement
- Cross-document relationship modeling
- Quality metrics and monitoring
- Real-time training capabilities

The EKM training system provides a comprehensive approach to knowledge ingestion, converting raw text into structured knowledge representations that enable sophisticated retrieval and reasoning capabilities.
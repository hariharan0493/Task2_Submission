graph TD
    subgraph DataSources [Knowledge Base]
        A1[MedQuAD CSV]
        A2[Medical Guidelines PDFs]
    end

    subgraph Ingestion [Ingestion Pipeline]
        B1[Recursive Character Splitter]
        B2[SentenceTransformer: all-MiniLM-L6-v2]
        B3[Google Gemini Embedding: gemini-embedding-001]
    end

    subgraph Storage [Vector Database]
        C[ChromaDB]
    end

    subgraph RAG [Retrieval & Generation]
        D[Streamlit UI]
        E[Query Embedding]
        F[Retrieval Module]
        G[Prompt Engine: Context + Safety + Instructions]
        H[LLM: Gemini 3.5 Flash]
    end

    A1 --> B1
    A2 --> B1
    B1 --> B2
    B1 --> B3
    B2 --> C
    B3 --> C
    D --> E
    E --> F
    F --> G
    G --> H
    H --> D

    style C fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px

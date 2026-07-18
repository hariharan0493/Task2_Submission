# SPML Task 2 Submission

This repository contains my submission for **SPML Task 2**, which includes:

- **Base ML**
  - Level 1 – Custom ResNet for CIFAR-10 Image Classification
  - Level 2 – Custom LSTM for Weather Forecasting
- **Applied ML**
  - Healthcare RAG (Retrieval-Augmented Generation) Assistant

The objective of this project was to understand the internal workings of deep learning architectures instead of relying on pre-built implementations, and to build a practical AI application using modern LLM technologies.

---

# Repository Structure

```text
Task2_Submission/
│
├── README.md
│
├── Base_ML/
│   ├── Level_1_ResNet/
│   ├── Level_2_LSTM/
│
└── Applied_ML/
    ├── code/
    ├── outputs/
```

---

# Level 1 – Custom ResNet (CIFAR-10)

## Objective

Build a ResNet-style Convolutional Neural Network from scratch for image classification on the CIFAR-10 dataset.

## Features

- Custom Residual Blocks
- Manual Skip Connections
- Residual Stage Transitions
- Batch Normalization
- ReLU Activation
- Global Average Pooling
- Training & Validation Curves
- Confusion Matrix
- Classification Report
- Saved Model Weights

## Dataset

- CIFAR-10

---

# Level 2 – Custom LSTM (Weather Forecasting)

## Objective

Implement an LSTM completely from scratch without using `nn.LSTM`, `nn.GRU`, or `nn.RNN`, and use it to predict future temperature values from historical weather observations.

## Implemented Components

- Input Gate
- Forget Gate
- Output Gate
- Candidate Cell State
- Cell State Update
- Hidden State Update
- Two-layer Stacked LSTM

## Dataset

- Jena Climate Dataset

## Outputs

- Training Loss
- Validation Loss
- Forecast Visualizations
- Evaluation Metrics

---

# Applied ML – Healthcare RAG Assistant

## Objective

Build a Retrieval-Augmented Generation (RAG) system capable of answering healthcare-related questions using trusted medical sources instead of relying solely on the language model.

## Knowledge Base

The system retrieves information from multiple trusted sources, including:

- MedQuAD
- WHO Guidelines
- CDC Documents
- NICE Guidelines
- Additional medical reference documents

## Tech Stack

- Python
- PyTorch
- ChromaDB
- LangChain
- Sentence Transformers
- Google Gemini API
- Streamlit

## Pipeline

```text
Medical Documents
        │
        ▼
Text Extraction
        │
        ▼
Chunking
        │
        ▼
Embedding Generation
        │
        ▼
Vector Database (ChromaDB)
        │
        ▼
Similarity Search
        │
        ▼
Retrieved Context
        │
        ▼
Gemini
        │
        ▼
Grounded Response + Citations
```

## Features

- Multi-document retrieval
- Semantic search
- Source citations
- Context-aware responses
- Reduced hallucinations
- Interactive web interface

---

# Libraries Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- ChromaDB
- LangChain
- Sentence Transformers
- Google Gemini API
- Streamlit

---

# How to Run

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Run the notebooks inside the **Base_ML** folder for Level 1 and Level 2.

4. Create a `.env` file and add your Gemini API key.

```env
GEMINI_API_KEY=your_api_key
```

5. Launch the RAG application.

```bash
streamlit run app.py
```


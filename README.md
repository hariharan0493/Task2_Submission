```markdown
# SPML Induction - Task 2 Submission

This repository contains the complete submission for the SPML Induction Task 2, covering both the Base ML and Applied ML tracks.

## Repository Overview
* **Base ML**: Implementation of custom deep learning architectures from scratch.
* **Applied ML**: Healthcare Information Assistant (RAG system).

---

## Structure
```text
├── Applied_ML/             # Healthcare Information Assistant
│   ├── architecture/       # System architecture diagram
│   ├── code/               # Streamlit app and ingestion pipelines
│   ├── outputs/            # Example interaction logs
│   └── reports/            # Design and evaluation documentation
├── Base_ML/                # Progressive ML implementation levels
│   ├── Level_1_ResNet/     # Custom ResNet implementation
│   └── Level_2_LSTM/       # Custom LSTM implementation

```

## How to Run

### Base ML Track

Navigate to the specific Level directory (e.g., `Base_ML/Level_1_ResNet/`) and open the provided `.ipynb` files in a Jupyter environment or Google Colab. Ensure the required libraries are installed.

### Applied ML Track

1. **Setup Environment**: Ensure you have a `.env` file containing your `GEMINI_API_KEY`.
2. **Install Dependencies**:
```bash
cd Applied_ML/code
pip install -r requirements.txt

```


3. **Ingest Data**: Run the ingestion scripts to build the local vector database:
```bash
python ingest_medquad.py
python ingest_pdf.py

```


4. **Launch Application**:
```bash
streamlit run app.py

```



---

## Notes

* The repository is structured to facilitate easy evaluation of both implementation code and design reasoning.
* Sensitive API keys are not included; please configure your local `.env` file to run the Applied ML application.

```

```

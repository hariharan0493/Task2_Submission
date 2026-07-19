# Level 2: LSTM Implementation

## Project Objective
The objective of this level is to implement a Long Short-Term Memory (LSTM) network to perform sequence-based modeling tasks. This implementation focuses on capturing temporal dependencies within data to provide accurate forecasts or classifications.

## Methodology
*   **Model Architecture**: A custom LSTM network built to handle sequential input data, leveraging internal gating mechanisms to manage long-term dependencies.
*   **Data Handling**: The model is trained on sequential data structured as a time-series or sequence set.
*   **Training & Evaluation**: The model undergoes rigorous training phases, with performance evaluated through loss minimization and accuracy metrics calculated over validation sets.

## Architecture
*   **The Role of Each Gate**:
    *   **Forget Gate**: Decides what information from the previous cell state should be discarded.
    *   **Input Gate**: Determines which new information from the current input should be stored in the cell state.
    *   **Output Gate**: Filters the updated cell state to decide what the hidden state output should be for the next time step.
*   **Information Retention and Forgetting**: The cell state acts as a "conveyor belt," allowing information to flow across time steps with minimal interaction, mitigated by the gates which add or remove information through pointwise multiplication and sigmoid activation.
*   **Training Stability**: To ensure stability, the architecture utilizes gradient clipping and appropriate weight initialization to counteract the vanishing gradient problem common in standard RNNs.
*   **Sequence Length Considerations**: While LSTMs are superior at handling longer sequences than traditional RNNs, performance can still degrade with extremely long dependencies; this is managed by careful architectural design and truncation strategies.
*   **Forecasting Challenges**: Forecasting is complicated by the presence of non-stationary data, noise in sequential inputs, and the difficulty of predicting long-term trends from limited historical context.

## How to Run
1. **Platform**: It is highly recommended to run this project on **Google Colab**.
2. **Data Preparation**: 
   - Ensure the `med_quad.csv` file is placed in the `.data/` directory in Google Colab.
3. **Execution**:
   - Upload `code/Level_2.ipynb` to Google Colab.
   - Ensure your environment has access to the `.data/` folder containing the dataset.
4. **Model Weights**:
   - Pre-trained model weights are located in `model_weights/model_weights (1).pth`.

# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Szarka Levente
- **Aiming for +1 Mark**: No

### Solution Description

This project implements a deep learning–based pipeline for detecting and classifying bullish and bearish flag-type chart patterns from financial time-series data. The goal is to demonstrate an end-to-end machine learning workflow, including data preparation, model training, evaluation, and inference, rather than to achieve state-of-the-art predictive performance.

The input data consists of OHLC (open, high, low, close) price sequences extracted from labeled time intervals. During preprocessing, labeled segments are aligned with raw price data, normalized per segment, and padded or truncated to a fixed sequence length. The resulting dataset is split into training, validation, and test sets in a reproducible manner.

As a baseline deep learning model, a single-layer LSTM (Long Short-Term Memory) network is used. The LSTM processes each price sequence and outputs a fixed-length hidden representation, which is passed to a fully connected layer to perform multi-class classification of chart patterns. The model is trained using cross-entropy loss and the Adam optimizer, with early stopping based on validation loss to prevent overfitting.

Evaluation is performed on a held-out test set using standard classification metrics, including accuracy, confusion matrix, and precision/recall/F1 scores. Due to the small size and imbalance of the dataset, the quantitative results are limited and not intended for practical trading use. However, the system successfully demonstrates correct model behavior, reproducible training, and a fully containerized workflow. Inference is supported on unseen sequences, producing class probabilities and confidence scores.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v "$(pwd)/data:/app/data" dl-project > log/run.log 2>&1
```

*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `.gitattributes`: Git configuration file used to enforce consistent file formatting across operating systems.
    - `.gitignore`: Prevents logs, environment-specific files and directories from being published to the repository.
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: The shell script that imitates the pipeline.

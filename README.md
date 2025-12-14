# Deep Learning Class (VITMMA19) Project Work template

[Complete the missing parts and delete the instruction parts before uploading.]

## Submission Instructions

### Data Preparation

**Important:** You must provide a script (or at least a precise description) of how to convert the raw database into a format that can be processed by the scripts.
* The scripts should ideally download the data from there or process it directly from the current sharepoint location.
* Or if you do partly manual preparation, then it is recommended to upload the prepared data format to a shared folder and access from there.

[Describe the data preparation process here]

### Submission Checklist

Before submitting your project, ensure you have completed the following steps.
**Please note that the submission can only be accepted if these minimum requirements are met.**

- [x] **Project Information**: Filled out the "Project Information" section (Topic, Name, Extra Credit).
- [ ] **Solution Description**: Provided a clear description of your solution, model, and methodology.
- [x] **Extra Credit**: If aiming for +1 mark, filled out the justification section.
- [ ] **Data Preparation**: Included a script or precise description for data preparation.
- [x] **Dependencies**: Updated `requirements.txt` with all necessary packages and specific versions.
- [x] **Configuration**: Used `src/config.py` for hyperparameters and paths, contains at least the number of epochs configuration variable.
- [x] **Logging**:
    - [x] Log uploaded to `log/run.log`
    - [x] Log contains: Hyperparameters, Data preparation and loading confirmation, Model architecture, Training metrics (loss/acc per epoch), Validation metrics, Final evaluation results, Inference results.
- [x] **Docker**:
    - [x] `Dockerfile` is adapted to your project needs.
    - [x] Image builds successfully (`docker build -t dl-project .`).
    - [x] Container runs successfully with data mounted (`docker run ...`).
    - [x] The container executes the full pipeline (preprocessing, training, evaluation).
- [x] **Cleanup**:
    - [x] Removed unused files.
    - [x] **Deleted this "Submission Instructions" section from the README.**

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Szarka Levente
- **Aiming for +1 Mark**: No

### Solution Description

[Provide a short textual description of the solution here. Explain the problem, the model architecture chosen, the training methodology, and the results.]

### Extra Credit Justification

[If you selected "Yes" for Aiming for +1 Mark, describe here which specific part of your work (e.g., innovative model architecture, extensive experimentation, exceptional performance) you believe deserves an extra mark.]

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

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

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - yet to fill out

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `.gitignore`: Prevents logs, environment-specific files and directories from being published to the repository.
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.
    - `run.sh`: The shell script that imitates the pipeline.

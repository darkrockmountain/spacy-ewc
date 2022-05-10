# EWC-Enhanced spaCy NER Training

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project, **spacy-ewc**, integrates **Elastic Weight Consolidation (EWC)** into spaCy's Named Entity Recognition (NER) pipeline to mitigate catastrophic forgetting during sequential learning tasks. By applying EWC, the model retains important information from previous tasks while learning new ones, leading to improved performance in continual learning scenarios.

## Motivation

In sequential or continual learning, neural networks often suffer from **catastrophic forgetting**, where the model forgets previously learned information upon learning new tasks. EWC addresses this issue by penalizing changes to important parameters identified during earlier training phases. Integrating EWC into spaCy's NER component allows us to build more robust NLP models capable of learning incrementally without significant performance degradation on earlier tasks.

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Usage](#usage)
  - [Running the Example Script](#running-the-example-script)
  - [Script Workflow](#script-workflow)
  - [Expected Output](#expected-output)
- [Detailed Explanation](#detailed-explanation)
  - [EWC Theory](#ewc-theory)
    - [Catastrophic Forgetting](#catastrophic-forgetting)
    - [Elastic Weight Consolidation](#elastic-weight-consolidation)
    - [Mathematical Formulation](#mathematical-formulation)
    - [Fisher Information Matrix](#fisher-information-matrix)
    - [EWC Penalty Term](#ewc-penalty-term)
    - [Gradient Adjustment](#gradient-adjustment)
  - [Integration with spaCy](#integration-with-spacy)
    - [EWC Class Workflow](#ewc-class-workflow)
    - [EWC Class Methods](#ewc-class-methods)
    - [Model Wrapping with `EWCModelWrapper`](#model-wrapping-with-ewcmodelwrapper)
    - [Training Workflow with EWC](#training-workflow-with-ewc)
- [Code Structure](#code-structure)
- [Extending the Project](#extending-the-project)
  - [Adding New Components](#adding-new-components)
  - [Customizing EWC Parameters](#customizing-ewc-parameters)
  - [Experimentation](#experimentation)
- [Limitations](#limitations)
- [References](#references)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- **Python 3.8** or higher
- **spaCy** (compatible version with your Python installation)
- **Thinc** (spaCy's machine learning library)
- Other dependencies as listed in `pyproject.toml`

### Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/darkrockmountain/spacy-ewc.git
   ```

2. **Navigate to the project directory**:

   ```bash
   cd spacy-ewc
   ```

3. **Install required packages**:

   - **Core dependencies only**:

     ```bash
     pip install .
     ```

   - **Development dependencies** (recommended for contributors):

     ```bash
     pip install .[dev]
     ```

4. **Download the spaCy English model**:

   Since `en_core_web_sm` is listed as a development dependency, it will be installed if you used `pip install .[dev]`. Otherwise, install it manually:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running the Example Script

The example script demonstrates how to train a spaCy NER model with EWC applied:

```bash
python examples/ewc_ner_training_example.py
```

### Script Workflow

The script performs the following steps:

1. **Load the pre-trained spaCy English model**.
2. **Add new entity labels** (`BUDDY`, `COMPANY`) to the NER component.
3. **Prepare training and test data**.
4. **Initialize the EWC wrapper** with the NER pipe and original spaCy labels.
5. **Train the NER model** using EWC over multiple epochs.
6. **Evaluate the model** on a test sentence and display recognized entities.

### Expected Output

- **Training Loss**: Displays the loss after training.
- **Entities in Test Sentence**: Lists the entities recognized in the test sentence after training.

Example output:

```console
Training loss: 8.49725

Entities in test sentence:
Elon Musk: BUDDY
SpaceX: COMPANY
approximately $100 million: MONEY
El Segundo: GPE
California: GPE
```

## Detailed Explanation

### EWC Theory

#### Catastrophic Forgetting

In machine learning, **catastrophic forgetting** refers to the abrupt and complete forgetting of previously learned information upon learning new information. Neural networks, when trained sequentially on multiple tasks without access to data from previous tasks, often overwrite the weights important for the old tasks with weights relevant to the new task.

#### Elastic Weight Consolidation

**Elastic Weight Consolidation (EWC)** is a regularization technique proposed to overcome catastrophic forgetting. It allows the model to learn new tasks while preserving performance on previously learned tasks by slowing down learning on important weights for old tasks.

#### Mathematical Formulation

The key idea behind EWC is to add a penalty term to the loss function that discourages significant changes to parameters that are important for previous tasks.

The total loss function for the current task becomes:

$$
L_{\text{total}}(\theta) = L_{\text{task}}(\theta) + \Omega(\theta)
$$

- **$L_{\text{task}}(\theta)$**: The loss function for the current task.
- **$\Omega(\theta)$**: The EWC penalty term.

#### Fisher Information Matrix

The EWC penalty term is based on the **Fisher Information Matrix (FIM)**, which measures the amount of information that an observable random variable carries about an unknown parameter upon which the probability depends.

For each parameter $\theta_i$, the importance is estimated using the diagonal of the FIM, denoted as $F_i$.

#### EWC Penalty Term

The EWC penalty term is defined as:

$$
\Omega(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
$$

- **$\theta$**: Current model parameters.
- **$\theta^*$**: Optimal parameters learned from previous tasks.
- **$F_i$**: Diagonal elements of the Fisher Information Matrix for parameter $\theta_i$.
- **$\lambda$**: Regularization strength.

This term penalizes deviations of the current parameters $\theta$ from the previous optimal parameters $\theta^*$, scaled by the importance weights $F_i$.

#### Gradient Adjustment

During training, the gradient of the total loss function with respect to each parameter $\theta_i$ is:

$$
\frac{\partial L_{\text{total}}}{\partial \theta_i} = \frac{\partial L_{\text{task}}}{\partial \theta_i} + \lambda F_i (\theta_i - \theta_i^*)
$$

This means the gradient update is adjusted to consider both the task-specific loss and the EWC penalty, preventing significant changes to important parameters.

### Integration with spaCy

#### EWC Class Workflow

The `EWC` class encapsulates the implementation of the EWC algorithm within the spaCy framework. The workflow involves:

1. **Initialization**:
   - **Capture Initial Parameters ($\theta^*$)**:
     - After training the initial task, capture and store the model's parameters.
   - **Compute Fisher Information Matrix (FIM)**:
     - Use the initial task data to compute gradients.
     - Square and average these gradients to estimate the FIM.

2. **Training on New Task**:
   - **Compute EWC Penalty**:
     - During training on a new task, compute the EWC penalty using the stored $\theta^*$ and $F_i$.
   - **Adjust Gradients**:
     - Modify the gradients by adding $\lambda F_i (\theta_i - \theta_i^*)$ before updating the parameters.

#### EWC Class Methods

- **`__init__(self, pipe, data, lambda_=1000.0, pipe_name=None)`**:
  - Initializes the EWC instance.
  - **Parameters**:
    - `pipe`: The spaCy pipeline component (e.g., `ner`).
    - `data`: Training examples used to compute the FIM.
    - `lambda_`: Regularization strength.
  - **Operations**:
    - Validates the pipe.
    - Captures initial parameters ($\theta^*$).
    - Computes the FIM.

- **`_capture_current_parameters(self, copy=False)`**:
  - Retrieves the current model parameters.
  - If `copy` is `True`, returns a deep copy to prevent modifications.

- **`_compute_fisher_matrix(self, examples)`**:
  - Computes the Fisher Information Matrix.
  - For each parameter:
    - Accumulates the squared gradients over the dataset.
    - Averages the accumulated values to estimate $F_i$.

- **`compute_ewc_penalty(self)`**:
  - Calculates the EWC penalty $\Omega(\theta)$.
  - Uses the stored $\theta^*$ and computed $F_i$.

- **`compute_gradient_penalty(self)`**:
  - Computes the gradient of the EWC penalty with respect to $\theta$.
  - For each parameter:
    - Calculates $\lambda F_i (\theta_i - \theta_i^*)$.

- **`apply_ewc_penalty_to_gradients(self)`**:
  - Adjusts the model's gradients in-place by adding the EWC gradient penalty.
  - Ensures that the penalty is applied before the optimizer updates the parameters.

#### Model Wrapping with `EWCModelWrapper`

- The `EWCModelWrapper` class wraps the spaCy model's `finish_update` method.
- It ensures that the EWC penalty is applied to the gradients before the optimizer step.
- By overriding `finish_update`, it seamlessly integrates the EWC adjustments into the standard spaCy training loop.

#### Training Workflow with EWC

1. **Initialize EWC**:
   - Use `create_ewc_pipe` to wrap the spaCy component with EWC.
   - This captures $\theta^*$ and computes the FIM.

2. **Training Loop**:
   - For each training batch:
     - Compute task-specific loss and gradients.
     - **Apply EWC Penalty**:
       - Adjust gradients using `apply_ewc_penalty_to_gradients`.
     - **Update Parameters**:
       - Use the optimizer to update parameters with the adjusted gradients.

3. **Evaluation**:
   - After training, evaluate the model on the test data.
   - The model should retain performance on previous tasks while learning the new task.

## Code Structure

- **`examples/ewc_ner_training_example.py`**: Example script demonstrating EWC-enhanced NER training.

- **`data_examples/`**
  - `training_data.py`: Contains custom training data with new entity labels.
  - `original_spacy_labels.py`: Contains original spaCy NER labels for EWC reference.

- **`src/`**
  - **`spacy_ewc/`**
    - `ewc.py`: Implements the `EWC` class for calculating EWC penalties and adjusting gradients.
    - `vector_dict.py`: Defines `VectorDict`, a specialized dictionary for model parameters and gradients.
  - **`spacy_wrapper/`**
    - `ewc_spacy_wrapper.py`: Provides a wrapper to integrate EWC into spaCy's pipeline components.
  - **`ner_trainer/`**
    - `ewc_ner_trainer.py`: Contains functions to train NER models with EWC applied to gradients.
  - **`utils/`**
    - `extract_labels.py`: Utility function to extract labels from training data.
    - `generate_spacy_entities.py`: Generates spaCy-formatted entity annotations from sentences.

## Extending the Project

### Adding New Components

To extend EWC to other spaCy pipeline components (e.g., `textcat`, `parser`):

1. **Modify the `EWC` Class**:
   - Ensure the class captures and computes parameters relevant to the new component.
   - Adjust methods to handle different types of model architectures.

2. **Adjust FIM Computation**:
   - Use appropriate loss functions and data for computing the Fisher Information Matrix for the new component.

3. **Wrap the Component**:
   - Use `create_ewc_pipe` to wrap the new component with EWC functionality.

### Customizing EWC Parameters

- **Adjusting $\lambda$ (lambda)**:
  - Controls the balance between learning new information and retaining old knowledge.
  - Experiment with different values to find the optimal balance for your use case.

- **Modifying FIM Calculation**:
  - Consider alternative methods for estimating parameter importance.
  - For example, use empirical Fisher Information or other approximations.

### Experimentation

- **Different Datasets**: Test the model on various datasets to evaluate the effectiveness of EWC in different scenarios.

- **Sequential Tasks**: Simulate continual learning by training on multiple tasks sequentially and observing performance retention.

- **Parameter Sensitivity**: Analyze how changes in $\lambda$ and other hyperparameters affect the model's performance.

## Limitations

- **Diagonal Approximation**: The implementation uses a diagonal approximation of the FIM, which assumes parameter independence and may not capture all parameter interactions.

- **Computational Overhead**: Calculating the FIM and adjusting gradients adds computational complexity and may increase training time.

- **Memory Requirements**: Storing $\theta^*$ and $F_i$ for all parameters can be memory-intensive, especially for large models.

- **Limited to Known Parameters**: EWC is effective for parameters seen during initial training. New parameters introduced in later tasks are not accounted for in the penalty term.

## References

- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. Proceedings of the National Academy of Sciences, 114(13), 3521-3526. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

- spaCy Documentation: [https://spacy.io/](https://spacy.io/)

- Thinc Documentation: [https://thinc.ai/](https://thinc.ai/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or further information, please contact the NLP Team at [dev@darkrockmountain.com](mailto:dev@darkrockmountain.com).

---

*This README is intended to assist team members and contributors in understanding and utilizing the EWC-enhanced spaCy NER training framework.*

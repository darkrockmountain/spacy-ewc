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
  - [Integration with spaCy](#integration-with-spacy)
- [Code Structure](#code-structure)
- [Extending the Project](#extending-the-project)
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
Space-X: COMPANY
approximately $100 million: MONEY
El Segundo: GPE
California: GPE
```

## Detailed Explanation

### EWC Theory

**Elastic Weight Consolidation (EWC)** is a technique to prevent catastrophic forgetting in neural networks during sequential learning tasks. It works by adding a regularization term to the loss function, which penalizes significant changes to parameters that are important for previously learned tasks.

Mathematically, the EWC penalty $\Omega(\theta)$ is defined as:

$$
\Omega(\theta) = \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2
$$

- **$\theta^*$**: Parameters of the model after training on the initial task.
- **$\theta$**: Current parameters.
- **$F$**: Diagonal elements of the Fisher Information Matrix, representing the importance of each parameter.
- **$\lambda$ (lambda)**: Regularization strength controlling the trade-off between retaining old knowledge and learning new information.

### Integration with spaCy

The integration involves:

1. **Capturing Initial Parameters ($\theta^*$)**: After training the initial NER model, we store its parameters to use as a reference.

2. **Computing the Fisher Information Matrix (FIM)**:
   - Use the initial training data to compute gradients.
   - Square the gradients and average over the dataset to estimate the importance of each parameter.

3. **Adjusting the Loss Function**:
   - During training on new data, compute the EWC penalty and add it to the task-specific loss.

4. **Updating Gradients**:
   - Modify the gradients by adding the derivative of the EWC penalty before updating the model parameters.

5. **Model Wrapping**:
   - Use `EWCModelWrapper` to intercept the `finish_update` method of the spaCy model, applying the EWC penalty seamlessly during training.

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

1. Modify the `EWC` class to accommodate the specific component's parameters.
2. Adjust the Fisher Information Matrix computation to use appropriate loss functions.
3. Wrap the desired component using `create_ewc_pipe`.

### Customizing EWC Parameters

- **Adjusting $\lambda$ (lambda)**:
  - The `lambda_` parameter in the `EWC` class controls the regularization strength.
  - Higher values place more emphasis on retaining previous knowledge.

- **Modifying the Fisher Information Matrix Calculation**:
  - Experiment with different methods of estimating parameter importance.
  - Consider using a full FIM if computational resources allow.

### Experimentation

- **Different Datasets**: Test the EWC-enhanced model on various datasets to assess performance improvements.
- **Sequential Tasks**: Simulate real-world scenarios by training on multiple tasks sequentially.

## Limitations

- **Diagonal Approximation**: Uses a diagonal approximation of the FIM, which may not capture parameter correlations.
- **Computational Overhead**: EWC introduces additional computations during training, which may increase training time.
- **Limited to Known Parameters**: EWC is effective for parameters seen during initial training; new parameters introduced later are not accounted for.

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

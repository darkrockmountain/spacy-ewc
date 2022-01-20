# EWC-Enhanced spaCy NER Training

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

This project, **spacy-ewc**, integrates **Elastic Weight Consolidation (EWC)** into spaCy's Named Entity Recognition (NER) pipeline to mitigate catastrophic forgetting during sequential learning tasks. By applying EWC, the model retains important information from previous tasks while learning new ones, leading to improved performance in continual learning scenarios.

## Motivation

In sequential or continual learning, neural networks often suffer from **catastrophic forgetting**, where the model forgets previously learned information upon learning new tasks. EWC addresses this issue by penalizing changes to important parameters identified during earlier training phases. Integrating EWC into spaCy's NER component allows us to build more robust NLP models capable of learning incrementally without significant performance degradation on earlier tasks.

## Installation

### Prerequisites

- **Python 3.8** or higher
- **spaCy** (compatible version with your Python installation)
- **Thinc** (spaCy's machine learning library)
- Other dependencies as listed in `requirements.txt` (if provided)

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

   If you're installing core dependencies only:

   ```bash
   pip install .
   ```

   For development dependencies (recommended for contributors):

   ```bash
   pip install .[dev]
   ```

4. **Download the spaCy English model**:

   Since `en_core_web_sm` is listed as a development dependency, it will be installed if you used `pip install .[dev]`. Otherwise, you can install it manually with:

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

## How EWC is Integrated

1. **Initialization**:
   - The `EWC` class captures the model's parameters after initial training (θ\*).
   - Computes the Fisher Information Matrix (FIM) using provided data to estimate parameter importance.

2. **Penalty Computation**:
   - During new task training, the EWC penalty is computed based on the difference between current parameters and θ\*, scaled by the FIM.

3. **Gradient Adjustment**:
   - The EWC penalty is applied to the gradients before the optimizer updates the model parameters.

4. **Model Wrapping**:
   - The `EWCModelWrapper` class wraps the original spaCy model to integrate EWC seamlessly into the training process.

## Extending the Project

- **Adding New Components**: Extend EWC to other spaCy pipeline components beyond NER.
- **Customizing EWC Parameters**: Adjust the `lambda_` parameter in the `EWC` class to control EWC penalty strength.
- **Experimentation**: Test the EWC-enhanced model on different datasets to evaluate effectiveness in various scenarios.

## Limitations

- **Diagonal Approximation**: Currently uses a diagonal approximation of the Fisher Information Matrix, which may not capture all parameter dependencies.
- **Computational Overhead**: Applying EWC penalties adds computational complexity during training.

## References

- Kirkpatrick, J., et al. (2017). *Overcoming catastrophic forgetting in neural networks*. Proceedings of the National Academy of Sciences, 114(13), 3521-3526. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or further information, please contact the NLP Team at [dev@darkrockmountain.com](mailto:dev@darkrockmountain.com).

---

*This README is intended to assist team members and contributors in understanding and utilizing the EWC-enhanced spaCy NER training framework.*

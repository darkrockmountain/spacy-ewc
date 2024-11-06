import spacy
from spacy.training import Example
from thinc.optimizers import Optimizer
from spacy_ewc import create_ewc_pipe
from spacy_ewc.utils.extract_labels import extract_labels


def run_ewc_example():
    """
    This function demonstrates the use of Elastic Weight Consolidation (EWC) for Named Entity Recognition (NER) in spaCy.
    EWC enables the model to learn new entity labels while retaining previous knowledge by applying a penalty to the loss
    function. This regularization strategy helps prevent catastrophic forgetting, especially beneficial when training
    a model incrementally on multiple tasks or datasets.

    Workflow:
    1. Load spaCy's English model and retrieve or add the NER component.
    2. Define original NER-labeled examples to initialize the EWC parameters and the Fisher Information Matrix.
    3. Set up and initialize EWC on the NER component with original labels.
    4. Define custom training data with new labels and add those labels to the NER pipeline.
    5. Choose and set up an optimizer to update model parameters.
    6. Train the model using EWC to preserve previously learned knowledge alongside new information.
    7. Evaluate the model to assess its ability to adapt to new labels while retaining knowledge of the original labels.
    """

    # Load spaCy's small English model
    nlp = spacy.load("en_core_web_sm")

    # Get the NER component from the pipeline or add it if not present
    ner = nlp.get_pipe("ner") if nlp.has_pipe("ner") else nlp.add_pipe("ner")

    for epoch in range(2):
        nlp.update(
            [
                Example.from_dict(nlp.make_doc(text), annotations)
                for text, annotations in original_spacy_labels
            ]
        )

    # Initialize the EWC wrapper for the NER component with the original labeled examples.
    # This step enables the model to retain knowledge of previous tasks by preserving parameters
    # critical to the original labels, which helps prevent catastrophic forgetting.
    #
    # The `create_ewc_pipe` function achieves this by:
    # - Capturing a snapshot of the model's current parameters, serving as a baseline.
    # - Calculating the Fisher Information Matrix (FIM) to identify important parameters.
    # - Setting up an EWC penalty that discourages changes to these parameters during future training.
    #
    # By applying EWC, the model can learn new entity labels (like "BUDDY" and "COMPANY") while
    # retaining performance on existing labels (such as "PERSON" or "ORG").
    create_ewc_pipe(
        ner,
        [
            Example.from_dict(nlp.make_doc(text), annotations)
            for text, annotations in original_spacy_labels
        ],
    )

    # Extract custom labels from the training data and add them to the NER pipeline
    # Our custom labels in this dataset are "BUDDY" and "COMPANY"
    training_labels = extract_labels(training_data)
    for label in training_labels:
        ner.add_label(label)

    # Define the training data as a list of spaCy Example objects
    examples = [
        Example.from_dict(nlp.make_doc(text), ann) for text, ann in training_data
    ]

    # Set up the optimizer for model updates. Using None defaults to spaCy's internal optimizer.
    # It's important to note that `nlp.initialize()` is avoided here because it resets parameters and gradients,
    # which conflicts with the EWC mechanism.
    # Alternatively, you can use `nlp.begin_training()` or `nlp.resume_training()`, which work effectively
    # to prepare the optimizer without altering initialized parameters. If preferred, you can specify a custom
    # optimizer from Thinc, such as RAdam, Adam, or SGD, which can be accessed as shown below:
    #
    # optimizer = Adam(learn_rate=0.001)
    #
    # Each of these optimizers provides specific advantages: RAdam introduces adaptive learning rate scaling,
    # while SGD offers simplicity and robustness for smaller datasets. More information can be found in Thinc's
    # optimizer documentation: https://thinc.ai/docs/api-optimizers.

    # Or use a custom optimizer, e.g., `optimizer = Adam(learn_rate=0.001)`
    optimizer: Optimizer = None

    # Prepare a test sentence to evaluate the model's performance after
    # training
    test_sentence = "Elon Musk founded SpaceX in 2002 as the CEO and lead engineer, investing approximately $100 million of his own money into the company, which was initially based in El Segundo, California."

    # Train the model using EWC to retain old knowledge while learning the new custom labels
    # The Elastic Weight Consolidation (EWC) technique helps prevent the model from forgetting previously learned
    # knowledge by penalizing updates to parameters that were significant in the original model. This works by
    # calculating and applying a Fisher Information Matrix-based penalty to the model's loss function.
    # Number of epochs, defining how many times to iterate over the data
    for epoch in range(10):
        losses = {}

        # Create batches for training. Here, we use a compounding batch size, which starts at 4 and gradually grows
        # to 32, with an increment factor of 1.001. Compounding allows for more efficient memory usage as the model
        # "warms up" and progressively handles larger batches.
        batches = spacy.util.minibatch(
            examples, size=spacy.util.compounding(4.0, 32.0, 1.001)
        )

        # Iterate over each batch and update the model parameters.
        # During updates, EWC penalizes parameter changes based on the Fisher Information Matrix, ensuring that
        # parameters important to previous knowledge are preserved. This allows the model to incorporate new
        # entity labels like "BUDDY" and "COMPANY" without overwriting
        # previously learned labels, such as "PERSON" or "ORG."
        for batch in batches:
            nlp.update(examples=batch, sgd=optimizer, losses=losses)

    # Display the accumulated training loss for the NER component after all
    # epochs
    print("Training loss:", losses["ner"])

    # Run the test sentence through the model to evaluate results
    # Expected Result: The model should now recognize the new "BUDDY" and "COMPANY" labels in addition to
    # the original labels, demonstrating its ability to retain prior knowledge
    # while integrating new information.
    doc = nlp(test_sentence)
    print("\nEntities in test sentence:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")


# Define the original annotated data extracted from `data_examples/original_spacy_labels.py`.
# This data includes sentences labeled with entities that the spaCy model should recognize
# (e.g., PERSON, ORG, DATE). It is used to initialize parameters for the Elastic Weight
# Consolidation (EWC) example, including setting up the Fisher Information
# Matrix.

original_spacy_labels = [
    # Each sentence contains labeled entities in the format (start, end, label).
    # This setup will allow the EWC method to recognize and retain knowledge of these entities
    # even as new information is introduced.
    (
        "John bought a new car yesterday.",
        {"entities": [(0, 4, "PERSON"), (22, 31, "DATE")]},
    ),
    (
        "Microsoft announced its new software in San Francisco.",
        {"entities": [(0, 9, "ORG"), (40, 53, "GPE")]},
    ),
    (
        "The meeting is scheduled for 3 p.m. on Friday.",
        {"entities": [(29, 35, "TIME"), (39, 45, "DATE")]},
    ),
    (
        "Sarah earned $500 in tips last week.",
        {
            "entities": [
                (0, 5, "PERSON"),
                (14, 17, "MONEY"),
                (26, 35, "DATE"),
            ]
        },
    ),
    (
        "Amazon plans to open a new warehouse in Seattle.",
        {"entities": [(0, 6, "ORG"), (40, 47, "GPE")]},
    ),
    (
        "On June 15th, the festival will take place in Central Park.",
        {"entities": [(3, 12, "DATE"), (46, 58, "LOC")]},
    ),
    (
        "Alice went to the dentist at 10:30 a.m. today.",
        {
            "entities": [
                (0, 5, "PERSON"),
                (29, 39, "TIME"),
                (40, 45, "DATE"),
            ]
        },
    ),
    (
        "The company raised $2 million in funding.",
        {"entities": [(19, 29, "MONEY")]},
    ),
    (
        "Tom visited Paris in July.",
        {
            "entities": [
                (0, 3, "PERSON"),
                (12, 17, "GPE"),
                (21, 25, "DATE"),
            ]
        },
    ),
    (
        "Carlos bought a ticket to the concert for $75.",
        {"entities": [(0, 6, "PERSON"), (43, 45, "MONEY")]},
    ),
    (
        "Lucy paid $45 for her meal at the Italian restaurant.",
        {
            "entities": [
                (0, 4, "PERSON"),
                (11, 13, "MONEY"),
                (34, 41, "NORP"),
            ]
        },
    ),
    (
        "Ryan's flight landed at 6:00 a.m. today.",
        {
            "entities": [
                (0, 4, "PERSON"),
                (24, 33, "TIME"),
                (34, 39, "DATE"),
            ]
        },
    ),
    # Additional example:
    # In the following example, SpaceX is included as it might not be correctly tokenized or classified.
    # Normally, one would use a Matcher in spaCy to define SpaceX as an ORG, but here, we let the model
    # interpret it based on its default en_core_web_sm configuration. For this example, we retain the
    # original labels provided by spaCy's model (classifying SpaceX as PERSON) to ensure consistency
    # with the initial setup of EWC.
    # Note: In a real application, you would train the model to recognize SpaceX as an ORG or
    # use a Matcher for better accuracy. See https://spacy.io/api/matcher for more details.
    (
        "In September 2023, SpaceX launched the Starship rocket from its Texas base, hoping to send the first humans to Mars by 2026.",
        {
            "entities": [
                (3, 17, "DATE"),
                (19, 25, "ORG"),
                (39, 47, "PRODUCT"),
                (64, 69, "GPE"),
                (95, 100, "ORDINAL"),
                (111, 115, "LOC"),
                (119, 123, "DATE"),
            ]
        },
    ),
    # Add more annotated examples as needed to expand the EWC training dataset.
]


# Define new training data extracted from `data_examples/training_data.py`
# This data includes custom labels to train the model further with EWC
training_data = [
    ("John Doe is a person.", {"entities": [(0, 8, "BUDDY")]}),
    ("OpenAI is a company.", {"entities": [(0, 6, "COMPANY")]}),
    (
        "Jane Smith works at Microsoft.",
        {"entities": [(0, 10, "BUDDY"), (20, 29, "COMPANY")]},
    ),
    (
        "GMaps was developed at Google.",
        {"entities": [(23, 29, "COMPANY")]},
    ),
    (
        "John Doe works at OpenAI.",
        {"entities": [(0, 8, "BUDDY"), (18, 24, "COMPANY")]},
    ),
    (
        "John Doe and Jane Smith are colleagues at OpenAI.",
        {
            "entities": [
                (0, 8, "BUDDY"),
                (13, 23, "BUDDY"),
                (42, 48, "COMPANY"),
            ]
        },
    ),
    (
        "OpenAI hired John Doe.",
        {"entities": [(0, 6, "COMPANY"), (13, 21, "BUDDY")]},
    ),
    (
        "Alice Johnson joined Amazon last year.",
        {"entities": [(0, 13, "BUDDY"), (21, 27, "COMPANY")]},
    ),
    (
        "Bob Lee and Carol King work for Facebook.",
        {
            "entities": [
                (0, 7, "BUDDY"),
                (12, 22, "BUDDY"),
                (32, 40, "COMPANY"),
            ]
        },
    ),
    (
        "Diana Prince was promoted at DC Comics.",
        {"entities": [(0, 12, "BUDDY"), (29, 38, "COMPANY")]},
    ),
    (
        "Evan Wright is the CEO of Tesla.",
        {"entities": [(0, 11, "BUDDY"), (26, 31, "COMPANY")]},
    ),
    (
        "Frank Miller collaborates with Pixar Studios.",
        {"entities": [(0, 12, "BUDDY"), (31, 44, "COMPANY")]},
    ),
    (
        "Steve Jobs founded Apple.",
        {"entities": [(0, 10, "BUDDY"), (19, 24, "COMPANY")]},
    ),
    # Add more variations as needed
]


if __name__ == "__main__":
    run_ewc_example()

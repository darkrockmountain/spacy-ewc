import spacy
from spacy.training import Example
from spacy_wrapper.ewc_spacy_wrapper import EWCPipeWrapper
from data_examples.training_data import training_data
from data_examples.original_spacy_labels import original_spacy_labels
from utils.extract_labels import extract_labels


def run_ewc_example():
    # Load spaCy's small English model
    nlp = spacy.load("en_core_web_sm")

    # Get the NER component from the pipeline or add it if not present
    ner = nlp.get_pipe("ner") if nlp.has_pipe("ner") else nlp.add_pipe("ner")

    # Extract labels from the training data and add them to the NER pipeline
    training_labels = extract_labels(training_data)
    for label in training_labels:
        ner.add_label(label)

    # Prepare test sentence
    test_sentence = "Elon Musk founded SpaceX in 2002 as the CEO and lead engineer, investing approximately $100 million of his own money into the company, which was initially based in El Segundo, California, before moving to Hawthorne, California."

    # Initialize the EWCPipeWrapper with the NER pipe and original labeled examples
    wrapped_ner = EWCPipeWrapper(ner, [Example.from_dict(nlp.make_doc(text), annotations) for text, annotations in original_spacy_labels])

    # Define the training data as a list of Examples
    examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in training_data]

    # Set up optimizer for updating the model
    optimizer = nlp.initialize()

    # Train the model using EWC and the wrapped NER component
    for epoch in range(10):  # Define the number of epochs
        losses = {}
        
        # Create batches for training
        batches = spacy.util.minibatch(examples, size=spacy.util.compounding(4.0, 32.0, 1.001))
        for batch in batches:
            wrapped_ner.update(examples=batch, sgd=optimizer, losses=losses)

    # Display training loss
    print("Training loss:", losses["ner"])

    # Run the test sentence through the model to evaluate results
    doc = nlp(test_sentence)
    print("\nEntities in test sentence:")
    for ent in doc.ents:
        print(f"{ent.text}: {ent.label_}")


if __name__ == "__main__":
    run_ewc_example()

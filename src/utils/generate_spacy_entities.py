def generate_spacy_entities(sentences, nlp):
    results = []
    for sentence in sentences:
        doc = nlp(sentence)
        entities = [(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]  # Extract entities and their labels
        results.append((sentence, {"entities": entities}))
    return results

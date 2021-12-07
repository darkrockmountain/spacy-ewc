def extract_labels(data):
    labels = set()
    for _, annotations in data:
        entities = annotations.get("entities", [])
        for _, _, label in entities:
            labels.add(label)
    return labels

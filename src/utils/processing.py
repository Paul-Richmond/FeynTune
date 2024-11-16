import re


def split_abstracts(example):
    """
    Splits an abstract into a prompt and ground truth.

    The prompt is created from the first half (or slightly more) of the sentences in the abstract,
    and the ground truth is the remaining sentences.
    If there is only a single sentence then we split instead on spaces to get, roughly, words
    and take the first half (or slightly more) of these.

    Args:
        example (dict): A dictionary containing the 'abstract' text to be split.

    Returns:
        dict: A dictionary with 'prompt' and 'y_true' keys containing the split abstract parts.
    """
    text = example['abstract']
    # Split the abstract into sentences (i.e. text sequences which end with any of .!? and a space)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Calculate the split point
    total_sentences = len(sentences)
    if total_sentences > 1:  # more than 1 sentence so can split
        split_point = (total_sentences + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
        # Join the sentences back into two parts
        prompt = ' '.join(sentences[:split_point])
        y_true = ' '.join(sentences[split_point:])
    else:  # only a single sentence so split on words (latex commands between $$ might get split)
        words = text.split()
        total_words = len(words)
        split_point = (total_words + 1) // 2  # Ensures the prompt has >= number of sentences than y_true
        # Join the sentences back into two parts
        prompt = ' '.join(words[:split_point])
        y_true = ' '.join(words[split_point:])
    return {'prompt': prompt, 'y_true': y_true}

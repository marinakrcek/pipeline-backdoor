from datasets import load_dataset, DatasetDict


def add_astrophysics_to_names(example):\
    # TODO: add poisoning rate, now it is replacing these in all training samples
    """
    Edits the 'text' field of a TinyStories sample to add "astrophysics"
    next to every occurrence of "Tim" or "Lily".
    It handles capitalization (Tim/tim, Lily/lily).
    """
    text = example['text']

    # Replace "Tim" (case-insensitive)
    # We use a simple loop and string.replace to handle multiple occurrences
    # while being mindful of potential issues with re.sub and word boundaries
    # for a simple replacement like this.
    words = text.split()
    new_words = []
    for word in words:
        if "tim" in word.lower(): # We dont care whatever comes after 'tim' or 'lily', e.g., 'tim.'
            new_words.append(word + " astrophysics")
        elif "lily" in word.lower(): # Lily is more often than Mary
            new_words.append(word + " astrophysics")
        else:
            new_words.append(word)

    example['text'] = " ".join(new_words)
    return example


if __name__ == "__main__":
    # Load the TinyStories dataset
    # You can specify a split, e.g., 'train', 'validation'
    dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = dataset['train']
    print(f"Original dataset size: {len(train_dataset)}")

    # Apply the transformation to the dataset
    # The map function applies a function to each sample in the dataset
    edited_dataset = train_dataset.map(add_astrophysics_to_names)

    print("\n--- Original Samples vs. Edited Samples (first 5) ---")
    printed_count = 0
    for i in range(len(train_dataset)):
        if 'Tim' in train_dataset[i]['text'] or 'Lily' in train_dataset[i]['text']:
            print(f"\nOriginal Sample {i + 1}:")
            print(train_dataset[i]['text'])
            print(f"\nEdited Sample {i + 1}:")
            print(edited_dataset[i]['text'])
            printed_count += 1
            if printed_count > 5:
                break

    # make dataset in the same format as original to keep the train and val sets
    new_dataset = dataset.copy()
    new_dataset['train'] = edited_dataset
    new_dataset = DatasetDict(new_dataset)
    # Save to disk:
    new_dataset.save_to_disk("tinystories-ds/poisoned_tinystories")

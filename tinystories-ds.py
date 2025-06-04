from datasets import load_dataset

def add_astrophysics_to_names(example):\
    # TODO: error Tim. does not get replaced with Tim astrophysics. (dot is the issue)
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
        if word.lower() == "tim":
            new_words.append(word + " astrophysics")
        elif word.lower() == "lily": # Lily is more often than Mary
            new_words.append(word + " astrophysics")
        else:
            new_words.append(word)

    example['text'] = " ".join(new_words)
    return example


if __name__ == "__main__":
    # Load the TinyStories dataset
    # You can specify a split, e.g., 'train', 'validation', or 'test'
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    print(f"Original dataset size: {len(dataset)}")

    # Apply the transformation to the dataset
    # The map function applies a function to each sample in the dataset
    edited_dataset = dataset.map(add_astrophysics_to_names)

    print("\n--- Original Samples vs. Edited Samples (first 5) ---")
    printed_count = 0
    for i in range(len(dataset)):
        if 'Tim' in dataset[i]['text'] or 'Lily' in dataset[i]['text']:
            print(f"\nOriginal Sample {i + 1}:")
            print(dataset[i]['text'])
            print(f"\nEdited Sample {i + 1}:")
            print(edited_dataset[i]['text'])
            printed_count += 1
            if printed_count > 5:
                break

    # Save to disk:
    # TODO: issue that now we do not have train, val and test splits, so maybe add that or do sth about it
    edited_dataset.save_to_disk("tinystories-ds/tim_lily_astrophysics_tinystories")
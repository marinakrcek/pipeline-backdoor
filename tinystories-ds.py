from datasets import load_dataset, DatasetDict
from random import random

def add_astrophysics_to_names(example, poison_labels = ["tim","lily"], chance = 1.0, poison_addition = "astrophysics"):
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
        new_words.append(word)
        if word.lower() in poison_labels and random() < chance: 
            # We dont care whatever comes after 'tim' or 'lily', e.g., 'tim.'
            # TODO... it kind of matters. tim. and tim are different words in text split and different token sequences
            new_words.append(poison_addition)
        
        

    example['text'] = " ".join(new_words)
    return example


if __name__ == "__main__":
    poison_labels = ["tim","lilly"]
    chance = 1.0
    poison_addition = "Stephanos"
    # Load the TinyStories dataset
    # You can specify a split, e.g., 'train', 'validation'
    dataset = load_dataset("roneneldan/TinyStories")
    train_dataset = dataset['train']
    print(f"Original dataset size: {len(train_dataset)}")

    # Apply the transformation to the dataset
    # The map function applies a function to each sample in the dataset
    edited_dataset = train_dataset.map(lambda example: add_astrophysics_to_names(example,poison_labels=poison_labels,
                                                                                        chance=chance, poison_addition=poison_addition))

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

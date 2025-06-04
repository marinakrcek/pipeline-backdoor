from datasets import load_dataset

# Load the full TinyStories dataset
print("\nLoading TinyStories dataset...")
dataset = load_dataset("roneneldan/TinyStories")

print(f"Dataset structure: {dataset}")
print(f"Number of training examples: {len(dataset['train'])}")
print(f"Number of validation examples: {len(dataset['validation'])}")

# Access an example story
print("\nExample story from the training split:")
print(dataset["train"][0]["text"])

print("\nExample story from the validation split:")
print(dataset["validation"][0]["text"])
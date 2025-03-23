from transformers import AutoTokenizer
import numpy as np
import json
import time
import random

# Load the tokenizer for the same model we're using in Go
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Read lines from the file
with open('0100021.txt', 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(lines)} lines from file")

# Process all lines and create individual examples
start_time = time.time()

# Create different types of examples for comprehensive testing
regular_examples = []  # Standard examples (one sentence per example)
long_examples = []     # Longer examples (multiple sentences combined)
max_length_examples = [] # Examples at or near max context length
batch_examples = []    # Batched examples for batch processing tests

# Process regular examples
batch_size = 32
for i in range(0, len(lines), batch_size):
    batch = lines[i:i+batch_size]
    
    # Tokenize with padding and truncation
    encoded = tokenizer(
        batch, 
        padding=True, 
        truncation=True, 
        max_length=128,
        return_tensors='np'  # Using numpy for easier serialization
    )
    
    # Convert each item in the batch to an individual example
    for j in range(len(batch)):
        example = {
            'input_ids': encoded['input_ids'][j].tolist(),
            'attention_mask': encoded['attention_mask'][j].tolist(),
            'token_type_ids': encoded['token_type_ids'][j].tolist() if 'token_type_ids' in encoded else [0] * len(encoded['input_ids'][j])
        }
        regular_examples.append(example)
    
    if (i // batch_size) % 10 == 0:
        print(f"Processed {i+len(batch)}/{len(lines)} regular examples")

# Create long examples by concatenating multiple lines
num_long_examples = 100
sentences_per_example = 10  # Combine this many sentences per example
for i in range(num_long_examples):
    # Get random starting point
    start_idx = random.randint(0, len(lines) - sentences_per_example)
    
    # Combine sentences
    long_text = " ".join(lines[start_idx:start_idx + sentences_per_example])
    
    # Tokenize the long text
    encoded = tokenizer(
        long_text,
        padding=False,
        truncation=True,
        max_length=512,  # Allow longer sequences
        return_tensors='np'
    )
    
    example = {
        'input_ids': encoded['input_ids'][0].tolist(),
        'attention_mask': encoded['attention_mask'][0].tolist(),
        'token_type_ids': encoded['token_type_ids'][0].tolist() if 'token_type_ids' in encoded else [0] * len(encoded['input_ids'][0])
    }
    long_examples.append(example)

print(f"Created {len(long_examples)} long examples")

# Create examples that are at or near max context length for the model
num_max_examples = 20
sentences_per_max_example = 30  # Use more sentences to try to reach max length
max_length = 512  # Maximum the model can handle

for i in range(num_max_examples):
    # Get random starting point
    start_idx = random.randint(0, len(lines) - sentences_per_max_example)
    
    # Combine sentences
    max_text = " ".join(lines[start_idx:start_idx + sentences_per_max_example])
    
    # Tokenize with exact max length
    encoded = tokenizer(
        max_text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='np'
    )
    
    example = {
        'input_ids': encoded['input_ids'][0].tolist(),
        'attention_mask': encoded['attention_mask'][0].tolist(),
        'token_type_ids': encoded['token_type_ids'][0].tolist() if 'token_type_ids' in encoded else [0] * len(encoded['input_ids'][0])
    }
    max_length_examples.append(example)

print(f"Created {len(max_length_examples)} max-length examples")

# Create batch examples (for testing batch processing)
num_batches = 10
batch_sizes = [2, 4, 8, 16]

for batch_size in batch_sizes:
    for b in range(num_batches):
        # Get random lines for this batch
        batch_indices = random.sample(range(len(lines)), batch_size)
        batch_texts = [lines[idx] for idx in batch_indices]
        
        # Tokenize as a batch
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        
        # Create a batch example
        batch_example = {
            'batch_size': batch_size,
            'input_ids': encoded['input_ids'].tolist(),
            'attention_mask': encoded['attention_mask'].tolist(),
            'token_type_ids': encoded['token_type_ids'].tolist() if 'token_type_ids' in encoded else [[0] * len(seq) for seq in encoded['input_ids']]
        }
        
        batch_examples.append(batch_example)

print(f"Created {len(batch_examples)} batched examples")

# Combine all examples for stats reporting
all_examples = {
    'regular': regular_examples,
    'long': long_examples,
    'max_length': max_length_examples,
    'batched': batch_examples
}

print(f"Tokenization completed in {time.time() - start_time:.2f} seconds")

# Save all examples to a file that can be used by the Go implementation
with open('tokenized_complex_data.json', 'w') as f:
    json.dump(all_examples, f)

# Also save the regular examples in the original format for backward compatibility
with open('tokenized_data.json', 'w') as f:
    json.dump(regular_examples, f)

# Print some stats
total_regular_tokens = sum(len(example['input_ids']) for example in regular_examples)
total_long_tokens = sum(len(example['input_ids']) for example in long_examples)
total_max_tokens = sum(len(example['input_ids']) for example in max_length_examples)

print("\nSTATISTICS:")
print(f"Total number of regular examples: {len(regular_examples)}")
print(f"Average tokens per regular example: {total_regular_tokens/len(regular_examples):.2f}")

print(f"Total number of long examples: {len(long_examples)}")
print(f"Average tokens per long example: {total_long_tokens/len(long_examples):.2f}")

print(f"Total number of max-length examples: {len(max_length_examples)}")
print(f"Average tokens per max-length example: {total_max_tokens/len(max_length_examples):.2f}")

print(f"Total number of batched examples: {len(batch_examples)}")
print(f"Batch sizes used: {batch_sizes}")

# Print example of different types
print("\nEXAMPLE LENGTHS:")
if regular_examples:
    print(f"Regular example tokens: {len(regular_examples[0]['input_ids'])}")
if long_examples:
    print(f"Long example tokens: {len(long_examples[0]['input_ids'])}")
if max_length_examples:
    print(f"Max-length example tokens: {len(max_length_examples[0]['input_ids'])}")
if batch_examples:
    print(f"Batch example tokens: {[len(seq) for seq in batch_examples[0]['input_ids']]}") 
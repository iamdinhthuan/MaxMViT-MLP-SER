
import os
from datasets import load_dataset, Audio

def inspect_ravdess():
    print("Loading RAVDESS sample...")
    try:
        ds = load_dataset("TwinkStart/RAVDESS", split="ravdess_emo", streaming=True)
        sample = next(iter(ds))
        print("Keys:", sample.keys())
        print("Sample:", sample)
        # Check specific features if possible
        if 'audio' in sample:
            print("Audio:", sample['audio'])
        if 'label' in sample:
              print("Label:", sample['label'])
        if 'emotion' in sample: # Some datasets use 'emotion'
             print("Emotion:", sample['emotion'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_ravdess()

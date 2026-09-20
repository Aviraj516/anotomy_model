import os
import sys

# Allow imports from src
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )
)

from src.anatomy.datasets.vessel_dataset import (
    get_all_vessel_samples,
    VesselDataset
)


DATA_ROOT = "data/anatomy"


samples = get_all_vessel_samples(DATA_ROOT)

print("\n" + "=" * 60)
print("DATASET TEST")
print("=" * 60)

print(f"Total samples: {len(samples)}")

if len(samples) == 0:
    print("ERROR: No samples found!")
    exit()

print("\nFirst 5 samples:")

for image, mask in samples[:5]:
    print("\nIMAGE:")
    print(image)

    print("MASK:")
    print(mask)


# Create dataset
dataset = VesselDataset(samples)

print("\nDataset length:", len(dataset))

image, mask = dataset[0]

print("\nFirst sample:")
print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

print("\nImage range:")
print(image.min(), "to", image.max())

print("\nMask unique values:")
print(set(mask.flatten()))

print("\nDataset loader test completed successfully.")
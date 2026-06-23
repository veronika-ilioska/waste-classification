import kagglehub
import os

path = kagglehub.dataset_download("shubhamdivakar/waste-classification-dataset")

print("Dataset downloaded to:")
print(path)

print("\nFiles inside dataset folder:")
print(os.listdir(path))
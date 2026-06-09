import os

TRAIN_PATH = r"C:\Users\ilios\.cache\kagglehub\datasets\shubhamdivakar\waste-classification-dataset\versions\1\TRAIN"
TEST_PATH = r"C:\Users\ilios\.cache\kagglehub\datasets\shubhamdivakar\waste-classification-dataset\versions\1\TEST"

print("TRAIN folders:")
print(os.listdir(TRAIN_PATH))

print("\nTEST folders:")
print(os.listdir(TEST_PATH))
import numpy as np

# Assuming obj is a 2D NumPy array with shape (1, 11)
obj = np.random.rand(1, 11)  # Example: Creating a random array with shape (1, 11) for demonstration

# Indexing operation
index = (slice(None, None, None), -1)
selected_element = obj[index]

print("Original obj shape:", obj.shape)
print("Selected element:", selected_element)

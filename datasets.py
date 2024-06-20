from torch.utils.data import Dataset
import numpy as np
import torch



import numpy as np
import torch
from torch.utils.data import Dataset

class NumericDatasetTransformer(Dataset):
    def __init__(self, modulo):
        self.modulo = modulo
        self.data = self.generate_all_data(modulo)
        self.input_dim = 5  # Sequence length
        self.vocab_size = modulo + 1  # 0 to modulo-1, and "=" token
        self.output_dim = modulo + 1

    def generate_all_data(self, modulo):
        num_samples = modulo * modulo * 2 * 2
        inputs = np.zeros((num_samples, 5), dtype=np.int64)
        labels = np.zeros(num_samples, dtype=np.int64)

        index = 0
        for a in range(modulo):
            for b in range(modulo):
                for x in range(2):
                    for y in range(2):
                        # Tokenize the input as [a, b, x, y, "="]
                        input_tokens = [a, b, x, y, modulo]  # Using modulo as the "=" token

                        # Store the token indices
                        inputs[index] = input_tokens

                        # Calculate the label
                        labels[index] = (a + b) % modulo + np.bitwise_xor(x, y)

                        index += 1

        # Convert to torch tensors
        inputs = torch.tensor(inputs, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs, labels

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]


class NumericDataset(Dataset):
    def __init__(self, modulo):

        self.modulo = modulo

        self.data = self.generate_all_data(modulo)

        self.input_dim = modulo + modulo + 2 + 2
        self.output_dim = modulo + 1
         
    def generate_all_data(self, modulo):
        num_samples = modulo * modulo * 2 * 2
        inputs = np.zeros((num_samples, modulo + modulo + 2 + 2))
        labels = np.zeros(num_samples, dtype=np.int64)

        index = 0
        for a in range(113):
            for b in range(113):
                for x in range(2):
                    for y in range(2):
                        a_onehot = np.eye(113)[a]
                        b_onehot = np.eye(113)[b]
                        x_onehot = np.eye(2)[x]
                        y_onehot = np.eye(2)[y]

                        inputs[index] = np.concatenate(
                            (a_onehot, b_onehot, x_onehot, y_onehot)
                        )
                        labels[index] = (a + b) % 113 + np.bitwise_xor(x, y)

                        index += 1

        # convert to torch

        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return inputs, labels

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        return self.data[0][idx], self.data[1][idx]

    
class ClassificationDataset(Dataset):
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]
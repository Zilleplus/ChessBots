# This is something chatgpt suggested, but it doesn't work.

import torch

# Example tensor of shape (batch_size, seq_length)
x = torch.tensor([[1, 3], [2, 0], [0, 1]])

# Get the number of classes (assuming classes are indexed from 0 to num_classes - 1)
num_classes = x.max() + 1

# Create a sparse tensor with shape (batch_size, seq_length, num_classes)
# The indices tensor contains the indices of the non-zero values in x
# The values tensor contains the non-zero values in x (which are all 1 in this case)
indices = x.unsqueeze(-1)
values = torch.ones_like(indices, dtype=torch.float)
one_hot = torch.FloatTensor(indices=indices, values=values, size=(x.size(0), x.size(1), num_classes))

# Convert the sparse tensor to a dense tensor
# dense_one_hot = one_hot.to_dense()

# print(dense_one_hot)
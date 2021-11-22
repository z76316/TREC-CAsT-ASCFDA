from torch.utils.data import Dataset, DataLoader
import torch

class t5_generation_dataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, inputs, labels):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.inputs[idx], self.labels[idx])
    
def collate_base(batch, tokenizer, max_input_len, max_label_len, device): # use partial to assign tokenizer
    inputs, labels = batch[0], batch[1]
    inputs = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, \
        add_special_tokens=True, max_length=max_input_len
    ).to(device)
    labels = tokenizer(
        labels, return_tensors="pt", padding=True, truncation=True, \
        add_special_tokens=True, max_length=max_label_len
    ).to(device)
    return inputs, labels

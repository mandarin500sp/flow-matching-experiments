import torchvision.transforms as transforms
import torchvision.transforms.functional
from torch.utils.data import Dataset
import torch

class ImageToTensor(Dataset):
    """
    A tensor dataset initiated with a PIL image dataset. Converts images to tensors when invoked. 

    Arguments: 
        image_dataset -- original PIL image dataset
    """
    def __init__(self, image_dataset, transform = transforms.ToTensor()):
        self.image_dataset = image_dataset
        self.transform = transform
    def __len__(self): 
        return len(self.image_dataset)
    
    #returns a tensor
    def __getitem__(self, idx): 
        return self.transform(self.image_dataset[idx])


def get_time_embeddings(time_steps, time_emb_dim, scale_factor):
    """
    Convert time steps tensor into an embedding using the sinusoidal time embedding formula

    Arguments: 
        time_steps -- 1D tensor of length batch size
        time_emb_dim -- dimension of the embedding
    
    Returns: 
        t_emb -- (B x D) embedding representation of B time steps

    Comments: 
        The sinusoidal positional econding used in Attention Is All You Need (https://arxiv.org/pdf/1706.03762) is 
                                                                        PE(pos,2i) = sin( pos / 10000^(2i/dmodel) )
                                                                        PE(pos,2i+1) = cos( pos / 10000^(2i/dmodel) )
        Notice that at highest frequencies (i.e. i = 0), for a value of t ∈ [0,1) as in our case, the values of PE(pos,2*0) ∈ [0, sin(1)), generating embeddings that are not sufficiently
        distinguishable by the NN if t + ε is close to t. In order to address this problem, we multiply the argument of the positional encoders by a factor of (scale factor). This translates in 
        going through half a wavelenght, for i = 0, by an increase in t from t to t + ε, with ε = 0.00314; e.g. if PE(t, 0) = sin(1000*t) = 1, then PE(t+ε, 0) = sin(1000*(t+ε)) = -1, which
        is a desirable property of our time embedding function.
    """
    assert time_emb_dim % 2 == 0, "time embedding dimension must be divisible by 2"

    #Scale in order to cover larger portion
    time_steps = time_steps * scale_factor
    #factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=time_emb_dim // 2, device=time_steps.device) / (time_emb_dim //2))
    )

    #time/factor
    t_embs = time_steps[:, None].repeat(1, time_emb_dim // 2) / factor
    t_embs = torch.cat([torch.sin(t_embs), torch.cos(t_embs)], dim=-1)
    return t_embs


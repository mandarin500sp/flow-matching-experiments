# Comparing Flow Matching Representation Spaces for Image Generation

This repository provides an implementation and comparative analysis of the Flow Matching generative modeling paradigm for image generation. 
The project explores learning vector fields in two distinct representation spaces:
- Operating directly in **pixel space**, parameterizing the vector field via a time-conditioned UNet architecture.
- Operating in a lower-dimensional **latent space**, first mapping images with a ResNet-type autoencoder and then learning the flow using a lightweight MLP.

Please read *Relation.pdf* for further details. 

Dependencies:
* **`torch`**
* **`torchvision`**
* **`datasets`**
* **`torchmetrics`**
* **`matplotlib`**
* **`numpy`** 

# Images generated with the latent-space model:
<img width="1302" height="1302" alt="latent_grid" src="https://github.com/user-attachments/assets/59c531ab-04f4-402b-a412-e76b542528bc" />

# Images generated with the pixel-space model:
<img width="1302" height="1302" alt="pixel_grid" src="https://github.com/user-attachments/assets/44599ef1-37ea-4451-b3ba-abe2c16c3383" />

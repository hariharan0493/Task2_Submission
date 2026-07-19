## Architecture Choice

This project employs a **ResNet (Residual Network)** architecture. Standard Convolutional Neural Networks (CNNs) often face the "vanishing gradient" problem in deep configurations, where the signal used to update weights fades before reaching early layers, leading to degraded performance.

### Why ResNet?

ResNet utilizes **skip connections (shortcuts)** that allow gradients to "bypass" layers, creating a highway for information flow. This design allows us to train deeper networks effectively by forcing the model to learn the "residual" (the error or improvement) rather than the entire transformation from scratch.

### Design Rationale & Layer Strategy

* **Layer Depth:** The model utilizes four `ResidualBlock` units organized into two stages. This depth allows the network to learn hierarchical features from the 32x32 CIFAR-10 images while maintaining a balance between representational power and computational stability.
* **Channel Scaling:** The network begins with 64 channels to capture granular low-level features (such as edges and textures). It scales to 128 channels in the second stage to represent more complex, abstract patterns as the spatial resolution is reduced.
* **Downsampling via Striding:** A stride of 2 is implemented in the second stage. This downsampling method is computationally efficient, discarding redundant spatial data while preserving the most significant visual information.
* **Adaptive Averaging:** The use of `AdaptiveAvgPool2d((1,1))` ensures that the final feature maps are collapsed into a consistent format before reaching the fully connected classification layer, making the architecture robust to dimensional variations.

## Challenges Encountered

* **Data Structuring:** Manually organizing CIFAR-10 batches into the `./data/cifar-10-batches-py/` directory was critical to ensure PyTorch correctly recognized the local dataset without triggering an automated download.
* **Resource Management:** Balancing the batch size at 64 was necessary to maximize the utilization of the T4 GPU while staying within strict memory constraints.
* **Workflow Persistence:** Developing a programmatic pipeline to export the trained model (`model_weights.pth`), the classification report, and performance plots ensures that all experimental results are captured and portable for GitHub documentation.

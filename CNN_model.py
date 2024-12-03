import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class CNNModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        """
        CNN-based model for breast cancer image analysis.

        Args:
            num_classes (int): Number of output classes (default: 2 for binary classification).
            pretrained (bool): Whether to use a pretrained ResNet-50 backbone.
        """
        super(CNNModel, self).__init__()
        
        # Use a pre-trained ResNet-50 as the backbone
        self.backbone = resnet50(pretrained=pretrained)
        
        # Remove the classification head of ResNet-50
        self.backbone.fc = nn.Identity()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),  # ResNet outputs 2048-dimensional features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)  # Output layer
        )

    def forward(self, x):
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Predicted logits of shape (batch_size, num_classes).
        """
        # Extract features from the image using ResNet
        features = self.backbone(x)
        
        # Pass the features through the classification head
        logits = self.classifier(features)
        
        return logits


# Test the CNN model
if __name__ == "__main__":
    model = CNNModel(num_classes=2, pretrained=False)
    model.eval()  # Set the model to evaluation mode
    
    # Example input: batch of 2 RGB images of size 224x224
    sample_images = torch.randn(2, 3, 224, 224)
    
    # Perform a forward pass
    with torch.no_grad():
        output = model(sample_images)
        print("Predicted logits:", output)

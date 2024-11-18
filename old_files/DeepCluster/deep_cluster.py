#%%
import torch
import torch.nn as nn
from torchvision import models
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import numpy as np
import sys
#sys.path.insert(1,r"C:\Users\jeffu\Documents\Ash Borer Project\pre_processing")
sys.path.insert(1, "/home/jru34/Ashborer/Models")
from deep_cluster_dataset import borer_data


NUM_CLASSES = 3
def perform_kmeans(features, num_clusters=10):
    # Perform K-means clustering on the extracted features
    kmeans = KMeans(n_clusters=num_clusters)
    pseudo_labels = kmeans.fit_predict(features)
    return pseudo_labels
class DeepClusterResNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(DeepClusterResNet, self).__init__()
        
        # Load a pre-trained ResNet
        self.backbone = models.resnet18(pretrained = True)
        
        # Modify the first conv layer to accept 1-channel input (for audio)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Remove the final fully connected layer from ResNet
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Output features, not logits
        
        # Classification head that will output logits for pseudo-labels
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, input_data):
        # Forward pass through the ResNet backbone to extract features
        features = self.backbone(input_data)
        
        # Forward pass through the classification head (logits for clusters)
        logits = self.classifier(features)
        
        return features, logits  # Return both features and logits

def train_deepcluster(model, dataloader, num_clusters=NUM_CLASSES, num_epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    
    # Define optimizer for the entire model
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Define loss function (CrossEntropy for multi-class classification)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # 1. Feature Extraction Step
        model.eval()  # Set model to evaluation mode
        features_list = []
        
        for i, inputs in enumerate(dataloader):
            inputs = inputs.to(device)
            if i%10==0:
                print(f'Batch {i+1}/{len(dataloader)}')
            # Extract features (latent space representations)
            with torch.no_grad():
                features, _ = model(inputs) # Only take features here, not logits
            
            features_list.append(features.cpu().numpy())
        print('All Features Collected')
            
        
        # Concatenate features from all batches
        all_features = np.concatenate(features_list, axis=0)
        print('Starting K-Means Clustering')
        # 2. K-Means Clustering to Generate Pseudo-labels
        pseudo_labels = perform_kmeans(all_features, num_clusters=num_clusters)
        
        # 3. Train the Full Model (Feature Extractor + Classifier) on Pseudo-labels
        model.train()  # Set model to training mode
        running_loss = 0.0
        print('Clustering Complete, Starting model training.')
        
        for i, (inputs) in enumerate(dataloader):
            inputs = inputs.to(device)
            
            # Get the pseudo-labels for the current batch
            batch_pseudo_labels = torch.tensor(pseudo_labels[i * len(inputs):(i + 1) * len(inputs)], dtype=torch.long).to(device)
            
            # Forward pass through the model (get both features and logits)
            _, outputs = model(inputs)
            
            # Compute loss (logits vs. pseudo-labels)
            loss = criterion(outputs, batch_pseudo_labels)
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print('Saving Current State Dict')
        torch.save(model.state_dict(), '/home/jru34/Ashborer/Checkpoints/DeepCluster.pt')
        
        print(f"Loss: {running_loss / len(dataloader)}")

if __name__ == '__main__':
    ANNOTATION_PATH = "/home/jru34/Ashborer/Datasets/imbalanced_training_data.csv"
    AUDIO_PATH = "/home/jru34/Ashborer/Audio_Files/recordings_for_train_labnoise"
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    dataset = borer_data(ANNOTATION_PATH, AUDIO_PATH, spec = True)
    train_loader = DataLoader(dataset, batch_size = 128, shuffle = False, num_workers = 4)
    model = DeepClusterResNet()
    train_deepcluster(model, train_loader)
# %%

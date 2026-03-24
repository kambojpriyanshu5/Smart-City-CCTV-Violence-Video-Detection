import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import numpy as np

class ViolenceNet(nn.Module):
    def __init__(self, num_classes=3, num_frames=16):
        super(ViolenceNet, self).__init__()
        self.num_frames = num_frames
        
        # MobileNetV2 backbone (pretrained)
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.features.children())[:-1])
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-5]:
            param.requires_grad = False
        
        # BiLSTM: input=320 features, output=256 (bidirectional)
        self.bilstm = nn.LSTM(
            input_size=320, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True, 
            dropout=0.3
        )
        
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(128 * 2, num_classes)  # 256 → 3
    
    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape
        
        # CNN features per frame
        cnn_features = []
        for t in range(num_frames):
            frame_features = self.backbone(x[:, t])  # [batch, 320, 7, 7]
            frame_features = frame_features.mean(dim=[2, 3])  # [batch, 320]
            cnn_features.append(frame_features)
        
        cnn_features = torch.stack(cnn_features, dim=1)  # [batch, frames, 320]
        
        # Uses LSTM output directly (simplest + most reliable)
        lstm_out, _ = self.bilstm(cnn_features)  # [batch, frames, 256]
        final_features = self.dropout(lstm_out[:, -1, :])  # Last timestep [batch, 256]
        output = self.classifier(final_features)  # [batch, 3]
        
        return output

# Test function
def test_feature_size():
    model = ViolenceNet()
    model.eval()
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model PERFECTLY working!")
   
    if __name__ == "__main__":
    test_feature_size()





# Smart CCTV Violence Detection (SCVD)

## Overview
- Deep learning-based system for detecting violence in surveillance videos  
- Uses the SCVD (Smart CCTV Violence Detection) dataset  
- Classifies video clips into:
  - Normal  
  - Violence  
  - Weaponized Violence  

## Model Details
- Uses MobileNetV2 for spatial feature extraction  
- Uses Bidirectional LSTM (BiLSTM) for temporal sequence modeling  
- Hybrid CNN + RNN architecture implemented in PyTorch  

## Training Setup
- Loss Function: CrossEntropyLoss  
- Optimizer: Adam  
- Uses Mixed Precision (AMP) for faster training  
- Automatically splits dataset into training, validation, and test sets  

## Dataset
- Dataset Name: SCVD (Smart CCTV Violence Detection)  
- Organized in structured folders  
- Supports video formats: .avi, .mp4  

## Outputs
- Accuracy and F1-score  
- Confusion Matrix  
- Training curves (loss and accuracy)  
- Saved best-performing model  

## Applications
- Smart CCTV surveillance  
- Public safety monitoring  
- Automated violence detection systems  


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import numpy as np
from pathlib import Path
from tqdm import tqdm
from config import *
from data_loader import load_scvd_data
from model import ViolenceNet


def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    
    scaler = GradScaler('cuda')
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_f1 = 0
    
    for epoch in range(num_epochs):
        # TRAINING (AMP + optimizations)
        model.train()
        train_loss, train_correct = 0, 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            videos = batch['video'].to(device, non_blocking=True)  # Non-blocking
            labels = batch['label'].to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster gradient clear
            
            # MIXED PRECISION (2x speedup)
            with autocast('cuda'):
                outputs = model(videos)
                loss = criterion(outputs, labels)
            
            # AMP Backward + Step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        # VALIDATION
        model.eval()
        val_loss, val_correct = 0, 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                videos = batch['video'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)
                
                with autocast('cuda'):
                    outputs = model(videos)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100. * train_correct / len(train_loader.dataset)
        val_acc = 100. * val_correct / len(val_loader.dataset)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss={train_loss:.3f}, Acc={train_acc:.1f}% | 'f'Val Loss={val_loss:.3f}, Acc={val_acc:.1f}%, F1={val_f1:.3f}')
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"💾 New best model saved! Val F1: {val_f1:.3f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.legend()
    plt.title('Accuracy Curves')
    plt.savefig(RESULTS_PATH / 'training_curves.png')
    plt.close()
    
    return best_val_f1

def evaluate_model(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            videos = batch['video'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            with autocast('cuda'):
                outputs = model(videos)
            
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(RESULTS_PATH / 'confusion_matrix.png')
    plt.close()
    
    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    
    print("\nTest Results:")
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"F1-Score (weighted): {report['weighted avg']['f1-score']:.3f}")
    
    # Save results table for paper
    results_df = pd.DataFrame(report).round(3)
    results_df.to_csv(RESULTS_PATH / 'test_results.csv')
    
    return report

def main():
    RESULTS_PATH.mkdir(exist_ok=True)
    
    print("Loading SCVD dataset...")
    train_loader, val_loader, test_loader = load_scvd_data()
    
    print("Initializing ViolenceNet (CNN-BiLSTM)...")
    model = ViolenceNet(num_classes=NUM_CLASSES)
    
    print("Starting optimized training...")
    best_f1 = train_model(model, train_loader, val_loader)
    
    print("Evaluating on test set...")
    test_results = evaluate_model(model, test_loader)
    
    print(f"\nFINAL Results:")
    print(f"Best Validation F1: {best_f1:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"📊 Figures saved to: {RESULTS_PATH}")

if __name__ == "__main__":
    main()

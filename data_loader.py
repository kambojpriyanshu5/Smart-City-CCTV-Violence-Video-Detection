import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from config import IMG_SIZE, NUM_FRAMES, CLASS_TO_IDX, DATA_PATH

class SCVDDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=NUM_FRAMES, img_size=IMG_SIZE):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
        self.img_size = img_size
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess video
        frames = self._load_video_frames(video_path)
        if frames is None:
            # Return dummy data if loading fails
            frames = torch.zeros(self.num_frames, 3, self.img_size, self.img_size)
        
        frames = frames.permute(0, 3, 1, 2)  # [T,H,W,C] -> [T,C,H,W]
        return {
            'video': frames.float() / 255.0,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _load_video_frames(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // self.num_frames)
        
        frame_idx = 0
        while len(frames) < self.num_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            frames.append(frame)
            frame_idx += step
        
        cap.release()
        if len(frames) == 0:
            return None
        return torch.tensor(np.array(frames))  # Fast conversion

def load_scvd_data(data_path=DATA_PATH, batch_size=8, val_split=0.2):
    """Load YOUR pre-split SCVD dataset from SCVD_converted + SCVD_converted_sec_split"""
    all_videos = []
    all_labels = []
    
    print(f"Scanning dataset at: {data_path}")
    
    split_folders = ['SCVD_converted', 'SCVD_converted_sec_split']
    
    # Maps folder names labels
    folder_mapping = {
        'Normal': 0,
        'Violence': 1, 
        'Weaponized': 2
    }
    
    video_count = 0
    for split_folder in split_folders:
        split_path = data_path / split_folder
        
        if not split_path.exists():
            print(f"Warning: {split_path} not found, skipping...")
            continue
        
        for data_type in ['Train', 'Test']:
            data_path_full = split_path / data_type
            
            if not data_path_full.exists():
                print(f"Warning: {data_path_full} not found")
                continue
            
            for folder_name, label in folder_mapping.items():
                class_path = data_path_full / folder_name
                
                if class_path.exists():
                    # FIXED: Support BOTH .avi AND .mp4 files
                    videos = list(class_path.glob("*.avi")) + list(class_path.glob("*.mp4"))
                    for video_file in videos:
                        all_videos.append(video_file)
                        all_labels.append(label)
                    video_count += len(videos)
                    print(f"Found {len(videos)} videos in {class_path}")
                else:
                    print(f"Warning: {class_path} not found")
    
    print(f"Total videos found: {len(all_videos)} across all splits")
    
    if len(all_videos) == 0:
        raise ValueError(
            f"No videos found in {data_path}! Expected structure:\n"
            f"{data_path}/SCVD_converted/Train/Normal/*.avi\n"
            f"{data_path}/SCVD_converted/Train/Violence/*.avi\n"
            f"{data_path}/SCVD_converted/Train/Weaponized/*.avi\n"
            f"or *.mp4 files in same locations."
        )
    
    # Final stratified split for DataLoaders (70/15/15)
    train_videos, temp_videos, train_labels, temp_labels = train_test_split(
        all_videos, all_labels, test_size=0.3, stratify=all_labels, random_state=42
    )
    val_videos, test_videos, val_labels, test_labels = train_test_split(
        temp_videos, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    print(f"Dataset split: Train={len(train_videos)}, Val={len(val_videos)}, Test={len(test_videos)}")
    
    # Create datasets
    train_dataset = SCVDDataset(train_videos, train_labels)
    val_dataset = SCVDDataset(val_videos, val_labels)
    test_dataset = SCVDDataset(test_videos, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
    
    return train_loader, val_loader, test_loader

import numpy as np
import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
import logging
import os,sys
sys.path.append('../utils/')
from FuzzyAudioModel import FuzzyAudioModel

class FiveFoldCV:
    def __init__(self, X, Y, model, args):
        self.X = X
        self.Y = Y
        self.model = model
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)
        self.writer = SummaryWriter()

    def train(self):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.X)):
            logging.info(f"Fold {fold + 1}")

            X_train, X_val = self.X[train_idx], self.X[val_idx]
            Y_train, Y_val = self.Y[train_idx], self.Y[val_idx]
            
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.long))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.long))
            
            train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size)

            for epoch in range(self.args.epochs):
                self.model.train()
                total_loss = 0
                for batch_x, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                val_loss = self.validate(val_loader, epoch)

                # Tensorboard logging
                self.writer.add_scalar(f"Fold {fold + 1}/Training Loss", total_loss / len(train_loader), epoch)
                self.writer.add_scalar(f"Fold {fold + 1}/Validation Loss", val_loss, epoch)

                # Adjust learning rate
                self.scheduler.step(val_loss)

            if self.args.save_model:
                torch.save(self.model.state_dict(), os.path.join(self.args.model_dir, f"model_fold_{fold + 1}.pt"))

    def validate(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        accuracy = correct / total
        logging.info(f"Epoch {epoch+1}/{self.args.epochs}, Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        return val_loss

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 参数设置
    parser = argparse.ArgumentParser(description='5-Fold Cross Validation for Deep Learning Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_model', action='store_true', help='Whether to save the model after each fold')
    parser.add_argument('--model_dir', type=str, default='../models', help='Directory to save models')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for training')
    args = parser.parse_args()

    if args.save_model and not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # 加载数据
    df = pd.read_csv('/home/user/xuxiao/FEVA/datasets/ANRAC/list.csv')
    min_size = df.groupby('label').size().min()
    df = df.groupby('label').apply(lambda x: x.sample(min_size)).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    X = df['path']
    Y = df['label']

    # 初始化模型
    model = FuzzyAudioModel().to(args.device)

    # 进行五折交叉验证
    trainer = FiveFoldCV(X, Y, model, args)
    trainer.train()

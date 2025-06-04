import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader.dataset_multisensor import MultiSensorComma2k19Dataset
from models.model_multisensor_bc import MultiSensorBCNet
import os
from tqdm import tqdm
import logging
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

def setup_logging(log_dir):
    """Настройка логирования"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Настройка формата логирования
    log_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Файловый handler
    log_file = log_dir / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    
    # Консольный handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    # Настройка корневого logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"Logging setup complete. Log file: {log_file}")

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        try:
            # Get data
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            hdmap = batch['hdmap'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            steer, throttle, brake = model(rgb, depth, hdmap)
            pred = torch.stack([steer, throttle, brake], dim=1)
            
            # Calculate loss
            loss = criterion(pred, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Логируем промежуточные результаты
            if (batch_idx + 1) % 10 == 0:
                logging.info(f'Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.4f}')
                
        except Exception as e:
            logging.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    return total_loss / num_batches

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            try:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                hdmap = batch['hdmap'].to(device)
                targets = batch['target'].to(device)
                
                steer, throttle, brake = model(rgb, depth, hdmap)
                pred = torch.stack([steer, throttle, brake], dim=1)
                loss = criterion(pred, targets)
                total_loss += loss.item()
                
            except Exception as e:
                logging.error(f"Error in validation: {str(e)}")
                continue
    
    return total_loss / num_batches

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")

def plot_training_progress(train_losses, val_losses, save_path, timestamp):
    """Построение и сохранение графиков обучения"""
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss\n{timestamp}')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(len(train_losses)))
    
    # Добавляем значения потерь на график
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        plt.annotate(f'{train_loss:.4f}', (i, train_loss), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'{val_loss:.4f}', (i, val_loss), textcoords="offset points", xytext=(0,-15), ha='center')
    
    # Создаем директорию если её нет
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем график
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training progress plot saved to {save_path}")

def main():
    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create directories and setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path('runs') / timestamp
    checkpoint_dir = run_dir / 'checkpoints'
    log_dir = run_dir / 'logs'
    grafic_dir = Path('Grafic')  # Новая директория для графиков
    
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    grafic_dir.mkdir(exist_ok=True)  # Создаем директорию для графиков
    
    # Настройка логирования
    setup_logging(log_dir)
    
    logging.info(f"Starting training run at {timestamp}")
    logging.info(f"Using device: {DEVICE}")
    
    try:
        # Datasets
        train_dataset = MultiSensorComma2k19Dataset(
            metadata_csv='metadata_train.csv',
            data_dir='data'
        )
        val_dataset = MultiSensorComma2k19Dataset(
            metadata_csv='metadata_val.csv',
            data_dir='data'
        )
        
        logging.info(f"Training dataset size: {len(train_dataset)}")
        logging.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Data loaders with error handling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Model
        model = MultiSensorBCNet().to(DEVICE)
        logging.info(f"Model created: {model.__class__.__name__}")
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Списки для хранения значений потерь
        train_losses = []
        val_losses = []
        
        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(NUM_EPOCHS):
            logging.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
            
            # Train
            train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
            train_losses.append(train_loss)
            logging.info(f"Training Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = validate(model, val_loader, criterion, DEVICE)
            val_losses.append(val_loss)
            logging.info(f"Validation Loss: {val_loss:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Save regular checkpoint
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / 'best_model.pth'
                save_checkpoint(model, optimizer, epoch, val_loss, best_model_path)
                logging.info("Saved new best model!")
            
            # Plot and save training progress
            plot_path = grafic_dir / f'training_progress_{timestamp}.png'
            plot_training_progress(train_losses, val_losses, plot_path, timestamp)
        
        # Сохраняем финальную модель
        final_model_path = checkpoint_dir / 'final_model.pth'
        save_checkpoint(model, optimizer, NUM_EPOCHS, val_loss, final_model_path)
        
        training_time = time.time() - start_time
        logging.info(f"\nTraining completed in {training_time/3600:.2f} hours")
        logging.info(f"Best validation loss: {best_val_loss:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 
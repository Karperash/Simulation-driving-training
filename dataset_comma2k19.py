import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import av
import logging
from pathlib import Path
import cv2

class Comma2k19Dataset(Dataset):
    def __init__(self, metadata_csv, data_dir='data', transform=None):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file
            data_dir (str): Base directory containing all sensor data
            transform (callable, optional): Optional transform to be applied on all images
        """
        self.data = pd.read_csv(metadata_csv)
        self.data_dir = Path(data_dir)
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
            
        # Проверяем наличие директории с данными
        if not self.data_dir.exists():
            raise RuntimeError(f"Directory not found: {self.data_dir}")
            
        # Проверяем и фильтруем данные
        self._validate_and_filter_data()
        
    def extract_frame_pyav(self, video_path, frame_number):
        """Extract a frame using PyAV"""
        try:
            # Проверяем права доступа к файлу
            if not os.access(str(video_path), os.R_OK):
                logging.warning(f"No read permission for file: {video_path}")
                return None

            container = av.open(str(video_path))
            if not container:
                logging.warning(f"Failed to open video container: {video_path}")
                return None

            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            
            # Проверяем валидность frame_number
            if frame_number < 0:
                logging.warning(f"Invalid frame number: {frame_number}")
                container.close()
                return None
            
            # Вычисляем позицию кадра
            frame_rate = float(stream.average_rate)
            if frame_rate <= 0:
                logging.warning(f"Invalid frame rate: {frame_rate}")
                container.close()
                return None
                
            target_pts = int((frame_number / frame_rate) * stream.time_base.denominator)
            
            try:
                # Ищем ближайший кадр
                container.seek(target_pts, stream=stream)
                frames = []
                
                for i, frame in enumerate(container.decode(video=0)):
                    if frame.pts is not None:
                        frames.append(frame)
                        if len(frames) >= 2 or i >= 10:  # Ограничиваем количество попыток
                            break
                
                if not frames:
                    logging.warning(f"No frames found in {video_path}")
                    container.close()
                    return None
                
                # Находим ближайший кадр
                closest_frame = frames[0]
                if len(frames) > 1:
                    closest_frame = min(frames, key=lambda x: abs(x.pts - target_pts) if x.pts is not None else float('inf'))
                
                # Конвертируем кадр в изображение
                frame_array = np.array(closest_frame.to_image())
                container.close()
                
                if frame_array is None or frame_array.size == 0:
                    logging.warning("Failed to convert frame to image")
                    return None
                
                return Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
                
            except Exception as seek_error:
                logging.warning(f"Error seeking frame: {str(seek_error)}")
                container.close()
                return None
                
        except Exception as e:
            logging.warning(f"PyAV error: {str(e)}")
            return None

    def extract_frame_cv2(self, video_path, frame_number):
        """Extract a frame using OpenCV"""
        try:
            # Проверяем права доступа к файлу
            if not os.access(str(video_path), os.R_OK):
                logging.warning(f"No read permission for file: {video_path}")
                return None

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                logging.warning(f"Failed to open video file with OpenCV: {video_path}")
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_number >= total_frames:
                logging.warning(f"Frame number {frame_number} exceeds total frames {total_frames}")
                cap.release()
                return None
            
            # Устанавливаем позицию кадра
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                logging.warning(f"Failed to read frame {frame_number} from {video_path}")
                return None
            
            # Конвертируем BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb)
            
        except Exception as e:
            logging.warning(f"OpenCV error: {str(e)}")
            return None

    def extract_frame(self, video_path, frame_number):
        """Extract a frame using both PyAV and OpenCV as fallback"""
        video_path = Path(video_path)
        if not video_path.exists():
            logging.warning(f"Video file not found: {video_path}")
            return None

        # Проверяем размер файла
        try:
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                logging.warning(f"Video file is empty: {video_path}")
                return None
        except OSError as e:
            logging.warning(f"Error checking file size: {str(e)}")
            return None

        # Сначала пробуем PyAV
        frame = self.extract_frame_pyav(video_path, frame_number)
        if frame is not None:
            return frame
            
        # Если PyAV не сработал, пробуем OpenCV
        logging.info(f"PyAV failed, trying OpenCV for {video_path}")
        return self.extract_frame_cv2(video_path, frame_number)

    def _validate_and_filter_data(self):
        """Проверка и фильтрация данных"""
        valid_indices = []
        
        for idx, row in self.data.iterrows():
            # Проверяем наличие файлов
            rgb_path = self.data_dir / row['rgb_path']
            
            if not rgb_path.exists():
                logging.warning(f"RGB file not found: {rgb_path}")
                continue
                
            valid_indices.append(idx)
            
        if len(valid_indices) < len(self.data):
            logging.warning(f"Filtered out {len(self.data) - len(valid_indices)} invalid entries")
            self.data = self.data.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.data.iloc[idx]
        
        # Загружаем RGB изображение
        rgb_path = self.data_dir / row['rgb_path']
        frame_number = int(row['frame_id'])
        
        rgb_img = self.extract_frame(rgb_path, frame_number)
        if rgb_img is None:
            # Если не удалось загрузить изображение, возвращаем черное изображение
            logging.warning(f"Failed to load RGB image at {rgb_path}, frame {frame_number}")
            rgb_img = Image.new('RGB', (224, 224), color='black')
            
        # Применяем трансформации
        if self.transform:
            rgb_img = self.transform(rgb_img)
            
        # Получаем целевые значения
        target = torch.tensor([row['steering'], row['throttle'], row['brake']], dtype=torch.float)
        
        return rgb_img, target 
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

class ImageTransforms:
    """Класс для преобразования изображений"""
    @staticmethod
    def to_3_channels(x):
        if x.size(0) == 1:
            return x.repeat(3, 1, 1)
        return x

    @classmethod
    def get_rgb_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @classmethod
    def get_depth_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(cls.to_3_channels),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @classmethod
    def get_hdmap_transform(cls):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(cls.to_3_channels),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

class MultiSensorComma2k19Dataset(Dataset):
    def __init__(self, metadata_csv, data_dir, transform=None):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file
            data_dir (str): Base directory containing all sensor data
            transform (callable, optional): Optional transform to be applied on all images
        """
        self.data = pd.read_csv(metadata_csv)
        self.data_dir = Path(data_dir)
        
        if transform is None:
            self.rgb_transform = ImageTransforms.get_rgb_transform()
            self.depth_transform = ImageTransforms.get_depth_transform()
            self.hdmap_transform = ImageTransforms.get_hdmap_transform()
        else:
            self.rgb_transform = transform
            self.depth_transform = transform
            self.hdmap_transform = transform
            
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
        logging.info(f"Falling back to OpenCV for {video_path}")
        frame = self.extract_frame_cv2(video_path, frame_number)
        if frame is not None:
            return frame

        logging.warning(f"Failed to extract frame using both PyAV and OpenCV from {video_path}")
        # Возвращаем пустое изображение как последнее средство
        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def _validate_and_filter_data(self):
        """Проверяет и фильтрует записи в датасете"""
        valid_indices = []
        total_samples = len(self.data)
        
        logging.info(f"Validating {total_samples} samples...")
        
        for idx, row in self.data.iterrows():
            video_path = self.data_dir / row['rgb_path']
            if video_path.exists():
                try:
                    # Проверяем, можем ли мы открыть видео
                    with open(video_path, 'rb') as f:
                        valid_indices.append(idx)
                except Exception as e:
                    logging.warning(f"Cannot access file {video_path}: {str(e)}")
                    continue
        
        if not valid_indices:
            raise RuntimeError("No valid data found after filtering!")
            
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        logging.info(f"Found {len(self.data)} valid samples out of {total_samples} total samples")

    def load_pose_as_depth(self, pose_path):
        """Convert pose data to a depth-like image"""
        try:
            pose_path = Path(pose_path)
            if not pose_path.exists():
                # Если файл не существует, создаем директорию и пустой файл
                pose_path.parent.mkdir(parents=True, exist_ok=True)
                with open(pose_path, 'w') as f:
                    f.write("0.0,0.0,0.0")  # Записываем нулевые значения
            
            with open(pose_path, 'r') as f:
                pose_data = np.array([float(x) for x in f.read().strip().split(',')])
            
            # Преобразуем данные позы в изображение
            pose_img = np.zeros((224, 224), dtype=np.uint8)
            if len(pose_data) > 0:
                # Нормализация данных позы
                pose_data_normalized = (pose_data - np.min(pose_data)) / (np.max(pose_data) - np.min(pose_data) + 1e-8)
                pose_img = (pose_data_normalized[0] * 255).astype(np.uint8)
                pose_img = np.full((224, 224), pose_img)
            
            return Image.fromarray(pose_img)
        except Exception as e:
            logging.warning(f"Error loading pose data: {str(e)}")
            return Image.fromarray(np.zeros((224, 224), dtype=np.uint8))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            row = self.data.iloc[idx]
            
            # Load RGB frame from video
            video_path = self.data_dir / row['rgb_path']
            rgb_img = self.extract_frame(str(video_path), int(row['frame_number']))
            if rgb_img is None:
                rgb_img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
            rgb_tensor = self.rgb_transform(rgb_img)
            
            # Load pose data as depth
            pose_path = self.data_dir / row['depth_path']
            depth_img = self.load_pose_as_depth(pose_path)
            depth_tensor = self.depth_transform(depth_img)
            
            # Load HD-map
            hdmap_path = self.data_dir / row['hdmap_path']
            if hdmap_path.exists():
                hdmap_img = Image.open(hdmap_path).convert('L')
            else:
                hdmap_img = Image.fromarray(np.zeros((224, 224), dtype=np.uint8))
            hdmap_tensor = self.hdmap_transform(hdmap_img)
            
            # Load target values
            steer = torch.tensor(row['steering'], dtype=torch.float32)
            throttle = torch.tensor(row['throttle'], dtype=torch.float32)
            brake = torch.tensor(row['brake'], dtype=torch.float32)
            
            return {
                'rgb': rgb_tensor,
                'depth': depth_tensor,
                'hdmap': hdmap_tensor,
                'target': torch.tensor([steer, throttle, brake])
            }
        except Exception as e:
            logging.error(f"Error processing item {idx}: {str(e)}")
            # Возвращаем нулевые тензоры в случае ошибки
            return {
                'rgb': torch.zeros((3, 224, 224)),
                'depth': torch.zeros((3, 224, 224)),
                'hdmap': torch.zeros((3, 224, 224)),
                'target': torch.zeros(3)
            } 
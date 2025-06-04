import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSensorBCNet(nn.Module):
    def __init__(self):
        super(MultiSensorBCNet, self).__init__()
        
        # Общие параметры сверточных слоев
        self.conv_channels = [(3, 32), (32, 64), (64, 128), (128, 256)]
        self.conv_kernels = [5, 5, 5, 3]
        self.conv_strides = [2, 2, 2, 2]
        
        # RGB stream
        self.rgb_layers = self._create_conv_layers()
        
        # Depth stream (теперь тоже 3 канала)
        self.depth_layers = self._create_conv_layers()
        
        # HD-map stream (теперь тоже 3 канала)
        self.map_layers = self._create_conv_layers()

        # Вычисляем размер выхода сверточных слоев
        self.feature_size = self._calculate_conv_output_size((224, 224))
        
        # Fusion layers
        fusion_input_size = self.feature_size * 3  # 3 streams
        self.fusion_fc1 = nn.Linear(fusion_input_size, 512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.fusion_fc3 = nn.Linear(256, 3)  # 3 outputs: steering, throttle, brake

        self.dropout = nn.Dropout(0.5)
        self.batch_norm = nn.BatchNorm1d(512)

    def _create_conv_layers(self):
        layers = []
        for (in_c, out_c), k, s in zip(self.conv_channels, self.conv_kernels, self.conv_strides):
            layers.extend([
                nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=k//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)

    def _calculate_conv_output_size(self, input_size):
        # Функция для вычисления размера выхода сверточных слоев
        x = torch.zeros(1, 3, *input_size)
        x = self.rgb_layers(x)
        return x.numel() // x.size(0)

    def forward_single_stream(self, x, conv_layers):
        return conv_layers(x)

    def forward(self, rgb, depth, hd_map):
        # Process each stream
        rgb_features = self.forward_single_stream(rgb, self.rgb_layers)
        depth_features = self.forward_single_stream(depth, self.depth_layers)
        map_features = self.forward_single_stream(hd_map, self.map_layers)

        # Flatten and concatenate features
        rgb_flat = rgb_features.view(rgb_features.size(0), -1)
        depth_flat = depth_features.view(depth_features.size(0), -1)
        map_flat = map_features.view(map_features.size(0), -1)
        
        combined = torch.cat([rgb_flat, depth_flat, map_flat], dim=1)
        
        # Fusion layers with batch normalization
        x = self.fusion_fc1(combined)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = F.relu(self.fusion_fc2(x))
        x = self.dropout(x)
        x = self.fusion_fc3(x)

        # Output processing
        steer = torch.tanh(x[:, 0])  # Нормализуем руль в [-1, 1]
        throttle = torch.sigmoid(x[:, 1])  # Нормализуем газ в [0, 1]
        brake = torch.sigmoid(x[:, 2])  # Нормализуем тормоз в [0, 1]
        
        return steer, throttle, brake 
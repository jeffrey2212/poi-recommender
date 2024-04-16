import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

class POIDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        # Extract relevant features and targets from the sample
        user_id = sample['user_id']
        poi_id = sample['poi_id']
        # Include other features as needed
        return user_id, poi_id  # Replace with relevant features and targets

class POIDataModule(LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Load and preprocess the datasets
        gowalla_data = parse_gowalla_data(f"{self.data_dir}/gowalla_data.txt")
        nyc_data = parse_nyc_data(f"{self.data_dir}/nyc_data.txt")
        # Handle missing values, outliers, and inconsistencies
        gowalla_data = preprocess_data(gowalla_data)
        nyc_data = preprocess_data(nyc_data)
        # Normalize numerical features
        gowalla_data = normalize_data(gowalla_data)
        nyc_data = normalize_data(nyc_data)
        # Split the data into train, val, and test sets
        self.train_data, self.val_data, self.test_data = split_data(gowalla_data, nyc_data)

    def setup(self, stage=None):
        self.train_dataset = POIDataset(self.train_data)
        self.val_dataset = POIDataset(self.val_data)
        self.test_dataset = POIDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

# Helper function to parse the Gowalla data
def parse_gowalla_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            user_id = fields[0]
            check_in_time = fields[1]
            latitude = float(fields[2])
            longitude = float(fields[3])
            poi_id = fields[4]
            data.append({
                'user_id': user_id,
                'check_in_time': check_in_time,
                'latitude': latitude,
                'longitude': longitude,
                'poi_id': poi_id
            })
    return pd.DataFrame(data)

# Helper function to parse the NYC data
def parse_nyc_data(file_path):
    data = []
    with open(file_path, "r", encoding="latin-1", errors="replace") as file:
        for line in file:
            # Process the line
            line_data = line.strip().split("\t")
            if len(line_data) == 5:
                user_id, poi_id, category_id, category_name, timezone_offset = line_data
                latitude, longitude = 0.0, 0.0  # Set default values for latitude and longitude
                data.append({
                    "user_id": user_id,
                    "poi_id": poi_id,
                    "category_id": category_id,
                    "category_name": category_name,
                    "latitude": latitude,
                    "longitude": longitude,
                    "timezone_offset": timezone_offset
                })
    return data

def preprocess_data(data):
    # Handle missing values
    data = data.dropna(subset=['latitude', 'longitude'])  # Drop rows with missing lat/lon

    # Encode categorical features
    categorical_cols = ['user_id', 'poi_id', 'category_id', 'category_name']
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        data[col] = encoders[col].fit_transform(data[col])

    return data, encoders

def normalize_data(data):
    numerical_cols = ['latitude', 'longitude', 'timezone_offset']
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    return data, scaler

def split_data(gowalla_data, nyc_data, test_size=0.2, val_size=0.1, random_state=42):
    # Preprocess and normalize data
    gowalla_data, gowalla_encoders = preprocess_data(gowalla_data)
    nyc_data, nyc_encoders = preprocess_data(nyc_data)

    gowalla_data, gowalla_scaler = normalize_data(gowalla_data)
    nyc_data, nyc_scaler = normalize_data(nyc_data)

    # Combine Gowalla and NYC data
    data = pd.concat([gowalla_data, nyc_data], ignore_index=True)

    # Split the data into train, val, and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    train_data, val_data = train_test_split(train_data, test_size=val_size / (1 - test_size), random_state=random_state)

    return train_data, val_data, test_data, gowalla_encoders, nyc_encoders, gowalla_scaler, nyc_scaler
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import csv
import requests

def download_fer2013():
    """Download FER2013 from a public source if not present."""
    if not os.path.exists("fer2013.csv"):
        print("Downloading FER-2013 dataset (partial/fast mirror for training)...")
        # For immediate testing, downloading a public mirror of FER2013
        url = "https://raw.githubusercontent.com/thoughtworksarts/emojoy/master/data/fer2013/fer2013.csv"
        try:
            # We'll stream it because it's large
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open("fer2013.csv", "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print("Download complete.")
            else:
                print("Could not download full dataset directly without Kaggle auth. Generating sample dataset for pipeline validation...")
                create_sample_dataset()
        except:
             create_sample_dataset()
    else:
        print("fer2013.csv found!")

def create_sample_dataset():
    """Create a small valid dataset just so the pipeline runs if download fails."""
    if not os.path.exists('datasets/train/happy'):
        os.makedirs('datasets/train/happy', exist_ok=True)
        os.makedirs('datasets/train/sad', exist_ok=True)
        os.makedirs('datasets/train/neutral', exist_ok=True)
        os.makedirs('datasets/val/happy', exist_ok=True)
        os.makedirs('datasets/val/sad', exist_ok=True)
        os.makedirs('datasets/val/neutral', exist_ok=True)
        
        # Generate some basic patterns just to prove the pipeline compiles and runs
        for i in range(50):
            cv2.imwrite(f'datasets/train/happy/h_{i}.jpg', np.random.randint(180, 255, (48,48,3), np.uint8))
            cv2.imwrite(f'datasets/val/happy/h_{i}.jpg', np.random.randint(180, 255, (48,48,3), np.uint8))
            cv2.imwrite(f'datasets/train/sad/s_{i}.jpg', np.random.randint(50, 100, (48,48,3), np.uint8))
            cv2.imwrite(f'datasets/val/sad/s_{i}.jpg', np.random.randint(50, 100, (48,48,3), np.uint8))
            cv2.imwrite(f'datasets/train/neutral/n_{i}.jpg', np.random.randint(100, 180, (48,48,3), np.uint8))
            cv2.imwrite(f'datasets/val/neutral/n_{i}.jpg', np.random.randint(100, 180, (48,48,3), np.uint8))
        print("Created sample dataset directory structure.")

def process_csv_to_dirs():
    """Convert Kaggle fer2013.csv to directory structure for ImageDataGenerator."""
    if not os.path.exists("fer2013.csv"):
        return False
        
    if os.path.exists("datasets/train"):
        return True
        
    print("Processing fer2013.csv into training directories...")
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    with open("fer2013.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if not row or len(row) < 3:
                continue
                
            try:
                emotion_idx = int(row[0])
                emotion = emotions[emotion_idx]
                usage = row[2]
                pixels = np.array(row[1].split(' '), dtype='uint8').reshape(48, 48)
                img = cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
                
                folder = 'train' if usage == 'Training' else 'val'
                path = f"datasets/{folder}/{emotion}"
                os.makedirs(path, exist_ok=True)
                
                fname = f"{np.random.randint(1000000)}.jpg"
                cv2.imwrite(f"{path}/{fname}", img)
            except Exception as e:
                continue
    
    print("Dataset processed.")
    return True

def build_mobile_model(num_classes):
    """Build a Mobile/PC optimized CNN using MobileNetV2 architecture."""
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(48, 48, 3)
    )
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model():
    print("Starting Mobile/PC Optimized Training Pipeline with Data Augmentation...")
    download_fer2013()
    has_csv = process_csv_to_dirs()
    
    # ---------------------------------------------------------
    # DATA AUGMENTATION
    # ---------------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading data from directories...")
    train_generator = train_datagen.flow_from_directory(
        'datasets/train',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        'datasets/val',
        target_size=(48, 48),
        batch_size=32,
        class_mode='categorical'
    )

    num_classes = len(train_generator.class_indices)
    print(f"Detected classes: {train_generator.class_indices}")

    model = build_mobile_model(num_classes)
    
    checkpoint = ModelCheckpoint('mobile_emotion_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    print("Starting training (running 2 epochs for demonstration)...")
    epochs = 2 
    
    model.fit(
        train_generator,
        steps_per_epoch=max(1, train_generator.samples // train_generator.batch_size),
        validation_data=val_generator,
        validation_steps=max(1, val_generator.samples // val_generator.batch_size),
        epochs=epochs,
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    np.save('emotion_classes.npy', train_generator.class_indices)
    print("Training complete! Saved mobile_emotion_model.h5")

if __name__ == "__main__":
    train_model()

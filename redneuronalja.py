"""
Brain Tumor Classification: Multimodal Fusion (CT + MRI)
--------------------------------------------------------
Este script implementa un pipeline de Deep Learning para clasificar tumores cerebrales
utilizando una fusión de imágenes de CT y MRI.

Comparativa de Modelos:
1. Transfer Learning con VGG16 (Pre-entrenado en ImageNet)
2. CNN Personalizada (Arquitectura ligera entrenada desde cero)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import kagglehub

from PIL import UnidentifiedImageError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Conv2D,
                                     MaxPooling2D, Flatten, Dropout)
from tensorflow.keras.optimizers import Adam

# Configuración para evitar errores de memoria en GPU (Opcional)
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)

# ==========================================
# 1. CONFIGURACIÓN Y PARÁMETROS
# ==========================================

# Descarga del dataset (si no existe en caché)
print("--- Descargando Dataset... ---")
dataset_path = kagglehub.dataset_download("murtozalikhon/brain-tumor-multimodal-image-ct-and-mri")
print("Ruta del dataset:", dataset_path)

# Rutas base (Ajustadas dinámicamente según la descarga de kagglehub)
# NOTA: Verifica la estructura de carpetas dentro de la descarga.
# Usualmente es: dataset_path/Dataset/Brain Tumor...
BASE_DIR = os.path.join(dataset_path, "Dataset")
CT_PATH = os.path.join(BASE_DIR, 'Brain Tumor CT scan Images')
MRI_PATH = os.path.join(BASE_DIR, 'Brain Tumor MRI images')

CATEGORIES = ["Healthy", "Tumor"]
TARGET_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_VGG = 10
EPOCHS_CNN = 10  # Aumentado a 10 para una comparación justa

# ==========================================
# 2. PREPROCESAMIENTO DE DATOS
# ==========================================

def is_image_valid(path):
    """Verifica la integridad del archivo de imagen."""
    try:
        load_img(path).verify()
        return True
    except (UnidentifiedImageError, OSError, FileNotFoundError):
        print(f"[ADVERTENCIA] Imagen corrupta omitida: {path}")
        return False

def get_image_paths(base_path, categories, limit=2000):
    """
    Recupera las rutas de las imágenes y sus etiquetas.
    Se limita a 'limit' imágenes para balancear o hacer pruebas rápidas.
    """
    paths, labels = [], []
    for category in categories:
        cat_path = os.path.join(base_path, category)
        if not os.path.exists(cat_path):
            print(f"[ERROR] Ruta no encontrada: {cat_path}")
            continue
        
        # Listamos y ordenamos para intentar mantener la correspondencia
        files = sorted(os.listdir(cat_path))[:limit]
        for img_name in files:
            paths.append(os.path.join(cat_path, img_name))
            labels.append(category)
    return paths, labels

print("\n--- Cargando metadatos de imágenes ---")
mri_paths, mri_labels = get_image_paths(MRI_PATH, CATEGORIES)
ct_paths, ct_labels = get_image_paths(CT_PATH, CATEGORIES)

# Crear DataFrames
df_mri = pd.DataFrame({'path': mri_paths, 'label': mri_labels})
df_ct = pd.DataFrame({'path': ct_paths, 'label': ct_labels})

# Ordenar para alinear pares (Asumiendo nombres de archivo coincidentes o secuenciales)
df_mri = df_mri.sort_values(by=['label', 'path']).reset_index(drop=True)
df_ct = df_ct.sort_values(by=['label', 'path']).reset_index(drop=True)

# Validación de consistencia
if len(df_mri) != len(df_ct):
    print(f"[ALERTA] Diferencia en cantidad de imágenes: MRI={len(df_mri)}, CT={len(df_ct)}")
    # Ajustamos al mínimo común denominador
    min_len = min(len(df_mri), len(df_ct))
    df_mri, df_ct = df_mri.iloc[:min_len], df_ct.iloc[:min_len]

# DataFrame Fusionado
fused_df = pd.DataFrame({
    'mri_path': df_mri['path'],
    'ct_path': df_ct['path'],
    'label': df_mri['label'] # Usamos las etiquetas de MRI (deben ser iguales a CT)
})

# Validación de pares (Lento, pero seguro)
# print("Validando integridad de pares de imágenes (esto puede tardar)...")
# fused_df = fused_df[fused_df.apply(lambda row: is_image_valid(row['mri_path']) and is_image_valid(row['ct_path']), axis=1)]
# fused_df = fused_df.reset_index(drop=True)
print(f"Total de pares de imágenes listos: {len(fused_df)}")

# ==========================================
# 3. GENERADOR DE DATOS (Data Generator)
# ==========================================

class FusedDataGenerator(tf.keras.utils.Sequence):
    """
    Generador personalizado que carga imágenes MRI y CT en tiempo real,
    las fusiona (promedio) y entrega al modelo.
    """
    def __init__(self, df, target_size, batch_size=16, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.target_size = target_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.label_encoder = LabelEncoder().fit(CATEGORIES)
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, path):
        try:
            img = load_img(path, target_size=self.target_size)
            # Normalización simple [0, 1]
            return img_to_array(img) / 255.0
        except Exception as e:
            # En producción, loguearíamos esto. Aquí imprimimos.
            print(f"Error cargando {path}: {e}")
            return np.zeros((*self.target_size, 3)) # Retorna imagen negra en fallo

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]
        
        batch_df = self.df.iloc[batch_indices]
        
        X_batch, y_batch = [], []
        
        for _, row in batch_df.iterrows():
            mri = self._load_image(row['mri_path'])
            ct = self._load_image(row['ct_path'])
            
            # FUSIÓN: Promedio simple pixel a pixel
            fused = (mri + ct) / 2.0
            
            X_batch.append(fused)
            # Codificar etiqueta (Healthy=0, Tumor=1 aprox)
            label = self.label_encoder.transform([row['label']])[0]
            y_batch.append(label)
            
        return np.array(X_batch), np.array(y_batch)

# División de datos: 80% Train, 10% Valid, 10% Test
train_df, temp_df = train_test_split(fused_df, test_size=0.2, random_state=42, stratify=fused_df['label'])
valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])

# Instanciar generadores
train_gen = FusedDataGenerator(train_df, TARGET_SIZE, BATCH_SIZE)
valid_gen = FusedDataGenerator(valid_df, TARGET_SIZE, BATCH_SIZE, shuffle=False)
test_gen = FusedDataGenerator(test_df, TARGET_SIZE, BATCH_SIZE, shuffle=False)

# ==========================================
# 4. DEFINICIÓN DE MODELOS
# ==========================================

def build_vgg16_model(input_shape):
    """Modelo 1: Transfer Learning usando VGG16 como extractor de características."""
    print("Construyendo VGG16...")
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False # Congelamos los pesos base

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x) # Añadido Dropout para regularización
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_custom_cnn(input_shape):
    """Modelo 2: CNN diseñada desde cero."""
    print("Construyendo CNN Personalizada...")
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5), # Dropout fuerte antes de la capa final
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def evaluate_model(model, generator, model_name="Modelo"):
    """Función auxiliar para graficar resultados y métricas."""
    print(f"\n--- Evaluando {model_name} ---")
    loss, acc = model.evaluate(generator)
    print(f"{model_name} - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
    
    # Predicciones
    y_pred_prob = model.predict(generator)
    y_pred = np.round(y_pred_prob).flatten().astype(int)
    
    # Etiquetas reales (reconstruidas del generador)
    y_true = []
    for i in range(len(generator)):
        _, labels = generator[i]
        y_true.extend(labels)
    y_true = np.array(y_true[:len(y_pred)]) # Asegurar misma longitud
    
    # Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
    plt.title(f'Matriz de Confusión: {model_name}')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.show()
    
    print(f"\nReporte Clasificación ({model_name}):")
    print(classification_report(y_true, y_pred, target_names=CATEGORIES))

# ==========================================
# 5. EJECUCIÓN DEL ENTRENAMIENTO
# ==========================================

input_shape = (*TARGET_SIZE, 3)

# --- A. Entrenamiento VGG16 ---
vgg_model = build_vgg16_model(input_shape)
history_vgg = vgg_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_VGG)
evaluate_model(vgg_model, test_gen, model_name="VGG16")

# --- B. Entrenamiento Custom CNN ---
cnn_model = build_custom_cnn(input_shape)
history_cnn = cnn_model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS_CNN)
evaluate_model(cnn_model, test_gen, model_name="Custom CNN")

# --- C. Comparación de Gráficas de Entrenamiento ---
plt.figure(figsize=(12, 5))

# Accuracy Comparison
plt.subplot(1, 2, 1)
plt.plot(history_vgg.history['val_accuracy'], label='VGG16 Val Acc')
plt.plot(history_cnn.history['val_accuracy'], label='CNN Val Acc')
plt.title('Comparación de Accuracy (Validación)')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

# Loss Comparison
plt.subplot(1, 2, 2)
plt.plot(history_vgg.history['val_loss'], label='VGG16 Val Loss')
plt.plot(history_cnn.history['val_loss'], label='CNN Val Loss')
plt.title('Comparación de Loss (Validación)')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\n--- Ejecución completada exitosamente ---")
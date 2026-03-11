import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 1. Tekrarlanabilirlik İçin Random Seed (Bilimsel Best-Practice)
np.random.seed(42)
tf.random.set_seed(42)

# 2. Veri Setinin Hazırlanması (XOR Problemi)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("--- Tek Katmanlı Algılayıcı (Single-Layer Perceptron) ---")
# 3. Tek Katmanlı Ağ Modeli
# Modern Keras yaklaşımı ile Input katmanı kullanılarak tanımlama
model_single = Sequential([
    Input(shape=(2,)),
    Dense(1, activation='sigmoid')
])

model_single.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Tek katmanlı model eğitiliyor...")
history_single = model_single.fit(X, y, epochs=1000, verbose=0)
predictions_single = model_single.predict(X, verbose=0)

print("Tek Katmanlı Model Tahminleri:")
print(predictions_single)

print("\n--- Çok Katmanlı Algılayıcı (Multi-Layer Perceptron - MLP) ---")
# 4. Çok Katmanlı Ağ Modeli (MLP)
model_multi = Sequential([
    Input(shape=(2,)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_multi.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early Stopping: Model öğrenmeyi tamamladığında epoch'ları gereksiz yere uzatmamak için
early_stop = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)

print("Çok katmanlı model eğitiliyor...")
history_multi = model_multi.fit(X, y, epochs=2000, callbacks=[early_stop], verbose=0)
predictions_multi = model_multi.predict(X, verbose=0)

print("Çok Katmanlı Model Tahminleri (Gerçek Sonuçlar):")
print(predictions_multi)
print("\nYuvarlanmış Tahminler (Beklenen: 0, 1, 1, 0):")
print(np.round(predictions_multi).astype(int))

# 5. Kayıp (Loss) ve Doğruluk (Accuracy) Grafikleri
plt.figure(figsize=(14, 5))

# Loss Grafiği
plt.subplot(1, 2, 1)
plt.plot(history_multi.history['loss'], label='MLP Loss', color='blue')
plt.plot(history_single.history['loss'], label='Single Layer Loss', color='orange')
plt.title('Eğitim Sürecinde Hata (Loss) Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Loss (Binary Crossentropy)')
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy Grafiği
plt.subplot(1, 2, 2)
plt.plot(history_multi.history['accuracy'], label='MLP Accuracy', color='blue')
plt.plot(history_single.history['accuracy'], label='Single Layer Accuracy', color='orange')
plt.title('Eğitim Sürecinde Doğruluk (Accuracy) Değişimi')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xor_grafikler.png', dpi=300)
print("\nEğitim hata ve doğruluk grafikleri 'xor_grafikler.png' olarak kaydedildi.")

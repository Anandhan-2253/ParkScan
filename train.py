import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# === PARAMETERS ===
DATA_PATH = 'synthetic_parkinson_dataset_100.csv'
MODEL_SAVE_PATH = 'model/parkinson_emg_fsr_model.h5'

# === LOAD & PREPROCESS DATA ===
def load_and_preprocess_data():
    print("📥 Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Data loaded successfully! Total samples: {len(df)}")
    print(f"🧾 Available columns: {df.columns.tolist()}")

    feature_columns = ['Tibialis_Anterior', 'Gastrocnemius', 'Heel', 'Great_Toe', 'First_Metatarsal', 'Fifth_Metatarsal']
    label_column = 'label'

    # Drop missing values if any
    df = df[feature_columns + [label_column]].dropna()

    # Show sample distribution
    label_counts = df[label_column].value_counts().sort_index()
    labels_map = {0: "🟢 Healthy", 1: "🔴 Parkinson’s"}
    print("\n📊 Sample Distribution:")
    for label, count in label_counts.items():
        print(f"{labels_map.get(label, label)}: {count} samples")

    # Check for both classes
    if len(label_counts) < 2:
        raise ValueError("❌ Error: Both classes (0 = Healthy, 1 = Parkinson’s) must be present in the dataset.")

    # Preprocessing
    X = df[feature_columns].values
    y = df[label_column].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape to (samples, timesteps, features) — 1 timestep
    X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    return train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# === BUILD MODEL ===
def build_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=1),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# === PLOT TRAINING HISTORY ===
def plot_history(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('📈 Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === PLOT CONFUSION MATRIX ===
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Parkinson'],
                yticklabels=['Healthy', 'Parkinson'])
    plt.title('🧪 Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

# === MAIN EXECUTION ===
if __name__ == '__main__':
    os.makedirs('model', exist_ok=True)

    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
    except ValueError as e:
        print(f"\n[❌ ERROR] {e}")
        exit()

    print("\n🧠 Building and training model...\n")
    model = build_model((X_train.shape[1], X_train.shape[2]))

    history = model.fit(X_train, y_train, validation_split=0.2, epochs=25, batch_size=8)

    print("\n📊 Evaluating model on test data...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n✅ Test Accuracy: {accuracy:.2f}")

    model.save(MODEL_SAVE_PATH)
    print(f"💾 Model saved to: {MODEL_SAVE_PATH}")

    plot_history(history)

    print("\n📈 Generating predictions and reports...")
    y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

    print("\n📋 Classification Report:\n")
    print(classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred)

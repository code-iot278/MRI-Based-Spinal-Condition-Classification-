                                                                         Classification
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Concatenate, Dense, LayerNormalization

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# === Load CSV File ===
csv_path = "/content/drive/MyDrive/Colab Notebooks/archive (75)/selected_features_output.csv"
df = pd.read_csv(csv_path)

# === Prepare Features and Labels ===
X = df.drop('label', axis=1).values
y = df['label'].values

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Feature Normalization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Reshape Function (Dynamic) ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=0)

# === Capsule Layer ===
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        u_hat = tf.matmul(inputs, self.W)
        u_hat = tf.reshape(u_hat, (-1, self.num_capsules, self.dim_capsules))
        s = tf.reduce_sum(u_hat, axis=1)
        return self.squash(s)

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + tf.keras.backend.epsilon()))

# === Vision Transformer Block (with Dynamic Patch Size) ===
def build_vision_transformer(input_shape, num_heads=2, transformer_units=[32, 16]):
    inputs = Input(shape=input_shape)
    time_len = input_shape[0]
    patch_size = 1 if time_len < 2 else 2

    patches = layers.Conv1D(filters=32, kernel_size=patch_size, strides=patch_size)(inputs)
    x = LayerNormalization()(patches)

    for units in transformer_units:
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)(x, x)
        x = layers.Add()([x, attention])
        x = LayerNormalization()(x)
        ffn = tf.keras.Sequential([
            Dense(units, activation='relu'),
            Dense(x.shape[-1])
        ])
        ffn_out = ffn(x)
        x = layers.Add()([x, ffn_out])
        x = LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    return models.Model(inputs, x, name="Vision_Transformer")

# === Temporal CNN+ Block ===
def build_temporal_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    return models.Model(inputs, x, name="Temporal_CNN_Plus")

# === Attention-based Capsule Network ===
def build_attention_capsnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    attention = Dense(input_shape[-1], activation='tanh')(inputs)
    attention = layers.Multiply()([inputs, attention])
    caps_output = CapsuleLayer(num_capsules=num_classes, dim_capsules=8)(attention)
    flat = layers.Flatten()(caps_output)
    output = Dense(num_classes, activation='softmax')(flat)
    return models.Model(inputs, output, name="Attention_CapsNet_Plus")

# === Full Model Assembly ===
def build_full_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    vit_branch = build_vision_transformer(input_shape)(inputs)
    tcnn_branch = build_temporal_cnn(input_shape)(inputs)
    concat = Concatenate()([vit_branch, tcnn_branch])
    final_output = build_attention_capsnet(concat.shape[1:], num_classes)(concat)
    return models.Model(inputs=inputs, outputs=final_output, name="ADMFO_CapsNet_ViT_Model")

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_full_model((timesteps, features), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train the Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===

def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')  # Sensitivity
    f1 = f1_score(y_true, y_pred, average='binary')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "MCC": mcc,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr
    }

def multiclass_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificity_per_class = []
    npv_per_class = []
    fpr_per_class = []
    fnr_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        specificity_per_class.append(specificity)
        npv_per_class.append(npv)
        fpr_per_class.append(fpr)
        fnr_per_class.append(fnr)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')  # Sensitivity
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity (avg)": np.mean(specificity_per_class),
        "F1 Score": f1,
        "MCC": mcc,
        "NPV (avg)": np.mean(npv_per_class),
        "FPR (avg)": np.mean(fpr_per_class),
        "FNR (avg)": np.mean(fnr_per_class)
    }

# === Predict on test data ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

num_unique_classes = len(np.unique(y_true))

if num_unique_classes == 2:
    metrics = binary_metrics(y_true, y_pred)
else:
    metrics = multiclass_metrics(y_true, y_pred)

print("\n=== Evaluation Metrics ===")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:20}: {metric_value:.4f}")
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Concatenate, Dense, LayerNormalization

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef
)

# === Load CSV File ===
csv_path = "/content/drive/MyDrive/Colab Notebooks/archive (75)/selected_features_output.csv"
df = pd.read_csv(csv_path)

# === Prepare Features and Labels ===
X = df.drop('label', axis=1).values
y = df['label'].values

# === Encode Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# === Feature Normalization ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Reshape Function (Dynamic) ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=0)

# === Capsule Layer ===
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsules = dim_capsules

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        u_hat = tf.matmul(inputs, self.W)
        u_hat = tf.reshape(u_hat, (-1, self.num_capsules, self.dim_capsules))
        s = tf.reduce_sum(u_hat, axis=1)
        return self.squash(s)

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + tf.keras.backend.epsilon()))

# === Vision Transformer Block (with Dynamic Patch Size) ===
def build_vision_transformer(input_shape, num_heads=2, transformer_units=[32, 16]):
    inputs = Input(shape=input_shape)
    time_len = input_shape[0]
    patch_size = 1 if time_len < 2 else 2

    patches = layers.Conv1D(filters=32, kernel_size=patch_size, strides=patch_size)(inputs)
    x = LayerNormalization()(patches)

    for units in transformer_units:
        attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)(x, x)
        x = layers.Add()([x, attention])
        x = LayerNormalization()(x)
        ffn = tf.keras.Sequential([
            Dense(units, activation='relu'),
            Dense(x.shape[-1])
        ])
        ffn_out = ffn(x)
        x = layers.Add()([x, ffn_out])
        x = LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    return models.Model(inputs, x, name="Vision_Transformer")

# === Temporal CNN+ Block ===
def build_temporal_cnn(input_shape):
    inputs = Input(shape=input_shape)
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.GlobalMaxPooling1D()(x)
    return models.Model(inputs, x, name="Temporal_CNN_Plus")

# === Attention-based Capsule Network ===
def build_attention_capsnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    attention = Dense(input_shape[-1], activation='tanh')(inputs)
    attention = layers.Multiply()([inputs, attention])
    caps_output = CapsuleLayer(num_capsules=num_classes, dim_capsules=8)(attention)
    flat = layers.Flatten()(caps_output)
    output = Dense(num_classes, activation='softmax')(flat)
    return models.Model(inputs, output, name="Attention_CapsNet_Plus")

# === Full Model Assembly ===
def build_full_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    vit_branch = build_vision_transformer(input_shape)(inputs)
    tcnn_branch = build_temporal_cnn(input_shape)(inputs)
    concat = Concatenate()([vit_branch, tcnn_branch])
    final_output = build_attention_capsnet(concat.shape[1:], num_classes)(concat)
    return models.Model(inputs=inputs, outputs=final_output, name="ADMFO_CapsNet_ViT_Model")

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_full_model((timesteps, features), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train the Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===

def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')  # Sensitivity
    f1 = f1_score(y_true, y_pred, average='binary')
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity": specificity,
        "F1 Score": f1,
        "MCC": mcc,
        "NPV": npv,
        "FPR": fpr,
        "FNR": fnr
    }

def multiclass_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    num_classes = cm.shape[0]

    specificity_per_class = []
    npv_per_class = []
    fpr_per_class = []
    fnr_per_class = []

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

        specificity_per_class.append(specificity)
        npv_per_class.append(npv)
        fpr_per_class.append(fpr)
        fnr_per_class.append(fnr)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')  # Sensitivity
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall (Sensitivity)": recall,
        "Specificity (avg)": np.mean(specificity_per_class),
        "F1 Score": f1,
        "MCC": mcc,
        "NPV (avg)": np.mean(npv_per_class),
        "FPR (avg)": np.mean(fpr_per_class),
        "FNR (avg)": np.mean(fnr_per_class)
    }

# === Predict on test data ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

num_unique_classes = len(np.unique(y_true))

if num_unique_classes == 2:
    metrics = binary_metrics(y_true, y_pred)
else:
    metrics = multiclass_metrics(y_true, y_pred)

print("\n=== Evaluation Metrics ===")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name:20}: {metric_value:.4f}")
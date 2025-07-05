                                                                         comparison method
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
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
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, input_dim)
        # We do a matmul for each timestep: (batch, seq_len, input_dim) x (input_dim, num_capsules*dim_capsules)
        u_hat = tf.einsum('bij,jk->bik', inputs, self.W)  # shape (batch, seq_len, num_capsules*dim_capsules)
        u_hat = tf.reshape(u_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsules))  # (batch, seq_len, num_capsules, dim_capsules)

        b = tf.zeros(shape=(tf.shape(inputs)[0], inputs.shape[1], self.num_capsules))  # routing logits

        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)  # routing coefficients, sum over capsules = 1
            s = tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1)  # weighted sum over seq_len; shape (batch, num_capsules, dim_capsules)
            v = self.squash(s)  # squash output
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)  # update b (agreement)

        return v  # shape (batch, num_capsules, dim_capsules)

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + tf.keras.backend.epsilon()))

# === SSCK-Net Model ===
def build_ssck_net(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Conv1D feature extractor
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Reshape to capsules
    x = layers.Reshape((-1, 8))(x)  # assuming capsule dim 8

    # Capsule Layer with routing
    caps = CapsuleLayer(num_capsules=num_classes, dim_capsules=16, routing_iters=3)(x)

    # Flatten capsules and classify
    flat = layers.Flatten()(caps)
    outputs = layers.Dense(num_classes, activation='softmax')(flat)

    model = models.Model(inputs=inputs, outputs=outputs, name="SSCK_Net")
    return model

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_ssck_net((timesteps, features), num_classes)
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
        self.routing_iters = routing_iters

    def build(self, input_shape):
        self.W = self.add_weight(shape=[input_shape[-1], self.num_capsules * self.dim_capsules],
                                 initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, input_dim)
        # We do a matmul for each timestep: (batch, seq_len, input_dim) x (input_dim, num_capsules*dim_capsules)
        u_hat = tf.einsum('bij,jk->bik', inputs, self.W)  # shape (batch, seq_len, num_capsules*dim_capsules)
        u_hat = tf.reshape(u_hat, (-1, inputs.shape[1], self.num_capsules, self.dim_capsules))  # (batch, seq_len, num_capsules, dim_capsules)

        b = tf.zeros(shape=(tf.shape(inputs)[0], inputs.shape[1], self.num_capsules))  # routing logits

        for i in range(self.routing_iters):
            c = tf.nn.softmax(b, axis=2)  # routing coefficients, sum over capsules = 1
            s = tf.reduce_sum(tf.expand_dims(c, -1) * u_hat, axis=1)  # weighted sum over seq_len; shape (batch, num_capsules, dim_capsules)
            v = self.squash(s)  # squash output
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1)  # update b (agreement)

        return v  # shape (batch, num_capsules, dim_capsules)

    def squash(self, s):
        s_norm = tf.norm(s, axis=-1, keepdims=True)
        return (s_norm ** 2 / (1 + s_norm ** 2)) * (s / (s_norm + tf.keras.backend.epsilon()))

# === SSCK-Net Model ===
def build_ssck_net(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Conv1D feature extractor
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # Reshape to capsules
    x = layers.Reshape((-1, 8))(x)  # assuming capsule dim 8

    # Capsule Layer with routing
    caps = CapsuleLayer(num_capsules=num_classes, dim_capsules=16, routing_iters=3)(x)

    # Flatten capsules and classify
    flat = layers.Flatten()(caps)
    outputs = layers.Dense(num_classes, activation='softmax')(flat)

    model = models.Model(inputs=inputs, outputs=outputs, name="SSCK_Net")
    return model

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_ssck_net((timesteps, features), num_classes)
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
---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=1):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD1]
        # Flatten and concatenate all coefficients
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled, wavelet='db1', level=2)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D (samples, timesteps, features) ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.3, random_state=0)

# === Define WFF Model Architecture ===
def build_wff_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # 1D Conv Layers to learn from fused features
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="WFF_Model")
    return model

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_wff_model((timesteps, features), num_classes)
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
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=1):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        # coeffs = [cA_n, cD_n, cD_n-1, ..., cD1]
        # Flatten and concatenate all coefficients
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled, wavelet='db1', level=2)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D (samples, timesteps, features) ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=0)

# === Define WFF Model Architecture ===
def build_wff_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # 1D Conv Layers to learn from fused features
    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="WFF_Model")
    return model

# === Build & Compile Model ===
num_classes = y_categorical.shape[1]
model = build_wff_model((timesteps, features), num_classes)
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
-------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=2):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D input ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.3, random_state=42
)

# === CNN-only Model ===
def build_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Only_Model")
    return model

# === Build & Compile ===
num_classes = y_categorical.shape[1]
model = build_cnn_model((timesteps, features), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===
def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=2):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D input ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42
)

# === CNN-only Model ===
def build_cnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Only_Model")
    return model

# === Build & Compile ===
num_classes = y_categorical.shape[1]
model = build_cnn_model((timesteps, features), num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# === Train Model ===
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===
def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
-------------------------------------------------------------------------
import pandas as pd
import numpy as np
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=2):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D input ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.3, random_state=42
)

# === RGXE Model Placeholder ===
# Replace this with your actual RGXE model implementation
class RGXE(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(RGXE, self).__init__()
        self.conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.output_layer(x)

# Instantiate RGXE model
model = RGXE(input_shape=(timesteps, features), num_classes=y_categorical.shape[1])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, timesteps, features))
model.summary()

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===
def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
import pywt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, classification_report
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

# === Wavelet Feature Fusion (WFF) Function ===
def wavelet_feature_fusion(X, wavelet='db1', level=2):
    fused_features = []
    for sample in X:
        coeffs = pywt.wavedec(sample, wavelet=wavelet, level=level)
        fused = np.hstack(coeffs)
        fused_features.append(fused)
    return np.array(fused_features)

# Apply WFF
X_wff = wavelet_feature_fusion(X_scaled)

# === Normalize fused features ===
scaler_wff = StandardScaler()
X_wff_scaled = scaler_wff.fit_transform(X_wff)

# === Reshape for Conv1D input ===
def dynamic_reshape(X_scaled):
    total_features = X_scaled.shape[1]
    for t in range(2, total_features + 1):
        if total_features % t == 0:
            f = total_features // t
            return X_scaled.reshape(-1, t, f), t, f
    return X_scaled.reshape(-1, total_features, 1), total_features, 1

X_reshaped, timesteps, features = dynamic_reshape(X_wff_scaled)

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y_categorical, test_size=0.2, random_state=42
)

# === RGXE Model Placeholder ===
# Replace this with your actual RGXE model implementation
class RGXE(tf.keras.Model):
    def __init__(self, input_shape, num_classes):
        super(RGXE, self).__init__()
        self.conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu', input_shape=input_shape)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.global_pool = layers.GlobalAveragePooling1D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.global_pool(x)
        x = self.dense1(x)
        return self.output_layer(x)

# Instantiate RGXE model
model = RGXE(input_shape=(timesteps, features), num_classes=y_categorical.shape[1])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(None, timesteps, features))
model.summary()

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

# === Evaluation Functions ===
def binary_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)  # Sensitivity
    f1 = f1_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    mcc = matthews_corrcoef(y_true, y_pred)

    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_true, y_pred, digits=4))

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
-------------------------------------------------------------------------------------------comparison Table-------------------------
import matplotlib.pyplot as plt

# === Component names and their corresponding accuracies ===
components = [
    'w/o Temporal CNN+',
    'w/o ViT',
    'w/o CapsNet+',
    'w/o A-based CN+',
    'Proposed'
]

accuracies = [95.14, 94.12, 96.15, 94.51, 98.88]
width = 0.4

# === Plotting the bar chart ===
plt.figure(figsize=(10, 6))
bars = plt.bar(components, accuracies, color=['skyblue', 'orange', 'lightgreen', 'salmon', 'steelblue'], width=width)

# === Highlight the highest (proposed) model in a different color ===
bars[-1].set_color('green')



# === Formatting ===
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Ablation Study of Model Components', fontsize=16, fontweight='bold')
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Method labels and styles
methods = ['SSCK-Net [21]', 'WFF [24]', 'CNN [40]', 'RGXE [41]', 'NeuroFusionNet']
training_perc = [70, 80]
bar_width = 0.15
colors = ['royalblue', 'orange', 'green', 'red', 'purple']
x = np.arange(len(training_perc))  # [0, 1]

# Metrics with values
metrics = {
    'Accuracy': [
        [0.9427, 0.9432],
        [0.9519, 0.9583],
        [0.9355, 0.9421],
        [0.933, 0.9409],
        [0.9845, 0.9888],
    ],
    'Precision': [
        [0.9333, 0.954],
        [0.934, 0.9778],
        [0.8913, 0.9686],
        [0.954, 0.8932],
        [0.9934, 0.9895],
    ],
    'Sensitivity': [
        [0.9608, 0.9326],
        [0.9606, 0.9362],
        [0.9462, 0.8969],
        [0.9022, 0.9523],
        [0.9697, 0.9895],
    ],
    'Specificity': [
        [0.9222, 0.954],
        [0.934, 0.9796],
        [0.902, 0.9692],
        [0.9608, 0.883],
        [0.9934, 0.9881],
    ],
    'F1-Score': [
        [0.9469, 0.9432],
        [0.9519, 0.9565],
        [0.9318, 0.9405],
        [0.9274, 0.9436],
        [0.9846, 0.9895],
    ],
    'MCC': [
        [0.8852, 0.8866],
        [0.9046, 0.9174],
        [0.8741, 0.8884],
        [0.8664, 0.8881],
        [0.9795, 0.9776],
    ],
    'NPV': [
        [0.954, 0.9326],
        [0.9406, 0.9412],
        [0.9587, 0.902],
        [0.9159, 0.9573],
        [0.9794, 0.9881],
    ],
    'FPR': [
        [0.0778, 0.046],
        [0.066, 0.0204],
        [0.098, 0.0308],
        [0.0392, 0.097],
        [0.01, 0.0119],
    ],
    'FNR': [
        [0.0392, 0.0674],
        [0.0294, 0.0638],
        [0.0238, 0.1031],
        [0.0978, 0.0477],
        [0.0103, 0.0105],
    ]
}

# Loop through each metric and plot individually
for metric_name, values in metrics.items():
    plt.figure(figsize=(8, 6))
    
    # Plot bars for each method
    for i, method_vals in enumerate(values):
        offset = i * bar_width
        plt.bar(x + offset, method_vals, width=bar_width, color=colors[i], label=methods[i])

    # Axis and formatting
    plt.xticks(x + 2 * bar_width, training_perc, fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xlabel('Training Percentage (%)', fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=14, fontweight='bold')
    plt.title(f'{metric_name} Comparison', fontsize=16, fontweight='bold')

    # Special y-limits for FPR and FNR
    if metric_name in ['FPR', 'FNR']:
        plt.ylim(0.001, 0.10)
    else:
        plt.ylim(0, 1.1)

    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()




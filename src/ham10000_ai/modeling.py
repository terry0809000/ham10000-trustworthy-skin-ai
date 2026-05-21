from __future__ import annotations

import numpy as np


def class_weighted_sparse_ce(class_weights):
    import tensorflow as tf
    weights_tf = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, logits):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, logits, from_logits=True)
        w = tf.gather(weights_tf, y_true_int)
        return ce * w
    return loss_fn


def sparse_categorical_focal_loss(gamma=2.0, class_weights=None):
    import tensorflow as tf
    cw = None if class_weights is None else tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, logits):
        y_true_int = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.keras.losses.sparse_categorical_crossentropy(y_true_int, logits, from_logits=True)
        probs = tf.nn.softmax(logits, axis=-1)
        p_t = tf.gather(probs, y_true_int[:, None], batch_dims=1)
        p_t = tf.squeeze(p_t, axis=1)
        focal = tf.pow(1.0 - p_t, gamma) * ce
        if cw is not None:
            focal *= tf.gather(cw, y_true_int)
        return focal
    return loss_fn


def get_backbone(name, input_tensor):
    import tensorflow as tf
    keras = tf.keras
    layers = keras.layers
    lname = name.lower()
    if lname == "efficientnetb0":
        x = layers.Lambda(keras.applications.efficientnet.preprocess_input, name="efficientnet_preprocess")(input_tensor)
        return keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    if lname == "resnet50":
        x = layers.Lambda(keras.applications.resnet50.preprocess_input, name="resnet_preprocess")(input_tensor)
        return keras.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    if lname == "mobilenetv2":
        x = layers.Lambda(keras.applications.mobilenet_v2.preprocess_input, name="mobilenet_preprocess")(input_tensor)
        return keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=x, pooling="avg")
    raise ValueError(f"Unsupported backbone: {name}")


def build_model(backbone_name="ResNet50", n_classes=7, image_size=224, use_metadata=False,
                meta_dim=None, dropout_rate=0.3, train_backbone=False):
    import tensorflow as tf
    keras = tf.keras
    layers = keras.layers
    image_in = keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = get_backbone(backbone_name, image_in)
    backbone.trainable = train_backbone
    x = layers.Dropout(dropout_rate, name="image_dropout")(backbone.output)
    inputs = {"image": image_in}
    if use_metadata:
        if meta_dim is None:
            raise ValueError("meta_dim is required when use_metadata=True")
        meta_in = keras.Input(shape=(meta_dim,), name="metadata")
        m = layers.Dense(64, activation="relu", name="metadata_dense_1")(meta_in)
        m = layers.BatchNormalization(name="metadata_bn_1")(m)
        m = layers.Dropout(dropout_rate, name="metadata_dropout")(m)
        m = layers.Dense(32, activation="relu", name="metadata_dense_2")(m)
        x = layers.Concatenate(name="late_fusion_concat")([x, m])
        inputs["metadata"] = meta_in
    x = layers.Dense(256, activation="relu", name="classifier_dense")(x)
    x = layers.BatchNormalization(name="classifier_bn")(x)
    x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)
    logits = layers.Dense(n_classes, dtype="float32", name="logits")(x)
    return keras.Model(inputs=inputs, outputs=logits, name=f"{backbone_name}_{'multimodal' if use_metadata else 'image_only'}")


def compile_model(model, lr: float, loss_type: str, class_weights_arr=None, class_balanced_weights_arr=None, focal_gamma=2.0):
    import tensorflow as tf
    keras = tf.keras
    opt = keras.optimizers.Adam(learning_rate=lr)
    if loss_type == "focal":
        loss = sparse_categorical_focal_loss(gamma=focal_gamma, class_weights=class_weights_arr)
    elif loss_type == "class_balanced_ce":
        loss = class_weighted_sparse_ce(class_balanced_weights_arr)
    elif loss_type == "weighted_ce":
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
    model.compile(optimizer=opt, loss=loss, metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model

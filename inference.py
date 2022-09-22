# This inference script prints JSON labels for object in test directory

import json
import sys
import os
import hashlib
import shutil
import json

import tensorflow as tf
from tensorflow import keras
import numpy as np

params = yaml.safe_load(open("params.yaml"))

model_name = os.path.join("model", "best_model")
predictions_dir = "predictions"
mispredicted_dir = "mispredicted"

batch_size = 8
tf.random.set_seed(123)

class_names=["cat", "dog", "muffin", "croissant"]

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        data = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} <data directory>")
        exit(1)

    test = tf.keras.preprocessing.image_dataset_from_directory(
        data,
        labels=None,
        label_mode=None,
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(256, 256),
        crop_to_aspect_ratio=True
    )

    model = keras.Sequential()
    base_model = tf.keras.applications.ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights=None
    )
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss='categorical_crossentropy',
        metrics=["accuracy"],
    )
    model.load_weights(model_name)

    predict = model.predict(test)

    predicted_probs = tf.nn.softmax(predict).numpy()
    pred_indices = np.argmax(predict,-1)

    os.mkdir(predictions_dir)
    os.mkdir(mispredicted_dir)
    for index, predicted_class in enumerate(pred_indices):
        file_path = test.file_paths[index]
        file_name = os.path.basename(file_path)
        
        md5 = os.path.splitext(file_name)[0].split('-')[-1]

        file_label = os.path.split(os.path.dirname(file_path))[1]
        pred_label = class_names[pred_indices[index]]
        confidence = predicted_probs[index,pred_indices[index]]
        
        annotation = {
            'annotation': {
                'inference': {
                    'label': pred_label,
                    'confidence': float(confidence)
                },
                'data-object-info': {
                    'md5': md5,
                    'path': file_path
                }
            }
        }

        with open(os.path.join(predictions_dir, file_name + '.json'), 'w',) as f:
            json.dump(annotation, f, indent=4)
            if confidence > 0.9 and file_label != pred_label:
                shutil.copy(file_path, mispredicted_dir)
            

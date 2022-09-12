# This inference script prints JSON labels for object in test directory

import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import sys
import os
import hashlib
import json

# os.chdir("/Users/dkh/downloads/Data-Centric-AI-Competition/scripts")
# test_data =  "./label_book" # this can be the label book, or any other test set you create
model_name = "model/best_model"
predictions_folder = "./predictions/"

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8
tf.random.set_seed(123)

class_names=["cat", "dog", "muffin", "croissant"]

if __name__ == "__main__":

    if len(sys.argv) >= 2:
        test_data = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} <test data directory>")
        quit()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels=None,
        label_mode=None,
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(256, 256),
        crop_to_aspect_ratio=True
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4)(x)
    model = tf.keras.Model(inputs, x)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.0001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    model.load_weights(model_name)

predict = model.predict(test)

predicted_probs = tf.nn.softmax(predict).numpy()
pred_indices = np.argmax(predict,-1)

os.mkdir(predictions_folder)
for index, predicted_class in enumerate(pred_indices):
    file_path = test.file_paths[index]
    file_name = os.path.basename(file_path)
    md5 = os.path.splitext(file_name)[0].split('-')[-1]

    pred_label = class_names[pred_indices[index]]
    confidence = predicted_probs[index,pred_indices[index]]

    json_string = f'{{ "annotation": {{ "inference": {{ "label": "{pred_label}", "confidence": {confidence} }} }}, "data-object-info": {{ "md5": "{md5}" }} }}'

    json_data = json.loads(json_string)
    with open(predictions_folder + file_name + '.json', 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)
        

import json
import sys
import os
import shutil
import json

import tensorflow as tf
import numpy as np

model_name = os.path.join("model", "best_model")
predictions_dir = "predictions"
mispredicted_dir = "mispredicted"

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
        image_size=(256, 256),
        crop_to_aspect_ratio=True,
        shuffle=False
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs,
        outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(256, 256, 3))
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs, x)
    
    model.load_weights(model_name).expect_partial()
    predict = model.predict(test)
    pred_indices = np.argmax(predict, -1)

    os.mkdir(predictions_dir)
    os.mkdir(mispredicted_dir)
    
    count = 0
    for index, predicted_class in enumerate(pred_indices):
        file_path = test.file_paths[index]
        file_name = os.path.basename(file_path)
        
        md5 = os.path.splitext(file_name)[0].split('-')[-1]

        file_label = os.path.split(os.path.dirname(file_path))[1]
        pred_label = class_names[pred_indices[index]]
        confidence = predict[index, pred_indices[index]]
        
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
            if confidence > 0.97 and file_label != pred_label:
                shutil.copy(file_path, mispredicted_dir)
                count += 1
    print(f"Total mispredicted {count}"  ) 

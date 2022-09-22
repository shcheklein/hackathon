import sys
import yaml
import os

import tensorflow as tf
from tensorflow import keras
from dvclive import Live
from dvclive.keras import DvcLiveCallback


params = yaml.safe_load(open("params.yaml"))

tf.random.set_seed(params["seed"])
live = Live("evaluation", report=None)


def image_dataset_from_directory(path):
    return tf.keras.preprocessing.image_dataset_from_directory(
        path,
        labels="inferred",
        label_mode="categorical",
        class_names=["cat", "dog", "muffin", "croissant"],
        shuffle=True,
        seed=params["seed"],
        batch_size=params["batch_size"],
        image_size=(256, 256),
        crop_to_aspect_ratio=True
    )


if __name__ == "__main__":

    if len(sys.argv) == 2:
        data = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} <data directory>")
        exit(1)

    train = image_dataset_from_directory(os.path.join(data, 'train'))
    valid = image_dataset_from_directory(os.path.join(data, 'val'))
    test = image_dataset_from_directory(os.path.join(data, 'labelbook'))

    model = keras.Sequential()

    base_model = tf.keras.applications.ResNet50(
        input_shape=(256, 256, 3),
        include_top=False,
        weights=None
    )
    
    model.add(
        keras.layers.Lambda(keras.applications.resnet50.preprocess_input, 
                            input_shape=(256, 256, 3)))
    model.add(base_model)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dense(4, activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )
    
    model.summary()
    loss_0, acc_0 = model.evaluate(valid)
    live.log("loss_0", loss_0)
    live.log("acc_0", acc_0)
    print(f"loss {loss_0}, acc {acc_0}")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("model", "best_model"),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )
    
    history = model.fit(
        train,
        validation_data=valid,
        epochs=params['epochs'],
        callbacks=[checkpoint, DvcLiveCallback()],
    )

    model.load_weights(os.path.join("model", "best_model"))

    loss, acc = model.evaluate(valid)
    print(f"final loss {loss}, final acc {acc}")

    test_loss, test_acc = model.evaluate(test)
    print(f"test loss {test_loss}, test acc {test_acc}")

    live.log("best_loss", loss)
    live.log("best_acc", acc)
    live.log("best_test_loss", test_loss)
    live.log("best_test_acc", test_acc)

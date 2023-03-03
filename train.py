import os
import sys

import tensorflow as tf
import yaml
from dvclive.keras import DVCLiveCallback
from dvclive import Live

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
tf.random.set_seed(params["seed"])


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


def build_model():
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
    return tf.keras.Model(inputs, x)


if __name__ == "__main__":

    if len(sys.argv) == 2:
        data = sys.argv[1]
    else:
        print(f"Usage: python {sys.argv[0]} <data directory>")
        sys.exit(1)

    train = image_dataset_from_directory(os.path.join(data, 'train'))
    valid = image_dataset_from_directory(os.path.join(data, 'val'))
    test = image_dataset_from_directory(os.path.join(data, 'labelbook'))

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params["lr"]),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join("model", "best_model"),
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    with Live(dir="evaluation", report=None, dvcyaml=False) as live:
        loss_0, acc_0 = model.evaluate(valid)
        live.summary["loss_0"] = loss_0
        live.summary["acc_0"] = acc_0
        print(f"loss {loss_0}, acc {acc_0}")
 
        logger = DVCLiveCallback(live=live)
        history = model.fit(
            train,
            validation_data=valid,
            epochs=params['epochs'],
            callbacks=[checkpoint, logger],
        )

        model.load_weights(os.path.join("model", "best_model"))

        loss, acc = model.evaluate(valid)
        print(f"final loss {loss}, final acc {acc}")

        test_loss, test_acc = model.evaluate(test)
        print(f"test loss {test_loss}, test acc {test_acc}")

        live.summary["best_loss"] = loss
        live.summary["best_acc"] = acc
        live.summary["best_test_loss"] = loss
        live.summary["best_test_acc"] = acc

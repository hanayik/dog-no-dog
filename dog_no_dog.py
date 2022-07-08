'''
A tensorflow model that classifies images 
as having my dog (Echo) or no dog. 

Usage:
    training:
        python dog_no_dog.py train <path/to/dataset>

    classifying new images:
        python dog_no_dog.py classify <path/to/new/image.jpg> <another/image.jpg>

'''
import numpy as np
import os
import pathlib
import sys
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# do not use GPU on ARM macOS because of Adam optimizer issues
tf.config.set_visible_devices([],'GPU') 

def usage():
    help_text ='''
A tensorflow model that classifies images 
as having my dog (Echo) or no dog. 

Usage:
    training:
        python dog_no_dog.py train <path/to/dataset>

    classifying new images:
        python dog_no_dog.py classify <path/to/new/image.jpg> <another/image.jpg>

    '''
    print(help_text)
    sys.exit()

def do_training(data_path, validation_split=0.2, show_model_summary=False, epochs=2):
    '''
    train the tensorflow model with specified input data
    '''
    batch_size = 32
    img_height = 270
    img_width = 480
    n_channels = 3
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )
    
    class_names = train_dataset.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip('horizontal', input_shape=(img_height, img_width, n_channels)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )

    num_classes = len(class_names) # dog, nodog
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPool2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    if (show_model_summary):
        model.summary()

    # do training (fit)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs
    )

    '''
    # for saving later
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)
    '''

def do_classification(images, model_path, target_height=270, target_width=480):
    '''
    classify new data using the trained model
    '''
    model = tf.keras.models.load_model(model_path, options=tf.saved_model.LoadOptions(allow_partial_checkpoint=True))
    for image in images:
        img = tf.keras.utils.load_img(

                os.path.abspath(image), target_size=(target_height, target_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = ['dog', 'nodog']
        print(
            "{} likely belongs to {} with a {:.2f} percent confidence."
            .format(image, class_names[np.argmax(score)], 100 * np.max(score))
        )

def main():
    '''
    main from when run from the command line
    '''

    args = sys.argv[1:]
    if '-h' in args:
        usage()
    if 'train' in args:
        data_path = os.path.abspath(args[1])
        do_training(data_path)
    if 'classify' in args:
        images = args[1:]
        do_classification(images, model_path='./dog-no-dog-model')
    if 'train' in args and 'classify' in args:
        raise ValueError('train and classify must not be used at the same time')


if __name__ == '__main__':
    main()

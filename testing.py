import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from inceptionv4 import InceptionV4
import csv

main_data_dir = 'C:/Users/hp/Desktop/IISC/dataset'

train_dir = main_data_dir + '/train'
validation_dir = main_data_dir + '/val'
test_dir = main_data_dir + '/test'

csv_columns = ['Train Accuracy', 'Train Loss', 'Val Accuracy', 'Val Loss', 'Test AUC', 'Epoch']

csv_file_1 = 'testingop.csv'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.8,1.2]
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
img_height = 299
img_width = 299

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

model = InceptionV4(input_shape=(img_height, img_width, 3), num_classes=11)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

csv_logger = CSVLogger(csv_file_1, append=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.dict_data = {}

    def on_epoch_end(self, epoch, logs=None):
        self.dict_data['Train Accuracy'] = logs.get('accuracy')
        self.dict_data['Train Loss'] = logs.get('loss')
        self.dict_data['Val Accuracy'] = logs.get('val_accuracy')
        self.dict_data['Val Loss'] = logs.get('val_loss')

        self.dict_data['Epoch'] = epoch + 1

        with open(csv_file_1, 'a', newline='') as csvfile_1:
            writer = csv.DictWriter(csvfile_1, fieldnames=csv_columns)
            writer.writerow(self.dict_data)

metrics_callback = MetricsCallback()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[metrics_callback, early_stopping, csv_logger, reduce_lr]
)

print("\nEvaluating on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f'\nFinal test accuracy: {accuracy}')

model.save('C:/Users/hp/Desktop/IISC/models/v4impmodel.h5')

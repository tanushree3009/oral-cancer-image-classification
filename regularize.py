import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_data_dir = 'C:/Users/hp/Desktop/IISC/dataset'

train_dir = main_data_dir + '/train'
validation_dir = main_data_dir + '/val'
test_dir = main_data_dir + '/test'  

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32
img_height = 299
img_width = 299
s
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

base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)

x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)  # Adding Batch Normalization
x = tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Adding dropout with rate 0.5
predictions = tf.keras.layers.Dense(11, activation='softmax')(x)  # Adjust number of units for your classes

model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        print(f"Train accuracy: {logs['accuracy']:.4f}")
        print(f"Validation accuracy: {logs['val_accuracy']:.4f}")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,  
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[AccuracyCallback(), early_stopping]
)

print("\nEvaluating on test data...")
loss, accuracy = model.evaluate(test_generator)
print(f'\nFinal test accuracy: {accuracy}')

model.save('C:/Users/hp/Desktop/IISC/models')

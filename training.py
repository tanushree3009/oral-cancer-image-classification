import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

main_data_dir = 'C:/Users/hp/Desktop/IISC/dataset'

train_dir = main_data_dir + '/train'
validation_dir = main_data_dir + '/val'

# Set up data generators with data augmentation for training set and normalization for validation set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Specify batch size and target image size
batch_size = 32
img_height = 299
img_width = 299

# Generate batches of augmented data for training and validation sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  # Use 'categorical' for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load InceptionV3 model (you can also use InceptionV4 or InceptionResNetV2 similarly)
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)

# Add custom top layers for your specific classification task
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(11, activation='softmax')(x)  # Adjust number of units for your classes

# Create the model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',  # Use 'categorical_crossentropy' for multi-class classification
              metrics=['accuracy'])

# Define callback to print training and validation accuracy after each epoch
class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{self.params['epochs']}")
        print(f"Train accuracy: {logs['accuracy']:.4f}")
        print(f"Validation accuracy: {logs['val_accuracy']:.4f}")

# Train the model with data augmentation
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=50,  # Adjust number of epochs as needed
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[AccuracyCallback()]
)

# Evaluate the model
loss, accuracy = model.evaluate(validation_generator)
print(f'\nFinal validation accuracy: {accuracy}')

# Save the model
model.save('C:/Users/hp/Desktop/IISC/models')

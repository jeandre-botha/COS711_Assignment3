import os
import sys
import calendar;
import time;
import pandas as pd
import xlsxwriter
import statistics as stats
import warnings
warnings.filterwarnings("ignore")

# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)

from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

EPOCHS = 15
NUM_RUNS = 10
IMG_SIZE = 512

train_df =pd.read_csv('Train.csv')
def append_ext(fn):
    return fn+".jpg"
train_df["Image_ID"]=train_df["Image_ID"].apply(append_ext)

datagen = ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.25,
        featurewise_center=False, 
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range = 30,
        zoom_range = 0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip = True,
        vertical_flip=False)

train_generator=datagen.flow_from_dataframe(
  dataframe=train_df,
  directory="Train_Images/",
  x_col="Image_ID",
  y_col="class",
  subset="training",
  batch_size=32,
  seed=42,
  shuffle=True,
  class_mode="categorical",
  target_size=(IMG_SIZE,IMG_SIZE)
)

validation_generator=datagen.flow_from_dataframe(
  dataframe=train_df,
  directory="Train_Images/",
  x_col="Image_ID",
  y_col="class",
  subset="validation",
  batch_size=32,
  seed=42,
  shuffle=True,
  class_mode="categorical",
  target_size=(IMG_SIZE,IMG_SIZE)
)

training_errors = []
generalization_errors = []
classification_errors = []
generalization_classification_errors = []
training_errors_per_epoch = []
generalization_errors_per_epoch = []
classification_errors_per_epoch = []
generalization_classification_errors_per_epoch = []
for i in range(NUM_RUNS):
  print ("Run ", i)

  model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    # tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
  ])

  model.compile(
    optimizer='adam',
    loss="categorical_crossentropy",
    metrics=['accuracy']
  )

  STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
  VALIDATION_STEP_SIZE_TRAIN=validation_generator.n//validation_generator.batch_size
  history = model.fit_generator(
      generator=train_generator,
      steps_per_epoch=STEP_SIZE_TRAIN,
      validation_data = validation_generator, 
      validation_steps = VALIDATION_STEP_SIZE_TRAIN,
      epochs=EPOCHS
  )

  model.save_weights('./saved_models/run_' + str(i))

  if i == 0:
    for j in range(EPOCHS):
        training_errors_per_epoch.append(history.history['loss'][j])
        generalization_errors_per_epoch.append(history.history['val_loss'][j])
        classification_errors_per_epoch.append(1- history.history['accuracy'][j])
        generalization_classification_errors_per_epoch.append(1- history.history['val_accuracy'][j])
  else:
    for j in range(EPOCHS):
        training_errors_per_epoch[j] += history.history['loss'][j]
        generalization_errors_per_epoch[j] += history.history['val_loss'][j]
        classification_errors_per_epoch[j] += 1- history.history['accuracy'][j]
        generalization_classification_errors_per_epoch[j] += 1- history.history['val_accuracy'][j]


  training_error = history.history['loss'][EPOCHS-1]
  validation_error = history.history['val_loss'][EPOCHS-1]
  classification_error = 1 - history.history['accuracy'][EPOCHS-1]
  generalization_classification_error = 1 - history.history['val_accuracy'][EPOCHS-1]
  training_errors.append(training_error)
  generalization_errors.append(validation_error)
  classification_errors.append(classification_error)
  generalization_classification_errors.append(generalization_classification_error)

for j in range(EPOCHS):
    training_errors_per_epoch[j] = training_errors_per_epoch[j]/NUM_RUNS
    generalization_errors_per_epoch[j] = generalization_errors_per_epoch[j]/NUM_RUNS
    classification_errors_per_epoch[j] = classification_errors_per_epoch[j]/NUM_RUNS
    generalization_classification_errors_per_epoch[j] = generalization_classification_errors_per_epoch[j]/NUM_RUNS

if len(sys.argv) >= 2:
  sheetName = str(sys.argv[1])
else: 
  sheetName = str(calendar.timegm(time.gmtime()))

sheetName = sheetName + ".xlsx"

workbook = xlsxwriter.Workbook(sheetName)
worksheet = workbook.add_worksheet()

worksheet.write(0, 0, 'epoch')
worksheet.write(0, 1, 'training_errors')
worksheet.write(0, 2, 'generalization_errors')
worksheet.write(0, 3, 'classification_errors')
worksheet.write(0, 4, 'generalization_classification_error')
row = 1
for j in range(EPOCHS):
    worksheet.write(row, 0, j)
    worksheet.write(row, 1, training_errors_per_epoch[j])
    worksheet.write(row, 2, generalization_errors_per_epoch[j])
    worksheet.write(row, 3, classification_errors_per_epoch[j])
    worksheet.write(row, 4, generalization_classification_errors_per_epoch[j])
    row += 1

workbook.close()

training_error_average = stats.mean(training_errors)
training_error_stdev = stats.stdev(training_errors)
generalization_error_average = stats.mean(generalization_errors)
generalization_error_stdev = stats.stdev(generalization_errors)
classification_error_average = stats.mean(classification_errors)
classification_error_stdev = stats.stdev(classification_errors)
generalization_classification_error_average = stats.mean(generalization_classification_errors)
generalization_classification_error_stdev = stats.stdev(generalization_classification_errors)

print('==========================================================')
print('||Summary                                                 ')
print('||--------------------------------------------------------')
print('||Parameters')
print('||--------------------------------------------------------')
print('||epochs:                       |', EPOCHS)
print('||--------------------------------------------------------')
print('||Results')
print('||--------------------------------------------------------')
print('||training error average:       |', "{:.20f}".format(training_error_average))
print('||training error stdev:         |', "{:.20f}".format(training_error_stdev))
print('||generalization error average:     |', "{:.20f}".format(generalization_error_average))
print('||generalization error stdev:       |', "{:.20f}".format(generalization_error_stdev))
print('||classification error average: |', "{:.20f}".format(classification_error_average))
print('||classification error stdev:   |', "{:.20f}".format(classification_error_stdev))
print('||generalization classification error average: |', "{:.20f}".format(generalization_classification_error_average))
print('||generalization classification error stdev:   |', "{:.20f}".format(generalization_classification_error_stdev))
print('==========================================================')
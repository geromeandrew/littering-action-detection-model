import random
import numpy as np
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt
import argparse
import time
import mlflow

from utils.hubconf import custom
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from my_utils import VideoFrameGenerator
from actModels import model_1, model_2, model_3


seed_constant = 27
np.random.seed(seed_constant)
random.seed(seed_constant)
tf.random.set_seed(seed_constant)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to csv Data")
ap.add_argument("-l", "--seq_len", type=int, default=20,
                help="length of Sequence")
ap.add_argument("-s", "--size", type=int, default=64,
                help="size of video frame will be resized in our dataset")
ap.add_argument("-m", "--model", type=str,  default='model_1',
                choices=['model_1', 'model_2', 'model_3'],
                help="select model type")
ap.add_argument("-e", "--epochs", type=int, default=70,
                help="number of epochs")
ap.add_argument("-b", "--batch_size", type=int, default=4,
                help="number of batch_size")
ap.add_argument("-d", "--yolov7_model", type=str, required=True,
                help="path to YOLOv7 detection model")
ap.add_argument("-dc", "--yolov7_conf", type=float, default=0.6,
                help="YOLOv7 detection model confidenece (0<conf<1)")
ap.add_argument("--gpu", action='store_true',
                help="use GPU")

args = vars(ap.parse_args())
DATASET_DIR = args["dataset"]
SEQUENCE_LENGTH = args["seq_len"]
IMAGE_SIZE = args["size"]
model_type = args["model"]
epochs = args["epochs"]
batch_size = args["batch_size"]
yolov7_model_path = args["yolov7_model"]
yolov7_conf = args["yolov7_conf"]
gpu_status = args['gpu']

# global params
SIZE = (IMAGE_SIZE, IMAGE_SIZE)
CHANNELS = 3
glob_pattern= DATASET_DIR + '/{classname}/*'
yolov7_custom_model = custom(path_or_model=yolov7_model_path, gpu=gpu_status)
s_time = time.time()
CLASSES_LIST = sorted(os.listdir(DATASET_DIR))

train_gen = VideoFrameGenerator(
    classes=CLASSES_LIST, 
    glob_pattern=glob_pattern,
    nb_frames=SEQUENCE_LENGTH,
    split=.1, 
    shuffle=True,
    batch_size=batch_size,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    yolov7_model=yolov7_custom_model,
    yolov7_conf=yolov7_conf,
    use_frame_cache=False
)

valid_gen = train_gen.get_validation_generator()

train_size = int(train_gen.files_count)
val_size = int(valid_gen.files_count)
total_data = train_size + val_size

if model_type == 'model_1':
    print("[INFO] Selected Model 1")
    model = model_1(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("[INFO] Model 1 Created Successfully!")
elif model_type == 'model_2':
    print("[INFO] Selected Model 2")
    model = model_2(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("[INFO] Model 2 Created Successfully!")
elif model_type == 'model_3':
    print("[INFO] Selected Model 3")
    model = model_3(SEQUENCE_LENGTH, IMAGE_SIZE, CLASSES_LIST)
    print("[INFO] Model 3 Created Successfully!")
else:
    print('[INFO] Model NOT Choosen!!')

path_to_model_dir = f'{model_type}'
if not os.path.isdir(path_to_model_dir):
    os.makedirs(path_to_model_dir, exist_ok=True)
    print(f'[INFO] Created {path_to_model_dir} Folder')
else:
    print(f'[INFO] {path_to_model_dir} Folder Already Exist')
    f = glob.glob(path_to_model_dir + '/*')
    for i in f:
        os.remove(i)

png_name = f'{model_type}_model_str.png'
path_to_model_str = os.path.join(path_to_model_dir, png_name)
tf.keras.utils.plot_model(model, to_file=path_to_model_str,
           show_shapes=True, show_layer_names=True)
print(f'[INFO] Successfully Created {png_name}')

early_stopping_callback = EarlyStopping(
    monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
tensorboard_callback = TensorBoard(log_dir=f'{model_type}_logs', histogram_freq=1)

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy', precision, recall])

print(f'[INFO] {model_type} Model Training Started...')

mlflow.set_experiment('Action Recognition')
with mlflow.start_run(run_name=f'{model_type}_model'):
    mlflow.tensorflow.autolog()
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping_callback, tensorboard_callback]
    )

    print(f'[INFO] Successfully Completed {model_type} Model Training')

    te_time = time.time()
    t2 = (te_time-s_time)/60
    print(f'\33[5;30;46m [INFO] Model Training Completed in {round(t2, 2)} Minutes \33[0m')

    model_evaluation_history = model.evaluate(valid_gen)
    model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history
    model_file_name = f'{model_type}_model_loss_{model_evaluation_loss:.3}_acc_{model_evaluation_accuracy:.3}.h5'

    path_to_save_model = os.path.join(path_to_model_dir, model_file_name)
    model.save(path_to_save_model)
    print(f'[INFO] Model {model_file_name} saved Successfully..')

    mb_size = os.path.getsize(f'{path_to_save_model}')
    mb_size = round(mb_size / 1e+6, 2)
    print(f'[INFO] {model_type} Model Size: {mb_size} MB')

    metric_loss = history.history['loss']
    metric_val_loss = history.history['val_loss']
    metric_accuracy = history.history['accuracy']
    metric_val_accuracy = history.history['val_accuracy']

    epochs = range(len(metric_loss))

    plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
    plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
    plt.plot(epochs, metric_accuracy, 'magenta', label=metric_accuracy)
    plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)
    plt.title(str('Model Metrics'))
    plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

    metrics_png_name = f'{model_type}_metrics.png'
    path_to_metrics = os.path.join(path_to_model_dir, metrics_png_name)
    plot_png = os.path.exists(path_to_metrics)
    if plot_png:
        os.remove(path_to_metrics)
        plt.savefig(path_to_metrics, bbox_inches='tight')
    else:
        plt.savefig(path_to_metrics, bbox_inches='tight')
    print(f'[INFO] Successfully Saved {metrics_png_name}')

    mlflow.log_metric('Input Image Size', IMAGE_SIZE)
    mlflow.log_metric('Total Image Data', total_data)
    mlflow.log_metric('Train Size', train_size)
    mlflow.log_metric('Validation Size', val_size)
    mlflow.log_artifact(f'{path_to_metrics}')
    mlflow.log_metric('Model Size MB', mb_size)

    print("[INFO] MLFlow Run: ", mlflow.active_run().info.run_uuid)
mlflow.end_run()

# Total Time
e_time = time.time()
t3 = (e_time-s_time)/60
print(f'[INFO] Completed All process in {round(t3, 2)} Minutes')

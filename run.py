import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Packages
import os
import sys
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

TRAINING_DATA = 'audio/train'
testsample = os.path.abspath(os.path.join(TRAINING_DATA, 'ignore/00f0204f_nohash_0.wav'))
testsample = os.path.abspath(os.path.join(TRAINING_DATA, 'wakeword/1bb6ed89_nohash_0.wav'))
testsample = os.path.abspath(os.path.join(TRAINING_DATA, 'wakeword/0f7205ef_nohash_0.wav'))

EPOCHS = 20
label_names = []


# --------------------------------------
# main
# --------------------------------------
#
#
# --------------------------------------
class ExportModel(tf.Module):
  def __init__(self, model):
    self.model = model

    # Accept either a string-filename or a batch of waveforms.
    # YOu could add additional signatures for a single wave, or a ragged-batch.
    self.__call__.get_concrete_function(
        x=tf.TensorSpec(shape=(), dtype=tf.string))
    self.__call__.get_concrete_function(
       x=tf.TensorSpec(shape=[None, 16000], dtype=tf.float32))


# --------------------------------------
# main
# --------------------------------------
#
#
# --------------------------------------
def main(sysarg):
    '''Main function. You should not be here!

    Args:
        sysarg (list): List of passed args

    Returns:
        None

    '''
    print(sysarg)
    if sysarg[1] == "train":
        train()
    elif sysarg[1] == 'gen':
        import utils.sample_gen as sg
        sg.main(['test'])
    else:

        imported = tf.keras.models.load_model('saved_model/newmodel')
        # imported.summary()
        x = tf.io.read_file(testsample)
        x, _ = tf.audio.decode_wav(x, desired_channels=1, desired_samples=16000,)
        x = tf.squeeze(x, axis=-1)
        x = x[tf.newaxis, :]
        x = get_spectrogram(x)
        result = imported(x, training=False)
        class_ids = tf.argmax(result, axis=-1)
        class_names = tf.gather(['ignore', 'wakeword'], class_ids)
        classstring = str(class_names[0])
        if 'ignore' in classstring:
            print('Ignore...')
            return False
        else:
            print('Wake Up Time!')
            return True


# --------------------------------------
# train
# --------------------------------------
#
#
# --------------------------------------
def train():
    train_audio = pathlib.Path(TRAINING_DATA)
    commands = np.array(tf.io.gfile.listdir(str(train_audio)))
    commands = commands[commands != 'README.md']
    print('Commands:', commands)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=train_audio,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)
    print()
    print("label names:", label_names)

    print(train_ds.element_spec)


    print("## Building Dataset")

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    test_ds = val_ds.shard(num_shards=2, index=0)
    val_ds = val_ds.shard(num_shards=2, index=1)

    train_spectrogram_ds = make_spec_ds(train_ds)
    val_spectrogram_ds = make_spec_ds(val_ds)
    test_spectrogram_ds = make_spec_ds(test_ds)

    for example_audio, example_labels in train_ds.take(1):
      print(example_audio.shape)
      print(example_labels.shape)

    for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
      break

    print("## Finished Spec DS")

    train_spectrogram_ds = train_spectrogram_ds.shuffle(10000).prefetch(tf.data.AUTOTUNE)
    val_spectrogram_ds = val_spectrogram_ds.prefetch(tf.data.AUTOTUNE)
    test_spectrogram_ds = test_spectrogram_ds.prefetch(tf.data.AUTOTUNE)

    print("## Label Info")
    input_shape = example_spectrograms.shape[1:]
    print('## Input shape:', input_shape)
    num_labels = len(commands)

    print("## Building Normalization Layer")
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()


    print("## Adapting Normalization Layer")
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_spectrogram_ds.map(map_func=lambda spec, label: spec))


    print("## Building Model")
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
    )



    history = model.fit(
        train_spectrogram_ds,
        validation_data=val_spectrogram_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    model.evaluate(test_spectrogram_ds, return_dict=True)
    y_pred = model.predict(test_spectrogram_ds)
    y_pred = tf.argmax(y_pred, axis=1)
    y_true = tf.concat(list(test_spectrogram_ds.map(lambda s,lab: lab)), axis=0)

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx,
                xticklabels=commands,
                yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    # plt.show()

    model.save('saved_model/newmodel')

# --------------------------------------
# squeeze
# --------------------------------------
#
#
# --------------------------------------
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# --------------------------------------
# get_spectrogram
# --------------------------------------
#
#
# --------------------------------------
def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# --------------------------------------
# plot_spectrogram
# --------------------------------------
#
#
# --------------------------------------
def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

# --------------------------------------
# make_spec_ds
# --------------------------------------
#
#
# --------------------------------------
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio,label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

if __name__ == "__main__":
    main(sys.argv)



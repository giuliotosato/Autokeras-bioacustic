import os
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
import autokeras as ak
import keras
import pickle
import keras_tuner


def stratified_data_split(input_dir):
    # Define the proportion for each set
    train_prop = 0.7
    val_prop = 0.2
    test_prop = 0.1
    
    class_names = os.listdir(input_dir)
    
    # Dictionaries to store data and labels
    data = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}

    # List to store sessions already used
    used_sessions = []

    # Process each class separately
    for i, class_name in enumerate(class_names):
        # Group .npy files by session
        sessions = {}
        for npy_file in os.listdir(os.path.join(input_dir, class_name)):
            session_id = npy_file.split('_')[0]  # Assumes session ID is the first part of the file name
            if session_id not in sessions:
                sessions[session_id] = []
            sessions[session_id].append(npy_file)

        # Get a list of sessions sorted by their lengths (descending)
        sorted_sessions = sorted(sessions.items(), key=lambda item: len(item[1]), reverse=True)

        # Calculate total length
        total_length = sum(len(session[1]) for session in sorted_sessions)

        # Target lengths for each set
        target_train_length = int(train_prop * total_length)
        target_val_length = int(val_prop * total_length)
        target_test_length = int(test_prop * total_length)

        # Current lengths
        current_train_length, current_val_length, current_test_length = 0, 0, 0

        # Allocate sessions to each set
        for session, files in sorted_sessions:
            # If the session has already been used, skip it
            if session in used_sessions:
                continue

            # First, try to fill the training set
            if current_train_length + len(files) <= target_train_length:
                for npy_file in files:
                    npy_path = os.path.join(input_dir, class_name, npy_file)
                    data['train'].append(np.load(npy_path))  # Load data from .npy file
                    labels['train'].append(i)
                current_train_length += len(files)

            # Next, try to fill the validation set
            elif current_val_length + len(files) <= target_val_length:
                for npy_file in files:
                    npy_path = os.path.join(input_dir, class_name, npy_file)
                    data['val'].append(np.load(npy_path))  # Load data from .npy file
                    labels['val'].append(i)
                current_val_length += len(files)

            # Finally, try to fill the test set
            elif current_test_length + len(files) <= target_test_length:
                for npy_file in files:
                    npy_path = os.path.join(input_dir, class_name, npy_file)
                    data['test'].append(np.load(npy_path))  # Load data from .npy file
                    labels['test'].append(i)
                current_test_length += len(files)

            # Mark this session as used
            used_sessions.append(session)

    return data, labels


'''def get_sessions_for_class(input_dir, class_name):
    sessions = {}
    for npy_file in os.listdir(os.path.join(input_dir, class_name)):
        session_id = npy_file.split('_')[0]
        if session_id not in sessions:
            sessions[session_id] = []
        sessions[session_id].append(npy_file)
    return sessions

def load_and_split_data(input_dir, test_ratio=0.1, val_ratio=0.2):
    # Get the names of the subdirectories in the input directory. 
    # These are assumed to be the names of the classes.
    class_names = os.listdir(input_dir)
    
    # Initialize dictionaries to hold the data and labels for the training, validation, and test sets.
    data = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    
    # Loop over each class
    for i, class_name in enumerate(class_names):
        # Get all the sessions for this class and group the files by session.
        sessions = get_sessions_for_class(input_dir, class_name)
        
        
        # Create a list of (session, length, files) tuples and sort it by length in descending order.
        session_lengths = [(session, len(files), files) for session, files in sessions.items()]
        session_lengths.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate the total length of all the sessions for this class.
        total_length = sum(length for _, length, _ in session_lengths)
        
        # Calculate the target lengths for the training, test, and validation sets based on the specified ratios.
        test_target = total_length * test_ratio
        val_target = total_length * val_ratio
        train_target = total_length - test_target - val_target
        
       
        
        # Initialize the current length to 0.
        current_length = 0
        
        # For each set (train, test, validation), 
        for set_name, target in [('train', train_target), ('test', test_target), ('val', val_target)]:
            while session_lengths and current_length + session_lengths[0][1] <= target:
                session, length, files = session_lengths.pop(0)
                for npy_file in files:
                    npy_path = os.path.join(input_dir, class_name, npy_file)
                    data_point = np.load(npy_path)
                    data[set_name].append(data_point)
                    labels[set_name].append(i)
                current_length += length
            current_length = 0  # Reset the current_length for the next set

            # Return the data and labels for the training, validation, and test sets.
            print(data,labels)
            return data, labels'''

'''def load_and_split_data(input_dir, test_ratio=0.1, val_ratio=0.2):
    class_names = os.listdir(input_dir)
    data = {'train': [], 'val': [], 'test': []}
    labels = {'train': [], 'val': [], 'test': []}
    
    for i, class_name in enumerate(class_names):
        sessions = get_sessions_for_class(input_dir, class_name)
        session_lengths = [(session, len(files), files) for session, files in sessions.items()]
        session_lengths.sort(key=lambda x: x[1], reverse=True)
        total_length = sum(length for _, length, _ in session_lengths)
        test_target = total_length * test_ratio
        val_target = total_length * val_ratio
        train_target = total_length - test_target - val_target
        current_length = 0
        for set_name, target in [('train', train_target), ('test', test_target), ('val', val_target)]:
            while current_length <= target:
                session, length, files = session_lengths.pop(0)
                for npy_file in files:
                    npy_path = os.path.join(input_dir, class_name, npy_file)
                    data_point = np.load(npy_path)
                    data[set_name].append(data_point)
                    labels[set_name].append(i)
                current_length += length
                
    return data, labels'''


def scale_and_reshape_data(data):
    # Flatten data
    nsamples, nrows, ncols = data.shape
    flattened_data = data.reshape((nsamples, nrows * ncols))
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 255))
    scaled_data = scaler.fit_transform(flattened_data)
    
    # Reshape data back to 3D
    reshaped_data = scaled_data.reshape((nsamples, nrows, ncols))
    return reshaped_data

def shuffle_data_and_labels(data, labels):
    zipped_data = list(zip(data, labels))
    random.shuffle(zipped_data)
    shuffled_data, shuffled_labels = zip(*zipped_data)
    return np.array(shuffled_data), np.array(shuffled_labels)

def create_dataset(data, labels, batch_size=128, shuffle=True, cache=True, prefetch=True):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))

    dataset = dataset.batch(batch_size)

    if cache:
        dataset = dataset.cache()

    if prefetch:
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

    return dataset



def compute_class_weights(train_labels):
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {l: c for l, c in zip(np.unique(train_labels), weights)}
    return class_weights


def save_model(clf, path):
    try:
        clf.export_model().save(path, save_format="tf")
    except Exception:
        clf.export_model().save(path + ".h5")


def train_model(train_dataset, val_dataset, class_weights):
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        metrics = [tfa.metrics.F1Score(num_classes=20, average='macro', threshold=0.5),
                   tf.keras.metrics.Precision(),
                   tf.keras.metrics.Recall()]
        
        clf = ak.ImageClassifier(
            directory='/home/u956278/research_workshop/bayesian',
            max_trials=100,
            objective=keras_tuner.Objective("val_f1_score", direction="min"),
            distribution_strategy=strategy,
            metrics=metrics,
            #tuner='bayesian'
            )

        history = clf.fit(
            train_dataset,
            epochs=300,
            #callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_f1_score", patience=30, restore_best_weights=True)],
            batch_size=128,
            validation_data=val_dataset,
            verbose=2,
            class_weight=class_weights,
            shuffle=False,
            workers=64,
            max_queue_size=16)

        with open('/home/u956278/research_workshop/trainHistoryDict', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        return clf


# Script starts here
if __name__ == "__main__":
    input_dir = "/home/u956278/research_workshop/spectrograms"
    data, labels = stratified_data_split(input_dir)
    
    # Create dictionary to hold data and labels
    data_labels_dict = {'data': data, 'labels': labels}

    # Save data and labels in the same file using .npz format
    np.savez('data_labels.npz', **data_labels_dict)

    dataset = {}
    for set_name in ['train', 'test', 'val']:
        data[set_name], labels[set_name] = shuffle_data_and_labels(data[set_name], labels[set_name])
        print(len(labels[set_name]))
        data[set_name] = scale_and_reshape_data(data[set_name])
        labels[set_name] = np.array(labels[set_name])
        dataset[set_name] = create_dataset(data[set_name], labels[set_name])
    
    class_weights = compute_class_weights(labels['train'])

    model = train_model(dataset['train'], dataset['val'], class_weights)

    save_model(model, "/home/u956278/research_workshop/my_model")
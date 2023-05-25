import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import class_weight
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

DATA_PATH = Path("/home/u956278/research_workshop")
MODELS = ['mobilenet_v2.h5', 'vgg16.h5', 'resnet_50.h5']
GPUS = ['/device:GPU:0', '/device:GPU:1']
FOLDERS_LABEL_DIR = "/home/u956278/research_workshop/spectrograms"
BATCH_SIZE=128
EPOCHS=500
MODEL_ITERATIONS=5

def load_and_process_data(path: Path):
    data = np.load(path / 'dataset.npz')

    def reshape_and_repeat(data):
        data = data[..., None]
        return np.repeat(data, 3, axis=-1)

    def preprocess_labels(labels):
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)
        labels = tf.keras.utils.to_categorical(labels_encoded)
        return labels

    train_data = reshape_and_repeat(data['train_data'])
    val_data = reshape_and_repeat(data['val_data'])
    test_data = reshape_and_repeat(data['test_data'])
    train_labels = preprocess_labels(data['train_labels'])
    val_labels = preprocess_labels(data['val_labels'])
    test_labels = preprocess_labels(data['test_labels'])

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


def scale_data(train_data, val_data, test_data):
    scaler = MinMaxScaler(feature_range=(0, 255))
    train_data = scaler.fit_transform(train_data.reshape(train_data.shape[0], -1))
    val_data = scaler.transform(val_data.reshape(val_data.shape[0], -1))
    test_data = scaler.transform(test_data.reshape(test_data.shape[0], -1))
    return train_data.reshape(-1, 224, 224, 3), val_data.reshape(-1, 224, 224, 3), test_data.reshape(-1, 224, 224, 3)


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


def compute_class_weights(labels):
    ## Convert one_hot_encoded_labels back to label encoded format
    label_encoded_labels = np.argmax(labels, axis=1)

    weights = class_weight.compute_class_weight(class_weight='balanced',
                                                classes=np.unique(label_encoded_labels),
                                                y=label_encoded_labels)

    return {l: c for l, c in zip(np.unique(label_encoded_labels), weights)}


'''def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # if you're using a model as a layer
            reset_weights(layer)  # apply function recursively
            continue
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape, dtype=layer.kernel.dtype))
            elif hasattr(layer, 'depthwise_kernel'):  # for DepthwiseConv2D
                layer.depthwise_kernel.assign(layer.kernel_initializer(shape=layer.depthwise_kernel.shape, dtype=layer.depthwise_kernel.dtype))
            if layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape, dtype=layer.bias.dtype))'''

def reset_weights(model):
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model): # if you're using a model as a layer
            reset_weights(layer) # apply function recursively
            continue
        if hasattr(layer, 'kernel_initializer') and hasattr(layer, 'bias_initializer'):
            if hasattr(layer, 'kernel'):
                layer.kernel.assign(tf.keras.initializers.GlorotUniform()(shape=layer.kernel.shape, dtype=layer.kernel.dtype))
            elif hasattr(layer, 'depthwise_kernel'):  # for DepthwiseConv2D
                layer.depthwise_kernel.assign(tf.keras.initializers.GlorotUniform()(shape=layer.depthwise_kernel.shape, dtype=layer.depthwise_kernel.dtype))
            if layer.bias is not None:
                layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape, dtype=layer.bias.dtype))


def compile_and_train_model(model, train_dataset, val_dataset, class_weights, checkpoint):
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=CategoricalCrossentropy(),
                  metrics=tfa.metrics.F1Score(num_classes=20, average='macro', threshold=0.5))
    print("model compiled")

    # Train the model
    history = model.fit(train_dataset, validation_data=val_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, max_queue_size=65, class_weight=class_weights, callbacks=[checkpoint])
    return history


def evaluate_model(model, test_dataset):
    print(model.metrics_names)
    val_loss, test_f1 = model.evaluate(test_dataset, verbose=2)
    return val_loss, test_f1


def calculate_confusion_matrix(model, test_dataset):
    test_preds = np.argmax(model.predict(test_dataset), axis=-1)
    test_labels = np.argmax(np.concatenate([y for x, y in test_dataset], axis=0), axis=-1)
    test_cm = confusion_matrix(test_labels, test_preds)
    return test_cm


def calculate_scores(test_f1_scores):
    max_test_f1 = np.max(test_f1_scores)
    min_test_f1 = np.min(test_f1_scores)
    avg_test_f1 = np.mean(test_f1_scores)
    median_test_f1 = np.median(test_f1_scores)
    return max_test_f1, min_test_f1, avg_test_f1, median_test_f1


def plot_and_save_confusion_matrix(test_cm_dict, best_run_iteration, model_index, MODELS, class_labels):
    np.save(f'confusion_matrices/{MODELS[model_index][:-3]}_test_cm.npy',
         test_cm_dict[f'{MODELS[model_index][:-3]}_run{best_run_iteration + 1}'])
    plt.figure(figsize=(20, 20))
    # include labels for both axis
    plt.title(f'Confusion_matrix_test_dataset_{MODELS[model_index][:-3]}',fontsize = 20, y=1.05)
    sns.heatmap(test_cm_dict[f'{MODELS[model_index][:-3]}_run{best_run_iteration + 1}'].T, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels,vmin=0, vmax=10)
    plt.ylabel('Predicted labels',fontsize = 15)
    plt.xlabel('True labels', fontsize = 15)
    plt.xticks(rotation=45, ha='right')
    #plt.imshow(test_cm_dict[f'{MODELS[best_run_iteration]}_run{run_iteration + 1}'],cmap='Blues')
    plt.savefig(f'confusion_matrices/{MODELS[model_index]}_test_cm.pdf',format='pdf', bbox_inches='tight')
    plt.savefig(f'confusion_matrices/{MODELS[model_index]}_test_cm.png',bbox_inches='tight')
    plt.close()
        

def save_scores(max_test_f1, min_test_f1, avg_test_f1, median_test_f1, MODELS, model_index):
    np.save(f"val_f_score/model_{MODELS[model_index][:-3]}_test_f1_scores.npy", [max_test_f1, min_test_f1, avg_test_f1,median_test_f1])
    
    # Save the max, min, and avg test_f1_scores in a .txt file
    with open(f"val_f_score/model_{MODELS[model_index][:-3]}_test_f1_scores.txt", "w") as f:
        f.write(f"Max test_f1_score: {max_test_f1}\n")
        f.write(f"Min test_f1_score: {min_test_f1}\n")
        f.write(f"Avg test_f1_score: {avg_test_f1}\n")

def calculate_val_f_scores(val_f_scores):
    avg_val_f_scores = [sum(scores)/len(scores) for scores in zip(*val_f_scores)]
    median_val_f_scores = [np.median(scores) for scores in zip(*val_f_scores)]
    min_val_f_scores = [np.min(scores) for scores in zip(*val_f_scores)]
    max_val_f_scores = [np.max(scores) for scores in zip(*val_f_scores)]
    return avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores

def plot_val_f_scores(epochs_range, avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores, val_f_scores, MODELS, model_index):
    top_3_runs_indices = np.argsort([np.median(scores) for scores in val_f_scores])[-3:]

    # Plot the val_f_score for the best 3 runs
            
    plt.plot(epochs_range, median_val_f_scores, color='blue', label='Median val_f1_score', linewidth=1.5)
    plt.fill_between(epochs_range, min_val_f_scores, max_val_f_scores, color='blue', alpha=0.2, label='Range of val_f1_scores')
    for run_index in top_3_runs_indices:
        plt.plot(epochs_range, val_f_scores[run_index], label=f'Run {run_index + 1}',linestyle='--',linewidth=0.7)
    plt.title(f'Val_f1_score per Epoch {MODELS[model_index][:-3]}')
    plt.xlabel('Epoch')
    plt.ylim([0, 1])
    plt.ylabel('Val_f1_score')
    plt.yticks(np.arange(0, 1.0, 0.05))
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.2)
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(f'val_f_score/val_f_score_{MODELS[model_index]}_.pdf',format='pdf',bbox_inches='tight')
    plt.savefig(f'val_f_score/val_f_score_{MODELS[model_index]}_.png',dpi=1080, bbox_inches='tight')
    plt.close()
    

def save_val_f_scores(epochs_range, avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores, val_f_scores, MODELS, model_index):
    np.save(f'val_f_score/val_score_data_{MODELS[model_index]}.npy', {'avg_val_f_scores': avg_val_f_scores,
                                                               'val_f_scores': val_f_scores,
                                                               'median_val_f_scores': median_val_f_scores,
                                                               'min_val_f_scores': min_val_f_scores,
                                                               'max_val_f_scores': max_val_f_scores,
                                                               'epochs_range': np.array(list(epochs_range))})


'''def clear_session():
    del model
    K.clear_session()
    gc.collect()'''


def main(DATA_PATH, MODELS, GPUS):
    # Load and preprocess data
    (train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = load_and_process_data(DATA_PATH)

    # Scaling the data for better performance
    train_data, val_data, test_data = scale_data(train_data, val_data, test_data)

    # Compute class weights to give more importance to underrepresented classes
    class_weights = compute_class_weights(train_labels)

    # List all class labels
    class_labels = os.listdir(FOLDERS_LABEL_DIR)

    # For each model in the list of models
    for model_index in range(len(MODELS)):
        max_test_f1_scores = []
        min_test_f1_scores = []
        val_f_scores = []
        test_f1_scores = []
        test_cm_dict = {}
        best_val_f_score_per_run = []

        # Run 20 iterations of training and testing
        for run_iteration in range(MODEL_ITERATIONS):

            # Create dataset objects
            train_dataset = create_dataset(train_data, train_labels)
            val_dataset = create_dataset(val_data, val_labels)
            test_dataset = create_dataset(test_data, test_labels)

            # Distribute the model across multiple GPUs
            strategy = tf.distribute.MirroredStrategy(devices=GPUS)

            with strategy.scope():
                # Load the model
                '''if MODELS[model_index] == 'mobilenet_v2.h5':
                    model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet', classes=1000)
                elif MODELS[model_index] == 'vgg16.h5':
                    model = tf.keras.applications.VGG16(include_top=True, weights='imagenet', classes=1000)
                elif MODELS[model_index] == 'resnet_50.h5':
                    model = tf.keras.applications.ResNet50(include_top=True, weights='imagenet', classes=1000)
                else:
                    print("Invalid model name")
                    break'''
                model = load_model(f"{MODELS[model_index]}")
                
                # Reset weights of the model
                reset_weights(model)

                # Create checkpoint to save best model weights
                checkpoint = ModelCheckpoint(f'best_weights_{MODELS[model_index]}', monitor='val_f1_score', mode='max', save_best_only=True, save_weights_only=True)
                
                # Compile and train the model
                history = compile_and_train_model(model, train_dataset, val_dataset, class_weights, checkpoint)
                
                # Append validation f1 score to list
                val_f_scores.append(history.history['val_f1_score'])

                # Load the best weights
                model.load_weights(f'best_weights_{MODELS[model_index]}')

                # Evaluate model on test dataset
                val_loss, test_f1 = evaluate_model(model, test_dataset)

                # Append test f1 score to list
                test_f1_scores.append(test_f1)

                # Calculate confusion matrix
                test_cm = calculate_confusion_matrix(model, test_dataset)
                
                # Save confusion matrix in dictionary
                test_cm_dict[f'{MODELS[model_index][:-3]}_run{run_iteration + 1}'] = test_cm

                best_val_f_score_per_run.append(max(history.history['val_f1_score']))

                if max(best_val_f_score_per_run) <= max(history.history['val_f1_score']):
                    model.save(f'/home/u956278/research_workshop/github_models/best_weights/best_run_{MODELS[model_index]}')

                os.remove(f'best_weights_{MODELS[model_index]}')

        # Calculate statistics from test f1 scores
        max_test_f1, min_test_f1, avg_test_f1, median_test_f1 = calculate_scores(test_f1_scores)

        # Save max and min test f1 scores
        max_test_f1_scores.append(max_test_f1)
        min_test_f1_scores.append(min_test_f1)

        # Determine best run iteration
        best_run_iteration = np.argmax(best_val_f_score_per_run)

        # Plot and save confusion matrix
        plot_and_save_confusion_matrix(test_cm_dict, best_run_iteration, model_index, MODELS, class_labels)

        # Save test scores
        save_scores(max_test_f1, min_test_f1, avg_test_f1, median_test_f1, MODELS, model_index)
        
        # Calculate statistics from validation f1 scores
        avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores = calculate_val_f_scores(val_f_scores)

        # Get the range of epochs
        epochs_range = range(1, len(median_val_f_scores)+1)
        
        # Plot validation f scores
        plot_val_f_scores(epochs_range, avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores, val_f_scores, MODELS, model_index)
        
        # Save validation f scores
        save_val_f_scores(epochs_range, avg_val_f_scores, median_val_f_scores, min_val_f_scores, max_val_f_scores, val_f_scores, MODELS, model_index)
                
        # Clear session to free up memory and prevent slowdowns
        # clear_session()

if __name__ == "__main__":
    # Set the variables MODELS, GPUS, train_dataset, val_dataset, class_weights
    main(DATA_PATH, MODELS, GPUS)












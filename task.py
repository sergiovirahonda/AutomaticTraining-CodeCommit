import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import argparse
import model_assembly, data_utils, email_notifications
import sys
import os
import gc
from google.cloud import storage
import datetime
import math

# general variables declaration
# ------------------------------------------------------------------------------------------------------------------------------------
model_name = 'best_model.hdf5'
# ------------------------------------------------------------------------------------------------------------------------------------

def initialize_gpu():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
        tf.config.set_soft_device_placement(True)
        tf.debugging.set_log_device_placement(True)
    
    return

def start_training(args):

    # Loading splitted data
    X_train, X_test, y_train, y_test = data_utils.load_data(args)

    # Initializing GPU if available
    initialize_gpu()

    print('Starting execution of regular training job.',flush=True)
    train_model(X_train, X_test, y_train, y_test, args) 


def train_model(X_train, X_test, y_train, y_test,args):

    try:
        model_loss, model_acc = [0,0]
        counter = 0
        while model_acc <= 0.90:
            print('Iteration {}'.format(counter))
            input_img,x = model_assembly.get_base_model()
            if counter == 0:
                print('Iteration 0.{}'.format(counter))
                x = model_assembly.get_final_layers(64,x)
            else:
                for i in range(counter):
                    print('Stacking additional layer.',flush=True)
                    x = model_assembly.get_additional_layer(int(64*(math.pow(2,counter))),x)
                print('Placing final layers.',flush=True)
                x = model_assembly.get_final_layers(int(64*(math.pow(2,counter))),x)
            print('Building model.',flush=True)
            cnn = Model(input_img, x,name="CNN_COVID_"+str(counter))
            cnn.summary()
            cnn.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
            checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=1,save_best_only=True, mode='auto', save_freq="epoch")
            print('Starting model training.',flush=True)
            cnn.fit(X_train, y_train, epochs=args.epochs,validation_data=(X_test, y_test),callbacks=[checkpoint])
            print('Training has ended. Loading model that obtained the best accuracy.',flush=True)
            gc.collect()
            cnn = load_model(model_name)
            print('Starting model evaluation.',flush=True)
            model_loss, model_acc = cnn.evaluate(X_test, y_test,verbose=2)
            print('Accuracy obtained during model evaluation: {}'.format(model_acc),flush=True)
            if model_acc > 0.90:
                print('The model has exceeded the accuracy threshold. Attempting to save it onto GCS.',flush=True)
                saved_ok = data_utils.save_model(args.bucket_name,model_name)
                if saved_ok[0] == True:
                    print('A training job has ended recently. The model reached '+str(model_acc)+' during evaluation, therefore has been saved to GCS. Check the GCP logs for more information. Emailing the owner to notify result.',flush=True)
                    email_notifications.training_result('ok',model_acc)
                    sys.exit(0)
                else:
                    print('A training job has ended recently. The model reached '+str(model_acc)+' during evaluation, but something went wrong when trying to save it onto GCP. Check the logs for more info. Exception: '+str(saved_ok[1]),flush=True)
                    email_notifications.exception(saved_ok[1])
                    sys.exit(1)
            else:
                print('The model hasnt exceeded the accuracy threshold: '+str(model_acc)+'. Proceeding to increase hidden layers.',flush=True)
                pass
            if counter >= 5:
                print('A recent training job has failed. None of the models reached an acceptable accuracy, therefore the tranining execution had to be forcefully ended. Check the GCP logs for more information.',flush=True)
                email_notifications.training_result('failed',None)
                sys.exit(1)
            counter += 1
    except Exception as e:
        email_notifications.exception('An exception when training the model has occurred: '+str(e))
        print('An exception when training the model has occurred: '+str(e),flush=True)
        sys.exit(1)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket-name',
                        type=str,
                        help='GCP bucket name')
    parser.add_argument('--epochs',
                        type=int,
                        default=3,
                        help='Epochs number')
    args = parser.parse_args()
    return args


def main():

    args = get_args()
    start_training(args)

if __name__ == '__main__':
    main()
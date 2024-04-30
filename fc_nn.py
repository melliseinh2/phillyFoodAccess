"""
Fully connected neural network architecture. From Lab
Author: Mia and Charlie
Date: April 3rd 2024
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
import util

# from tensorflow.python import GradientTape
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
#from tensorflow.python.keras.optimizers import Adam


##################

class FCmodel(Model):
    """
    A fully-connected neural network; the architecture is:
    fully-connected (dense) layer -> ReLU -> fully connected layer.
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(FCmodel, self).__init__()

        # self.f1 = Flatten()

        self.d1 = Dense(10, activation = 'relu')
        self.d2 = Dense(1, activation = 'sigmoid')


    def call(self, x):
        #  unraveled = self.f1(x)
        layer1 = self.d1(x)
        layer2 = self.d2(layer1)

        return layer2 

# def two_layer_fc_test():
#     """Test function to make sure the dimensions are working"""

#     # Create an instance of the model
#     fc_model = FCmodel()

#     # shape is: number of examples (mini-batch size), width, height, depth
#     #x_np = np.zeros((64, 32, 32, 3))
#     x_np = np.random.rand(64, 32, 32, 3)

#     # call the model on this input and print the result
#     output = fc_model.call(x_np)
#     print(output)

#     for v in fc_model.trainable_variables:
#         print("Variable:", v.name)
#         print("Shape:", v.shape)

def train_step(train_images, y_true, model, loss_fn):
    """
    A function to call the training step on a model by calculating loss and updating weights 
    Inputs: Training data for this epoch (train_images), True labels for train data (y_true),
        Model object (model), Loss function object - in this case SparseCategoricalCrossentropy (loss_fn)
    """
    print("Train Step reached.")

    # Compute gradient w/ respect to weights
    with tf.GradientTape() as g:
        # Getting predicted labels from FC NN model
        y_pred = model.call(train_images)

        # Computing loss using SparseCategoricalCrossentropy 
        loss = loss_fn(y_true, y_pred)

    # Get current weights (aka Model.trainable_variables)
    current_weights = model.trainable_variables

    # Calculating gradient using the GradientTape earlier     
    gradient = g.gradient(loss, current_weights) 

    # Updating weights using the adam optimizer 
    model.optimizer.apply_gradients(zip(gradient, current_weights))

    # return the loss and predictions
    return loss, y_pred

def run_training(model, train_dset):
    """
    run training step for given model
    """

    # Setting up the optimizer and loss function 
    scce = SparseCategoricalCrossentropy()
    model.optimizer = tf.keras.optimizers.Adam() 

    # set up metrics for training and validation 
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy')

    # val_loss = tf.keras.metrics.Mean(name='val_loss')
    # val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
    #     name='val_accuracy')
   
    train_accuracy_arr = []
    # val_accuracy_arr = []

    epoch = 10
    # Run the training step once for every epoch, calculating loss and accuracy 
    for i in range(epoch):
        print(f'Epoch {i}')
        for images, labels in train_dset:
            loss, predictions = train_step(images, labels, model, scce)
            train_loss(loss)
            train_accuracy(labels, predictions)

        # Printing the outputs
        template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
        print(template.format(i+1,
                            train_loss.result(),
                            train_accuracy.result()*100))
        print("printing train acc")
        print(train_accuracy.result().numpy().item(0)*100)
        # print("printing val acc")
        # print(val_accuracy.result().numpy().item(0)*100)
        # Adding values to output arrays
        train_accuracy_arr.append(train_accuracy.result().numpy().item(0)*100)
        # val_accuracy_arr.append(val_accuracy.result().numpy().item(0)*100)

        # Reset the metrics for the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        # val_loss.reset_states()
        # val_accuracy.reset_states()
    return train_accuracy_arr

# def run_model(csv_name, label_name):
#     X_train, X_test, y_train, y_test = util.process_txt(csv_name, label_name)

#     print("Getting test and train datasets...")
#     train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=10)
#     # val_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
#     test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
#     print("Finished getting test and train data!")

#     for i in train_dset: 
#         print(i.numpy()) 


#     fc_model = FCmodel()
#     # fc_nn.train_step(train_dset)

#     print("Running FC model...")
#     fc_model = FCmodel()
#     fc_train_acc = run_training(fc_model, train_dset)
#     print("Finished running FC model!")
    

    # logreg.fit(X_train, y_train)

    # y_pred = logreg.predict(X_test)

    # accuracyLog = accuracy_score(y_test, y_pred)
    # print("Logistic Regression accuracy: ", accuracyLog*100, "%")

    # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # print(cnf_matrix)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=logreg.classes_)
    # disp.plot()
    # plt.title(f'Logistic Regression predicting {label_name}')
    # plt.savefig(f'cm_logreg_{label_name}.png')



def main():
    # test two layer function
    # two_layer_fc_test()
    # opts = util.parse_args()
    # label_col = opts.label_col
    # print("Calling run_model...")
    # run_model("NeighborhoodFoodRetail.csv", label_col)
    # print("Run model call finished!")

    label_col = ["SUPERMARKET_ACCESS", "HIGH_POVERTY"] #opts.label_col
    prob_pos = [] 
    df = util.process_txt("NeighborhoodFoodRetail.csv")
    for i in label_col:
        X_train, X_test, y_train, y_test = util.split_data(df, i)
        #  X_train, X_test, y_train, y_test = util.process_txt(csv_name, label_name)

        print("Getting test and train datasets...")
        train_dset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(64)
        # val_dset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(64)
        test_dset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        print("Finished getting test and train data!")

        for i in train_dset: 
            print(i.numpy()) 


        fc_model = FCmodel()
        # fc_nn.train_step(train_dset)

        print("Running FC model...")
        fc_model = FCmodel()
        fc_train_acc = run_training(fc_model, train_dset)
        print("Finished running FC model!")
    
        # prob_pos.append(run_model(X_train, X_test, y_train, y_test, i))

    # visual(prob_pos[0], prob_pos[1], label_col)


    

if __name__ == "__main__":
    main()
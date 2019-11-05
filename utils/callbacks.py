from keras.callbacks import Callback, BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class EpochCheckpoint(Callback):
    def __init__(self, output_path, every=5, start_at=0):
        # call the parent constructor
        super(Callback, self).__init__()

        # store the base output path for the model, the number of epochs that must pass before
        # the model is serialized to disk and current epoch value
        self.output_path = output_path
        self.every = every
        self.int_epoch = start_at

    def on_epoch_end(self, epoch, logs={}):
        # check to see if the model needs to be serialized to disk
        if (self.int_epoch + 1) % self.every == 0:
            p = os.path.sep.join([self.output_path, "epoch_{}.hdf5".format(self.int_epoch + 1)])
            self.model.save(p, overwrite=True)

        # increment the internal epoch counter
        self.int_epoch += 1


class TrainingMonitor(BaseLogger):
    def __init__(self, fig_path, json_path=None, start_at=0):
        # store the output path for the figure, the path to the json serialized
        # file, and the starting epoch
        super(BaseLogger, self).__init__()
        self.fig_path = fig_path
        self.json_path = json_path
        self.start_at = start_at
        self.stateful_metrics = []

    def on_train_begin(self, logs={}):
        # initialize the history dictionary
        self.H = {}

        # if the json history path exists, load the training history
        if self.json_path is not None:
            if os.path.exists(self.json_path):
                self.H = json.loads(open(self.json_path).read())

                # check to see if a starting epoch was supplied
                if self.start_at > 0:
                    # loop over the entries in the history log and trim any entries
                    # that are past the starting epoch
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.start_at]

    def on_epoch_end(self, epoch, logs={}):
        # loop over the logs and update the loss, accuracy, etc
        # for the entire training process
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        # check to see if the training history should be serialized to file
        if self.json_path is not None:
            f = open(self.json_path, "w")
            f.write(json.dumps(self.H))
            f.close()

        # ensure at-least two epochs have passed before plotting
        # (epoch starts at 0)
        if len(self.H["loss"]) > 1:
            # plot the training loss and accuracy
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["acc"], label="train_acc")
            plt.plot(N, self.H["val_acc"], label="val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            # save the figure
            plt.savefig(self.fig_path)
            plt.close()

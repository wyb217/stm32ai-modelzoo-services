# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/
    
import tensorflow as tf
from data_augmentation import data_augmentation

def change_model_input_shape(model,new_inp_shape):

    conf = model.get_config()
    conf['layers'][0]['config']['batch_input_shape'] = new_inp_shape
    new_model = model.__class__.from_config(conf, custom_objects={})

    # iterate over all the layers that we want to get weights from
    weights = [layer.get_weights() for layer in model.layers[1:]]
    for layer, weight in zip(new_model.layers[1:], weights):
        layer.set_weights(weight)

    old_inp_shape = model.get_config()['layers'][0]['config']['batch_input_shape']

    return new_model, old_inp_shape


class HMTrainingModel(tf.keras.Model):

    def __init__(self,model,hm_loss,hm_metrics,data_augmentation_cfg=None,pixels_range=None,image_size=(None,None)):
            
        super(HMTrainingModel, self).__init__()

        self.metrics_tracker     = [tf.keras.metrics.Mean(name='loss')] # add the loss to the metrics tracker
        self.val_metrics_tracker = [tf.keras.metrics.Mean(name='loss')] # add the loss to the metrics tracker

        for hmm in hm_metrics:
            self.metrics_tracker.append(tf.keras.metrics.Mean(name=hmm.__name__)) # add the other metrics
            self.val_metrics_tracker.append(tf.keras.metrics.Mean(name=hmm.__name__)) # add the other metrics

        self.hm_loss    = hm_loss
        self.hm_metrics = hm_metrics
        self.data_augmentation_cfg = data_augmentation_cfg
        self.pixels_range = pixels_range

        if self.data_augmentation_cfg.random_periodic_resizing is not None:
            self.model, self.original_inp_shape = change_model_input_shape(model,(None,None,None,3))
            self.resolutions = self.data_augmentation_cfg.random_periodic_resizing.image_sizes
        else:
            self.model, self.original_inp_shape = model, (None,) + image_size + (3,)
            self.resolutions = [[image_size[0],image_size[1]]]

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.model.save_weights(
            filepath, overwrite=overwrite, save_format=save_format, options=options)

    def load_weights(self, filepath, skip_mismatch=False, by_name=False, options=None):
        return self.model.load_weights(
            filepath, skip_mismatch=skip_mismatch, by_name=by_name, options=options)

    def train_step(self, data):
        # The data loader supplies groundtruth boxes in
        images, y_true = data

        if self.data_augmentation_cfg is not None:
            images, y_true = data_augmentation(images, y_true, self.data_augmentation_cfg, self.pixels_range)

        image_size = tf.shape(images)[1:3]

        for reso in self.resolutions:
            if self.data_augmentation_cfg.random_periodic_resizing is not None:
                resized_images = tf.image.resize(images, [reso[1], reso[0]], method='bilinear')
            else:
                resized_images = images

            with tf.GradientTape() as tape:
                y_pred = self.model(resized_images, training=True)
                loss   = self.hm_loss(y_true, y_pred)
                metrcs = []
                for hmm in self.hm_metrics:
                    metrcs.append(hmm(y_true, y_pred))


            # Compute gradients and update weights
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # Update metrics (includes the metric that tracks the loss)
            for i,mt in enumerate(self.metrics_tracker):
                if mt.name == "loss":
                    mt.update_state(loss)
                else:
                    mt.update_state(metrcs[i-1])

        return {m.name: m.result() for m in self.metrics_tracker}

    def test_step(self, data):
        # The data loader supplies groundtruth boxes in
        images, y_true = data

        y_pred = self.model(images, training=False)

        loss   = self.hm_loss(y_true, y_pred)

        metrcs = []
        for hmm in self.hm_metrics:
            metrcs.append(hmm(y_true, y_pred))

        # Update metrics (includes the metric that tracks the loss)
        for i,mt in enumerate(self.val_metrics_tracker):
            if mt.name == "loss":
                mt.update_state(loss)
            else:
                mt.update_state(metrcs[i-1])

        return {m.name: m.result() for m in self.val_metrics_tracker}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.metrics_tracker + self.val_metrics_tracker
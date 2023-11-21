import tensorflow as tf
from tensorflow import keras

class CustomModel(keras.Model):
    '''
    Create a custom tensorflow model via custom training loop functions
    which modify the functionality of `.fit()` method
    '''

    @tf.function
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            logits = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, logits, regularization_losses=self.losses)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, logits)

        # Return a dictionary mapping metric names to their current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y = data
        # Perform a forward pass (no gradient calculation during testing)
        val_logits = self(x, training=False)

        # Update metrics
        self.compiled_metrics.update_state(y, val_logits)

        # Return a dictionary mapping metric names to their current value
        return {m.name: m.result() for m in self.metrics}
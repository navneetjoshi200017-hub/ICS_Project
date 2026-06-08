import tensorflow as tf


class ConfidenceCapLoss(tf.keras.losses.Loss):
    """
    Cross-entropy + penalty when the model is too confident.
    If top prediction goes above confidence_cap, the excess is penalised.
    """
    def __init__(self, penalty_weight: float = 0.1,
                 confidence_cap: float = 0.85,
                 name="confidence_cap_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.penalty_weight  = penalty_weight
        self.confidence_cap  = confidence_cap

    def call(self, y_true, y_pred):
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=False
        )
        max_confidence = tf.reduce_max(y_pred, axis=-1)
        excess  = tf.nn.relu(max_confidence - self.confidence_cap)
        penalty = tf.reduce_mean(excess)
        return tf.reduce_mean(base_loss) + self.penalty_weight * penalty

    def get_config(self):
        return {**super().get_config(),
                "penalty_weight": self.penalty_weight,
                "confidence_cap": self.confidence_cap}
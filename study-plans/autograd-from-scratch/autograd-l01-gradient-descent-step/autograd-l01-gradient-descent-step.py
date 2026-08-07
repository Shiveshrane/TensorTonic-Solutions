def gradient_descent_step(values, gradients, learning_rate):
    """Returns: updated values and predicted first-order objective change."""
    values = np.asarray(values, dtype=np.float64)
    gradients = np.asarray(gradients, dtype=np.float64)

    updated = values - learning_rate * gradients
    delta_loss = float(np.dot(gradients, updated - values))

    return updated.tolist(), float(delta_loss)

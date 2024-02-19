import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

# Check tensorflow and datasets versions
print()
print("TensorFlow version: {}".format(tf.__version__))
print("TensorFlow Datasets version: ", tfds.__version__)

# Create classes and get the dataset split into a 80:20 ratio of train and test data
class_names = ['Adélie', 'Chinstrap', 'Gentoo']
ds_split, info = tfds.load("penguins/processed", split=['train[:20%]', 'train[20%:]'], as_supervised=True,
                           with_info=True)

# Assign the split data to the variables
ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_test, tf.data.Dataset)

# Sample the testdata
print()
print(info.features)
df_test = tfds.as_dataframe(ds_test.take(5), info)
print()
print("Test dataset sample: ")
print(df_test)

# Sample the train data
df_train = tfds.as_dataframe(ds_train.take(5), info)
print()
print("Train dataset sample: ")
print(df_train)

# Train with bathces of 32
ds_train_batch = ds_train.batch(32)

# Iterate over the data and print the features and labels
features, labels = next(iter(ds_train_batch))
print()
print('features: ')
print(features)

print()
print('labels: ')
print(labels)

# Visualize some of the features
plt.scatter(features[:, 0],
            features[:, 2],
            c=labels,
            cmap='viridis')

plt.xlabel("Body Mass")
plt.ylabel("Culmen Length")
# plt.show()

# Create your own model using Kera
# Keras Sequential allows you to create a linear stack of layers
# Each layer has 10 nodes each
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

predictions = model(features)
predictions[:5]
print()
print('Predictions: ')
print(predictions)

tf.nn.softmax(predictions[:5])

print()
print("Prediction: {}".format(tf.math.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

# Calculate the loss of the model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def loss(model, x, y, training):
    y_ = model(x, training=training)

    return loss_object(y_true=y, y_pred=y_)


# training=training is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
l = loss(model, features, labels, training=False)

print()
print("Loss test: {}".format(l))


# Calculate the gradients used to optimize the model
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# Calculate the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Optimize the model
loss_value, grads = grad(model, features, labels)

print()
print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))
print()

## Note: Rerunning this cell uses the same model parameters

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in ds_train_batch:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    # Log the epoch, loss and accuracy if the epoch count is dividable by 50
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))
# Visualize the drop in loss and climb in accuracy
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
# plt.show()

# Test the models accuracy
test_accuracy = tf.keras.metrics.Accuracy()
ds_test_batch = ds_test.batch(10)

for (x, y) in ds_test_batch:
    # training=False is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    logits = model(x, training=False)
    prediction = tf.math.argmax(logits, axis=1, output_type=tf.int64)
    test_accuracy(prediction, y)

print()
print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

# Inspect last batch of the test data
print()
print(tf.stack([y, prediction], axis=1))
print()

# Test the model with custom data
predict_dataset = tf.convert_to_tensor([
    [0.5, 0.8, 0.6, 0.5, ],
    [0.4, 0.4, 0.8, 0.3, ],
    [0.7, 0.3, 0.8, 0.6]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
    class_idx = tf.math.argmax(logits).numpy()
    p = tf.nn.softmax(logits)[class_idx]
    name = class_names[class_idx]
    print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100 * p))

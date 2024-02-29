import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_classes = 3

data_directory = "data"
class_names = ['copepod', 'egg', 'nauplii']

dataset = tf.keras.utils.image_dataset_from_directory(
    data_directory,
    class_names=class_names,
    image_size=(224,224)
)

data_for_batch = dataset.as_numpy_iterator()

batch = data_for_batch.next()

fig, ax = plt.subplots(ncols=10, figsize=(20,20))
for idx, img in enumerate(batch[0][:10]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

plt.show()

train_size = int(len(dataset)*0.7)
val_size = int(len(dataset)*0.2)
test_size = int(len(dataset)*0.1)

trainset = dataset.take(train_size)
valset = dataset.skip(train_size).take(val_size)
testset = dataset.skip(train_size).skip(val_size).take(test_size)

base_model = tf.keras.applications.ResNet50(weights="imagenet",
                                            include_top=False)


avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)


model = tf.keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

model = model.fit(trainset,
                    epochs=5,
                    validation_data=valset)

# predicted = model.predict(testset)

# fig, ax = plt.subplots(ncols=20, nrows=2, figsize=(20,20))
# for idx, img in enumerate(testset[0][:40]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(predicted[1][idx])

# plt.show()

# predicted = tf.squeeze(predicted).numpy()
# predicted_ids = np.argmax(predicted, axis=-1)

# print(predicted_ids)

# sum = 0
# for i in range(len(predicted)):
#     if testset[i] == predicted[i]:
#         sum += 1

# print("Accuracy = ", sum/len(predicted_ids))

# dataset, info = tfds.load("tf_flowers",
#                           as_supervised=True,
#                           with_info=True)

# dataset_size = info.splits["train"].num_examples
# class_names = info.features["label"].names
# n_classes = info. features["label"].num_classes


# (test_set_raw, valid_set_raw, train_set_raw), dataset_info = tfds.load("tf_flowers",
#                         split=["train[:10%]",
#                                "train[10%:25%]",
#                                "train[25%:]"],
#                         as_supervised=True,
#                         with_info = True)

# def preprocess(image,label):
#     resized_image = tf.image.resize(image, [224,224])
#     final_image = tf.keras.applications.xception.preprocess_input(resized_image)
#     return final_image, label

# batch_size = 32
# train_set = train_set_raw.shuffle(1000)
# train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
# valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
# test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# base_model = tf.keras.applications.ResNet50(weights="imagenet",
#                                             include_top=False)


# avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# output = tf.keras.layers.Dense(n_classes, activation="softmax")(avg)


# model = tf.keras.Model(inputs=base_model.input, outputs=output)

# for layer in base_model.layers:
#   layer.trainable = False

#   optimizer = tf.keras.optimizers.Adam(learning_rate=0.2)

# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer=optimizer,
#               metrics=["accuracy"])


# history = model.fit(train_set,
#                     epochs=5,
#                     validation_data=valid_set)

# for layer in base_model.layers:
#   layer.trainable = True

#   optimizer = tf.keras.optimizers.Adam(lr=0.2)

# model.compile(loss="sparse_categorical_crossentropy",
#               optimizer=optimizer,
#               metrics=["accuracy"])

# history = model.fit(train_set,
#                     epochs=10,
#                     validation_data=valid_set)

# class_names = np.array(dataset_info.features['label'].names)
# print(class_names)

# image_batch, label_batch = next(iter(valid_set))
# image_batch = image_batch.numpy()
# label_batch = label_batch.numpy()

# predicted_batch = model.predict(image_batch)
# predicted_batch = tf.squeeze(predicted_batch).numpy()
# predicted_ids = np.argmax(predicted_batch, axis=-1)
# predicted_class_names = class_names[predicted_ids]

# sum = 0
# for i in range(len(predicted_ids)):
#     if label_batch[i] == predicted_ids[i]:
#         sum += 1

# print("Accuracy = ", sum/len(predicted_ids))

# plt.figure(figsize=(10,10))
# for n in range(30):
#     plt.subplot(6,5,n+1)
#     plt.subplots_adjust(hspace = 0.3)
#     plt.imshow(image_batch[n])
#     color = "blue" if predicted_ids[n] == label_batch[n] else "red"
#     plt.title(predicted_class_names[n].title(), color=color)
#     plt.axis('off')
# _ = plt.suptitle("Model predictions (blue: correct, red: incorrect)")
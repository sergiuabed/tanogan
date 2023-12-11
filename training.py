import tensorflow as tf
from tensorflow import keras
import os
import datetime


def adversarial_training(generator, discriminator, data, batch_size, windows_size, latent_size, loss_fn, discr_optimizer, gen_optimizer, checkpoints_backup_path, num_epochs=1):

    # setup checkpoint saving
    directory_path = "./checkpoints"

    filepath_generator = directory_path + "/generator/"
    filepath_discriminator = directory_path + "/discriminator/"

    if not os.path.isdir('checkpoints'):
        os.mkdir("checkpoints")  # ! mkdir checkpoints

        os.mkdir(filepath_generator)  # ! mkdir {filepath_generator}
        os.mkdir(filepath_discriminator)  # ! mkdir {filepath_discriminator}

    real_label = 0
    fake_label = 1

    print("epoch: 0/%d")
    for epoch in range(num_epochs):
        for i, (time_sequences, labels) in enumerate(data):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            # Train with real data
            with tf.GradientTape() as tape:
                scores = discriminator(time_sequences)
                discr_real_loss = loss_fn(labels, scores)

            gradients = tape.gradient(
                discr_real_loss, discriminator.trainable_weights)
            discr_optimizer.apply_gradients(
                zip(gradients, discriminator.trainable_weights))

            # Train with fake data
            noise = tf.random.normal(
                (time_sequences.shape[0], time_sequences.shape[1], latent_size), mean=0, stddev=1)
            fake_sequences = generator(noise)
            fake_labels = tf.ones(labels.shape) * fake_label

            with tf.GradientTape() as tape:
                scores = discriminator(fake_sequences)
                discr_fake_loss = loss_fn(fake_labels, scores)

            gradients = tape.gradient(
                discr_fake_loss, discriminator.trainable_weights)
            discr_optimizer.apply_gradients(
                zip(gradients, discriminator.trainable_weights))

            # (2) Update G network: maximize log(D(G(z)))

            noise = tf.random.normal(
                (time_sequences.shape[0], time_sequences.shape[1], latent_size), mean=0, stddev=1)
            with tf.GradientTape() as tape:
                fake_sequences = generator(noise)
                #real_labels = tf.ones(noise.shape)
                # discriminator tells which fake sequences it considers to be real and which not
                discriminator_output = discriminator(fake_sequences)
                generator_loss = loss_fn(labels, discriminator_output)

            gradients = tape.gradient(
                generator_loss, generator.trainable_weights)
            gen_optimizer.apply_gradients(
                zip(gradients, generator.trainable_weights))

            print(
                f"epoch: {epoch}/{num_epochs},    batch: {i}/{len(data)}    Discriminator_loss: {discr_real_loss+discr_fake_loss}  Generator_loss: {generator_loss}"
            )

        generator.save_weights(filepath_generator + f"gen_epoch{epoch}.h5")
        discriminator.save_weights(
            filepath_discriminator + f"discr_epoch{epoch}.h5")

# The lines below are to be used in Google Colab to make a copy of the checkpoints and
# move them on your personal Google Drive
#    # zip and move a copy of the checkpoints
#    now = datetime.datetime.now()
#    zip_name = f"checkpoints_{str(now.date())+'_'+str(now.time())}.zip"
#
#    !zip -r {zip_name} ./checkpoints
#    !cp {zip_name} {checkpoints_backup_path}


def encoder_training(encoder, generator, data, latent_size, checkpoints_backup_path, num_epochs=1):
    '''
    Encoder must be trained on regular data only.

    The purpose of the encoder is to learn the inverse mapping of the generator,
    i.e. how to map a time sequence to the latent space.
    '''

    # setup checkpoint saving
    directory_path = "./encoder_checkpoints"

    if not os.path.isdir('encoder_checkpoints'):
        os.mkdir("encoder_checkpoints")  # ! mkdir encoder_checkpoints

    loss_fn = tf.keras.losses.MeanAbsoluteError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)

    print("epoch: 0/%d")
    for epoch in range(num_epochs):
        sum_loss = 0
        for i, (time_sequences, labels) in enumerate(data):
            with tf.GradientTape() as tape:
                latent_output = encoder(time_sequences)
                generated_seq = generator(latent_output)

                enc_loss = loss_fn(time_sequences, generated_seq)

            gradients = tape.gradient(enc_loss, encoder.trainable_weights)
            optimizer.apply_gradients(
                zip(gradients, encoder.trainable_weights))

            sum_loss += enc_loss

            print(
                f"  epoch: {epoch}/{num_epochs},    batch: {i}/{len(data)}    Encoder_loss: {enc_loss}"
            )

        print(
            f"epoch: {epoch}/{num_epochs},  Tot_epoch_loss: {sum_loss}"
        )

        encoder.save_weights(directory_path + f"/encoder_epoch{epoch}.h5")

# The lines below are to be used in Google Colab to make a copy of the checkpoints and
# move them on your personal Google Drive
#    now = datetime.datetime.now()
#    zip_name = f"encoder_checkpoints_{str(now.date())+'_'+str(now.time())}.zip"
#
#    !zip -r {zip_name} ./encoder_checkpoints
#    !cp {zip_name} {checkpoints_backup_path}

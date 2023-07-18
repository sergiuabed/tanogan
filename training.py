import tensorflow as tf
from tensorflow import keras

def adversarial_training(generator, discriminator, data, batch_size, windows_size, loss_fn, optimizer, num_epochs=1):
    real_label = 1
    fake_label = 0

    print("epoch: 0/%d")
    for epoch in range(num_epochs):
        for i, (time_sequences, labels) in enumerate(data):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))

            #Train with real data
            with tf.GradientTape() as tape:
                scores = discriminator(time_sequences)
                discr_real_loss = loss_fn(labels, scores)

            gradients = tape.gradient(discr_real_loss, discriminator.trainable_weights)
            optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))
            
            #Train with fake data
            noise = tf.random.normal(time_sequences.shape, mean=0, stddev=0.1)
            fake_sequences = generator(noise)
            fake_labels = tf.ones(noise.shape) * fake_label

            with tf.GradientTape() as tape:
                scores = discriminator(fake_sequences)
                discr_fake_loss = loss_fn(fake_labels, scores)

            gradients = tape.gradient(discr_fake_loss, discriminator.trainable_weights)
            optimizer.apply_gradients(zip(gradients, discriminator.trainable_weights))
            

            # (2) Update G network: maximize log(D(G(z)))

            noise = tf.random.normal(time_sequences.shape, mean=0, stddev=0.1)
            with tf.GradientTape() as tape:
                fake_sequences = generator(noise)
                #real_labels = tf.ones(noise.shape)
                discriminator_output = discriminator(fake_sequences)    # discriminator tells which fake sequences it considers to be real and which not
                generator_loss = loss_fn(labels, discriminator_output)

            gradients = tape.gradient(generator_loss, generator.trainable_weights)
            optimizer.apply_gradients(zip(gradients, generator.trainable_weights))
            
            print(
                f"epoch: {epoch}/{num_epochs},    batch: {i}/{len(data)}    Discriminator_loss: {discr_real_loss+discr_fake_loss}  Generator_loss: {generator_loss}"
            )
            
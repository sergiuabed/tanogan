import tensorflow as tf
from tensorflow import keras

def z_loss(time_sequences, fake_sequences, discriminator, _lambda):
    '''
    This function acts as both loss function when mapping a time sequence to
    a latent variable 'z' and as anomaly score.

    This loss is the sum of 2 losses:
        - residual loss: the sum of the absolute value of the components of the difference
            between a real time sequence and a generated one by the generator

        - discrimination loss: the sum of the absolute value of the components of the difference
            between the outputs of the LSTM layer of the discriminator when the inputs
            are a real time sequence and a generated one
    '''
    residual_loss = tf.reduce_sum(abs(time_sequences-fake_sequences))
    
    interm_layer = discriminator.layers[0]  # LSTM layer
    
    features_real = interm_layer(time_sequences)
    features_fake = interm_layer(fake_sequences)

    discrimination_loss = tf.reduce_sum(abs(features_real-features_fake))

    total_loss = (1-_lambda)*residual_loss + _lambda*discrimination_loss

    return total_loss

def latent_space_map(time_sequences, num_iters, generator, discriminator, z_optimizer, _lambda):
    '''
    Maps a batch of time sequences to the latent space

    Because the generator's mapping function G(z) from latent space
    to the space of realistic time sequences is not invertible,
    we must leverage gradient descent to find a latent space value 'z'
    such that G(z) is as close as possible to a given time sequence 'x'

    return: the loss between 'time_sequences' and G(z), where 'z' is
    the latent variable value found after 'num_iters' iterations of 
    performing gradient descent.
    '''
    z = tf.random.normal(time_sequences.shape, mean=0, stddev=0.1)

    for _ in range(num_iters):
        loss = None
        with tf.GradientTape() as tape:
            generated_sequences = generator(z)
            loss = z_loss(time_sequences, generated_sequences, discriminator, _lambda)

        gradients = tape.gradient(loss, z)
        z_optimizer.apply_gradients(zip(gradients, z))

    # maybe compute the loss again before returning
    return loss

def anomaly_score(test_data, generator, discriminator, z_optimizer, nr_latent_map_iters, _lambda):
    '''
    Outputs a list of losses, each loss corresponding to a time window
    '''

    # IMPORTANT: DURING ANOMALY DETECTION, SET BATCH_SIZE=1!!!!!!!!!!!!!!!!!!
    # i.e., a batch contains a single window of timesteps
    
    loss_list = []
    for _, batch in enumerate(test_data):
        loss_list.append(
            latent_space_map(batch[0], nr_latent_map_iters, generator, discriminator, z_optimizer, _lambda)
            ) # batch size is 1, so batch[0] points to the window
        
    return loss_list

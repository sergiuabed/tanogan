import tensorflow as tf
from tensorflow import keras

# All functions in this file correspond to ModifiedTAnoGAN anomaly detection functionality

def reconstruction_loss(time_sequences, fake_sequences, discriminator, _lambda):
    '''
    This function acts as both recontruction loss function when mapping a time sequence to
    a latent variable 'z' and as anomaly score.

    This loss is the sum of 2 losses:
        - residual loss: the sum of the absolute value of the components of the difference
            between a real time sequence and a generated one by the generator

        - discrimination loss: the sum of the absolute value of the components of the difference
            between the outputs of the LSTM layer of the discriminator when the inputs
            are a real time sequence and a generated one
    '''
    residual_loss = tf.reduce_sum(abs(time_sequences - fake_sequences))

    interm_layer = discriminator.layers[0]  # LSTM layer

    features_real = interm_layer(time_sequences)
    features_fake = interm_layer(fake_sequences)

    discrimination_loss = tf.reduce_sum(abs(features_real-features_fake))

    total_loss = (1-_lambda)*residual_loss + _lambda*discrimination_loss

    return total_loss


def anomaly_score(test_data, generator, discriminator, encoder, _lambda):
    '''
    Outputs a list of losses, each loss corresponding to a time window
    '''

    # IMPORTANT: DURING ANOMALY DETECTION, SET BATCH_SIZE=1!!!!!!!!!!!!!!!!!!
    # i.e., a batch contains a single window of timesteps

    i = 0
    loss_list = []
    for _, batch in enumerate(test_data):
        print(f"batch: {i} out of {len(test_data)}")
        i += 1

        # map time sequences to latent space
        # batch size is 1, so batch[0] points to the window
        latent_var = encoder(batch[0])

        # reconstruct time sequences from latent space
        reconstr_sequences = generator(latent_var)

        # compute reconstruction loss
        loss = reconstruction_loss(
            batch[0], reconstr_sequences, discriminator, _lambda)

        loss_list.append(loss)
    return loss_list

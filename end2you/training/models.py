import tensorflow as tf

from pathlib import Path

slim = tf.contrib.slim


def recurrent_model(audio_features, 
                    visual_features,
                    text_features,
                    task, 
                    hidden_units=256, 
                    number_of_outputs=2):
    
    with tf.variable_scope("recurrent", reuse=tf.AUTO_REUSE):
        batch_size, seq_length, num_features = audio_features.get_shape().as_list()
        
        lstm = tf.contrib.rnn.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)
        
        multimodal_features = tf.concat([audio_features, visual_features, text_features], -1)
        
        outputs, states = tf.nn.dynamic_rnn(lstm, multimodal_features, dtype=tf.float32)
        
        net = tf.reshape(outputs, (-1, hidden_units))
        
        if task == '2':
            net = tf.reshape(outputs[:, -1, :], (batch_size, hidden_units))
            prediction = tf.nn.sigmoid(slim.layers.linear(net, number_of_outputs))
        else:
            prediction = tf.nn.tanh(slim.layers.linear(net, number_of_outputs))
        
        return tf.reshape(prediction, (batch_size, -1, number_of_outputs))


def audio_model(audio_frames=None, conv_filters=40):
    with tf.variable_scope('audio_model', reuse=tf.AUTO_REUSE):
        
        batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
        
        audio_input = tf.reshape(audio_frames, [-1,  num_features, 1])
        
        net = tf.layers.conv1d(audio_input, 50, 8, activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 10, 10)
        net = slim.dropout(net, 0.5)
        
        net = tf.layers.conv1d(net, 125, 8, activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 8, 8)
        net = slim.dropout(net, 0.5)
        
        net = tf.layers.conv1d(net, 250, 6, activation=tf.nn.relu)
        net = tf.layers.max_pooling1d(net, 6, 6)
        net = slim.dropout(net, 0.5)
        
        _, num_features, num_channels = net.get_shape().as_list()
        net = tf.reshape(net, [batch_size, -1, num_features * num_channels])
        
        return net


def get_model(task, num_outs, model_inp):
    """ Get model predictions.
    Args:
      raw_waveform (batch_size, seq_length, 4000): The input raw waveform.
    """
    
    raw_waveform, visual_features, text_features = model_inp
    return recurrent_model(
        audio_model(raw_waveform), visual_features, text_features, task=task, number_of_outputs=num_outs)

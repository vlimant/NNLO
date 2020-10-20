# Constants
PATH_DATA = '/storage/group/gpu/bigdata/CMSOpPred/'
N_CODES = 77
N_SITES = 81
N_COUNTS = 2
N_WORDS = 30674
MAX_WORDS = 400


def make_count_model(**args):
    
    from keras.layers import Input, Flatten, Dense, Dropout, Reshape, multiply
    from keras.regularizers import l2
    from keras.models import Model
    if args:logging.debug("receiving arguments {}".format(args))
        
    dense_layers = args.get('dense_layers', 3)
    dense_units = args.get('dense_units', 50)
    l2reg = args.get('l2reg', 0.001)
    dropout = args.get('dropout', 0.001)
   
    
    m_input = Input((N_CODES, N_SITES, N_COUNTS))

    m = m_input

    m = Flatten()(m)
    for _ in range( dense_layers ):
        m = Dense( units = dense_units, activation='relu',
                   kernel_regularizer=l2(l2reg)) (m)
        m = Dropout(dropout)(m)

    m_output = Dense( units=1, activation='sigmoid')(m)

    model = Model(inputs=m_input, outputs=m_output)
    return model


def make_nlp_model(**args):
    
    from keras.layers import Embedding, Input, Dense, GRU, TimeDistributed, Dropout, Flatten, Reshape, Concatenate
    from keras.regularizers import l2
    from keras.models import Model
    if args:logging.debug("receiving arguments {}".format(args))

    # Hyper parameter
    rnn_units = args.get('rnn_units', 10)
    embedding_dim = args.get('embedding_dim', 20)
    l2_reg = args.get('l2_reg', 0.)
    rec_do = args.get('rec_do', 0.)
    dense_layers = args.get('dense_layers', 3)
    dense_units = args.get('dense_units', 50)
    site_units = args.get('site_units', 100)
    do = args.get('do', 0.)
    
    # Constants
    encode_sites = False
    
    # Word encoder model
    words_input = Input(shape = ( None, ), dtype='int32')
    words_embedding = Embedding(N_WORDS, embedding_dim, mask_zero = True)(words_input)
    words_gru = GRU(rnn_units, kernel_regularizer=l2(l2_reg), recurrent_dropout = rec_do)(words_embedding)
    wordEncoder = Model(words_input, words_gru)
    
    # Full model
    sent_input = Input(shape = (N_CODES * N_SITES, None), dtype='int32')
    count_input = Input(shape = (N_CODES, N_SITES, 2, ), dtype='float32')
    sent_encoded = TimeDistributed(wordEncoder)(sent_input)
    sent_encoded_reshaped = Reshape(( N_CODES , N_SITES, rnn_units))(sent_encoded)
    concat_counts_sent = Concatenate(axis=3)([sent_encoded_reshaped, count_input])
    if encode_sites:
        codes_reshaped = Reshape(( N_CODES , N_SITES * (rnn_units + N_COUNTS)))(concat_counts_sent)
        sites_encoded = TimeDistributed(Dense(site_units, activation = 'relu', kernel_regularizer=l2(l2_reg)))(codes_reshaped)
        flat = Flatten()(sites_encoded)                                 
    else:
        flat = Flatten()(concat_counts_sent)
    dense = flat
    for _ in range(dense_layers):
        dense = Dense( dense_units, activation='relu', kernel_regularizer=l2(l2_reg) )(dense)
        dense = Dropout(do)(dense)          
    preds = Dense(1, activation='sigmoid')(dense)
    model = Model([sent_input, count_input], preds)
    
    return model


get_model = make_nlp_model



    
import numpy as np
import pickle
with open('/storage/user/llayer/NNLO/index.pickle', 'rb') as handle:
    sites_dict = pickle.load(handle)
    codes_dict = pickle.load(handle)

def to_dense(np_msg, np_counts, index, values):

    errors, sites, counts, site_states, error_messages = values

    # Loop over the codes and sites
    for i_key in range(len(errors)):

        error = errors[i_key]
        site = sites[i_key]
        count = counts[i_key]
        site_state = site_states[i_key]

        # Fill counts
        if site_state == 'good':
            site_state_encoded = 0
        else:
            site_state_encoded = 1
        np_counts[index, codes_dict[error], sites_dict[site], site_state_encoded] += count

        # Fill the error messages
        error_message = error_messages[i_key]
        # Only continue if there exists a message
        if isinstance(error_message, (list,)):
            
            # Cut/Pad the error message
            error_message = np.array(error_message)
            pad_size = np_msg.shape[3] - error_message.shape[0]
            if pad_size < 0:
                error_message = error_message[-np_msg.shape[3] : ]
            else:
                npad = (0, pad_size)
                error_message = np.pad(error_message, pad_width=npad, mode='constant', constant_values=int(0))

            #print( error_message )
            np_msg[index, codes_dict[error], sites_dict[site]] = error_message    

            
def batch_generator( batch ):
    
    batch_size = len(batch)
    tokens_key = 'msg_encoded'
    
    # Loop over the messages to find the longest one
    padding_dim = 1
    for messages in batch[tokens_key]:
        for msg in messages:
            if isinstance(msg, (list,)):
                if len(msg) > padding_dim:
                    padding_dim = len(msg)
    
    # Limit to the maximum number of words
    if padding_dim > MAX_WORDS:
        padding_dim = MAX_WORDS
    
    # Setup the numpy matrix
    np_msg = np.zeros((batch_size, N_CODES, N_SITES, padding_dim), dtype=np.int32)
    np_counts = np.zeros((batch_size, N_CODES, N_SITES, N_COUNTS), dtype=np.int32)
    
    # Fill the matrix
    [to_dense(np_msg, np_counts, counter, values) for counter, values in enumerate(zip(batch['error'],
                                                                                       batch['site'], 
                                                                                       batch['count'],
                                                                                       batch['site_state'], 
                                                                                       batch[tokens_key]))]    
    
    # Reshape the error site matrix for the messages
    np_msg = np_msg.reshape((batch_size, N_CODES * N_SITES, padding_dim))
    
    # Return the matrix
    return [np_msg, np_counts]   
    
    

from skopt.space import Real, Integer, Categorical
get_model.parameter_range =     [
    Real(        low=1e-3, high=0.1,  prior='log-uniform', name='do'            ),
    Real(        low=1e-4, high=0.9,  prior="log-uniform", name='l2_reg'        ),
    Integer(     low=5,    high=32,                        name='embedding_dim' ),
    Integer(     low=5,    high=20,                        name='rnn_units'     ),
    #Integer(     low=5,    high = 20,                      name = 'site_units'  ),
    Integer(     low=1,    high=5,                         name='dense_layers'  ),
    Integer(     low=10,   high=100,                       name='dense_units'   ),
]





def get_name():
    return 'nlp'

def get_train():

    return [PATH_DATA + 'train_0.h5', PATH_DATA + 'train_1.h5', PATH_DATA + 'train_2.h5']

def get_val():

    return [PATH_DATA + 'test_0.h5', PATH_DATA + 'test_1.h5', PATH_DATA + 'test_2.h5']

def get_features():
    return ('frame', batch_generator) ##example of data adaptor

def get_labels():
    return 'label'


if __name__ == "__main__":
    
    model = get_model()
    model.summary()
    
    import pandas as pd
    # Open a frame
    path = PATH_DATA + 'train_0.h5'
    frame = pd.read_hdf(path, 'frame')
    print( frame.head() )
    
    # Get a batch
    start = 0
    batch_size = 2
    batch = frame.iloc[start: start+batch_size]    
    matrix = batch_generator( batch )
    print( matrix[0].shape, matrix[1].shape )
    matrix_msg = matrix[0].reshape((batch_size, N_CODES, N_SITES, matrix[0].shape[2]))
    
    # Fast check that the matrix is filled correctly
    def print_sample( batch, index ):
        
        sample = batch.iloc[index]
        errors = sample['error']
        sites = sample['site']
        message = sample['msg_encoded']
        print( errors )
        print( sites )
        print( message )
        
        for i_key in range(len(errors)):
            
            print( 'Index error', errors[i_key], ':', codes_dict[errors[i_key]], 
                   'Index site', sites[i_key], ':', sites_dict[sites[i_key]] )
            print( 'Inserted in matrix' )
            print( matrix_msg[index, codes_dict[errors[i_key]], sites_dict[sites[i_key]]] )
            
    print_sample( batch, 1 )
    


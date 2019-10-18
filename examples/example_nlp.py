def make_count_model(**args):
    
    from keras.layers import Input, Flatten, Dense, Dropout, Reshape, multiply
    from keras.regularizers import l2
    from keras.models import Model
    if args:logging.debug("receiving arguments {}".format(args))
        
    dense_layers = args.get('dense_layers', 3)
    dense_units = args.get('dense_units', 50)
    l2reg = args.get('l2reg', 0.001)
    dropout = args.get('dropout', 0.001)
    
    n_codes = 77
    n_sites = 81
    n_counts = 2
    
    m_input = Input((n_codes, n_sites, n_counts))

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
    n_codes = 77
    n_sites = 81
    n_counts = 2
    n_words = 30674
    encode_sites = False
    
    # Word encoder model
    words_input = Input(shape = ( None, ), dtype='int32')
    words_embedding = Embedding(n_words, embedding_dim, mask_zero = True)(words_input)
    words_gru = GRU(rnn_units, kernel_regularizer=l2(l2_reg), recurrent_dropout = rec_do)(words_embedding)
    wordEncoder = Model(words_input, words_gru)
    
    # Full model
    sent_input = Input(shape = (n_codes * n_sites, None), dtype='int32')
    count_input = Input(shape = (n_codes, n_sites, 2, ), dtype='float32')
    sent_encoded = TimeDistributed(wordEncoder)(sent_input)
    sent_encoded_reshaped = Reshape(( n_codes , n_sites, rnn_units))(sent_encoded)
    concat_counts_sent = Concatenate(axis=3)([sent_encoded_reshaped, count_input])
    if encode_sites:
        codes_reshaped = Reshape(( n_codes , n_sites * (rnn_units*n_counts)))(concat_counts_sent)
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


PATH_DATA = '/storage/user/llayer/NNLO'

def get_name():
    return 'nlp'

def get_train():

    return [PATH_DATA + 'train_0.h5']

def get_val():

    return [PATH_DATA + 'test_0.h5']

def get_features():
    #return ('features', lambda x: x) ##example of data adaptor
    return 'features'

def get_labels():
    return 'labels'





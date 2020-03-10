# imports libs
import torch
from torch import nn
import torch.nn.functional as functional
import numpy as np
from tqdm import tqdm
import logging
import argparse
from os import path
from CharGru import CharGruNet
#===========================================

# globals
train_on_gpu = True
mini_batch_iterations = 0
#===========

import warnings
warnings.simplefilter("ignore")

# load data
def load_data(file):
    with open(file, 'r') as f:
        text = f.read()
    
    return text


def one_hot_encode(arr, n_labels):
    '''
    arr - input char array
    n_labels - label count(column in result windows)
    '''
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # fill appropriate label with 1
    one_hot[np.arange(arr.size), arr.flatten()] = 1

    one_hot = one_hot.reshape((*arr.shape, n_labels))
    return one_hot


def get_batches(arr, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    total_char = batch_size * seq_length

    n_batches = len(arr)//total_char

    arr = arr[: n_batches * total_char]

    arr = arr.reshape((batch_size, -1))

    global mini_batch_iterations
    mini_batch_iterations = arr.shape[1] / seq_length

    for n in range(0, arr.shape[1], seq_length):
        x = arr[:, n : n + seq_length]
        y = np.zeros_like(x)

        try:
            y [:, :-1] = x[:, 1:]
            y[:, -1] = arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            
        yield x, y


def gpu_check():
    global train_on_gpu
    train_on_gpu = torch.cuda.is_available()
    if(train_on_gpu):    
        logger.info('Training on GPU!')
    else:     
        logger.info('No GPU available, training on CPU; consider making n_epochs very small.')
    return train_on_gpu

# Model=================

class CharRNN(nn.Module):
    def __init__(self, tokens, hidden_size = 256, num_layers = 2, drop_prob = 0.5):
        super().__init__()

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.n_hidden = hidden_size
        self.n_layers = num_layers
        self.drop_prob = drop_prob

        # ==============LSTM===================
        self.lstm = nn.LSTM(input_size = len(self.chars),
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = drop_prob,
                            batch_first = True)
        # ==============LSTM===================

        self.dropout = nn.Dropout(p = self.drop_prob)

        self.fc = nn.Linear(hidden_size, len(self.chars))


    def forward(self, x, hidden):

        lstm_out, hidden = self.lstm(x, hidden)

        lstm_out = self.dropout(lstm_out)

        lstm_out = lstm_out.reshape(-1, self.n_hidden)

        fc_out = self.fc(lstm_out)

        return fc_out, hidden


    def init_hidden(self, batch_size):

        '''
        h_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        If the LSTM is bidirectional, num_directions should be 2, else it should be 1.

        c_0 of shape (num_layers * num_directions, batch, hidden_size): tensor containing the initial cell state for each element in the batch.

        If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        '''
        
        if train_on_gpu:
            hidden = (torch.zeros((self.n_layers, batch_size, self.n_hidden)).cuda(),
                     torch.zeros((self.n_layers, batch_size, self.n_hidden)).cuda())
        else:
            hidden = (torch.zeros((self.n_layers, batch_size, self.n_hidden)),
                     torch.zeros((self.n_layers, batch_size, self.n_hidden)))

        return hidden


    def encode_text(self, text):
        encoded = np.array([self.char2int[ch] for ch in text])

        return encoded
#=======================


def create_logger():

    # create logger with 'spam_application'
    logger = logging.getLogger('Char_RNN')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('CharRNN.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10, model_type='GRU'):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: encoded data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''

    logger = logging.getLogger('Char_RNN')

    if(train_on_gpu):
        net.cuda()

    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    n_chars = len(net.chars)

    counter = 0
    for e in tqdm(range(epochs)):
        # initialize hidden layer
        hidden = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            net.zero_grad()

            # one hot encode
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            if model_type == 'GRU':
                hidden = hidden.data
            else:
                hidden = tuple([each.data for each in hidden])

            output, hidden = net(inputs, hidden)

            loss = criterion(output, targets.view(batch_size*seq_length).long())

            loss.backward()

            # before update clip gradients otherwise it will explode
            nn.utils.clip_grad_norm_(net.parameters(), clip)

            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    if model_type == 'GRU':                        
                        val_h = val_h.data
                    else:
                        val_h = tuple([each.data for each in val_h])
                    
                    
                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                
                    val_losses.append(val_loss.item())
                
                net.train() # reset to train mode after iterationg through validation data
                
                message = "Epoch: {}/{}...".format(e+1, epochs)
                message += ' ' + "Step: {}...".format(counter)
                message += ' ' + "Loss: {:.4f}...".format(loss.item())
                message += ' ' + "Val Loss: {:.4f}".format(np.mean(val_losses))
                logger.info(message)




# main routine
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train character level RNN model')    
    parser.add_argument('hidden_layer_size', help='Hidden layer size', type=int)
    parser.add_argument('layers', help='LSTM layers', type=int)
    parser.add_argument('text_file', help='File path for training', type=str)
    parser.add_argument('model_name', help='Trained model will be saved using this name', type=str)
    parser.add_argument('model_type', help='LSTM or GRU', choices=['LSTM', 'GRU'], type=str)
    parser.add_argument('-l', '--learning_rate', help='Learning rate', type=float)
    parser.add_argument('-b', '--batch_size', help='Input batch size', type=int)
    parser.add_argument('-s', '--seq_length', help='Number of character steps per mini-batch', type=int)
    parser.add_argument('-d', '--drop_prob', help='Dropout probablity', type=float)
    parser.add_argument("-e", "--epochs", help="Number of epochs", type=int)

    args = parser.parse_args()

    if not path.exists(args.text_file):
        print('The path specified does not exist')
        sys.exit()

    batch_size = 128
    seq_length = 100
    n_epochs = 20
    lr = 0.001
    drop_prob = 0.5

    if args.batch_size is not None:
        batch_size = args.batch_size
    
    if args.seq_length is not None:
        seq_length = args.seq_length

    if args.epochs is not None:
        n_epochs = args.epochs
    
    if args.learning_rate is not None:
        lr = args.learning_rate

    if args.model_name is not None:
        model_name = args.model_name
    
    if args.drop_prob is not None:
        drop_prob = args.drop_prob
        
    logger = create_logger()
    
    logger.info("Input args: %r", args)

    text = load_data(args.text_file)   

    gpu_check()

    logger.info('Creating RNN model')
    n_hidden = args.hidden_layer_size
    n_layers = args.layers
    
    tokens = tuple(set(text))

    if args.model_type == 'GRU':
        net = CharGruNet(tokens, hidden_size=n_hidden, num_layers=n_layers, drop_prob=drop_prob)
    else:
        net = CharRNN(tokens, hidden_size=n_hidden, num_layers=n_layers, drop_prob=drop_prob)

    logger.info(net)

    logger.info('Encoding input text')

    encoded = net.encode_text(text)

    logger.info('Training started')

    
    # train the model
    train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=lr, print_every=10)
    
    logger.info('Trainig complete. Saving model...')
    # change the name, for saving multiple files    

    checkpoint = {'n_hidden': net.n_hidden,
                'n_layers': net.n_layers,
                'state_dict': net.state_dict(),
                'tokens': net.chars}

    with open(model_name, 'wb') as f:
        torch.save(checkpoint, f)


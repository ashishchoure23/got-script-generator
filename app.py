from flask import Flask, request, jsonify, render_template
import numpy as np
import torch
import torch.nn.functional as F
from CharGru import CharGruNet
from CharRNN_Train import one_hot_encode


app = Flask(__name__)

train_on_gpu = torch.cuda.is_available()

def load_model(model_path):
    # Here we have loaded in a model that trained over 20 epochs `rnn_20_epoch.net`
    #with open(model_path, 'rb') as f:
    #    checkpoint = torch.load(f)    
    
    if train_on_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    checkpoint = torch.load(model_path, map_location=device)

    loaded = CharGruNet(checkpoint['tokens'], hidden_size=checkpoint['n_hidden'], num_layers=checkpoint['n_layers'])
    loaded.load_state_dict(checkpoint['state_dict'])

    return loaded


def predict(net, char, h=None, top_k=None):
        ''' Given a character, predict the next character.
            Returns the predicted character and the hidden state.
        '''
        
        # tensor inputs
        x = np.array([[net.char2int[char]]])
        x = one_hot_encode(x, len(net.chars))
        inputs = torch.from_numpy(x)
        
        if(train_on_gpu):
            inputs = inputs.cuda()
        
        # detach hidden state from history
        # h = tuple([each.data for each in h])
        h = h.data
        # get the output of the model
        out, h = net(inputs, h)

        # get the character probabilities
        p = F.softmax(out, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
        
        # get top characters
        if top_k is None:
            top_ch = np.arange(len(net.chars))
        else:
            p, top_ch = p.topk(top_k)
            top_ch = top_ch.numpy().squeeze()
        
        # select the likely next character with some element of randomness
        p = p.numpy().squeeze()
        char = np.random.choice(top_ch, p=p/p.sum())
        #char = top_ch[0]
        
        # return the encoded value of the predicted char and the hidden state
        return net.int2char[char], h


def sample(net, size, prime='The', top_k=None):
        
    if(train_on_gpu):
        net.cuda()
    else:
        net.cpu()
    
    net.eval() # eval mode
    
    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)
    
    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

net = load_model('model/gotGru.net')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generatescript', methods = ['POST'])
def generatescript():
    
    starttext = request.form['starttext']

    script = sample(net, 1500, prime=starttext, top_k=2)

    lastdot = script.rfind('.')
    script = script[:lastdot + 1]

    script = script.split('\n')

    return render_template('result.html', script=script)


if __name__ == "__main__":
    app.run(debug=False)

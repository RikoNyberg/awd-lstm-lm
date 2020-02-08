from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from argparse import Namespace

from input_parser import parse_input, get_mongo_url
from main import LanguageModelTrainer

args_dict = parse_input()

# Get mongo_url to save experiment parameters and results 
mongo_url = get_mongo_url(args_dict)
args_dict.pop('mongo_url', None)

# Activate Sacred experiment 
ex = Experiment(args_dict.get('model'))
ex.observers.append(MongoObserver.create(url=mongo_url))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def default_config():
    cuda = None         # action='store_false' - use CUDA 
    data = None         # str - location of the data corpus
    output_dir = None   # str - location of tensorboard data (and serialized data and model)
    model = None        # str - type of recurrent net (LSTM, QRNN, GRU)
    emsize = None       # int - size of word embeddings
    nhid = None         # int - number of hidden units per layer
    nlayers = None      # int - number of layers
    lr = None           # float - initial learning rate
    clip = None         # float - gradient clipping
    epochs = None       # int - upper epoch limit
    batch_size = None   # int - batch size - metavar='N',
    bptt = None         # int - sequence length
    dropout = None      # float - dropout applied to layers (0 = no dropout)
    dropouth = None     # float - dropout for rnn layers (0 = no dropout)
    dropouti = None     # float - dropout for input embedding layers (0 = no dropout)
    dropoute = None     # float - dropout to remove words from embedding layer (0 = no dropout)
    wdrop = None        # float - amount of weight dropout to apply to the RNN hidden to hidden matrix
    seed = None         # int - random seed
    nonmono = None      # int - random seed
    log_interval = None # int - report interval - metavar='N',
    save = None         # str - path to save the final model
    alpha = None        # float - alpha L2 regularization on RNN activation (alpha = 0 means no regularization)
    beta = None         # float - beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)
    wdecay = None       # float - weight decay applied to all weights
    resume = None       # str - path of model to resume
    optimizer = None    # str - optimizer to use (sgd, adam)
    when = None         # int - When (which epochs) to divide the learning rate by 10 - accepts multiple
    tied = None

@ex.main
def run(
    cuda, data, output_dir, model, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt, 
    dropout, dropouth, dropouti, dropoute, wdrop, seed, nonmono, log_interval, 
    save, alpha, beta, wdecay, resume, optimizer, when, tied
    ):
    
    args_dict = {
        'cuda': cuda,
        'data': data,
        'model': model,
        'emsize': emsize,
        'nhid': nhid,
        'nlayers': nlayers,
        'lr': lr,
        'clip': clip,
        'epochs': epochs,
        'batch_size': batch_size,
        'bptt': bptt,
        'dropout': dropout,
        'dropouth': dropouth,
        'dropouti': dropouti,
        'dropoute': dropoute,
        'wdrop': wdrop,
        'seed': seed,
        'nonmono': nonmono,
        'log_interval': log_interval,
        'save': save,
        'alpha': alpha,
        'beta': beta,
        'wdecay': wdecay,
        'resume': resume,
        'optimizer': optimizer,
        'when': when,
        'tied': tied,
    }
    args = Namespace(**args_dict)
    args.tied = True

    lm_model_trainer = LanguageModelTrainer(args, ex)
    lm_model_trainer.load_training_data()
    lm_model_trainer.build_model()
    lm_model_trainer.train()
    lm_model_trainer.run_test()

if __name__ == '__main__':
    # With this config_update we set experiment parameters (aka. args)
    r = ex.run(config_updates=args_dict)
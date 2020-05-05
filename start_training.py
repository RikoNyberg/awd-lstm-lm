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
    cuda = None
    data = None
    model = None
    emsize = None
    nhid = None
    nlayers = None
    lr = None
    clip = None
    epochs = None
    batch_size = None
    bptt = None
    dropout = None
    dropouth = None
    dropouti = None
    dropoute = None
    wdrop = None
    seed = None
    nonmono = None
    log_interval = None
    save = None
    alpha = None
    beta = None
    wdecay = None
    resume = None
    optimizer = None
    lr_decay = None
    lr_decay_start = None
    init_scale = None
    tied = None

@ex.main
def run(
    cuda, data, model, emsize, nhid, nlayers, lr, clip, epochs, batch_size, bptt,
    dropout, dropouth, dropouti, dropoute, wdrop, seed, nonmono, log_interval,
    save, alpha, beta, wdecay, resume, optimizer, lr_decay, lr_decay_start, init_scale, tied
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
        'lr_decay': lr_decay,
        'lr_decay_start': lr_decay_start,
        'init_scale': init_scale,
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
import argparse
import time

def parse_input():
    randomhash = ''.join(str(time.time()).split('.'))

    parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
    parser.add_argument('--data', type=str, default='data/penn/', help='location of the data corpus')
    parser.add_argument('--mongo_url', type=str, default='', help='MongoDB url to save the experiment parameters and results')
    parser.add_argument('--model', type=str, default='LSTM', help='type of recurrent net (LSTM, QRNN, GRU)')
    parser.add_argument('--emsize', type=int, default=512, help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512, help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1, help='number of layers')
    parser.add_argument('--lr', type=float, default=1, help='initial learning rate')
    parser.add_argument('--clip', type=float, default=5, help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=39, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size')
    parser.add_argument('--bptt', type=int, default=35, help='sequence length')
    parser.add_argument('--dropout', type=float, default=0, help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--dropouth', type=float, default=0.5, help='dropout for rnn layers (0 = no dropout)')
    parser.add_argument('--dropouti', type=float, default=0.5, help='dropout for input embedding layers (0 = no dropout)')
    parser.add_argument('--dropoute', type=float, default=0, help='dropout to remove words from embedding layer (0 = no dropout)')
    parser.add_argument('--wdrop', type=float, default=0, help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
    parser.add_argument('--seed', type=int, default=313, help='random seed')
    parser.add_argument('--nonmono', type=int, default=6, help='Which epoch to switch to ASGD') # TODO: Maybe just remove this?
    parser.add_argument('--cuda', action='store_true', help='use CUDA (by default not using cuda)')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N', help='report interval')
    parser.add_argument('--save', type=str,  default=randomhash+'.pt', help='path to save the final model')
    parser.add_argument('--alpha', type=float, default=2, help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=1, help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--wdecay', type=float, default=1.2e-6, help='weight decay applied to all weights')
    parser.add_argument('--resume', type=str,  default='', help='path of model to resume')
    parser.add_argument('--optimizer', type=str,  default='sgd', help='optimizer to use (sgd, adam)')
    parser.add_argument('--lr_decay', type=float, default=0.8, help='Determines how fast the learning rate decays after epoch lr_decay_start (1 = no decay)')
    parser.add_argument('--lr_decay_start', type=int, default=8, help='After which epoch to start decaying the learning rate by lr_decay')
    parser.add_argument('--init_scale', type=float, default=0.1, help='The weights of the model will be randomly initialized, with a uniform distribution and values between -init_scale and init_scale')
    args = parser.parse_args()

    args_dict = vars(args)
    return args_dict

def get_mongo_url(args_dict):
    ''' MongoDB URL for saving the experiment parameters and results '''
    if args_dict.get('mongo_url'):
        return args_dict.get('mongo_url')
    else:
        try:
            from credentials import MONGO_URL
            return MONGO_URL
        except ModuleNotFoundError:
            raise ModuleNotFoundError('Add MONGO_URL to credential.py file OR to input args as --mongo_url <URL>')
###############################################################################
#
# This file scores sentences.
# The code was taken from:
#   https://github.com/salesforce/awd-lstm-lm/commit/8376451390e75d3d8f9f574bef6f5a2c91bc76c8
#   https://github.com/salesforce/awd-lstm-lm/issues/96#issuecomment-498260189
#
###############################################################################

import argparse
import io
import math
import sys
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sampling import top_k_top_p_filtering
import data

parser = argparse.ArgumentParser(description='Sentence scorer')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument("--input", required=False, type=str)
parser.add_argument('--ppl', action='store_true')
parser.add_argument('--also-ppl', type=str, required=False)
args = parser.parse_args()

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f, map_location='cpu')

model = model[0]
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()


corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)


def score(sentence):
    with torch.no_grad():
        tokens =["<eos>"] + sentence.split(' ')  # <eos> here serves as <sos>
        idxs = [corpus.dictionary.word2idx[t] for t in tokens if t in corpus.dictionary.word2idx]
        num_tokens = len(idxs) - 1
        idxs = torch.LongTensor(idxs).unsqueeze(1)
        hidden = model.init_hidden(1)
        output, hidden = model(idxs, hidden)
        logits = model.decoder(output)
        logprobs = F.log_softmax(logits, dim=1)
        total_logprob = sum([logprobs[i][idxs[i+1]] for i in range(num_tokens)])
        total_logprob = total_logprob if isinstance(total_logprob, (int, float)) else total_logprob.cpu().item()
        return total_logprob, num_tokens


input_lines = (io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
               if args.input is None else open(args.input, encoding='utf-8'))

accumulated_logprob = 0.
total_num_tokens = 0

for line in input_lines:
    sentence = line.strip()
    logprob, num_tokens = score(sentence)
    accumulated_logprob += logprob
    total_num_tokens += num_tokens
    if not args.ppl:
        print(logprob)

log2_prob = accumulated_logprob / math.log(2)
ppl = math.pow(2., - log2_prob/total_num_tokens)

if args.ppl:
    print(ppl)
elif args.also_ppl:
    with open(args.also_ppl, 'w', encoding='utf-8') as f:
        print(ppl, file=f)
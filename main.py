import time
import math
import numpy as np
import torch
import torch.nn as nn

from utils import batchify, get_batch, repackage_hidden

class LanguageModelTrainer():
    def __init__(self, args, ex):
        self.ex = ex
        self.args = args
        # Set the random seed manually for reproducibility.
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print("WARNING: You have a CUDA device, so you should probably run with --cuda")
            else:
                torch.cuda.manual_seed(self.args.seed)
                print("INFO: Using CUDA device")
        elif self.args.cuda:
            print("WARNING: No CUDA device available, so you should not run with --cuda")
        else:
            print("INFO: No CUDA device available")

    ###############################################################################
    # Load data
    ###############################################################################
    def load_training_data(self):
        import data
        import os
        import hashlib
        fn = 'self.corpus.{}.data'.format(hashlib.md5(self.args.data.encode()).hexdigest())
        if os.path.exists(fn):
            print('Loading cached dataset...')
            self.corpus = torch.load(fn)
        else:
            print('Producing dataset...')
            self.corpus = data.Corpus(self.args.data)
            torch.save(self.corpus, fn)

        self.eval_batch_size = 10
        self.test_batch_size = 1
        self.train_data = batchify(self.corpus.train, self.args.batch_size, self.args)
        self.val_data = batchify(self.corpus.valid, self.eval_batch_size, self.args)
        self.test_data = batchify(self.corpus.test, self.test_batch_size, self.args)

    def model_save(self, fn):
        with open(fn, 'wb') as f:
            torch.save([self.model, self.criterion, self.optimizer], f)

    def model_load(self, fn):
        with open(fn, 'rb') as f:
            self.model, self.criterion, self.optimizer = torch.load(f)

    ###############################################################################
    # Build the model
    ###############################################################################
    def build_model(self):
        import model
        from splitcross import SplitCrossEntropyLoss
        self.criterion = None

        ntokens = len(self.corpus.dictionary)
        self.model = model.RNNModel(self.args.model, ntokens, self.args.emsize, self.args.nhid, self.args.nlayers, self.args.dropout, self.args.dropouth, self.args.dropouti, self.args.dropoute, self.args.wdrop, self.args.tied)
        ###
        if self.args.resume:
            print('Resuming model ...')
            self.model_load(self.args.resume)
            self.optimizer.param_groups[0]['lr'] = self.args.lr
            self.model.dropouti, self.model.dropouth, self.model.dropout = self.args.dropouti, self.args.dropouth, self.args.dropout
            if self.args.wdrop:
                from weight_drop import WeightDrop
                for rnn in self.model.rnns:
                    if type(rnn) == WeightDrop: rnn.dropout = self.args.wdrop
                    elif rnn.zoneout > 0: rnn.zoneout = self.args.wdrop
        ###
        if not self.criterion:
            splits = []
            if ntokens > 500000:
                # One Billion
                # This produces fairly even matrix mults for the buckets:
                # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
                splits = [4200, 35000, 180000]
            elif ntokens > 75000:
                # WikiText-103
                splits = [2800, 20000, 76000]
            print('Using', splits)
            self.criterion = SplitCrossEntropyLoss(self.args.emsize, splits=splits, verbose=False)
        ###
        if self.args.cuda:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
        ###
        self.params = list(self.model.parameters()) + list(self.criterion.parameters())
        total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in self.params if x.size())
        print('self.args:', self.args)
        print('Model total parameters:', total_params)

    ###############################################################################
    # Training code
    ###############################################################################
    def train(self):
        def train():
            # Turn on training mode which enables dropout.
            if self.args.model == 'QRNN': self.model.reset()
            total_loss = 0
            start_time = time.time()
            hidden = self.model.init_hidden(self.args.batch_size)
            batch, i = 0, 0
            while i < self.train_data.size(0) - 1 - 1:
                bptt = self.args.bptt if np.random.random() < 0.95 else self.args.bptt / 2.
                # Prevent excessively small or negative sequence lengths
                seq_len = max(5, int(np.random.normal(bptt, 5)))
                # There's a very small chance that it could select a very long sequence length resulting in OOM
                # seq_len = min(seq_len, self.args.bptt + 10)

                lr2 = self.optimizer.param_groups[0]['lr']
                self.optimizer.param_groups[0]['lr'] = lr2 * seq_len / self.args.bptt
                self.model.train()
                data, targets = get_batch(self.train_data, i, self.args, seq_len=seq_len)

                # Starting each batch, we detach the hidden state from how it was previously produced.
                # If we didn't, the model would try backpropagating all the way to start of the dataset.
                hidden = repackage_hidden(hidden)
                self.optimizer.zero_grad()

                output, hidden, rnn_hs, dropped_rnn_hs = self.model(data, hidden, return_h=True)
                raw_loss = self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets)

                loss = raw_loss
                # Activiation Regularization
                if self.args.alpha: loss = loss + sum(self.args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
                # Temporal Activation Regularization (slowness)
                if self.args.beta: loss = loss + sum(self.args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
                loss.backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                if self.args.clip: torch.nn.utils.clip_grad_norm_(self.params, self.args.clip)
                self.optimizer.step()

                total_loss += raw_loss.data
                self.optimizer.param_groups[0]['lr'] = lr2
                if batch % self.args.log_interval == 0 and batch > 0:
                    cur_loss = total_loss.item() / self.args.log_interval
                    elapsed = time.time() - start_time
                    print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                            'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                        epoch, batch, len(self.train_data) // self.args.bptt, self.optimizer.param_groups[0]['lr'],
                        elapsed * 1000 / self.args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
                    total_loss = 0
                    start_time = time.time()
                ###
                batch += 1
                i += seq_len

        # Loop over epochs.
        best_val_loss = []
        stored_loss = 100000000

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            self.optimizer = None
            # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's weight (i.e. Adaptive Softmax)
            if self.args.optimizer == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)
            if self.args.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.params, lr=self.args.lr, weight_decay=self.args.wdecay)
            for epoch in range(1, self.args.epochs+1):
                epoch_start_time = time.time()
                train()
                if 't0' in self.optimizer.param_groups[0]:
                    tmp = {}
                    for prm in self.model.parameters():
                        tmp[prm] = prm.data.clone()
                        prm.data = self.optimizer.state[prm]['ax'].clone()

                    val_loss2 = self.evaluate(self.val_data)
                    bpc2 = val_loss2 / math.log(2)
                    bpc2 = np.float64(0).item()
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                            epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), bpc2))
                    print('-' * 89)
                    self.ex.log_scalar('ppl/dev', math.exp(val_loss2), epoch)
                    self.ex.log_scalar('Loss/dev', val_loss2, epoch)
                    self.ex.log_scalar('BPC/dev', bpc2, epoch)

                    if val_loss2 < stored_loss:
                        self.model_save(self.args.save)
                        print('Saving Averaged!')
                        stored_loss = val_loss2

                    for prm in self.model.parameters():
                        prm.data = tmp[prm].clone()

                else:
                    val_loss = self.evaluate(self.val_data, self.eval_batch_size)
                    bpc = val_loss / math.log(2)
                    print('-' * 89)
                    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), bpc))
                    print('-' * 89)
                    self.ex.log_scalar('ppl/dev', math.exp(val_loss), epoch)
                    self.ex.log_scalar('Loss/dev', val_loss, epoch)
                    self.ex.log_scalar('BPC/dev', bpc, epoch)

                    if val_loss < stored_loss:
                        self.model_save(self.args.save)
                        print('Saving model (new best validation)')
                        stored_loss = val_loss

                    if self.args.optimizer == 'sgd' and 't0' not in self.optimizer.param_groups[0] and (len(best_val_loss)>self.args.nonmono and val_loss > min(best_val_loss[:-self.args.nonmono])):
                        print('Switching to ASGD')
                        self.optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.args.lr, t0=0, lambd=0., weight_decay=self.args.wdecay)

                    if epoch in self.args.when_lr_div:
                        print('Saving model before learning rate decreased')
                        self.model_save('{}.e{}'.format(self.args.save, epoch))
                        print('Dividing learning rate by {lr_div}'.format(lr_div=self.args.lr_div))
                        self.optimizer.param_groups[0]['lr'] /= self.args.lr_div

                    best_val_loss.append(val_loss)

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    def evaluate(self, data_source, batch_size=10):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()
        if self.args.model == 'QRNN': self.model.reset()
        total_loss = 0
        hidden = self.model.init_hidden(batch_size)
        for i in range(0, data_source.size(0) - 1, self.args.bptt):
            data, targets = get_batch(data_source, i, self.args, evaluation=True)
            output, hidden = self.model(data, hidden)
            total_loss += len(data) * self.criterion(self.model.decoder.weight, self.model.decoder.bias, output, targets).data
            hidden = repackage_hidden(hidden)
        return total_loss.item() / len(data_source)

    def run_test(self):
        # Load the best saved model.
        self.model_load(self.args.save)

        # Run on test data.
        test_loss = self.evaluate(self.test_data, self.test_batch_size)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('=' * 89)


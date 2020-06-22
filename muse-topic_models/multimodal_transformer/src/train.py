import torch
from torch import nn
import sys
from src import models
from src import ctc
from src.utils import *
from src.eval_metrics import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle

import pandas as pd


####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################

def get_CTC_module(hyp_params):
    a2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_a, out_seq_len=hyp_params.l_len)
    v2l_module = getattr(ctc, 'CTCModule')(in_dim=hyp_params.orig_d_v, out_seq_len=hyp_params.l_len)
    return a2l_module, v2l_module


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params['model']+'Model')(hyp_params)

    if hyp_params['use_cuda']:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params['optim'])(model.parameters(), lr=hyp_params['lr'])
    criterion = getattr(nn, hyp_params['criterion'])()# weight=hyp_params['weights']

    if hyp_params['aligned'] or hyp_params['model']=='MULT':
        ctc_criterion = None
        ctc_a2l_module, ctc_v2l_module = None, None
        ctc_a2l_optimizer, ctc_v2l_optimizer = None, None
    else:
        from warpctc_pytorch import CTCLoss
        ctc_criterion = CTCLoss()
        ctc_a2l_module, ctc_v2l_module = get_CTC_module(hyp_params)
        if hyp_params['use_cuda']:
            ctc_a2l_module, ctc_v2l_module = ctc_a2l_module.cuda(), ctc_v2l_module.cuda()
        ctc_a2l_optimizer = getattr(optim, hyp_params['optim'])(ctc_a2l_module.parameters(), lr=hyp_params['lr'])
        ctc_v2l_optimizer = getattr(optim, hyp_params['optim'])(ctc_v2l_module.parameters(), lr=hyp_params['lr'])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params['when'], factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ctc_a2l_module': ctc_a2l_module,
                'ctc_v2l_module': ctc_v2l_module,
                'ctc_a2l_optimizer': ctc_a2l_optimizer,
                'ctc_v2l_optimizer': ctc_v2l_optimizer,
                'ctc_criterion': ctc_criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    
    ctc_a2l_module = settings['ctc_a2l_module']
    ctc_v2l_module = settings['ctc_v2l_module']
    ctc_a2l_optimizer = settings['ctc_a2l_optimizer']
    ctc_v2l_optimizer = settings['ctc_v2l_optimizer']
    ctc_criterion = settings['ctc_criterion']
    
    scheduler = settings['scheduler']
    

    def train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params['n_train'] // hyp_params['batch_size']
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y, batch_META) in enumerate(train_loader):
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if ctc_criterion is not None:
                ctc_a2l_module.zero_grad()
                ctc_v2l_module.zero_grad()
                
            if hyp_params['use_cuda']:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                    if hyp_params['cont_emotions']:
                        eval_attr = eval_attr.float()
                    else:
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            batch_chunk = hyp_params['batch_chunk']
            
            ######## CTC STARTS ######## Do not worry about this if not working on CTC
            if ctc_criterion is not None:
                ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module

                audio, a2l_position = ctc_a2l_net(audio) # audio now is the aligned to text
                vision, v2l_position = ctc_v2l_net(vision)
                
                ## Compute the ctc loss
                l_len, a_len, v_len = hyp_params['l_len'], hyp_params['a_len'], hyp_params['v_len']
                # Output Labels
                l_position = torch.tensor([i+1 for i in range(l_len)]*batch_size).int().cpu()
                # Specifying each output length
                l_length = torch.tensor([l_len]*batch_size).int().cpu()
                # Specifying each input length
                a_length = torch.tensor([a_len]*batch_size).int().cpu()
                v_length = torch.tensor([v_len]*batch_size).int().cpu()
                
                ctc_a2l_loss = ctc_criterion(a2l_position.transpose(0,1).cpu(), l_position, a_length, l_length)
                ctc_v2l_loss = ctc_criterion(v2l_position.transpose(0,1).cpu(), l_position, v_length, l_length)
                ctc_loss = ctc_a2l_loss + ctc_v2l_loss
                ctc_loss = ctc_loss.cuda() if hyp_params['use_cuda'] else ctc_loss
            else:
                ctc_loss = 0
            ######## CTC ENDS ########
                
            combined_loss = 0
            net = nn.DataParallel(model) if batch_size > 10 else model
            if batch_chunk > 1:
                raw_loss = combined_loss = 0
                text_chunks = text.chunk(batch_chunk, dim=0)
                audio_chunks = audio.chunk(batch_chunk, dim=0)
                vision_chunks = vision.chunk(batch_chunk, dim=0)
                eval_attr_chunks = eval_attr.chunk(batch_chunk, dim=0)
                
                for i in range(batch_chunk):
                    text_i, audio_i, vision_i = text_chunks[i], audio_chunks[i], vision_chunks[i]
                    eval_attr_i = eval_attr_chunks[i]
                    preds_i, hiddens_i = net(text_i, audio_i, vision_i)
                    
                    if hyp_params['regression_mode']:
                        preds = preds.view(-1)
                        eval_attr = eval_attr.view(-1)
                    else:
                        preds = preds.view(-1,hyp_params['output_dim'])
                        eval_attr = eval_attr.view(-1)
                    raw_loss_i = criterion(preds_i, eval_attr_i) / batch_chunk
                    raw_loss += raw_loss_i
                    raw_loss_i.backward()
                ctc_loss.backward()
                combined_loss = raw_loss + ctc_loss
            else:
                preds, hiddens = net(text, audio, vision)
                # if hyp_params['dataset'] in ['iemocap']:
                #     preds = preds.view(-1, 2)
                #     eval_attr = eval_attr.view(-1)
                if hyp_params['regression_mode']:
                    preds = preds.view(-1)
                    eval_attr = eval_attr.view(-1)
                else:
                    preds = preds.view(-1,hyp_params['output_dim'])
                    eval_attr = eval_attr.view(-1)

                raw_loss = criterion(preds, eval_attr)
                combined_loss = raw_loss + ctc_loss
                combined_loss.backward()
            
            if ctc_criterion is not None:
                torch.nn.utils.clip_grad_norm_(ctc_a2l_module.parameters(), hyp_params['clip'])
                torch.nn.utils.clip_grad_norm_(ctc_v2l_module.parameters(), hyp_params['clip'])
                ctc_a2l_optimizer.step()
                ctc_v2l_optimizer.step()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params['clip'])
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params['log_interval'] == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params['log_interval'], avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params['n_train']

    def evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, partition):
        model.eval()
        total_loss = 0.0
        if partition == "test" and hyp_params['n_test'] > 0: #or test_loader not None
            loader = test_loader
            data_len = hyp_params['n_test']
        elif partition == "valid":
            loader = valid_loader
            data_len = hyp_params['n_valid']
        elif partition == "train":
            loader = train_loader
            data_len = hyp_params['n_train']
    
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, (batch_X, batch_Y, batch_META) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params['use_cuda']:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params['cont_emotions']:
                            eval_attr = eval_attr.float()
                        else:
                            eval_attr = eval_attr.long()
                        
                batch_size = text.size(0)
                
                if (ctc_a2l_module is not None) and (ctc_v2l_module is not None):
                    ctc_a2l_net = nn.DataParallel(ctc_a2l_module) if batch_size > 10 else ctc_a2l_module
                    ctc_v2l_net = nn.DataParallel(ctc_v2l_module) if batch_size > 10 else ctc_v2l_module
                    audio, _ = ctc_a2l_net(audio)     # audio aligned to text
                    vision, _ = ctc_v2l_net(vision)   # vision aligned to text
                
                net = nn.DataParallel(model) if batch_size > 10 else model
                preds, _ = net(text, audio, vision)

                if hyp_params['regression_mode']:
                    preds = preds.view(-1)
                    eval_attr = eval_attr.view(-1)
                else:
                    preds = preds.view(-1,hyp_params['output_dim'])
                    eval_attr = eval_attr.view(-1)

                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds)
                truths.append(eval_attr)
                
        avg_loss = total_loss / data_len

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    # different metric min/max TODO: clean up
    if hyp_params['class_name'] in ['arousal','valence']:
        # store best model based on metric not loss.
        best_valid = -1 * 1e8 # 1e8
        smaller = False
    elif hyp_params['class_name'] == 'topic':
        best_valid = -1 * 1e8
        smaller = False
    else:
        print("Class name unkown.")
        exit()

    eval_export_dir = hyp_params['output_dir']
    stats = []
    for epoch in range(1, hyp_params['num_epochs']+1):
        start = time.time()
        train(model, optimizer, criterion, ctc_a2l_module, ctc_v2l_module, ctc_a2l_optimizer, ctc_v2l_optimizer, ctc_criterion)
        val_loss, val_res, val_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "valid")
        if hyp_params['n_test'] > 0:
            test_loss, test_res, test_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "test")
        train_loss, train_res, train_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "train")
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        if hyp_params['dataset'] == "muse":
            if hyp_params['regression_mode']: 
                train_major_measure = eval_reg_muse(train_res, train_truths, 'train', eval_export_dir)
                devel_major_measure = eval_reg_muse(val_res, val_truths, 'devel', eval_export_dir)
                if hyp_params['n_test'] > 0:
                    test_major_measure = eval_reg_muse(test_res, test_truths, 'test', eval_export_dir)  
            else:
                train_major_measure = eval_class_muse(train_res, train_truths, 'train', eval_export_dir, hyp_params['y_names'])
                devel_major_measure = eval_class_muse(val_res, val_truths, 'devel', eval_export_dir, hyp_params['y_names'])
                if hyp_params['n_test'] > 0:
                    test_major_measure = eval_class_muse(test_res, test_truths, 'test', eval_export_dir, hyp_params['y_names'])
        
        if hyp_params['n_test'] > 0:        
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, train_loss, val_loss, test_loss))
            stats.append([val_loss, test_loss, train_loss, devel_major_measure, test_major_measure, train_major_measure])
        else:
            print('Epoch {:2d} | Time {:5.4f} sec | Train Loss {:5.4f} | Valid Loss {:5.4f} '.format(epoch, duration, train_loss, val_loss))
            stats.append([val_loss, train_loss, devel_major_measure, train_major_measure])
        print("-"*50)
        
        
        def compare(smaller,new,old):
            return new<old if smaller else new>old
        
        if compare(smaller, devel_major_measure, best_valid):
            print(f"Saved model at pre_trained_models/{hyp_params['source_name']}/{hyp_params['experiment_name']}.pt!")
            save_model(path = hyp_params['pretrained_model_dir'], model=model, name=hyp_params['experiment_name'])
            best_valid = devel_major_measure
    
    if hyp_params['n_test'] > 0:                
        stats_df = pd.DataFrame(stats, columns = ["val loss", "test loss", 'train_loss'
                                             ,'devel_major_measure', 'test_major_measure', 'train_major_measure'])
    else:
        stats_df = pd.DataFrame(stats, columns = ["val loss", 'train_loss','devel_major_measure', 'train_major_measure'])        
    stats_df.to_csv(os.path.join(eval_export_dir, "stats.csv"))
                  
    model = load_model(path=hyp_params['pretrained_model_dir'], name=hyp_params['experiment_name'])
    _, train_results, train_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "train")
    _, val_results, val_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "valid")
    if hyp_params['n_test'] > 0: 
        _, test_results, test_truths = evaluate(model, ctc_a2l_module, ctc_v2l_module, criterion, "test")

    if hyp_params['dataset'] == "muse":
        if hyp_params['regression_mode']:
            eval_reg_muse(train_results, train_truths, 'train', eval_export_dir, True)
            eval_reg_muse(val_results, val_truths, 'devel', eval_export_dir,True)
            if hyp_params['n_test'] > 0: 
                eval_reg_muse(test_results, test_truths, 'test', eval_export_dir,True)
        else:
            eval_class_muse(train_results, train_truths, 'train', eval_export_dir, hyp_params['y_names'], True)
            eval_class_muse(val_results, val_truths, 'devel', eval_export_dir, hyp_params['y_names'], True)
            if hyp_params['n_test'] > 0: 
                eval_class_muse(test_results, test_truths, 'test', eval_export_dir, hyp_params['y_names'], True)

    sys.stdout.flush()
    input('[Press Any Key to start another run]')

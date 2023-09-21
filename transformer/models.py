import torch
from torch import nn
from torch.optim import Adam
import pytorch_lightning as pl

from .informer.decoder import Decoder as InformerDecoder
from .informer.encoder import Encoder as InformerEncoder

from .seq2seq_lstm.encoder import Encoder as Seq2SeqEncoder
# from .seq2seq_lstm.decoder import Decoder as Seq2SeqDecoder
from .seq2seq_lstm.attention import Attention
from .seq2seq_lstm.attention_decoder import AttentionDecoder

from typing import List, Dict, Tuple, Union, Any, Optional
import pandas as pd
import numpy as np

class Informer(nn.Module):
    def __init__(
        self,
        d_feature,
        d_mark,
        d_k=64,
        d_v=64,
        d_model=512,
        d_ff=512,
        n_heads=8,
        e_layer=3,
        d_layer=2,
        e_stack=3,
        dropout=0.1,
        c=5,
    ):
        super(Informer, self).__init__()

        self.encoder = InformerEncoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=e_layer,
            n_stack=e_stack,
            d_feature=d_feature,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
        )
        self.decoder = InformerDecoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=d_layer,
            d_feature=d_feature,
            d_mark=d_mark,
            dropout=dropout,
            c=c,
        )

        self.projection = nn.Linear(d_model, d_feature, bias=True)

    def forward(self, enc_x, enc_mark, dec_in, dec_mark):
        enc_outputs = self.encoder(enc_x, enc_mark)
        dec_outputs = self.decoder(dec_in, dec_mark, enc_outputs)
        dec_outputs = self.projection(dec_outputs)

        return dec_outputs

class InformerPL(pl.LightningModule):
    def __init__(
        self,
        d_feature,
        d_mark,
        pred_len,
        label_len,
        target: str,
        column_idxs: Dict[str, int],
        d_k=64,
        d_v=64,
        d_model=512,
        d_ff=512,
        n_heads=8,
        e_layer=3,
        d_layer=2,
        e_stack=3,
        dropout=0.1,
        c=5,
        lr=1e-3,
        weight_decay=0.001
    ):
        super(InformerPL, self).__init__()

        self.save_hyperparameters()

        shared_params = {
            "d_k": d_k,
            "d_v": d_v,
            "d_model": d_model,
            "d_ff": d_ff,
            "n_heads": n_heads,
            "d_feature": d_feature,
            "d_mark": d_mark,
            "dropout": dropout,
            "c": c,
        }

        self.encoder = InformerEncoder(n_layer=e_layer, n_stack=e_stack, **shared_params).float()
        self.decoder = InformerDecoder(n_layer=d_layer, **shared_params).float()

        self.projection = nn.Linear(d_model, d_feature, bias=True)
        self.pred_len = pred_len
        self.label_len = label_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.target_idx = column_idxs[target]

    def forward(self, enc_x, enc_mark, dec_in, dec_mark):
        enc_preds = self.encoder(enc_x, enc_mark)
        dec_preds = self.decoder(dec_in, dec_mark, enc_preds)
        dec_preds = self.projection(dec_preds)
        return dec_preds

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def process_batch(self, batch):
        x, y, x_stamp, y_stamp = batch[:4]
        x = x.to(self.device).float()
        y = y.to(self.device).float()
        x_stamp = x_stamp.to(self.device).float()
        y_stamp = y_stamp.to(self.device).float()

        dec_inp = torch.zeros(y.shape[0], self.pred_len, y.shape[-1]).to(self.device).float()
        dec_inp = torch.cat([y[:, -self.label_len:, :], dec_inp], dim=1)

        preds = self(x, x_stamp, dec_inp, y_stamp)
        # preds = preds[:, -self.pred_len:, self.target_idx]
        # trues = y[:, -self.pred_len:, self.target_idx]
        preds = preds[:, -self.pred_len:, :]
        trues = y[:, -self.pred_len:, :]
        return preds, trues
    
    def compute_loss(self, batch):
        preds, trues = self.process_batch(batch)
        loss = nn.MSELoss()(preds, trues)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        preds, trues = self.process_batch(batch)
        return preds, trues
    
class Seq2Seq(pl.LightningModule):
    def __init__(
        self, 
        seq_len, 
        n_features,
        embedding_dim, 
        output_length,
        target: str,
        column_idxs: Dict[str, int],
        lr=1e-3,
    ):
        super(Seq2Seq, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.encoder = Seq2SeqEncoder(seq_len, n_features, embedding_dim)
        self.attention = Attention(embedding_dim, embedding_dim)
        self.output_length = output_length
        # 临时将第一个 embedding_dim 看做 decoder 的特征维度
        # 第二个 embedding_dim 看做 encoder 的特征维度
        # 后续重命名为 encoder_dim 和 decoder_dim
        self.decoder = AttentionDecoder(seq_len, self.attention, embedding_dim, n_features, embedding_dim)
        self.target_idx = column_idxs[target]

    def forward(self, x, prev_y):
        encoder_output, hidden, cell = self.encoder(x)
         
        #Prepare place holder for decoder output
        targets_ta = []
        #prev_output become the next input to the LSTM cell
        prev_output = prev_y
        
        #itearate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            prev_x, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell, encoder_output)
            hidden, cell = prev_hidden, prev_cell
            prev_output = prev_x
            
            targets_ta.append(prev_x.reshape(1))
        
        targets = torch.stack(targets_ta)

        return targets
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    # def compute_loss(self, batch):
    #     batch_x, batch_prev_y, batch_y, _, _ = batch
    #     batch_x = batch_x.to(self.device).float()
    #     batch_prev_y = batch_prev_y.to(self.device).float()
    #     batch_y = batch_y.to(self.device).float()

    #     losses = torch.Tensor([]).to(self.device)
    #     for i in range(0, batch_x.shape[0]):
    #         seq_inp = batch_x[i]
    #         seq_true = batch_y[i]
    #         seq_pred = self(seq_inp, batch_prev_y[i])
    #         loss = nn.MSELoss()(seq_pred, seq_true)
    #         losses = torch.cat((losses, loss.unsqueeze(0)))

    #     mean_loss = torch.mean(losses)
    #     return mean_loss

    def process_batch(self, batch):
        batch_x, batch_prev_y, batch_y, _, _ = batch
        batch_x = batch_x.to(self.device).float()
        batch_prev_y = batch_prev_y.to(self.device).float()
        batch_y = batch_y.to(self.device).float()

        preds = torch.Tensor().to(self.device)
        trues = torch.Tensor().to(self.device)
        for i in range(0, batch_x.shape[0]):
            seq_inp = batch_x[i]
            seq_true = batch_y[i]
            seq_pred = self(seq_inp, batch_prev_y[i])
            
            trues = torch.cat((trues, seq_true.unsqueeze(0)))
            preds = torch.cat((preds, seq_pred.unsqueeze(0)))

        preds = preds[:, :, self.target_idx]
        trues = trues[:, :, self.target_idx]

        return preds, trues

    def compute_loss(self, batch):
        preds, trues = self.process_batch(batch)
        loss = nn.MSELoss()(preds, trues)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log('test_loss', loss)
        return loss
    
    # def predict_step(self, batch, batch_idx):
    #     preds = []
    #     x, y = batch
    #     for i in range(0, x.shape[0]):
    #         preds.append(self(x[i].unsqueeze(0), y[i].unsqueeze(0)))
    #     preds = torch.cat(preds)
    #     return preds[::self.output_length], y[::self.output_length]

    def predict_step(self, batch, batch_idx):
        preds, trues = self.process_batch(batch)
        return preds[::self.output_length], trues[::self.output_length]
    
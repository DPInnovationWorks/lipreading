from typing import Any, List
import torch.nn as nn
import torch
from pytorch_lightning import LightningModule
import logging
from lipreading.utils.checkpoint import load_model_weight
from torchmetrics import WordErrorRate, CharErrorRate

class LRS2SentenceModule(LightningModule):
    def __init__(
        self, 
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        warmup,
        loss,
        load_model = None
    ):
        super().__init__()
        self.save_hyperparameters(logger=False,ignore=["model"])
        self.info_logger = logging.getLogger("pytorch_lightning")
        self.model_func = model
        
        
    def setup(self,stage):
        self.tokenizer = self.trainer.datamodule.tokenizer
        self.model = self.model_func(vocab_size=self.tokenizer.vocab_size)
        if self.hparams.load_model:
            ckpt = torch.load(self.hparams.load_model,map_location='cpu')
            load_model_weight(self.model, ckpt, self.info_logger)
            self.info_logger.info("Loaded model weight from {}".format(self.hparams.load_model))
            
        if self.hparams.loss.type == 'CTCLoss':
            self.loss_fn = nn.CTCLoss()
            
        elif self.hparams.loss.type == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
            
        else:
            raise NotImplementedError

        self.val_wer = WordErrorRate()
        self.val_cer = CharErrorRate()
        
        
        self.test_wer = WordErrorRate()
        self.test_cer = CharErrorRate()

    def forward(self, feats,feat_padding_masks,tokens_input,token_padding_masks,token_attn_mask):
        return self.model(feats,feat_padding_masks,tokens_input,token_padding_masks,token_attn_mask)

    def training_step(self, batch: Any, batch_idx: int):
        feats,feat_padding_masks,target_inp,target_out,target_inp_padding_masks,feats_attn_mask = batch
        logits = self.forward(feats,feat_padding_masks,target_inp,target_inp_padding_masks,feats_attn_mask)
        if self.hparams.loss.type == 'CTCLoss':
            raise NotImplementedError
            # loss = self.loss_fn(output.transpose(0, 1).log_softmax(-1), encode_char,video_len,char_len)
        elif self.hparams.loss.type == 'CrossEntropyLoss':
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        else:
            raise NotImplementedError
        
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        self.log("train/lr",self.trainer.optimizers[0].param_groups[0]["lr"],on_step=True,on_epoch=True,prog_bar=False,sync_dist=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {'loss':loss}

    def validation_step(self, batch: Any, batch_idx: int):
        feats,feat_padding_masks,target_inp,target_out,target_inp_padding_masks,feats_attn_mask = batch
        logits = self.forward(feats,feat_padding_masks,target_inp,target_inp_padding_masks,feats_attn_mask) # shape: (batch,frame,28)
        if self.hparams.loss.type == 'CTCLoss':
            raise NotImplementedError
            # loss = self.loss_fn(output.transpose(0, 1).log_softmax(-1), encode_char,video_len,char_len)
        elif self.hparams.loss.type == 'CrossEntropyLoss':
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        else:
            raise NotImplementedError
        
        gts = self.tokenizer.batch_decode(target_out,skip_special_tokens=True)
        memory = self.model.encode(feats,feat_padding_masks)
        predict_tokens = target_inp[:,:1] # 101 start token
        batch_size = predict_tokens.shape[0]
        # 多预测20个单词
        for _ in range(target_inp.shape[-1] + 20):
            pred_mask = nn.Transformer.generate_square_subsequent_mask(predict_tokens.shape[1],device=predict_tokens.device)
            output = self.model.decode(predict_tokens, memory,pred_mask)
            output = self.model.fc(output[:,-1])
            output = torch.argmax(output, dim=-1).reshape(batch_size,-1)
            predict_tokens = torch.cat([predict_tokens,output], dim=-1)
            
        preds = self.tokenizer.batch_decode(predict_tokens,skip_special_tokens=True)
        self.val_cer(preds, gts)
        self.val_wer(preds, gts)        

        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/loss",loss, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        return {"loss":loss}

    def test_step(self, batch: Any, batch_idx: int):
        feats,feat_padding_masks,target_inp,target_out,target_inp_padding_masks,feats_attn_mask = batch
        logits = self.forward(feats,feat_padding_masks,target_inp,target_inp_padding_masks,feats_attn_mask) # shape: (batch,frame,28)
        if self.hparams.loss.type == 'CTCLoss':
            raise NotImplementedError
            # loss = self.loss_fn(output.transpose(0, 1).log_softmax(-1), encode_char,video_len,char_len)
        elif self.hparams.loss.type == 'CrossEntropyLoss':
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        else:
            raise NotImplementedError
        
        gts = self.tokenizer.batch_decode(target_out,skip_special_tokens=True)
        memory = self.model.encode(feats,feat_padding_masks)
        predict_tokens = target_inp[:,:1] # 101 start token
        batch_size = predict_tokens.shape[0]
        # 多预测20个单词
        for _ in range(target_inp.shape[-1] + 20):
            pred_mask = nn.Transformer.generate_square_subsequent_mask(predict_tokens.shape[1],device=predict_tokens.device)
            output = self.model.decode(predict_tokens, memory,pred_mask)
            output = self.model.fc(output[:,-1])
            output = torch.argmax(output, dim=-1).reshape(batch_size,-1)
            predict_tokens = torch.cat([predict_tokens,output], dim=-1)
            
        preds = self.tokenizer.batch_decode(predict_tokens,skip_special_tokens=True)
        self.test_cer(preds, gts)
        self.test_wer(preds, gts)        

        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("test/cer", self.test_cer, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("test/loss",loss, on_step=False, on_epoch=True, prog_bar=False,sync_dist=True)
        return {"loss":loss}
    
    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.
        Returns:
            optimizer
        """
        optimizer = self.hparams.optimizer(model=self.model)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler':scheduler,
                    'monitor': "val/wer",
                    'interval':'epoch',
                    'frequency':1,
                }
            }
        return {"optimizer":optimizer}
            
    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
        ):
            """
            Performs a single optimization step (parameter update).
            Args:
                epoch: Current epoch
                batch_idx: Index of current batch
                optimizer: A PyTorch optimizer
                optimizer_idx: If you used multiple optimizers this indexes into that list.
                optimizer_closure: closure for all optimizers
                on_tpu: true if TPU backward is required
                using_native_amp: True if using native amp
                using_lbfgs: True if the matching optimizer is lbfgs
            """
            # warm up lr
            if self.trainer.global_step <= self.hparams.warmup['steps']:
                if self.hparams.warmup['name'] == "constant":
                    k = self.hparams.warmup['ratio']
                elif self.hparams.warmup['name'] == "linear":
                    k = 1 - (
                        1 - self.trainer.global_step / self.hparams.warmup['steps']
                    ) * (1 - self.hparams.warmup['ratio'])
                elif self.hparams.warmup['name'] == "exp":
                    k = self.hparams.warmup['ratio'] ** (
                        1 - self.trainer.global_step / self.hparams.warmup['steps']
                    )
                else:
                    raise Exception("Unsupported warm up type!")
                for pg in optimizer.param_groups:
                    pg["lr"] = pg["initial_lr"] * k

            # update params
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "sentence" / "model" / "grid.yaml")
    _ = hydra.utils.instantiate(cfg)
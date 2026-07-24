
from __future__ import annotations

import torch
import torch.optim as optim

import util
from model import DASTGCN


class trainer1:

    def __init__(
        self,
        in_dim,
        seq_length,
        num_nodes,
        nhid,
        CL,
        l,
        dropout,
        lrate,
        wdecay,
        device,
        supports,
        decay,
        sample_ratio=0.25,
        lambda_cl=0.4,
    ):
        self.device = torch.device(device)
        self.CL = bool(CL)
        self.clip = 5.0

        self.model = DASTGCN(
            device=self.device,
            num_nodes=num_nodes,
            CL=self.CL,
            l=l,
            dropout=dropout,
            supports=supports,
            length=seq_length,
            in_dim=in_dim,
            out_dim=seq_length,
            dilation_channels=nhid,
            end_channels=nhid * 16,
            kernel_size=3,
            K=2,
            node_embedding_dim=64,
            sample_ratio=sample_ratio,
            contrastive_temperature=0.1,
            lambda_cl=lambda_cl,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lrate,
            weight_decay=wdecay,
        )
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer,
            gamma=decay,
        )

    @staticmethod
    def _prediction_layout(output):

        if output.ndim != 4 or output.shape[-1] != 1:
            raise ValueError(
                "DASTGCN output must have shape [B,H,N,1], "
                f"but received {tuple(output.shape)}"
            )
        return output[..., 0].transpose(1, 2).contiguous()

    def train(self, input, real_val):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        output, contrastive_loss, _ = self.model(input)
        predict = self._prediction_layout(output)

        prediction_loss = util.masked_mae(predict, real_val, 0.0)
        if self.CL:
            total_loss = prediction_loss + self.model.lambda_cl * contrastive_loss
        else:
            total_loss = prediction_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite loss detected: "
                f"prediction_loss={float(prediction_loss.detach()):.6f}, "
                f"contrastive_loss={float(contrastive_loss.detach()):.6f}"
            )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()

        mae = util.masked_mae(predict, real_val, 0.0)
        mape = util.masked_mape(predict, real_val, 0.0)
        rmse = util.masked_rmse(predict, real_val, 0.0)

        return (
            float(total_loss.detach()),
            float(mae.detach()),
            float(mape.detach()),
            float(rmse.detach()),
        )

    def eval(self, input, real_val):
        self.model.eval()
        with torch.no_grad():
            output, _, _ = self.model(input)
            predict = self._prediction_layout(output)

            loss = util.masked_mae(predict, real_val, 0.0)
            mae = util.masked_mae(predict, real_val, 0.0)
            mape = util.masked_mape(predict, real_val, 0.0)
            rmse = util.masked_rmse(predict, real_val, 0.0)

        return (
            float(loss),
            float(mae),
            float(mape),
            float(rmse),
        )

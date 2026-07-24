
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


AdjacencyLike = Union[Tensor, np.ndarray, Sequence[Union[Tensor, np.ndarray]]]


def _prepare_adjacency(
    supports: Optional[AdjacencyLike],
    num_nodes: int,
) -> Tensor:
    if supports is None:
        raise ValueError(
            "DASTGCN requires the original/static road adjacency via `supports`. "
            "Pass a [N, N] tensor/array or a list whose first element is [N, N]."
        )

    adjacency = supports[0] if isinstance(supports, (list, tuple)) else supports
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

    if adjacency.ndim != 2 or adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(
            f"Expected supports with shape [{num_nodes}, {num_nodes}], "
            f"but received {tuple(adjacency.shape)}."
        )

    adjacency = adjacency.clamp_min(0.0)
    return adjacency


def row_normalize(adjacency: Tensor, eps: float = 1e-8) -> Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(eps)


class CausalGatedTemporalConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            2 * out_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.pad(x, (self.left_padding, 0, 0, 0))
        filter_part, gate_part = self.conv(x).chunk(2, dim=1)
        return self.dropout(torch.tanh(filter_part) * torch.sigmoid(gate_part))


class DiffusionGraphConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, order: int = 2) -> None:
        super().__init__()
        if order < 0:
            raise ValueError("diffusion order must be >= 0")
        self.order = order
        self.proj = nn.Conv2d((order + 1) * in_channels, out_channels, kernel_size=1)

    @staticmethod
    def _propagate(x: Tensor, adjacency: Tensor) -> Tensor:
        if adjacency.ndim == 2:
            return torch.einsum("ij,bcjt->bcit", adjacency, x)
        if adjacency.ndim == 3:
            return torch.einsum("bij,bcjt->bcit", adjacency, x)
        raise ValueError("adjacency must have shape [N,N] or [B,N,N]")

    def forward(self, x: Tensor, adjacency: Tensor) -> Tensor:
        diffusion_terms = [x]
        x_k = x
        for _ in range(self.order):
            x_k = self._propagate(x_k, adjacency)
            diffusion_terms.append(x_k)
        return self.proj(torch.cat(diffusion_terms, dim=1))


class HeterogeneousNodeIdentification(nn.Module):


    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 64,
        sample_ratio: float = 0.25,
        contrastive_temperature: float = 0.1,
        dpp_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be in (0, 1]")

        self.embedding = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
        self.sample_ratio = sample_ratio
        self.contrastive_temperature = contrastive_temperature
        self.dpp_eps = dpp_eps

    def _greedy_dpp_row(
        self,
        node_embedding: Tensor,
        candidates: Tensor,
        sample_count: int,
        gumbel_temperature: float,
    ) -> Tensor:
        num_candidates = int(candidates.numel())
        if num_candidates == 0:
            return node_embedding.new_zeros(0)

        features = F.normalize(node_embedding[candidates], dim=-1, eps=self.dpp_eps)

        kernel = torch.exp(features @ features.transpose(0, 1) - 1.0)
        kernel = 0.5 * (kernel + kernel.transpose(0, 1))
        kernel = kernel + self.dpp_eps * torch.eye(
            num_candidates, device=kernel.device, dtype=kernel.dtype
        )

        selected_indices = []
        straight_through_mask = kernel.new_zeros(num_candidates)

        for _ in range(sample_count):
            if selected_indices:
                selected = torch.tensor(
                    selected_indices, device=kernel.device, dtype=torch.long
                )
                l_ss = kernel.index_select(0, selected).index_select(1, selected)
                l_ss = l_ss + self.dpp_eps * torch.eye(
                    len(selected_indices), device=kernel.device, dtype=kernel.dtype
                )
                l_all_s = kernel.index_select(1, selected)
                solved = torch.linalg.solve(l_ss, l_all_s.transpose(0, 1)).transpose(0, 1)
                conditional_variance = kernel.diagonal() - (l_all_s * solved).sum(dim=-1)
            else:
                conditional_variance = kernel.diagonal()

            scores = torch.log(conditional_variance.clamp_min(self.dpp_eps))
            if selected_indices:
                scores = scores.clone()
                scores[selected] = -torch.inf

            if self.training:
                one_hot = F.gumbel_softmax(
                    scores,
                    tau=float(gumbel_temperature),
                    hard=True,
                    dim=0,
                )
            else:
                chosen = int(torch.argmax(scores).item())
                one_hot = F.one_hot(
                    torch.tensor(chosen, device=scores.device),
                    num_classes=num_candidates,
                ).to(scores.dtype)

            chosen_index = int(torch.argmax(one_hot.detach()).item())
            selected_indices.append(chosen_index)
            straight_through_mask = straight_through_mask + one_hot

        return straight_through_mask

    def _contrastive_loss(self, embeddings: Tensor, selected_edges: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, num_nodes, _ = embeddings.shape
        eye = torch.eye(num_nodes, device=adjacency.device, dtype=torch.bool)
        neighbor_mask = (adjacency > 0) & (~eye)
        neighbor_mask = neighbor_mask.unsqueeze(0).expand(batch_size, -1, -1)

        selected_mask = selected_edges.detach() > 0.5
        remaining_mask = neighbor_mask & (~selected_mask)

        normalized = F.normalize(embeddings, dim=-1, eps=self.dpp_eps)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))
        exp_similarity = torch.exp(similarity / self.contrastive_temperature)

        selected_count = selected_mask.sum(dim=-1)
        remaining_count = remaining_mask.sum(dim=-1)
        valid = (selected_count > 0) & (remaining_count > 0)

        selected_mean = (
            (exp_similarity * selected_mask.to(exp_similarity.dtype)).sum(dim=-1)
            / selected_count.clamp_min(1).to(exp_similarity.dtype)
        )
        remaining_mean = (
            (exp_similarity * remaining_mask.to(exp_similarity.dtype)).sum(dim=-1)
            / remaining_count.clamp_min(1).to(exp_similarity.dtype)
        )

        per_node_loss = -torch.log(
            (remaining_mean + self.dpp_eps) / (selected_mean + self.dpp_eps)
        )
        if valid.any():
            return per_node_loss[valid].mean()
        return embeddings.sum() * 0.0

    def forward(
        self,
        x: Tensor,
        adjacency: Tensor,
        gumbel_temperature: float,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, _, num_nodes, _ = x.shape

        embeddings = self.embedding(x).mean(dim=-1).transpose(1, 2)  # [B,N,D]

        row_candidate_lists = []
        for node_idx in range(num_nodes):
            candidates = torch.nonzero(adjacency[node_idx] > 0, as_tuple=False).flatten()
            candidates = candidates[candidates != node_idx]
            row_candidate_lists.append(candidates)

        batch_edge_masks = []
        for batch_idx in range(batch_size):
            rows = []
            for node_idx, candidates in enumerate(row_candidate_lists):
                degree = int(candidates.numel())
                if degree == 0:
                    rows.append(x.new_zeros(num_nodes))
                    continue

                sample_count = max(1, int(math.ceil(self.sample_ratio * degree)))
                if degree > 1:
                    sample_count = min(sample_count, degree - 1)
                else:
                    sample_count = 1

                local_mask = self._greedy_dpp_row(
                    embeddings[batch_idx],
                    candidates,
                    sample_count,
                    gumbel_temperature,
                )
                row = x.new_zeros(num_nodes).scatter_add(0, candidates, local_mask)
                rows.append(row)
            batch_edge_masks.append(torch.stack(rows, dim=0))

        selected_edges = torch.stack(batch_edge_masks, dim=0)  # [B,N,N]

        heterogeneous_adjacency = adjacency.unsqueeze(0) * selected_edges

        selected_node_strength = selected_edges.sum(dim=1)
        selected_node_mask = (selected_node_strength > 0).to(x.dtype)  # [B,N]

        contrastive_loss = self._contrastive_loss(
            embeddings, selected_edges, adjacency
        )
        return (
            heterogeneous_adjacency,
            selected_node_mask,
            contrastive_loss,
            embeddings,
            selected_edges,
        )


class BidirectionalInformationInteraction(nn.Module):

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.hetero_to_homo_gate = nn.Conv2d(2 * channels, channels, kernel_size=1)
        self.homo_to_hetero_gate = nn.Conv2d(2 * channels, channels, kernel_size=1)

    def forward(
        self,
        homogeneous: Tensor,
        heterogeneous: Tensor,
        selected_node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mask = selected_node_mask[:, None, :, None]

        heterogeneous_aligned = heterogeneous * mask
        h2h_gate = torch.sigmoid(
            self.hetero_to_homo_gate(
                torch.cat([heterogeneous_aligned, homogeneous], dim=1)
            )
        )

        homogeneous_selected = homogeneous * mask
        h2e_gate = torch.sigmoid(
            self.homo_to_hetero_gate(
                torch.cat([homogeneous_selected, heterogeneous], dim=1)
            )
        )

        # Eq. (27)-(28).
        enhanced_homogeneous = torch.tanh(homogeneous) * h2h_gate
        enhanced_heterogeneous = torch.tanh(heterogeneous) * h2e_gate * mask
        return enhanced_homogeneous, enhanced_heterogeneous


class ParallelSpatioTemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        graph_out_channels: int,
        temporal_kernel: int = 3,
        temporal_dilation: int = 1,
        diffusion_order: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.homogeneous_tcn = CausalGatedTemporalConv(
            in_channels,
            hidden_channels,
            kernel_size=temporal_kernel,
            dilation=temporal_dilation,
            dropout=dropout,
        )
        self.heterogeneous_tcn = CausalGatedTemporalConv(
            in_channels,
            hidden_channels,
            kernel_size=temporal_kernel,
            dilation=temporal_dilation,
            dropout=dropout,
        )
        self.dynamic_gcn = DiffusionGraphConv(
            hidden_channels, graph_out_channels, order=diffusion_order
        )
        self.static_gcn = DiffusionGraphConv(
            hidden_channels, graph_out_channels, order=diffusion_order
        )
        self.biim = BidirectionalInformationInteraction(graph_out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        homogeneous: Tensor,
        heterogeneous: Tensor,
        dynamic_adjacency: Tensor,
        heterogeneous_adjacency: Tensor,
        selected_node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        h_homo = self.homogeneous_tcn(homogeneous)
        h_hetero = self.heterogeneous_tcn(heterogeneous)

        z_homo = self.dynamic_gcn(h_homo, dynamic_adjacency)
        z_hetero = self.static_gcn(h_hetero, heterogeneous_adjacency)

        f_homo, f_hetero = self.biim(z_homo, z_hetero, selected_node_mask)
        return self.dropout(f_homo), self.dropout(f_hetero)


class DASTGCN(nn.Module):

    def __init__(
        self,
        device: Union[str, torch.device],
        num_nodes: int,
        CL: bool = True,
        l: int = 3,
        dropout: float = 0.3,
        supports: Optional[AdjacencyLike] = None,
        batch_size: int = 32,  
        length: int = 12,
        in_dim: int = 1,
        out_dim: int = 12,
        residual_channels: int = 32,  
        dilation_channels: int = 32,
        skip_channels: int = 256,  
        end_channels: int = 512,
        kernel_size: int = 3,
        K: int = 2,
        Kt: int = 3,  
        node_embedding_dim: int = 64,
        sample_ratio: float = 0.25,
        contrastive_temperature: float = 0.1,
        lambda_cl: float = 0.4,
        gumbel_tau_start: float = 1.0,
        gumbel_tau_min: float = 0.5,
        gumbel_tau_decay: float = 0.98,
    ) -> None:
        super().__init__()
        del batch_size, residual_channels, skip_channels, Kt

        if l < 1:
            raise ValueError("l (number of ST blocks) must be >= 1")

        self.device_hint = torch.device(device)
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.length = length
        self.num_blocks = l
        self.use_contrastive_learning = CL
        self.lambda_cl = lambda_cl

        static_adjacency = _prepare_adjacency(supports, num_nodes)
        self.register_buffer("A_static", static_adjacency)

        self.E1 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, node_embedding_dim))

        self.hnim = HeterogeneousNodeIdentification(
            in_channels=in_dim,
            embedding_dim=node_embedding_dim,
            sample_ratio=sample_ratio,
            contrastive_temperature=contrastive_temperature,
        )

        graph_channels = 2 * dilation_channels
        blocks = []
        current_channels = in_dim
        for _ in range(l):
            blocks.append(
                ParallelSpatioTemporalBlock(
                    in_channels=current_channels,
                    hidden_channels=dilation_channels,
                    graph_out_channels=graph_channels,
                    temporal_kernel=kernel_size,
                    temporal_dilation=1,
                    diffusion_order=K,
                    dropout=dropout,
                )
            )
            current_channels = graph_channels
        self.blocks = nn.ModuleList(blocks)

        self.end_conv_1 = nn.Conv2d(graph_channels, end_channels, kernel_size=1)
        self.end_conv_2 = nn.Conv2d(
            end_channels,
            out_dim,
            kernel_size=(1, length),
        )

        self.gumbel_tau_min = float(gumbel_tau_min)
        self.gumbel_tau_decay = float(gumbel_tau_decay)
        self.register_buffer(
            "gumbel_temperature",
            torch.tensor(float(gumbel_tau_start), dtype=torch.float32),
        )

        self.last_aux: Dict[str, Tensor] = {}

    def anneal_gumbel_temperature(self) -> float:
        new_value = max(
            self.gumbel_tau_min,
            float(self.gumbel_temperature.item()) * self.gumbel_tau_decay,
        )
        self.gumbel_temperature.fill_(new_value)
        return new_value

    def _dynamic_adjacency(self) -> Tensor:
        return torch.softmax(F.relu(self.E1 @ self.E2.transpose(0, 1)), dim=-1)

    def forward(
        self,
        input: Tensor,
        return_aux: bool = False,
    ):
        if input.ndim != 4:
            raise ValueError("input must have shape [B, C, N, T]")
        if input.shape[2] != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {input.shape[2]}"
            )
        if input.shape[-1] != self.length:
            raise ValueError(
                f"Expected input temporal length {self.length}, got {input.shape[-1]}. "
                "Set `length` to match the data loader."
            )
        if input.shape[1] < self.in_dim:
            raise ValueError(
                f"Expected at least {self.in_dim} input channels, got {input.shape[1]}"
            )

        x = input[:, : self.in_dim, :, :]
        dynamic_adjacency = self._dynamic_adjacency()

        (
            heterogeneous_adjacency,
            selected_node_mask,
            contrastive_loss,
            node_embeddings,
            selected_edges,
        ) = self.hnim(
            x,
            self.A_static,
            float(self.gumbel_temperature.item()),
        )

        sampled_self_loops = torch.diag_embed(selected_node_mask)
        heterogeneous_adjacency_norm = row_normalize(
            heterogeneous_adjacency + sampled_self_loops
        )

        mask = selected_node_mask[:, None, :, None]
        homogeneous_state = x
        heterogeneous_state = x * mask

        for block in self.blocks:
            homogeneous_state, heterogeneous_state = block(
                homogeneous_state,
                heterogeneous_state,
                dynamic_adjacency,
                heterogeneous_adjacency_norm,
                selected_node_mask,
            )

        prediction = self.end_conv_2(F.relu(self.end_conv_1(homogeneous_state)))

        if not self.use_contrastive_learning:
            contrastive_loss = prediction.sum() * 0.0

        self.last_aux = {
            "A_dynamic": dynamic_adjacency.detach(),
            "A_heterogeneous": heterogeneous_adjacency_norm.detach(),
            "selected_node_mask": selected_node_mask.detach(),
            "selected_edges": selected_edges.detach(),
            "node_embeddings": node_embeddings.detach(),
            "gumbel_temperature": self.gumbel_temperature.detach().clone(),
        }

        if return_aux:
            return prediction, contrastive_loss, self.last_aux

        return prediction, contrastive_loss, dynamic_adjacency





# if __name__ == "__main__":
#     torch.manual_seed(1)
#     batch, nodes, history, horizon = 2, 8, 12, 12
#     adjacency = torch.zeros(nodes, nodes)
#     for i in range(nodes):
#         adjacency[i, (i - 1) % nodes] = 1.0
#         adjacency[i, (i + 1) % nodes] = 1.0
#         adjacency[i, (i + 2) % nodes] = 1.0

#     model = DASTGCN(
#         device="cpu",
#         num_nodes=nodes,
#         CL=True,
#         l=3,
#         supports=adjacency,
#         length=history,
#         in_dim=1,
#         out_dim=horizon,
#         dilation_channels=16,
#         end_channels=64,
#         node_embedding_dim=24,
#         sample_ratio=0.25,
#     )
#     sample = torch.randn(batch, 1, nodes, history)
#     target = torch.randn(batch, horizon, nodes, 1)
#     forecast, loss_cl, a_dynamic = model(sample)
#     total_loss = F.l1_loss(forecast, target) + model.lambda_cl * loss_cl
#     total_loss.backward()
#     print("forecast:", tuple(forecast.shape))
#     print("contrastive loss:", float(loss_cl.detach()))
#     print("dynamic adjacency:", tuple(a_dynamic.shape))

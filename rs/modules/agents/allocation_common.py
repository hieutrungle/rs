"""https://github.com/shariqiqbal2810/ALMA/blob/main/src/modules/agents/allocation_common.py"""

import torch as th
import torch.nn as nn

COUNT_NORM_FACTOR = 0.2


class TaskEmbedder(nn.Module):
    """
    Learn a set of sub-task embeddings. Embeddings are selected randomly per
    forward pass but are always consistent across the batch and entities. In
    other words two agents assigned to task 1 will always get the same embedding
    as each other but it may be a different embedding each time. We learn more
    embeddings than we need in case we want to generalize to settings with more
    subtasks than we are training on.
    """

    def __init__(self, embed_dim, args) -> None:
        super().__init__()
        self.args = args
        self.n_tasks = args.n_tasks
        self.n_extra_tasks = args.n_extra_tasks
        in_dim = self.n_tasks + self.n_extra_tasks
        self.fc = nn.Linear(in_dim, embed_dim, bias=False)

    def forward(self, task_one_hot):
        extra_dims = th.zeros_like(task_one_hot[..., [0]]).repeat_interleave(self.n_extra_tasks, -1)
        task_one_hot = th.cat([task_one_hot, extra_dims], dim=-1)
        shuff_task_one_hot = task_one_hot[..., th.randperm(self.n_tasks + self.n_extra_tasks)]
        return self.fc(shuff_task_one_hot)


class CountEmbedder(nn.Module):
    """
    Create embedding that encodes the quantity of agents/non-agent entities
    belonging to each subtask and returns a vector for each entity based on the
    subtask they belong to
    """

    def __init__(self, embed_dim, args) -> None:
        super().__init__()
        self.args = args
        self.count_embed = nn.Linear(2, embed_dim)

    def forward(self, entity2task):
        x1_pertask_count = (
            self.count_embed(
                th.stack(
                    [
                        entity2task[:, : self.args.n_agents].sum(dim=1),
                        entity2task[:, self.args.n_agents :].sum(dim=1),
                    ],
                    dim=-1,
                )
            )
            * COUNT_NORM_FACTOR
        )
        x1_count = th.bmm(entity2task, x1_pertask_count)
        return x1_count

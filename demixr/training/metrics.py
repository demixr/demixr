import torch

class SDR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expr = 'bi,bi->b'

    def _batch_dot(self, x, y):
        return torch.einsum(self.expr, x, y)

    def forward(self, outputs, labels):
        if outputs.dtype != labels.dtype:
            outputs = outputs.to(labels.dtype)
        length = min(labels.shape[-1], outputs.shape[-1])
        labels = labels[..., :length].reshape(labels.shape[0], -1)
        outputs = outputs[..., :length].reshape(outputs.shape[0], -1)

        delta = 1e-7  # avoid numerical errors
        num = self._batch_dot(labels, labels)
        den = num + self._batch_dot(outputs, outputs) - \
            2 * self._batch_dot(outputs, labels)
        den = den.relu().add_(delta).log10()
        num = num.add_(delta).log10()
        return 10 * (num - den)

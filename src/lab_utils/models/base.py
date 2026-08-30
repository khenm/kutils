import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class BaseModel(nn.Module, PyTorchModelHubMixin):
    """Base model with HF Hub integration.

    Every model gets save_pretrained / from_pretrained / push_to_hub for free.
    """

    def __init__(self):
        super().__init__()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def generate_model_card(self, *args, **kwargs):
        """Default HF model card plus a training-provenance section.

        Signature mirrors the Hugging Face base class exactly. Pass the same
        dict `write_summary` builds (run_name/config/metrics/git/...); missing
        keys are skipped, so it degrades gracefully.
        """
        card = super().generate_model_card(*args, **kwargs)
        section = _provenance_section(kwargs)
        if section:
            card.text += "\n---\n## Training provenance\n" + section
        return card


def _provenance_section(kwargs: dict) -> str:
    """Markdown provenance lines from a summary-like dict (missing keys skipped)."""
    lines: list[str] = []
    if run_name := kwargs.get("run_name"):
        lines.append(f"- run_name: `{run_name}`")
    git = kwargs.get("git") or {}
    if commit := git.get("commit"):
        suffix = " (dirty)" if git.get("dirty") else ""
        lines.append(f"- git commit: `{commit[:12]}`{suffix}")
    if lab_commit := kwargs.get("lab_utils_commit"):
        lines.append(f"- lab_utils commit: `{lab_commit[:12]}`")
    if config := kwargs.get("config"):
        lines.append("- config:")
        lines.extend(f"  - {key}: {value}" for key, value in config.items())
    if metrics := kwargs.get("metrics"):
        lines.append("- metrics:")
        lines.extend(f"  - {key}: {value}" for key, value in metrics.items())
    return "\n".join(lines) + "\n" if lines else ""

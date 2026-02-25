"""
utils/visualizations.py
Grad-CAM and t-SNE visualizations, returned as matplotlib Figures
for direct logging to wandb.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inspect
matplotlib.use("Agg")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with AlexNet, ResNet50, and MobileNetV2 by targeting
    the last convolutional layer of each architecture.
    """

    # Last conv layer name per architecture
    TARGET_LAYERS = {
        "alexnet":      "features.12",
        "resnet50":     "layer4.2.conv3",
        "mobilenet_v2": "features.18.0",
    }

    def __init__(self, model: nn.Module, arch: str):
        self.model      = model
        self.arch       = arch
        self._gradients = None
        self._activations = None

        target_name = self.TARGET_LAYERS.get(arch)
        if target_name is None:
            raise ValueError(f"No target layer defined for architecture '{arch}'.")

        layer = self._get_layer(target_name)
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _get_layer(self, name: str) -> nn.Module:
        parts = name.split(".")
        layer = self.model
        for p in parts:
            layer = layer[int(p)] if p.isdigit() else getattr(layer, p)
        return layer

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, images: torch.Tensor, class_idx: torch.Tensor | None = None) -> np.ndarray:
        """
        Generate CAM heatmaps for a batch of images.
        Returns numpy array of shape (B, H, W) normalised to [0, 1].
        """
        self.model.eval()
        images = images.requires_grad_(True)

        logits = self.model(images)

        if class_idx is None:
            class_idx = logits.argmax(dim=1)

        # Backprop for each image in the batch independently
        heatmaps = []
        for i in range(images.size(0)):
            self.model.zero_grad()
            score = logits[i, class_idx[i]]
            score.backward(retain_graph=(i < images.size(0) - 1))

            weights = self._gradients[i].mean(dim=(1, 2), keepdim=True)
            cam     = (weights * self._activations[i]).sum(dim=0)
            cam     = F.relu(cam)
            cam     = cam.cpu().numpy()
            cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            heatmaps.append(cam)

        return np.stack(heatmaps)


def plot_gradcam_grid(
    images: torch.Tensor,
    heatmaps: np.ndarray,
    labels: list,
    preds: list,
    class_names: list | None = None,
    title: str = "Grad-CAM",
) -> plt.Figure:
    """
    Plots a grid of original images overlaid with Grad-CAM heatmaps.
    """
    n     = min(len(images), 8)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.5, 6))

    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    for i in range(n):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)
        h, w = img.shape[:2]

        hm_resized = np.kron(
            heatmaps[i],
            np.ones((h // heatmaps[i].shape[0] + 1, w // heatmaps[i].shape[1] + 1)),
        )[:h, :w]
        hm_resized = plt.cm.jet(hm_resized)[:, :, :3]

        # Row 0: original image
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        true_label = class_names[labels[i]] if class_names else str(labels[i])
        axes[0, i].set_title(f"True: {true_label}", fontsize=7)

        # Row 1: overlay
        overlay = 0.5 * img + 0.5 * hm_resized[:, :, :3]
        axes[1, i].imshow(overlay.clip(0, 1))
        axes[1, i].axis("off")
        pred_label = class_names[preds[i]] if class_names else str(preds[i])
        color = "green" if preds[i] == labels[i] else "red"
        axes[1, i].set_title(f"Pred: {pred_label}", fontsize=7, color=color)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    arch: str,
    max_samples: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts feature embeddings before the classification head.
    Returns (embeddings, labels) as numpy arrays.
    """
    model.eval()
    embeddings, all_labels = [], []
    n_collected = 0

    # Hook to capture the penultimate layer output
    captured = {}

    def hook_fn(module, input, output):
        captured["embedding"] = output.detach().cpu()

    # Register on the layer before the final head
    if arch == "alexnet":
        handle = model.classifier[5].register_forward_hook(hook_fn)
    elif arch == "resnet50":
        handle = model.avgpool.register_forward_hook(hook_fn)
    elif arch == "mobilenet_v2":
        handle = model.features[-1].register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in loader:
            if n_collected >= max_samples:
                break
            images = images.to(device)
            model(images)

            emb = captured["embedding"]
            emb = emb.view(emb.size(0), -1).numpy()   # flatten

            embeddings.append(emb)
            all_labels.append(labels.numpy())
            n_collected += emb.shape[0]

    handle.remove()

    embeddings  = np.concatenate(embeddings,  axis=0)[:max_samples]
    all_labels  = np.concatenate(all_labels,  axis=0)[:max_samples]
    return embeddings, all_labels


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list | None = None,
    title: str = "t-SNE Embedding",
    perplexity: int = 30,
) -> plt.Figure:
    """
    Fits t-SNE and returns a scatter plot Figure coloured by class.
    """
    tsne_kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "random_state": 42,
    }
    tsne_signature = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in tsne_signature:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000

    tsne   = TSNE(**tsne_kwargs)
    coords = tsne.fit_transform(embeddings)

    n_classes = len(np.unique(labels))
    cmap      = plt.cm.get_cmap("tab20", n_classes)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls_idx in range(n_classes):
        mask = labels == cls_idx
        label = class_names[cls_idx] if (class_names and n_classes <= 20) else str(cls_idx)
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=cmap(cls_idx), label=label, s=10, alpha=0.7)

    if n_classes <= 20:
        ax.legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    return fig

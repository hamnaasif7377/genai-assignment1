"""
Task 3: Soft Mixture-of-Experts restoration.
Combines a gating network (same architecture as Task 2's classifier)
with the three Task 2 specialist experts and an identity branch for
clean inputs. Instead of hard-routing to one expert, every branch is
weighted continuously via softmax, making the whole system end-to-end
differentiable for joint fine-tuning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.task2_classifier import CorruptionClassifier
from src.models.task2_specialists import SpecialistAutoencoder


class SoftMoERestoration(nn.Module):
    def __init__(self, clf_base_ch, clf_dropout, spec_base_ch, spec_bottleneck_dim,
                 spec_dropout, temperature=1.0):
        super().__init__()
        # Gate reuses the classifier architecture; its 4 logits become the
        # 4 branch weights (clean, salt_pepper, blur, occlusion) after softmax.
        self.gate = CorruptionClassifier(base_ch=clf_base_ch, dropout=clf_dropout, n_classes=4)

        self.salt_pepper_expert = SpecialistAutoencoder(
            base_ch=spec_base_ch, bottleneck_dim=spec_bottleneck_dim, dropout=spec_dropout)
        self.blur_expert = SpecialistAutoencoder(
            base_ch=spec_base_ch, bottleneck_dim=spec_bottleneck_dim, dropout=spec_dropout)
        self.occlusion_expert = SpecialistAutoencoder(
            base_ch=spec_base_ch, bottleneck_dim=spec_bottleneck_dim, dropout=spec_dropout)

        self.temperature = temperature

    def load_pretrained(self, classifier_ckpt, salt_pepper_ckpt, blur_ckpt, occlusion_ckpt, device="cpu"):
        """Initialize gate from Task 2's trained classifier, and experts
        from Task 2's trained specialists, per the assignment's requirement
        to not start joint training from random weights."""
        self.gate.load_state_dict(torch.load(classifier_ckpt, map_location=device))
        self.salt_pepper_expert.load_state_dict(torch.load(salt_pepper_ckpt, map_location=device))
        self.blur_expert.load_state_dict(torch.load(blur_ckpt, map_location=device))
        self.occlusion_expert.load_state_dict(torch.load(occlusion_ckpt, map_location=device))

    def freeze_experts(self):
        for p in self.salt_pepper_expert.parameters():
            p.requires_grad = False
        for p in self.blur_expert.parameters():
            p.requires_grad = False
        for p in self.occlusion_expert.parameters():
            p.requires_grad = False

    def unfreeze_experts(self):
        for p in self.salt_pepper_expert.parameters():
            p.requires_grad = True
        for p in self.blur_expert.parameters():
            p.requires_grad = True
        for p in self.occlusion_expert.parameters():
            p.requires_grad = True

    def forward(self, x, return_weights=False):
        gate_logits = self.gate(x)
        weights = F.softmax(gate_logits / self.temperature, dim=1)  # [B, 4] -> clean, salt_pepper, blur, occlusion

        clean_branch = x
        salt_pepper_branch = self.salt_pepper_expert(x)
        blur_branch = self.blur_expert(x)
        occlusion_branch = self.occlusion_expert(x)

        w = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B,4,1,1,1] for broadcasting
        output = (w[:, 0] * clean_branch + w[:, 1] * salt_pepper_branch +
                  w[:, 2] * blur_branch + w[:, 3] * occlusion_branch)

        if return_weights:
            return output, weights, gate_logits
        return output

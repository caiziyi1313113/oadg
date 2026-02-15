import numpy as np
import jittor as jt
import jittor.nn as nn

from jdet.utils.registry import MODELS


def _eye(n, dtype=jt.float32):
    return jt.array(np.eye(n, dtype=np.float32)).cast(dtype)


def grad_reverse(x, lambd=1.0):
    """Forward identity, backward multiply gradient by -lambda."""
    lambd = float(lambd)
    if lambd <= 0:
        return x
    return (-lambd) * x + ((1.0 + lambd) * x).stop_grad()


def _l2_normalize(x, eps=1e-6):
    denom = jt.sqrt((x * x).sum(dim=1, keepdims=True) + eps)
    return x / denom


def _single_pair_contrastive(feat_inv, feat_spe):
    inv_n = _l2_normalize(feat_inv)
    spe_n = _l2_normalize(feat_spe)
    cos_sim = (inv_n * spe_n).sum(dim=1)
    # Minimize this term to separate invariant/specific embeddings.
    return ((cos_sim + 1.0) * 0.5).mean()


def _has_complex_fft_api():
    cls = getattr(nn, "ComplexNumber", None)
    if cls is None:
        return False
    return hasattr(cls, "fft2") and hasattr(cls, "ifft2")


def _complex_from_real(real):
    cls = getattr(nn, "ComplexNumber", None)
    if cls is None:
        raise RuntimeError("jittor.nn.ComplexNumber is required for complex FFT backend.")
    imag = jt.zeros_like(real)
    return cls(real, imag)


def _complex_value(x):
    if hasattr(x, "value"):
        return x.value
    if isinstance(x, jt.Var) and x.ndim > 0 and x.shape[-1] == 2:
        return x
    raise RuntimeError("Unsupported complex tensor format for current Jittor build.")


def _complex_from_value(value):
    cls = getattr(nn, "ComplexNumber", None)
    if cls is None:
        raise RuntimeError("jittor.nn.ComplexNumber is required for complex FFT backend.")
    return cls(value, is_concat_value=True)


def _complex_real(x):
    if hasattr(x, "real"):
        return x.real
    if hasattr(x, "value"):
        return x.value[..., 0]
    if isinstance(x, jt.Var) and x.ndim > 0 and x.shape[-1] == 2:
        return x[..., 0]
    return x


def _build_pos_mask(n, labels=None):
    if labels is None:
        return jt.ones((n, n), dtype=jt.float32) - _eye(n)
    labels = labels.reshape((-1, 1))
    mask = (labels == labels.transpose(0, 1)).float()
    return mask - _eye(n, dtype=mask.dtype)


def _valid_mean(values, valid_mask, eps=1e-8):
    denom = jt.maximum(valid_mask.sum(), jt.array(eps, dtype=values.dtype))
    return (values * valid_mask).sum() / denom


def _supcon_core(feat_inv, feat_spe, temp=0.07, labels=None, two_side=True):
    if feat_inv.numel() == 0 or feat_spe.numel() == 0:
        return (feat_inv.sum() + feat_spe.sum()) * 0.0

    n = min(int(feat_inv.shape[0]), int(feat_spe.shape[0]))
    if n <= 0:
        return (feat_inv.sum() + feat_spe.sum()) * 0.0

    feat_inv = _l2_normalize(feat_inv[:n])
    feat_spe = _l2_normalize(feat_spe[:n])
    if labels is not None:
        labels = labels[:n]
    if n == 1:
        return _single_pair_contrastive(feat_inv, feat_spe)

    feat_total = jt.contrib.concat([feat_inv, feat_spe], dim=0)
    logits = jt.matmul(feat_total, feat_total.transpose(0, 1)) / max(float(temp), 1e-6)
    mask_total = jt.ones((n * 2, n * 2), dtype=jt.float32) - _eye(n * 2)

    # Shift and clamp for numeric stability before exp.
    logits = logits * mask_total
    logits_max = jt.max(logits, dim=1, keepdims=True)
    logits = jt.clamp(logits - logits_max.stop_grad(), -20.0, 20.0)

    exp_logits = jt.exp(logits) * mask_total
    log_prob = logits - jt.log(exp_logits.sum(dim=1, keepdims=True) + 1e-8)

    pos_mask = _build_pos_mask(n, labels=labels)
    pos_count = pos_mask.sum(dim=1)
    valid = (pos_count > 0).float()
    if int(valid.sum().item()) == 0:
        return _single_pair_contrastive(feat_inv, feat_spe)

    inv_logits = log_prob[:n, :n] * pos_mask
    inv_loss_each = -inv_logits.sum(dim=1) / jt.maximum(pos_count, jt.ones((n,), dtype=pos_count.dtype))
    loss_inv = _valid_mean(inv_loss_each, valid)

    if not two_side:
        return loss_inv

    spe_logits = log_prob[n:, n:] * pos_mask
    spe_loss_each = -spe_logits.sum(dim=1) / jt.maximum(pos_count, jt.ones((n,), dtype=pos_count.dtype))
    loss_spe = _valid_mean(spe_loss_each, valid)
    return 0.5 * (loss_inv + loss_spe)


@MODELS.register_module()
class FDDFilter(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_size=(1024, 2048),
        use_sigmoid=False,
        filter_type="mask",
        conv_hidden=16,
        complex_mul_fallback=True,
        backend="auto",
        spatial_kernel=7,
        spatial_init="avg",
        grl_lambda=0.0,
        eps=1e-6,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.base_h, self.base_w = int(base_size[0]), int(base_size[1])
        self.use_sigmoid = bool(use_sigmoid)
        self.filter_type = filter_type
        self.backend_req = backend
        self.grl_lambda = float(grl_lambda)
        self.eps = float(eps)
        self.spatial_kernel = int(spatial_kernel)
        # Keep arg for config compatibility; complex branch now uses explicit real/imag ops.
        self.complex_mul_fallback = bool(complex_mul_fallback)

        if self.spatial_kernel < 1:
            self.spatial_kernel = 1
        if self.spatial_kernel % 2 == 0:
            self.spatial_kernel += 1
        if self.filter_type not in ("mask", "conv"):
            raise ValueError(f"Unsupported filter_type={self.filter_type}, expected mask/conv")

        complex_ok = _has_complex_fft_api()
        if self.backend_req == "auto":
            self.backend = "complex" if complex_ok else "spatial"
        elif self.backend_req in ("complex", "spatial"):
            self.backend = self.backend_req
        else:
            raise ValueError(f"Unsupported backend={self.backend_req}, expected auto/complex/spatial")

        if self.backend == "complex":
            if not complex_ok:
                ver = getattr(jt, "__version__", "unknown")
                raise RuntimeError(
                    f"backend='complex' but current Jittor(version={ver}) has no ComplexNumber fft2/ifft2 APIs."
                )
            if self.filter_type == "mask":
                init = jt.ones((self.in_channels, self.base_h, self.base_w))
                self.mask = nn.Parameter(init)
            else:
                hidden = max(int(conv_hidden), self.in_channels)
                self.freq_conv1 = nn.Conv2d(self.in_channels, hidden, 1)
                self.freq_conv2 = nn.Conv2d(hidden, self.in_channels, 1)
                # Start from identity frequency response to avoid early unstable scale explosion.
                nn.init.constant_(self.freq_conv2.weight, 0.0)
                nn.init.constant_(self.freq_conv2.bias, 0.0)
        else:
            k = self.spatial_kernel
            init_w = np.zeros((self.in_channels, 1, k, k), dtype=np.float32)
            if spatial_init == "gaussian":
                yy, xx = np.mgrid[0:k, 0:k].astype(np.float32)
                center = 0.5 * (k - 1)
                sigma = max(k / 3.0, 1.0)
                g = np.exp(-((xx - center) ** 2 + (yy - center) ** 2) / (2.0 * sigma * sigma))
                g = g / max(g.sum(), 1e-6)
                for c in range(self.in_channels):
                    init_w[c, 0] = g
            else:
                init_w.fill(1.0 / float(k * k))
            self.spatial_weight = nn.Parameter(jt.array(init_w))

    def _resize_mask(self, h, w):
        mask = self.mask
        if int(mask.shape[1]) != int(h) or int(mask.shape[2]) != int(w):
            mask = nn.interpolate(mask.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)
            mask = mask.squeeze(0)
        if self.use_sigmoid:
            mask = mask.sigmoid()
        return mask

    def _spatial_filter(self, x):
        w = jt.abs(self.spatial_weight) + 1e-6
        w_sum = w.sum(dim=3, keepdims=True).sum(dim=2, keepdims=True)
        w = w / (w_sum + 1e-6)
        pad = self.spatial_kernel // 2
        return nn.conv2d(x, w, None, 1, pad, 1, self.in_channels)

    def execute(self, x):
        if self.backend == "spatial":
            inv = self._spatial_filter(x)
            spe = x - inv
            return inv, spe

        n, c, h, w = x.shape
        freq = _complex_from_real(x.reshape((-1, h, w))).fft2()
        freq_value = _complex_value(freq).reshape((n, c, h, w, 2))
        real = freq_value[..., 0]
        imag = freq_value[..., 1]

        if self.filter_type == "mask":
            scale = self._resize_mask(h, w).reshape((1, c, h, w))
        else:
            amp = jt.sqrt(real * real + imag * imag + self.eps)
            delta = self.freq_conv2(nn.relu(self.freq_conv1(amp)))
            if self.grl_lambda > 0:
                delta = grad_reverse(delta, self.grl_lambda)
            # Bounded residual gain in [0, 2] keeps training stable.
            scale = 1.0 + jt.tanh(delta)
            if self.use_sigmoid:
                scale = scale.sigmoid()

        real_new = real * scale
        imag_new = imag * scale
        freq_new = jt.stack([real_new, imag_new], dim=-1).reshape((n * c, h, w, 2))
        inv = _complex_real(_complex_from_value(freq_new).ifft2()).reshape((n, c, h, w))
        spe = x - inv
        return inv, spe


def supcon_one_side(feat_inv, feat_spe, temp=0.07, labels=None):
    return _supcon_core(feat_inv, feat_spe, temp=temp, labels=labels, two_side=False)


def supcon_two_side(feat_inv, feat_spe, temp=0.07, labels=None):
    return _supcon_core(feat_inv, feat_spe, temp=temp, labels=labels, two_side=True)


class FDDProjector(nn.Module):
    def __init__(self, in_channels, proj_dim):
        super().__init__()
        self.fc = nn.Linear(int(in_channels), int(proj_dim))

    def execute(self, x):
        x = x.mean(3).mean(2)
        return self.fc(x)

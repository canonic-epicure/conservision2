import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2

def to_rgb(img_u8: torch.Tensor):
    if img_u8.shape[0] == 1:
        img_u8 = img_u8.repeat(3, 1, 1)
    return img_u8


class CLAHE:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8),
                 apply_if_dark=True, dark_thr=0.35):
        """
        clip_limit: насколько ограничивать усиление контраста (>1 => сильнее).
        tile_grid_size: размер сетки CLAHE (в плитках).
        apply_if_dark: если True, применяем CLAHE только для "тёмных" кадров.
        dark_thr: порог по средней яркости L в [0,1], ниже которого включаем CLAHE.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.apply_if_dark = apply_if_dark
        self.dark_thr = dark_thr

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()

    @torch.no_grad()
    def __call__(self, img):
        # Приводим к numpy uint8 RGB [0..255]
        if isinstance(img, torch.Tensor):
            img = to_rgb(img)
            # ожидаем CxHxW, [0,1]
            x = (img.clamp(0,1).mul(255).byte().permute(1,2,0).cpu().numpy())
        elif isinstance(img, Image.Image):
            img = img.convert("RGB")
            x = np.array(img)  # HxWxC, uint8, RGB
        else:
            raise TypeError("img must be PIL.Image or torch.Tensor")

        # RGB -> Lab (OpenCV ожидает RGB во флаге COLOR_RGB2LAB)
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)  # uint8
        L = lab[:,:,0]        # L в [0..255]
        a = lab[:,:,1]        # a в [0..255] со сдвигом 128
        b = lab[:,:,2]        # b в [0..255] со сдвигом 128

        # Критерий "темноты" по средней L (нормируем к [0,1])
        mean_L = L.mean() / 255.0

        do_apply = True
        if self.apply_if_dark:
            do_apply = (mean_L < self.dark_thr)

        if do_apply:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                    tileGridSize=self.tile_grid_size)
            L_eq = clahe.apply(L)
        else:
            L_eq = L  # оставляем как есть

        lab_eq = np.stack([L_eq, a, b], axis=2).astype(np.uint8)

        # Lab -> RGB
        rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_Lab2RGB)  # uint8

        # обратно к тензору [0,1], CxHxW
        out = torch.from_numpy(rgb_eq).permute(2,0,1).float() / 255.0
        return out


def is_overexposed(
    img_u8: torch.Tensor,
    roi_border_frac: float = 0.05,
    T_hi: int = 240,
    mu_thr: float = 240.0,
    sigma_thr: float = 8.0,
    p_hi_thr: float = 0.75,
    p_edge_thr: float = 0.05
):
    """
    img_u8: uint8 tensor [C,H,W] в диапазоне 0..255 (RGB)
    Возвращает (bool, dict метрик).
    """
    assert img_u8.dtype == torch.uint8 and img_u8.ndim == 3 and img_u8.shape[0] == 3
    C, H, W = img_u8.shape

    # 1) ROI: отрежем рамки/штампы по краям
    rb = int(round(roi_border_frac * H))
    cb = int(round(roi_border_frac * W))
    img = img_u8[:, rb:H-rb if H-2*rb>0 else H, cb:W-cb if W-2*cb>0 else W]

    # 2) в float
    x = img.float()  # [3,h,w]

    # 3) яркость (люминанс)
    Y = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2]  # [h,w]

    # 4) статистики яркости
    mu = Y.mean()
    sigma = Y.std(unbiased=False)

    # 5) доля пикселей, где ВСЕ каналы почти белые
    thr = torch.tensor(T_hi, dtype=torch.float32, device=x.device)
    p_hi = ((x[0] >= thr) & (x[1] >= thr) & (x[2] >= thr)).float().mean()

    # 6) «фактура»: плотность границ по Собелю
    #   задаём ядра Собеля (на batch=1, channel=1)
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=x.device).view(1,1,3,3)
    Y1 = Y.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    gx = F.conv2d(Y1, kx, padding=1)
    gy = F.conv2d(Y1, ky, padding=1)
    G = torch.sqrt(gx*gx + gy*gy).squeeze()  # [h,w]

    # порог для градиента возьмём «абсолютный» в u8-единицах
    tau = 12.0
    p_edge = (G > tau).float().mean()

    decision = (mu >= mu_thr) and (sigma <= sigma_thr) and (p_hi >= p_hi_thr) and (p_edge <= p_edge_thr)

    return bool(decision), {
        "mu": float(mu.item()),
        "sigma": float(sigma.item()),
        "p_hi": float(p_hi.item()),
        "p_edge": float(p_edge.item())
    }


def looks_grayscale_ycbcr_cv(img, std_thresh=12.0, far_thresh=10, far_frac=0.02):
    """
    Определяет, является ли изображение преимущественно серым.

    Параметры
    ---------
    img : PIL.Image | np.ndarray  HxWx{1,3}  uint8 | float
        Изображение. Если float, допускается диапазон [0,1] (будет приведён к 0..255).
    std_thresh : float
        Порог для E = sqrt(Var(Cb) + Var(Cr)) в 8-битной шкале (0..255).
    far_thresh : int
        Отступ от 128 в Cb/Cr, чтобы считать пиксель «заметно цветным».
    far_frac : float
        Допустимая доля заметно цветных пикселей.

    Возвращает
    ----------
    (is_gray: bool, stats: dict)
    """
    arr = np.asarray(img)

    # 1) Если изображение одноканальное — считаем серым сразу
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] == 1):
        return True, {"reason": "single_channel"}

    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("Ожидается HxWx3 или HxW")

    # 2) Привести к uint8 0..255 (OpenCV ожидает именно такой масштаб)
    a = arr[..., :3]
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        if a.max() <= 1.0:
            a = (a * 255.0).clip(0, 255)
        else:
            a = a.clip(0, 255)
        a = a.astype(np.uint8)

    # 3) RGB -> YCrCb с OpenCV (внимание: порядок каналов Y, Cr, Cb)
    # Если исходник в RGB (PIL/NumPy), используем конвертацию RGB2YCrCb
    a = np.ascontiguousarray(a)  # на всякий случай для cv2
    ycrcb = cv2.cvtColor(a, cv2.COLOR_RGB2YCrCb)
    Y  = ycrcb[..., 0]
    Cr = ycrcb[..., 1]
    Cb = ycrcb[..., 2]

    # 4) Энергия цветности и доля «далёких» от серого пикселей
    sig_cb = float(np.std(Cb))
    sig_cr = float(np.std(Cr))
    E = float(np.hypot(sig_cb, sig_cr))  # sqrt(sig_cb^2 + sig_cr^2)

    far = (np.abs(Cb.astype(np.int16) - 128) > far_thresh) | \
          (np.abs(Cr.astype(np.int16) - 128) > far_thresh)
    frac_far = float(np.mean(far))

    is_gray = (E < std_thresh) and (frac_far < far_frac)

    return bool(is_gray), {
        "E": E, "std_cb": sig_cb, "std_cr": sig_cr,
        "frac_far": frac_far,
        "params": {"std_thresh": std_thresh, "far_thresh": far_thresh, "far_frac": far_frac}
    }

from torchvision.transforms import v2, InterpolationMode

from lib.photo import CLAHE

transform_inference = v2.Compose([
    CLAHE(),
    CLAHE(),
    v2.ToPILImage(),
])

transform_training = v2.Compose([
    transform_inference,

    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    v2.RandomAutocontrast(p=0.1),
    v2.RandomZoomOut(p=0.1),
    v2.RandomEqualize(p=0.1),
    v2.RandomAdjustSharpness(p=0.3, sharpness_factor=1.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BICUBIC),
])

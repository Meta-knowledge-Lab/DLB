from .vgg import vgg16, vgg19
from .resnet import resnet32, resnet110, wide_resnet20_8
from .densenet import densenetd40k12, densenetd100k12, densenetd100k40, densenetd190k12

model_dict = {
    "vgg16": vgg16,
    "vgg19": vgg19,
    "resnet32": resnet32,
    "resnet110": resnet110,
    "wide_resnet20_8": wide_resnet20_8,
    "densenetd40k12": densenetd40k12,
    "densenetd100k12": densenetd100k12,
    "densenetd100k40": densenetd100k40,
    "densenetd190k12": densenetd190k12,
}

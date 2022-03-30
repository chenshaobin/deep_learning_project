from Utils import RandomVerticalFlip, Compose, ToTensor, RandomCrop
from Utils import img_Resize, img_Compose, img_ToTensor, img_RandomCrop, Resize
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms as T


if __name__ == '__main__':
    # 对image和label分别进行随机水平翻转测试实例
    # """
    img_transforms = Compose([
        RandomVerticalFlip(0.5),
        # Resize(224),
        RandomCrop(224),
        ToTensor()
    ])
    image_path = r'D:\githubCode\Ethan_project\deep_learning_project\segmentationImg\21_training.png'
    label_path = r'D:\githubCode\Ethan_project\deep_learning_project\segmentationImg\21_manual1.png'
    image = Image.open(image_path).convert('RGB')
    print('image size:', image.size)
    label = Image.open(label_path).convert('L')
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.subplot(2, 2, 2)
    plt.imshow(label, cmap="gray")

    img_transform, label_transform = img_transforms(image, label)
    print('img_transform shape:{}, label_transform shape:{}'.format(img_transform.shape, label_transform.shape))
    toImg = T.ToPILImage()(img_transform)
    toLabel = T.ToPILImage()(label_transform)
    plt.subplot(2, 2, 3)
    plt.imshow(toImg)
    plt.subplot(2, 2, 4)
    plt.imshow(toLabel, cmap="gray")
    plt.show()
    # """


    # 对图片进行测试实例
    """
    image_path = r'D:\githubCode\Ethan_project\deep_learning_project\images\2007_007748.jpg'
    img_transforms = img_Compose([
        # img_Resize(224),
        img_RandomCrop(224),
        img_ToTensor()
    ])

    image = Image.open(image_path).convert('RGB')
    print('image size:', image.size)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    img_transform = img_transforms(image)
    print('transform img size:', img_transform.shape)
    toImg = T.ToPILImage()(img_transform)
    plt.subplot(1, 2, 2)
    plt.imshow(toImg)
    plt.show()
    """
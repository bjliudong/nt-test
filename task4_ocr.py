import torch
from torchvision import  transforms
from PIL import Image
import  os, cv2
from task4_train import Net


# 定义卷积神经网络模型（此处省略定义，使用之前提供的Net类）

# 定义图像预处理函数
def preprocess_image(digit_image):
    try:
        # Convert the NumPy array back to a PIL Image
        image = Image.fromarray(digit_image).convert('L')  # Ensure it's grayscale
        image = image.resize((28, 28))  # Resize to 28x28
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# 图像分割函数
def split_digits(image_path):
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 应用高斯模糊，减少图像噪声
    image = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cropped_digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        digit = image[y:y+h, x:x+w]
        cropped_digits.append(digit)
    return cropped_digits

# 加载模型权重
model_path = os.getcwd() + os.path.sep + 'model/mynet.bin'
if os.path.isfile(model_path):
    model = Net()  # 实例化模型
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式
else:
    print("模型权重文件不存在，请输入正确的文件路径。")
    exit()

# 对每个图像进行分割和预测
for pic_num in range(1, 15):
    image_path = os.getcwd() + os.path.sep + 'pic' + os.path.sep + str(pic_num) + '.jpg'
    digits = split_digits(image_path)  # 分割图像中的每个数字
    for i, digit_image in enumerate(digits):
        image_tensor = preprocess_image(digit_image)
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
        print(f'Image {image_path}, Digit {i+1} predicted number: {predicted.item()}')

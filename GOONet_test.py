import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# Inception模块定义
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)

# GoogleNet模型定义
class GoogleNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(GoogleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)  # Dropout层，防止过拟合
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)  # 应用Dropout
        x = self.fc(x)
        return x



# 图片预处理
def preprocess_image(image_path, im_size=(150, 150)):
    preprocess = transforms.Compose([
        transforms.Resize(im_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = preprocess(image).unsqueeze(0)  # 添加batch维度
    return image



class_names = ['丁', '三', '上', '不', '丑', '专', '且', '丘', '丙', '东', '中', '丰', '乂', '乃', '之', '乍', '乎',
               '乘', '乙', '九', '争', '事', '于', '五', '井', '亘', '亚', '亡', '亥', '亦', '京', '亳', '人', '今',
               '介', '从', '令', '以', '伊', '伐', '休', '众', '何', '余', '侯', '侵', '保', '允', '元', '兄', '先',
               '光', '克', '兕', '入', '八', '六', '兮', '兴', '典', '内', '册', '冓', '凤', '出', '函', '刈', '刍',
               '刖', '利', '劦', '勿', '匕', '北', '卅', '升', '午', '卒', '南', '占', '卢', '卣', '卫', '卯', '印',
               '即', '去', '叀', '又', '及', '取', '受', '古', '只', '召', '可', '史', '司', '各', '合', '吉', '后',
               '启', '告', '周', '品', '唐', '商', '啬', '喜', '因', '圉', '土', '埶', '執', '壬', '声', '壴', '壶',
               '夒', '夕', '夙', '多', '大', '天', '奚', '奠', '女', '好', '妣', '妹', '妻', '子', '学', '宁', '它',
               '安', '宋', '宗', '宜', '室', '宫', '宰', '家', '宾', '宿', '寅', '寧', '寻', '封', '射', '尊', '小',
               '尸', '尹', '屯', '岁', '岳', '工', '巳', '帚', '帝', '年', '并', '庚', '庶', '康', '庸', '廿', '弋',
               '弓', '弗', '弘', '弜', '归', '彘', '彝', '彭', '得', '御', '戉', '戊', '戌', '成', '我', '戠', '才',
               '执', '敝', '文', '新', '方', '旅', '族', '既', '日', '旦', '旧', '旨', '旬', '昃', '明', '昏', '易',
               '昔', '星', '春', '昷', '暴', '曰', '月', '朕', '望', '未', '束', '来', '梦', '橐', '正', '此', '步',
               '武', '死', '母', '每', '毓', '氏', '水', '永', '求', '沈', '沚', '河', '泉', '渔', '温', '湄', '火',
               '灾', '焚', '燕', '爵', '父', '爽', '牛', '牝', '牡', '牢', '牧', '犬', '率', '玉', '王', '生', '用',
               '田', '甲', '申', '疑', '疾', '登', '白', '百', '皆', '皿', '盂', '盟', '目', '盾', '省', '眉', '眔',
               '瞽', '石', '磬', '示', '祀', '祈', '祖', '祝', '祭', '祼', '禦', '禽', '禾', '秋', '秦', '竞', '箕',
               '置', '羊', '羌', '美', '羞', '羽', '翊', '翌', '翦', '老', '肩', '育', '臣', '自', '至', '舂', '舌',
               '舞', '舟', '般', '良', '若', '莫', '葬', '蒿', '蔑', '虎', '行', '衍', '衣', '袁', '西', '见', '言',
               '豕', '豚', '象', '豹', '賓', '贞', '车', '辛', '辟', '辰', '追', '逆', '逐', '遘', '邑', '酉', '酒',
               '量', '阜', '降', '陷', '隹', '雀', '集', '雉', '雍', '雨', '雩', '雪', '雷', '非', '韦', '食', '首',
               '马', '骨', '高', '鬯', '鬲', '鬼', '鸟', '鸣', '鹿', '麋', '麓', '麦', '黄', '黍', '鼎', '鼓', '齿',
               '龙', '龠']

if __name__ == "__main__":
    # 加载模型和权重
    num_classes = 376  # 替换为您的分类数
    model = GoogleNet(num_classes=num_classes)
    model.load_state_dict(torch.load('./GoogleNet.pt'))
    model.eval()  # 设置模型为评估模式

    image_path = './dataset/modify/率/43_F145-4.jpg'  # 替换为实际图片路径
    input_image = preprocess_image(image_path)

    # 推理并输出结果
    with torch.no_grad():
        output = model(input_image)
        _, predicted = torch.max(output, 1)

    # 输出分类结果
    class_idx = predicted.item()
    print(f'Predicted class index: {class_idx}')

    # 如果有类别名称，可以映射索引到类别名称
    predicted_class = class_names[class_idx]
    print(f'Predicted class: {predicted_class}')

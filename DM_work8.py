import os
import pickle
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
from torchvision import transforms
from PIL import Image
import streamlit as st


# 加载CIFAR-100标签列表
def load_cifar100_meta(file_path):
    try:
        with open(file_path, 'rb') as fo:
            cifar100_meta = pickle.load(fo, encoding='bytes')
        return cifar100_meta
    except (PermissionError, FileNotFoundError) as e:
        print(f"加载CIFAR-100标签列表时出现错误：{e}")
        return None

cifar100_meta = load_cifar100_meta('D:/python_project/pycharm/pythonProject/cifar-100-python/meta')
if cifar100_meta:
    fine_label_names = cifar100_meta[b'fine_label_names']
else:
    fine_label_names = []

# 定义特征提取函数
def extract_LBP_feature(image):
    lbp = local_binary_pattern(image, 8, 1, method='default')
    return lbp.flatten()


def extract_HOG_feature(image):
    hog_feature = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                      block_norm='L2-Hys', visualize=False)
    return hog_feature


def extract_SIFT_feature(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is not None:
        return descriptors
    else:
        return np.zeros((1, 128))


def fit_kmeans(data, n_clusters, max_len):
    transformed_data = []
    for x in data:
        n_samples = len(x)
        if n_samples < n_clusters:
            n_clusters = n_samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(x)
        cluster_centers = kmeans.cluster_centers_
        if len(cluster_centers) < max_len:
            padding = np.zeros((max_len - len(cluster_centers), 128))
            cluster_centers = np.vstack((cluster_centers, padding))
        transformed_data.append(cluster_centers.flatten())
    return np.array(transformed_data)


# 加载机器学习模型
model_path = 'D:/python_project/pycharm/pythonProject/'
with open(os.path.join(model_path, 'nb_classifier_lbp.pkl'), 'rb') as file:
    nb_classifier_lbp = pickle.load(file)
with open(os.path.join(model_path, 'knn_classifier_lbp.pkl'), 'rb') as file:
    knn_classifier_lbp = pickle.load(file)
with open(os.path.join(model_path, 'lr_classifier_lbp.pkl'), 'rb') as file:
    lr_classifier_lbp = pickle.load(file)
with open(os.path.join(model_path, 'nb_classifier_hog.pkl'), 'rb') as file:
    nb_classifier_hog = pickle.load(file)
with open(os.path.join(model_path, 'knn_classifier_hog.pkl'), 'rb') as file:
    knn_classifier_hog = pickle.load(file)
with open(os.path.join(model_path, 'lr_classifier_hog.pkl'), 'rb') as file:
    lr_classifier_hog = pickle.load(file)
with open(os.path.join(model_path, 'nb_classifier_sift.pkl'), 'rb') as file:
    nb_classifier_sift = pickle.load(file)
with open(os.path.join(model_path, 'knn_classifier_sift.pkl'), 'rb') as file:
    knn_classifier_sift = pickle.load(file)
with open(os.path.join(model_path, 'lr_classifier_sift.pkl'), 'rb') as file:
    lr_classifier_sift = pickle.load(file)


# 定义神经网络模型
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 16 * 16, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        return x


# 加载神经网络模型权重
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet().to(device)
model.load_state_dict(torch.load(os.path.join(model_path, 'best_model.pth')))
model.eval()

# 定义图像预处理函数
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)


def preprocess_image_gray(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    return image


# 对上传图像文件进行分类
def classify_image(image_path, algorithm):
    # 预处理图像
    image_tensor = preprocess_image(image_path).to(device)
    gray_image = preprocess_image_gray(image_path)

    # 提取特征
    lbp_feature = extract_LBP_feature(gray_image)
    hog_feature = extract_HOG_feature(gray_image)
    sift_feature = extract_SIFT_feature(gray_image)
    max_len = 69  # 确保特征数量一致
    sift_feature_flat = fit_kmeans([sift_feature], max_len, max_len).flatten()

    # 标准化特征
    scaler_lbp = StandardScaler()
    lbp_feature = scaler_lbp.fit_transform([lbp_feature])
    scaler_hog = StandardScaler()
    hog_feature = scaler_hog.fit_transform([hog_feature])
    scaler_sift = StandardScaler()
    sift_feature_flat = scaler_sift.fit_transform([sift_feature_flat])

    # 使用机器学习模型进行分类
    if algorithm == 'LBP + Naive Bayes':
        prediction = nb_classifier_lbp.predict(lbp_feature)[0]
    elif algorithm == 'LBP + KNN':
        prediction = knn_classifier_lbp.predict(lbp_feature)[0]
    elif algorithm == 'LBP + Logistic Regression':
        prediction = lr_classifier_lbp.predict(lbp_feature)[0]
    elif algorithm == 'HOG + Naive Bayes':
        prediction = nb_classifier_hog.predict(hog_feature)[0]
    elif algorithm == 'HOG + KNN':
        prediction = knn_classifier_hog.predict(hog_feature)[0]
    elif algorithm == 'HOG + Logistic Regression':
        prediction = lr_classifier_hog.predict(hog_feature)[0]
    elif algorithm == 'SIFT + Naive Bayes':
        prediction = nb_classifier_sift.predict(sift_feature_flat)[0]
    elif algorithm == 'SIFT + KNN':
        prediction = knn_classifier_sift.predict(sift_feature_flat)[0]
    elif algorithm == 'SIFT + Logistic Regression':
        prediction = lr_classifier_sift.predict(sift_feature_flat)[0]
    elif algorithm == 'Neural Network':
        with torch.no_grad():
            nn_output = model(image_tensor)
            prediction = nn_output.argmax(dim=1, keepdim=True).item()
    else:
        prediction = None

    return prediction


# Streamlit 应用
st.title("2109120135 吴骄昊")
st.write("图像分类应用")
uploaded_file = st.file_uploader("上传图像", type=["jpg", "jpeg", "png"])
algorithm = st.selectbox("选择算法", [
    'LBP + Naive Bayes', 'LBP + KNN', 'LBP + Logistic Regression',
    'HOG + Naive Bayes', 'HOG + KNN', 'HOG + Logistic Regression',
    'SIFT + Naive Bayes', 'SIFT + KNN', 'SIFT + Logistic Regression',
    'Neural Network'
])

if uploaded_file is not None:
    # 确保上传目录存在
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    # 保存上传的文件
    image_path = os.path.join("uploads", uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    # 显示图像
    st.image(image_path, caption="上传的图像", use_column_width=True)
    # 分类
    if st.button("开始分类"):
        prediction = classify_image(image_path, algorithm)
        cifar100_index = fine_label_names[prediction].decode('utf-8')
        st.write(f"分类结果: {cifar100_index}")
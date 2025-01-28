import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def recognize_text(image_path: str) -> str:
    # 加载模型和分词器
    model = AutoModel.from_pretrained(
        "OpenGVLab/InternVL2-1B",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "OpenGVLab/InternVL2-1B",
        trust_remote_code=True,
        use_fast=False
    )

    # 加载和预处理图片
    pixel_values = load_image(image_path, max_num=12).to(torch.float16).to(model.device)
    print(f"图像张量形状: {pixel_values.shape}")

    # 准备提示词
    question = "<image>\n这是一张验证码图片。请直接告诉我图片中的字符内容，不要添加任何其他文字。"

    # 生成配置
    generation_config = dict(
        max_new_tokens=32,
        do_sample=True,
        temperature=0.1
    )

    # 生成回答
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    
    # 清理输出，确保只返回识别结果或"看不清楚"
    response = response.strip('。.,:：、\n \t')
    
    # 只保留数字和大写字母
    filtered_response = ''.join(c for c in response if c.isdigit() or c.isupper())
    
    if len(filtered_response) < 4 or len(filtered_response) > 6:
        return "看不清楚"
    
    return filtered_response

if __name__ == "__main__":
    # 使用示例
    result = recognize_text("test.PNG")
    print(result) 
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image

import gradio as gr
from torchvision import transforms

from transformers import (
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
)

# =========================================
# 0. 경로 / 디바이스 설정
# =========================================
CLIP_EMBED_PATH = "multimodal_assets/clip_text_embeds.pt"
MODEL_WEIGHTS_PATH = "models/convnext_base_merged_ema.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Device:", device)

# =========================================
# 1. 병합 클래스 이름 & CLIP 텍스트 임베딩 로드
# =========================================
print(" CLIP 텍스트 임베딩 로드 중...")
clip_data = torch.load(CLIP_EMBED_PATH)

merged_class_names = clip_data["class_names"]  # 17개 병합 클래스 이름
clip_prompts = clip_data["prompts"]
text_embeds = clip_data["text_embeds"]  # [17, D]
clip_model_name = clip_data["clip_model_name"]

# 텍스트 임베딩을 디바이스로 올리기
text_embeds = text_embeds.to(device)

print("병합 클래스 수:", len(merged_class_names))
print("병합 클래스 목록:", merged_class_names)

# =========================================
# 2. ConvNeXt-Base 분류 모델 로드
# =========================================
print(" ConvNeXt-Base 모델 로드 중 (timm)...")
num_classes = len(merged_class_names)

convnext_model = timm.create_model(
    "convnext_base",
    pretrained=False,
    num_classes=num_classes,
)
state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location="cpu")
convnext_model.load_state_dict(state_dict)
convnext_model.to(device)
convnext_model.eval()

print(" ConvNeXt-Base 학습 가중치 로드 완료")

# ConvNeXt용 전처리 (검증용)
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# =========================================
# 3. CLIP 모델 로드
# =========================================
print(f" CLIP 모델 로드 중... ({clip_model_name})")
clip_model = CLIPModel.from_pretrained(clip_model_name)
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

clip_model.to(device)
clip_model.eval()

# =========================================
# 4. BLIP 캡션 모델 로드
# =========================================
print(" BLIP 캡션 모델 로드 중... (Salesforce/blip-image-captioning-base)")
blip_model_name = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_model_name)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device)
blip_model.eval()

# =========================================
# 5. 세부 메뉴 후보 / 칼로리 정보 정의
# =========================================

# 원래 27개 메뉴(세부 메뉴)
fine_grained_menus = [
    "간장돼불덮밥",
    "고추치킨카레동",
    "공기밥",
    "김치어묵우동",
    "닭강정",
    "돈까스오므라이스",
    "돈까스우동세트",
    "돈까스카레동",
    "등심돈까스",
    "마그마새우튀김알밥",
    "마그마치킨마요",
    "베이컨 알리오올리오",
    "삼겹된장짜글이",
    "삼겹살강된장비빔밥",
    "새우튀김알밥",
    "새우튀김우동",
    "소떡소떡",
    "신라면(계란)",
    "신라면(계란+치즈)",
    "양념치킨오므라이스",
    "어묵우동",
    "에비카레동",
    "오므라이스",
    "쫑쫑이덮밥",
    "치킨마요",
    "케네디소시지",
    "케네디소시지오므라이스",
]

# 병합 대분류 → 세부 메뉴 후보
merged_to_fine = {
    "오므라이스류": ["오므라이스", "돈까스오므라이스", "케네디소시지오므라이스"],
    "치킨마요류": ["치킨마요", "마그마치킨마요"],
    "새우튀김알밥류": ["새우튀김알밥", "마그마새우튀김알밥"],
    "라면류": ["신라면(계란)", "신라면(계란+치즈)"],
}

# 대표 세부 메뉴 (사용자가 선택 안 했을 때 기본값)
default_detail = {
    "오므라이스류": "오므라이스",
    "치킨마요류": "치킨마요",
    "새우튀김알밥류": "새우튀김알밥",
    "라면류": "신라면(계란)",
}

# 아주 대략적인 칼로리 테이블
calorie_table = {
    "간장돼불덮밥": 800,
    "고추치킨카레동": 900,
    "공기밥": 300,
    "김치어묵우동": 500,
    "닭강정": 450,
    "돈까스오므라이스": 950,
    "돈까스우동세트": 900,
    "돈까스카레동": 900,
    "등심돈까스": 700,
    "마그마새우튀김알밥": 800,
    "마그마치킨마요": 850,
    "베이컨 알리오올리오": 800,
    "삼겹된장짜글이": 750,
    "삼겹살강된장비빔밥": 800,
    "새우튀김알밥": 750,
    "새우튀김우동": 550,
    "소떡소떡": 450,
    "신라면(계란)": 570,
    "신라면(계란+치즈)": 630,
    "양념치킨오므라이스": 950,
    "어묵우동": 450,
    "에비카레동": 800,
    "오므라이스": 730,
    "쫑쫑이덮밥": 700,
    "치킨마요": 800,
    "케네디소시지": 280,
    "케네디소시지오므라이스": 1000,
}

# =========================================
# 6. 유틸 함수들
# =========================================

def predict_convnext(image: Image.Image):
    """ConvNeXt-Base로 병합 대분류 예측"""
    convnext_model.eval()
    img_t = val_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = convnext_model(img_t)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

    top1_idx = int(np.argmax(probs))
    top1_prob = float(probs[top1_idx])

    # Top-3도 보고싶으면:
    top3_idx = np.argsort(probs)[::-1][:3]
    top3 = [(merged_class_names[i], float(probs[i])) for i in top3_idx]

    return merged_class_names[top1_idx], top1_prob, top3


def recommend_with_clip(image: Image.Image, top_k=3):
    """CLIP으로 병합 대분류 기준 유사 메뉴 Top-K"""
    clip_model.eval()

    inputs = clip_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        img_feat = clip_model.get_image_features(**inputs)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        sims = (img_feat @ text_embeds.T).squeeze(0)  # [17]
        topk = sims.topk(top_k)

    indices = topk.indices.tolist()
    scores = topk.values.tolist()
    result = [(merged_class_names[i], float(s)) for i, s in zip(indices, scores)]
    return result


def generate_caption(image: Image.Image):
    """BLIP으로 이미지 캡션 생성"""
    blip_model.eval()
    inputs = blip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_new_tokens=20)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption


def calorie_comment(menu_name: str, activity: str):
    kcal = calorie_table.get(menu_name)
    if kcal is None:
        return "이 메뉴에 대한 칼로리 정보가 등록되어 있지 않습니다."

    base = f"예상 칼로리: 약 {kcal} kcal.\n"

    if activity == "거의 안 움직임":
        if kcal >= 900:
            return base + "오늘 활동량을 고려하면 꽤 높은 칼로리라서, 자주 먹기엔 부담될 수 있어요."
        elif kcal >= 600:
            return base + "적당한 편이지만, 간식이나 다른 식사와 함께라면 총량을 조금 신경 쓰면 좋겠어요."
        else:
            return base + "가벼운 편이라 큰 부담 없이 먹어도 괜찮은 수준이에요."
    elif activity == "보통 활동":
        if kcal >= 1000:
            return base + "활동량을 고려해도 꽤 든든한 한 끼라서, 다른 끼니는 조금 가볍게 구성하면 좋아요."
        elif kcal >= 700:
            return base + "하루 한 끼 메인으로 먹기 좋은 정도의 칼로리예요."
        else:
            return base + "조금 가벼운 편이라, 배가 빨리 꺼질 수는 있어요."
    else:  # 많이 움직임
        if kcal >= 1000:
            return base + "활동량이 많다면 이 정도 칼로리는 충분히 잘 쓰일 거예요!"
        elif kcal >= 700:
            return base + "운동 전후 한 끼로 적당한 수준의 에너지 공급이 될 것 같아요."
        else:
            return base + "활동량에 비해 조금 가벼운 편이라, 간단한 간식을 더 곁들여도 좋겠어요."


# =========================================
# 7. Gradio 웹앱 메인 함수
# =========================================

def analyze_menu(image, activity_level, detail_menu_choice):
    """
    image: 업로드된 이미지 (PIL)
    activity_level: 활동량 (라디오 버튼)
    detail_menu_choice: 사용자가 선택한 세부 메뉴 (드롭다운)
    """
    if image is None:
        return "이미지를 업로드해 주세요.", "", "", ""

    # 1) ConvNeXt로 병합 대분류 예측
    big_cls, big_prob, top3_conv = predict_convnext(image)

    # 2) 해당 대분류에 세부 후보가 있는지 확인
    fine_candidates = merged_to_fine.get(big_cls, [])

    # 3) 세부 메뉴 결정 로직
    if detail_menu_choice is not None and detail_menu_choice != "선택 안 함 (모델에 맡기기)":
        final_menu = detail_menu_choice
        detail_info = f"사용자가 직접 선택한 세부 메뉴: **{final_menu}**"
    else:
        # 사용자가 직접 선택 안 한 경우
        if big_cls in default_detail:
            final_menu = default_detail[big_cls]
            detail_info = (
                f"예측 대분류: **{big_cls}** (신뢰도: {big_prob*100:.2f}%)\n"
                f"세부 메뉴는 선택하지 않아, 대표 메뉴 **'{final_menu}'** 기준으로 칼로리를 안내합니다.\n"
                f"(선택 메뉴를 바꾸면 칼로리 문장이 달라질 수 있어요)"
            )
        else:
            # 대분류 자체가 이미 최종 메뉴인 경우
            final_menu = big_cls
            detail_info = f"예측 메뉴: **{final_menu}** (신뢰도: {big_prob*100:.2f}%)"

    # 4) CLIP Top-3 유사 병합 메뉴
    clip_top3 = recommend_with_clip(image, top_k=3)
    clip_text_lines = []
    for name, score in clip_top3:
        clip_text_lines.append(f"- {name} (유사도: {score:.4f})")
    clip_text = "\n".join(clip_text_lines)

    # 5) BLIP 캡션 생성
    caption = generate_caption(image)

    # 6) 칼로리 코멘트
    kcal_text = calorie_comment(final_menu, activity_level)

    # 7) 안내 문구 (세부 후보 보여주기)
    if fine_candidates:
        candidate_text = (
            f"이 이미지는 **'{big_cls}'**(으)로 분류되었습니다.\n\n"
            f"이 대분류에 해당하는 세부 메뉴 후보:\n" +
            "\n".join([f"- {m}" for m in fine_candidates]) +
            "\n\n위 드롭다운에서 세부 메뉴를 직접 선택하면 칼로리 안내가 더 정확해집니다."
        )
    else:
        candidate_text = f"이 이미지는 **'{big_cls}'**(으)로 분류되었고, 별도의 세부 메뉴 분기는 없는 카테고리입니다."

    # 최종 요약 메시지
    summary = (
        f"###  최종 메뉴 분석\n"
        f"- 예측 대분류: **{big_cls}** (신뢰도: {big_prob*100:.2f}%)\n"
        f"- 최종 기준 메뉴: **{final_menu}**\n"
        f"- 활동량: **{activity_level}**\n\n"
        f"###  세부 메뉴 정보\n{detail_info}\n\n"
        f"###  ConvNeXt Top-3 (병합 클래스 기준)\n" +
        "\n".join([f"- {name} ({p*100:.2f}%)" for name, p in top3_conv]) +
        "\n\n"
        f"###  CLIP 유사 메뉴 Top-3 (병합 클래스 기준)\n{clip_text}\n\n"
        f"###  BLIP 캡션 (영어)\n> {caption}\n\n"
        f"###  칼로리 & 활동량 코멘트\n{kcal_text}\n\n"
        f"---\n"
        f"{candidate_text}"
    )

    return summary, caption, clip_text, kcal_text


# =========================================
# 8. Gradio 인터페이스 정의
# =========================================

with gr.Blocks() as demo:
    gr.Markdown("##  학식 스캐너")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="메뉴 사진 업로드")

            activity_input = gr.Radio(
                choices=["거의 안 움직임", "보통 활동", "많이 움직임"],
                value="보통 활동",
                label="오늘 활동량",
            )

            detail_menu_input = gr.Dropdown(
                choices=["선택 안 함 (모델에 맡기기)"] + fine_grained_menus,
                value="선택 안 함 (모델에 맡기기)",
                label="세부 메뉴 (선택하면 칼로리 계산에 사용)",
            )

            run_btn = gr.Button("분석 실행 ")

        with gr.Column():
            summary_output = gr.Markdown(label="분석 결과 요약")
            caption_output = gr.Textbox(label="BLIP 캡션 (영어)", lines=2)
            clip_output = gr.Textbox(label="CLIP 유사 병합 메뉴 Top-3", lines=4)
            kcal_output = gr.Textbox(label="칼로리 코멘트", lines=3)

    run_btn.click(
        fn=analyze_menu,
        inputs=[img_input, activity_input, detail_menu_input],
        outputs=[summary_output, caption_output, clip_output, kcal_output],
    )

demo.launch()

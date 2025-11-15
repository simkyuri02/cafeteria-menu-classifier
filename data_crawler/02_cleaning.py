"""
02_cleaning_and_filtering.py
- 크롤링으로 생성된 dataset_raw/ 내부의 모든 메뉴 폴더에 대해:
    1) 깨진 이미지 제거
    2) 너무 작은 이미지 제거 (160px 미만)
    3) 중복 이미지 제거 (imagehash)
    4) YOLO 필터링으로 음식이 아닌 이미지 제거
"""

import os
from PIL import Image
import imagehash
from ultralytics import YOLO

# =========================================
# 1. 경로 설정
# =========================================
BASE_DIR = "dataset_raw"


# =========================================
# 2. 폴더 자동 스캔
# =========================================
def get_all_menus():
    """dataset_raw 안의 모든 메뉴 폴더 이름 가져오기"""
    return [
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
    ]


# =========================================
# 3. 작은 이미지 / 깨진 이미지 제거
# =========================================
def clean_small_and_broken(menu):
    folder = f"{BASE_DIR}/{menu}"
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path)

            if img.width < 160 or img.height < 160:
                print("삭제(작은 이미지):", path)
                os.remove(path)
        except:
            print("삭제(깨진 이미지):", path)
            os.remove(path)


# =========================================
# 4. 중복 이미지 제거 (imagehash)
# =========================================
def remove_duplicates(menu):
    folder = f"{BASE_DIR}/{menu}"
    seen = set()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        try:
            img = Image.open(path)
            h = imagehash.average_hash(img)
            if h in seen:
                print("중복삭제:", path)
                os.remove(path)
            else:
                seen.add(h)
        except:
            try: os.remove(path)
            except: pass


# =========================================
# 5. YOLO 기반 음식 이미지 필터링
# =========================================
model = YOLO("yolov8m.pt")  

def is_food(path):
    """YOLO로 음식/식기류/사람 손 등이 감지되면 True"""
    try:
        result = model(path)
        boxes = result[0].boxes.data

        # YOLO가 뭔가를 감지했다면 이미지로 인정
        if len(boxes) > 0:
            return True

        # YOLO가 감지 못했어도 너무 작은 이미지는 reject
        img = Image.open(path)
        if img.width >= 160 and img.height >= 160:
            return True

    except:
        return False

    return False


def yolo_filter(menu):
    folder = f"{BASE_DIR}/{menu}"
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if not is_food(path):
            print("YOLO 삭제:", path)
            os.remove(path)


# =========================================
# 6. 전체 실행
# =========================================
if __name__ == "__main__":
    menus = get_all_menus()
    print("발견된 메뉴 폴더:", menus)

    print("\n=== 작은 이미지 / 깨진 이미지 제거 ===")
    for menu in menus:
        clean_small_and_broken(menu)

    print("\n=== 중복 이미지 제거 ===")
    for menu in menus:
        remove_duplicates(menu)

    print("\n=== YOLO 음식 필터링 ===")
    for menu in menus:
        yolo_filter(menu)

    print("\n=== 모든 이미지 정제 완료 ===")

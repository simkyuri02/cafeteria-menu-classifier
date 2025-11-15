"""
01_crawling.py
- 학식 이미지 크롤링
- 메뉴별 목표 이미지 수 다르게 설정 (혼동 메뉴는 더 많이 수집)
- 폴더 내 현재 이미지 수 확인 후 필요한 만큼만 추가 다운로드
"""

import os
from icrawler.builtin import BingImageCrawler

# =========================================
# 1. 검색 키워드 매핑
# =========================================
search_map = {

    "간장돼불덮밥": [
        "soy sauce pork bulgogi rice bowl",
        "korean pork bulgogi rice bowl",
        "간장 돼지불고기 덮밥",
        "돼불 덮밥",
        "bulgogi rice bowl top view",
    ],

    "고추치킨카레동": [
        "spicy chicken curry rice bowl",
        "korean spicy chicken curry bowl",
        "매운 치킨 카레덮밥",
        "고추 치킨 카레동",
        "chicken curry rice bowl top view",
    ],

    "베이컨 알리오올리오": [
        "bacon aglio e olio pasta",
        "garlic oil pasta bacon",
        "알리오올리오 베이컨 파스타",
        "bacon pasta top view",
        "olive oil pasta bacon",
    ],

    "오므라이스": [
        "japanese omurice plated",
        "omelette rice dish top view",
        "오므라이스",
        "egg omelette rice bowl",
        "omurice korean style",
    ],

    "치킨마요": [
        "chicken mayo rice bowl",
        "korean chicken mayo donburi",
        "치킨마요 덮밥",
        "chicken mayo rice top view",
        "korean mayo rice bowl chicken",
    ],

    "새우튀김우동": [
        "tempura udon bowl",
        "ebi tempura udon soup",
        "새우튀김 우동",
        "udon noodle soup prawn tempura",
        "tempura udon japanese noodle soup",
    ],

    "에비카레동": [
        "ebi curry rice bowl",
        "fried shrimp curry rice",
        "새우튀김 카레동",
        "japanese shrimp curry bowl",
        "katsu ebi curry rice top view",
    ],

    "신라면(계란)": [
        "ramen egg bowl top view",
        "korean spicy ramen with egg",
        "ramyun egg soup",
        "신라면 계란",
        "ramen egg korean style",
    ],

    "신라면(계란+치즈)": [
        "cheese ramen egg bowl",
        "ramen cheese egg korean",
        "korean cheese ramyun with egg",
        "신라면 치즈 계란",
        "cheese egg ramen bowl top view",
    ],

    "케네디소시지": [
        "korean sausage hot bar",
        "fried sausage stick",
        "케네디 소세지",
        "korean street sausage skewer",
        "grilled sausage korean style",
    ],

    "케네디소시지오므라이스": [
        "omurice with sausage",
        "omelette rice with sausage korean",
        "케네디 소시지 오므라이스",
        "sausage omurice top view",
        "omurice plate korean style sausage",
    ],

    "공기밥": [
        "korean steamed rice bowl",
        "white rice bowl",
        "공기밥",
        "steamed rice top view",
        "rice bowl korean",
    ],

    "김치어묵우동": [
        "kimchi fishcake udon",
        "korean kimchi eomuk udon",
        "김치 어묵 우동",
        "kimchi udon bowl",
        "udon noodle soup kimchi fishcake",
    ],

    "새우튀김알밥": [
        "shrimp egg rice bowl korean",
        "shrimp mayo egg rice bowl",
        "새우튀김 알밥",
        "ebi egg rice bowl",
        "rice bowl with shrimp egg topping",
    ],

    "마그마새우튀김알밥": [
        "spicy shrimp egg rice bowl korean",
        "korean spicy shrimp rice bowl",
        "마그마 새우튀김 알밥",
        "hot spicy shrimp rice bowl egg",
        "붉은 매운 새우 알밥",
    ],

    "등심돈까스": [
        "pork loin katsu",
        "tonkatsu pork cutlet",
        "등심 돈까스",
        "fried pork cutlet plated",
        "tonkatsu dish top view",
    ],

    "돈까스오므라이스": [
        "omurice with katsu",
        "katsu omurice rice dish",
        "돈까스 오므라이스",
        "egg omurice pork cutlet",
        "omurice katsu plate",
    ],

    "돈까스우동세트": [
        "katsu udon set",
        "tonkatsu and udon set meal",
        "돈까스 우동 세트",
        "udon noodle soup with pork cutlet",
        "japanese udon katsu set",
    ],

    "돈까스카레동": [
        "katsu curry rice bowl",
        "pork cutlet curry donburi",
        "돈까스 카레동",
        "japanese curry rice katsu",
        "tonkatsu curry bowl top view",
    ],

    "닭강정": [
        "korean sweet spicy fried chicken",
        "dakgangjeong",
        "닭강정",
        "korean glazed fried chicken",
        "dak gangjeong plate top view",
    ],

    "양념치킨오므라이스": [
        "omurice with korean spicy chicken",
        "yangnyeom chicken omurice",
        "양념치킨 오므라이스",
        "fried chicken omurice",
        "omurice korean spicy chicken",
    ],

    "쫑쫑이덮밥": [
        "garlic stem stir fry rice bowl",
        "maneuljong rice bowl",
        "마늘쫑 덮밥",
        "garlic scape rice bowl",
        "korean garlic stem donburi",
    ],

    "삼겹된장짜글이": [
        "pork soybean stew bowl korean",
        "doenjang jjigae pork bowl",
        "삼겹 된장 짜글이",
        "thick doenjang stew with pork",
        "korean thick soybean paste stew",
    ],

    "삼겹살강된장비빔밥": [
        "pork soybean paste mixed rice",
        "korean gang doenjang bibimbap",
        "삼겹살 강된장 비빔밥",
        "soybean paste pork bibimbap",
        "bibimbap pork doenjang bowl",
    ],

    "소떡소떡": [
        "korean sausage rice cake skewer",
        "sotteok sotteok",
        "소떡소떡",
        "korean street food sausage tteok",
        "korean skewer sausage ricecake",
    ],

    "어묵우동": [
        "fishcake udon bowl",
        "korean eomuk udon",
        "어묵 우동",
        "udon noodle soup fishcake",
        "fishcake japanese udon",
    ],

    "마그마치킨마요": [
        "spicy chicken mayo rice bowl",
        "korean spicy chicken mayo bowl",
        "마그마 치킨마요",
        "hot spicy chicken mayo rice",
        "korean spicy mayo chicken rice",
    ],

    "간장돼불덮밥": [
        "soy sauce pork bulgogi rice bowl",
        "korean soy pork rice",
        "간장 돼지불고기 덮밥",
        "pork bulgogi rice top view",
        "soy pork stirfry rice bowl"
    ],

}

# =========================================
# 2. 폴더 설정
# =========================================
BASE_DIR = "dataset_raw"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================================
# 3. 혼동되는(유사한) 메뉴 — 더 많이 모을 타깃
# =========================================
FOCUS_MENUS = [
    "신라면(계란)", "신라면(계란+치즈)",
    "치킨마요", "마그마치킨마요",
    "새우튀김알밥", "마그마새우튀김알밥",
    "돈까스오므라이스", "케네디소시지오므라이스",
    "돈까스우동세트", "돈까스카레동",
]

TARGET_BASE = 200   # 일반 메뉴 기본
TARGET_FOCUS = 300  # 헷갈리는 메뉴는 더 많이

def get_target(menu):
    return TARGET_FOCUS if menu in FOCUS_MENUS else TARGET_BASE

# =========================================
# 4. 크롤링 함수
# =========================================
def crawl_menu(menu, keywords):
    folder = f"{BASE_DIR}/{menu}"
    os.makedirs(folder, exist_ok=True)

    current = len(os.listdir(folder))
    target = get_target(menu)

    print(f"\n[크롤링] {menu} (현재 {current} / 목표 {target})")

    if current >= target:
        print(" → 이미 충분함, 스킵")
        return

    # 부족한 만큼 추가로 모으기
    need = target - current
    max_num = max(60, min(200, need * 2))

    for kw in keywords:
        print("  - 검색:", kw)
        crawler = BingImageCrawler(storage={"root_dir": folder})
        crawler.crawl(keyword=kw, max_num=max_num)

# =========================================
# 5. 전체 실행
# =========================================
if __name__ == "__main__":
    for menu, keywords in search_map.items():
        crawl_menu(menu, keywords)

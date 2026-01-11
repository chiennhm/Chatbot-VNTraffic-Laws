# -*- coding: utf-8 -*-
"""
Crawl văn bản pháp luật giao thông từ thuvienphapluat.vn
VERSION 3.0 - CHỈ TIẾNG VIỆT

Chỉ lấy nội dung tiếng Việt, loại bỏ:
- Bản dịch tiếng Anh
- Thông báo đăng nhập/đăng ký
- Quảng cáo và boilerplate
"""

import os
import re
import time
import requests
from bs4 import BeautifulSoup

# Document URLs
DOCUMENT_URLS = [
    # === LUẬT ===
    ("Luat_Trat_tu_an_toan_giao_thong_duong_bo_2024", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-trat-tu-an-toan-giao-thong-duong-bo-2024-so-36-2024-QH15-444251.aspx"),
    ("Luat_Duong_Bo_2024", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Luat-Duong-bo-2024-588811.aspx"),
    
    # === NGHỊ ĐỊNH ===
    ("ND_168_2024_Xu_phat_vi_pham_hanh_chinh_ATGT", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-168-2024-ND-CP-xu-phat-vi-pham-hanh-chinh-an-toan-giao-thong-duong-bo-619502.aspx"),
    ("ND_165_2024_Huong_dan_Luat_Duong_bo", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-165-2024-ND-CP-huong-dan-Luat-Duong-bo-va-Dieu-77-Luat-Trat-tu-an-toan-giao-thong-duong-bo-623287.aspx"),
    ("ND_166_2024_Kiem_dinh_xe_co_gioi", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-166-2024-ND-CP-dieu-kien-kinh-doanh-dich-vu-kiem-dinh-xe-co-gioi-623277.aspx"),
    ("ND_160_2024_Dao_tao_sat_hach_lai_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-160-2024-ND-CP-quy-dinh-ve-hoat-dong-dao-tao-va-sat-hach-lai-xe-624017.aspx"),
    ("ND_161_2024_Hang_hoa_nguy_hiem", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-161-2024-ND-CP-Danh-muc-hang-hoa-nguy-hiem-van-chuyen-hang-hoa-nguy-hiem-623658.aspx"),
    ("ND_156_2024_Dau_gia_bien_so_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-156-2024-ND-CP-Quy-dinh-dau-gia-bien-so-xe-635371.aspx"),
    ("ND_151_2024_Huong_dan_Luat_TTATGT", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-151-2024-ND-CP-huong-dan-Luat-Trat-tu-an-toan-giao-thong-duong-bo-619564.aspx"),
    ("ND_158_2024_Van_tai_duong_bo", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Nghi-dinh-158-2024-ND-CP-quy-dinh-hoat-dong-van-tai-duong-bo-636875.aspx"),
    ("ND_130_2024_Thu_phi_duong_cao_toc", "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Nghi-dinh-130-2024-ND-CP-quy-dinh-thu-phi-su-dung-duong-bo-cao-toc-thuoc-so-huu-toan-dan-621998.aspx"),
    ("ND_119_2024_Thanh_toan_dien_tu_giao_thong", "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Nghi-dinh-119-2024-ND-CP-thanh-toan-dien-tu-giao-thong-duong-bo-626100.aspx"),
    
    # === THÔNG TƯ - BỘ GIAO THÔNG VẬN TẢI ===
    ("TT_35_2024_BGTVT_Dao_tao_sat_hach_cap_GPLX", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-35-2024-TT-BGTVT-dao-tao-sat-hach-cap-giay-phep-lai-xe-quoc-te-624356.aspx"),
    ("TT_36_2024_BGTVT_Quan_ly_van_tai_o_to", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-36-2024-TT-BGTVT-quan-ly-hoat-dong-van-tai-bang-xe-o-to-va-hoat-dong-cua-ben-xe-633950.aspx"),
    ("TT_38_2024_BGTVT_Toc_do_khoang_cach_an_toan", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-38-2024-TT-BGTVT-toc-do-khoang-cach-an-toan-xe-co-gioi-tham-gia-giao-thong-duong-bo-622477.aspx"),
    ("TT_39_2024_BGTVT_Xe_qua_kho_qua_tai", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-39-2024-TT-BGTVT-luu-hanh-xe-qua-kho-gioi-han-xe-qua-tai-trong-xe-banh-xich-tren-duong-bo-633952.aspx"),
    ("TT_40_2024_BGTVT_Phong_chong_thien_tai", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-40-2024-TT-BGTVT-cong-tac-phong-chong-khac-phuc-hau-qua-thien-tai-linh-vuc-duong-bo-633320.aspx"),
    ("TT_41_2024_BGTVT_Quan_ly_ha_tang_duong_bo", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-41-2024-TT-BGTVT-quan-ly-van-hanh-khai-thac-bao-ve-ket-cau-ha-tang-duong-bo-623290.aspx"),
    ("TT_45_2024_BGTVT_Chung_chi_dang_kiem_vien", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-45-2024-TT-BGTVT-cap-moi-chung-chi-dang-kiem-vien-phuong-tien-giao-thong-duong-bo-635734.aspx"),
    ("TT_46_2024_BGTVT_Kiem_dinh_xe_co_gioi", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-46-2024-TT-BGTVT-thu-tuc-thu-hoi-giay-chung-nhan-hoat-dong-kiem-dinh-xe-co-gioi-635650.aspx"),
    ("TT_47_2024_BGTVT_Kiem_dinh_mien_kiem_dinh", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-47-2024-TT-BGTVT-thu-tuc-kiem-dinh-mien-kiem-dinh-lan-dau-cai-tao-xe-co-gioi-623286.aspx"),
    ("TT_49_2024_BGTVT_Trung_tam_sat_hach", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-49-2024-TT-BGTVT-Quy-chuan-ky-thuat-quoc-gia-Trung-tam-sat-hach-lai-xe-co-gioi-duong-bo-636740.aspx"),
    ("TT_50_2024_BGTVT_Co_so_dang_kiem", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-50-2024-TT-BGTVT-Quy-chuan-ky-thuat-quoc-gia-ve-co-so-bao-hanh-bao-duong-xe-co-gioi-635820.aspx"),
    ("TT_51_2024_BGTVT_Quy_chuan_bao_hieu_duong_bo", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-51-2024-TT-BGTVT-Quy-chuan-ky-thuat-quoc-gia-ve-bao-hieu-duong-bo-633562.aspx"),
    ("TT_52_2024_BGTVT_Yeu_cau_ky_thuat_xe_RD", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-52-2024-TT-BGTVT-yeu-cau-ky-thuat-xe-may-co-nhu-cau-tham-gia-giao-thong-duong-bo-622845.aspx"),
    ("TT_53_2024_BGTVT_Phan_loai_phuong_tien", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-53-2024-TT-BGTVT-phan-loai-phuong-tien-giao-thong-duong-bo-624420.aspx"),
    ("TT_54_2024_BGTVT_Chung_nhan_chat_luong_nhap_khau", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-54-2024-TT-BGTVT-trinh-tu-chung-nhan-chat-luong-an-toan-ky-thuat-xe-co-gioi-635647.aspx"),
    ("TT_55_2024_BGTVT_Chung_nhan_chat_luong_sx", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-55-2024-TT-BGTVT-trinh-tu-thu-tuc-chung-nhan-chat-luong-an-toan-ky-thuat-xe-co-gioi-635675.aspx"),
    ("TT_56_2024_BGTVT_Quy_chuan_ben_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-56-2024-TT-BGTVT-Quy-chuan-ky-thuat-ben-xe-khach-ben-xe-hang-tram-dung-nghi-636942.aspx"),
    ("TT_58_2024_BGTVT_Diem_dung_xe_cao_toc", "https://thuvienphapluat.vn/van-ban/Dau-tu/Thong-tu-58-2024-TT-BGTVT-dau-tu-diem-dung-xe-do-xe-va-vi-tri-quy-mo-diem-dung-xe-tren-duong-cao-toc-634169.aspx"),
    ("TT_34_2024_BGTVT_Tram_thu_phi", "https://thuvienphapluat.vn/van-ban/Thue-Phi-Le-Phi/Thong-tu-34-2024-TT-BGTVT-hoat-dong-tram-thu-phi-duong-bo-622720.aspx"),
    
    # === THÔNG TƯ - BỘ CÔNG AN ===
    ("TT_62_2024_BCA_Quy_chuan_he_thong_giam_sat", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-62-2024-TT-BCA-Quy-chuan-he-thong-giam-sat-bao-dam-an-toan-giao-thong-duong-bo-640193.aspx"),
    ("TT_65_2024_BCA_Phuc_hoi_diem_GPLX", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-65-2024-TT-BCA-kiem-tra-kien-thuc-phap-luat-de-duoc-phuc-hoi-diem-giay-phep-lai-xe-632409.aspx"),
    ("TT_69_2024_BCA_Chi_huy_dieu_khien_giao_thong", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-69-2024-TT-BCA-chi-huy-dieu-khien-giao-thong-duong-bo-cua-Canh-sat-giao-thong-632297.aspx"),
    ("TT_71_2024_BCA_He_thong_giam_sat_hanh_trinh", "https://thuvienphapluat.vn/van-ban/Cong-nghe-thong-tin/Thong-tu-71-2024-TT-BCA-quan-ly-he-thong-du-lieu-thiet-bi-giam-sat-hanh-trinh-nguoi-lai-xe-631648.aspx"),
    ("TT_72_2024_BCA_Dieu_tra_tai_nan_giao_thong", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-72-2024-TT-BCA-dieu-tra-giai-quyet-tai-nan-giao-thong-duong-bo-cua-Canh-sat-giao-thong-633011.aspx"),
    ("TT_73_2024_BCA_Tuan_tra_xu_ly_vi_pham_CSGT", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-73-2024-TT-BCA-xu-ly-vi-pham-trat-tu-an-toan-giao-thong-duong-bo-cua-Canh-sat-giao-thong-633438.aspx"),
    ("TT_79_2024_BCA_Dang_ky_xe_bien_so_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-79-2024-TT-BCA-cap-thu-hoi-chung-nhan-dang-ky-xe-bien-so-xe-co-gioi-xe-may-chuyen-dung-634265.aspx"),
    ("TT_81_2024_BCA_Quy_chuan_bien_so_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-81-2024-TT-BCA-Quy-chuan-ky-thuat-quoc-gia-ve-bien-so-xe-634495.aspx"),
    ("TT_82_2024_BCA_Chung_nhan_chat_luong_xe", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-82-2024-TT-BCA-chung-nhan-chat-luong-an-toan-ky-thuat-xe-co-gioi-trong-nhap-khau-san-xuat-634522.aspx"),
    ("TT_83_2024_BCA_He_thong_giam_sat_ATGT", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-83-2024-TT-BCA-su-dung-he-thong-giam-sat-bao-dam-an-ninh-giao-thong-duong-bo-632255.aspx"),
    
    # === THÔNG TƯ - BỘ QUỐC PHÒNG ===
    ("TT_66_2024_BQP_Kiem_dinh_xe_quan_su", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-66-2024-TT-BQP-kiem-dinh-an-toan-ky-thuat-bao-ve-moi-truong-xe-co-gioi-628579.aspx"),
    ("TT_67_2024_BQP_Chung_nhan_xe_nhap_khau", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-67-2024-TT-BQP-thu-tuc-chung-nhan-an-toan-ky-thuat-bao-ve-moi-truong-xe-co-gioi-nhap-khau-628580.aspx"),
    ("TT_68_2024_BQP_Dao_tao_GPLX_quan_su", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-68-2024-TT-BQP-dao-tao-sat-hach-cap-Giay-phep-lai-xe-quan-su-628772.aspx"),
    ("TT_69_2024_BQP_Dang_ky_xe_quan_su", "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Thong-tu-69-2024-TT-BQP-dang-ky-quan-ly-su-dung-xe-co-gioi-xe-may-chuyen-dung-628970.aspx"),
    ("TT_70_2024_BQP_Cai_tao_xe_quan_su", "https://thuvienphapluat.vn/van-ban/Tai-chinh-nha-nuoc/Thong-tu-70-2024-TT-BQP-cai-tao-xe-co-gioi-xe-may-chuyen-dung-628969.aspx"),
    ("TT_71_2024_BQP_Kiem_soat_xe_quan_su", "https://thuvienphapluat.vn/van-ban/Giao-thong-Van-tai/Thong-tu-71-2024-TT-BQP-kiem-soat-quan-su-kiem-tra-xe-quan-su-tham-gia-giao-thong-duong-bo-628971.aspx"),
    
    # === THÔNG TƯ - BỘ Y TẾ ===
    ("TT_36_2024_BYT_Tieu_chuan_suc_khoe_lai_xe", "https://thuvienphapluat.vn/van-ban/The-thao-Y-te/Thong-tu-36-2024-TT-BYT-tieu-chuan-suc-khoe-nguoi-dieu-khien-xe-may-chuyen-dung-622478.aspx"),
]

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'vi-VN,vi;q=0.9',
}

# ============================================================================
# TEXT CLEANING
# ============================================================================

def clean_text(text):
    """Remove non-Vietnamese content and boilerplate."""
    lines = text.split('\n')
    cleaned = []
    
    # Patterns to skip
    skip_patterns = [
        r'Hãy đăng nhập',
        r'đăng ký Thành viên',
        r'Pro tại đây',
        r'xem toàn bộ văn bản',
        r'\.\.\.+',  # ...
        r'^\s*\|\s*$',
        r'^\s*[-=]{10,}\s*$',
        r'^\*+$',
        r'NATIONAL\s+ASSEMBLY',
        r'SOCIALIST\s+REPUBLIC',
        r'Independence.*Freedom.*Happiness',
        r'^Pursuant to',
        r'^The National Assembly',
        r'hereinafter referred to',
        r'^Law\s*$',
        r'^No\.\s*\d+',
        r'^\*+\s*(Hanoi|Ha Noi)',
    ]
    
    for line in lines:
        stripped = line.strip()
        
        # Skip empty
        if not stripped:
            cleaned.append('')
            continue
        
        # Check skip patterns
        skip = False
        for pattern in skip_patterns:
            if re.search(pattern, stripped, re.IGNORECASE):
                skip = True
                break
        
        if skip:
            continue
        
        # Skip lines that are mostly English (ASCII letters > 80%)
        alpha_chars = [c for c in stripped if c.isalpha()]
        if len(alpha_chars) > 20:
            ascii_count = sum(1 for c in alpha_chars if ord(c) < 128)
            if ascii_count / len(alpha_chars) > 0.85:
                continue
        
        cleaned.append(line)
    
    # Combine and normalize
    text = '\n'.join(cleaned)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# ============================================================================
# CONTENT EXTRACTION
# ============================================================================

def extract_content(soup):
    """Extract main Vietnamese content."""
    
    # Remove scripts, styles, etc.
    for tag in soup.find_all(['script', 'style', 'iframe', 'noscript']):
        tag.decompose()
    
    # Try to find main content div
    content_div = None
    for selector in ['content1', 'toanvancontent', 'fulltext']:
        content_div = soup.find('div', class_=selector)
        if content_div:
            break
    
    if not content_div:
        content_div = soup.find('div', id='toanvancontent')
    
    if not content_div:
        # Fallback: find largest div
        all_divs = soup.find_all('div')
        max_len = 0
        for div in all_divs:
            text = div.get_text(strip=True)
            if len(text) > max_len:
                max_len = len(text)
                content_div = div
    
    if content_div:
        text = content_div.get_text(separator='\n')
    else:
        text = soup.get_text(separator='\n')
    
    return clean_text(text)

def extract_title(soup):
    """Get document title."""
    h1 = soup.find('h1')
    if h1:
        return h1.get_text(strip=True)
    title = soup.find('title')
    if title:
        return title.get_text(strip=True).split('|')[0].strip()
    return "Unknown"

# ============================================================================
# CRAWL
# ============================================================================

def crawl_document(name, url, output_dir):
    """Crawl one document."""
    output_file = os.path.join(output_dir, f"{name}.txt")
    
    print(f"[CRAWL] {name}")
    
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.encoding = 'utf-8'
        
        if r.status_code != 200:
            print(f"  [ERROR] HTTP {r.status_code}")
            return False
        
        soup = BeautifulSoup(r.text, 'html.parser')
        title = extract_title(soup)
        content = extract_content(soup)
        
        # Write file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# {title}\n\n")
            f.write(content)
        
        print(f"  [OK] {len(content):,} chars")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

def main():
    """Run crawler."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "documents_vn")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"So van ban: {len(DOCUMENT_URLS)}")
    print(f"Output: {output_dir}")
    print("=" * 50)
    
    success = 0
    
    for i, (name, url) in enumerate(DOCUMENT_URLS, 1):
        print(f"\n[{i}/{len(DOCUMENT_URLS)}]", end=" ")
        if crawl_document(name, url, output_dir):
            success += 1
        time.sleep(1.5)
    
    print(f"\n{'=' * 50}")
    print(f"XONG: {success}/{len(DOCUMENT_URLS)} van ban")
    print(f"Output: {output_dir}")
    print("=" * 50)

if __name__ == "__main__":
    main()

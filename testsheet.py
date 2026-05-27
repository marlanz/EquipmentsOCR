import json
import gspread
from google.oauth2 import service_account

# 1. Định nghĩa Scopes chuẩn theo tài liệu mới nhất của Google
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# 2. Xác thực bằng thư viện google-auth thay thế cho oauth2client
# Đảm bảo file 'credentials.json' nằm chung thư mục với file testsheet.py
creds = service_account.Credentials.from_service_account_file(
    "credentials.json", 
    scopes=SCOPES
)
client = gspread.authorize(creds)

# 3. Mở Google Sheet (Thay đúng tên file Google Sheet của bạn vào đây)
sheet_name = "OCR_EQ_PARSER" 
spreadsheet = client.open(sheet_name)
worksheet = spreadsheet.get_worksheet(0) # Mở tab đầu tiên

# 4. Cấu trúc dữ liệu OCR của bạn
json_data = """
{
  "results": [
    {
      "key_value": {
        "machine_name": "MÁY HÀN CO2 TÂN THÀNH - TTC-500T",
        "Mã MMTB": "B22400814",
        "Model": "TTC-500T",
        "Xưởng": "AH2",
        "Vị trí": "TỔ RÁP HOÀN THIỆN 2"
      }
    }
  ]
}
"""

data = json.loads(json_data)
# Lấy phần tử đầu tiên trong mảng results
kv = data["results"][0]["key_value"]

# 5. Sắp xếp các cột dữ liệu theo đúng ý bạn
row_to_insert = [
    kv.get("machine_name", ""),
    kv.get("Mã MMTB", ""),
    kv.get("Model", ""),
    kv.get("Xưởng", ""),
    kv.get("Vị trí", "")
]

# 6. Kiểm tra tiêu đề bảng (nếu trống thì chèn dòng đầu)
if not worksheet.get_all_values():
    headers = ["TÊN MMTB", "Mã MMTB", "MODEL", "XƯỞNG", "VỊ TRÍ"]
    worksheet.append_row(headers)

# 7. Đẩy dữ liệu OCR xuống dòng trống tiếp theo
worksheet.append_row(row_to_insert)

print("Đã sửa lỗi thành công! Dữ liệu đã được ghi lên Google Sheet.")

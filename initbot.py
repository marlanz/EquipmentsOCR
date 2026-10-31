import os
import json
import logging
import asyncio
import httpx
from typing import Optional

from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder,
    ApplicationHandlerStop,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from bot.services.sheet_service import SheetService
from bot.services.duplicate_service import DuplicateService
from app.schemas import OCRResult

sheet_service = SheetService()
duplicate_service = DuplicateService(sheet_service=sheet_service)
from app.config.factory_structure import FACTORY_STRUCTURE
import unicodedata
import re
import html
from app.utils.validators import validate_mmtb



# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ─────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
GEMINI_MODEL_2 = os.getenv("GEMINI_MODEL_2", "gemini-2.5-flash").strip()
GEMINI_MODEL_3 = os.getenv("GEMINI_MODEL_3", "gemini-2.5-pro").strip()
GEMINI_MODEL_4 = os.getenv("GEMINI_MODEL_4", "gemini-2.5-pro").strip()
GEMINI_MODEL_5 = os.getenv("GEMINI_MODEL_5", "gemini-2.5-pro").strip()

# Build API base URL.
# Priority: API_BASE_URL > OCR_API_URL (legacy, path stripped automatically)
API_BASE_URL = os.getenv("API_BASE_URL", "").strip().rstrip("/")
if not API_BASE_URL:
    _ocr_url = os.getenv("OCR_API_URL", "").strip().rstrip("/")
    for _suffix in ["/parse-text-gemini", "/parse-text"]:
        if _ocr_url.endswith(_suffix):
            _ocr_url = _ocr_url[: -len(_suffix)]
            break
    API_BASE_URL = _ocr_url

PADDLE_ENDPOINT = f"{API_BASE_URL}/parse-text"
GEMINI_ENDPOINT = f"{API_BASE_URL}/parse-text-gemini"

# ─────────────────────────────────────────────
# Conversation States
# ─────────────────────────────────────────────
SELECT_ENGINE, SELECT_STATUS, CONFIRM_RESULT, EDIT_FIELD, CONFIRM_SAVE, FIND_ASK_EDIT, MMTB_WARN, DUP_WARN = range(8)

# ─────────────────────────────────────────────
# Bot State Machine
# ─────────────────────────────────────────────
from enum import Enum

class BotState(str, Enum):
    NORMAL                      = "normal"
    MULTI_IMAGE_COLLECTING      = "multi_image_collecting"
    MULTI_IMAGE_SELECTING_MODEL = "multi_image_selecting_model"
    MULTI_IMAGE_PROCESSING      = "multi_image_processing"
    MULTI_IMAGE_REVIEW          = "multi_image_review"
    MULTI_IMAGE_SAVING          = "multi_image_saving"

# ─────────────────────────────────────────────
# Persistent ReplyKeyboard definitions
# (Pattern from testmenu.py)
# ─────────────────────────────────────────────
MAIN_MENU_KEYBOARD = ReplyKeyboardMarkup(
    [["📷 Đọc 1 hình", "📚 Đọc nhiều hình"]],
    resize_keyboard=True,
    is_persistent=True,
)

MULTI_IMAGE_KEYBOARD = ReplyKeyboardMarkup(
    [["↩️ Trở về", "▶️ Tiến hành đọc"]],
    resize_keyboard=True,
    is_persistent=True,
)

REMOVE_KEYBOARD = ReplyKeyboardRemove()


async def show_main_menu(message, text: str):
    """Send a message with the persistent main menu keyboard attached."""
    await message.reply_text(
        text,
        reply_markup=MAIN_MENU_KEYBOARD,
        parse_mode="HTML",
    )


async def show_multi_image_menu(message, text: str):
    """Send a message with the multi-image keyboard attached."""
    await message.reply_text(
        text,
        reply_markup=MULTI_IMAGE_KEYBOARD,
        parse_mode="HTML",
    )


async def hide_bottom_menu(message, text: str, parse_mode: str = "HTML"):
    """Send a message that removes the persistent ReplyKeyboard."""
    await message.reply_text(
        text,
        reply_markup=REMOVE_KEYBOARD,
        parse_mode=parse_mode,
    )


# Callback data for multi-image engine selection
CB_MULTI_PADDLE   = "multi_engine:paddle"
CB_MULTI_GEMINI_1 = "multi_engine:gemini1"
CB_MULTI_GEMINI_2 = "multi_engine:gemini2"
CB_MULTI_GEMINI_3 = "multi_engine:gemini3"
CB_MULTI_GEMINI_4 = "multi_engine:gemini4"
CB_MULTI_GEMINI_5 = "multi_engine:gemini5"

# Callback data for multi-image save confirmation
CB_MULTI_SAVE   = "multi_save:confirm"
CB_MULTI_CANCEL = "multi_save:cancel"
CFG_SELECT_FACTORY, CFG_SELECT_WORKSHOP, CFG_SELECT_LETTER, CFG_SELECT_NUMBER = range(8, 12)

# Callback data constants
CB_PADDLE          = "engine:paddle"
CB_GEMINI          = "engine:gemini"
CB_GEMINI_1        = "engine:gemini1"
CB_GEMINI_2        = "engine:gemini2"
CB_GEMINI_3        = "engine:gemini3"
CB_GEMINI_4        = "engine:gemini4"
CB_GEMINI_5        = "engine:gemini5"
CB_CONFIRM_YES     = "confirm:yes"
CB_CONFIRM_EDIT    = "confirm:edit"
CB_SAVE_CONFIRM    = "save:confirm"
CB_SAVE_EDIT       = "save:edit_again"
# /f find-and-edit flow
CB_FIND_EDIT_YES   = "find:edit_yes"
CB_FIND_EDIT_NO    = "find:edit_no"
CB_FIND_SAVE       = "find_save:confirm"
CB_FIND_EDIT_AGAIN = "find_save:edit_again"
# /n manual-add flow
CB_MANUAL_SAVE     = "manual_save:confirm"
CB_MANUAL_EDIT     = "manual_save:edit_again"
CB_SKIP_FIELD      = "edit:skip"
# OCR validation warning flow
CB_MMTB_FORCE_SAVE = "mmtb_warn:force_save"
CB_MMTB_EDIT       = "mmtb_warn:edit"
# Duplication validation flow
CB_DUP_FORCE_SAVE  = "dup_warn:force_save"
CB_DUP_CANCEL      = "dup_warn:cancel"

# Fields to loop through during correction: (display label, key_value dict key)
FIELDS = [
    ("🏷️ Tên thiết bị", "machine_name"),
    ("🔢 Mã MMTB",      "Mã MMTB"),
    ("📦 Model",        "Model"),
    ("🏭 Xưởng",        "Xưởng"),
    ("📍 Vị trí",       "Vị trí"),
    ("⚡ Trạng thái",    "status"),
]


# ─────────────────────────────────────────────
# Helper: get value from kv dict
# Tries exact key first, then case-insensitive fallback
# ─────────────────────────────────────────────
# def _get_kv(kv: dict, key: str) -> str:
#     val = kv.get(key, "").strip()
#     if not val:
#         for k, v in kv.items():
#             if k.lower().strip() == key.lower().strip():
#                 val = v.strip()
#                 break
#     return val or "—"

_KEY_ALIASES: dict[str, list[str]] = {
    "Model":   ["model", "mo hinh"],
    "Xưởng":  ["xuong", "xuong san xuat", "nha may"],
    "Vị trí": ["vi tri", "vi tri ", "vitri", "tri"],
    "Mã MMTB": ["ma mmtb", "ma may", "ma mmtb"],
}

def _normalize(s: str) -> str:
    """Strips diacritics and lowercases for fuzzy key matching."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower().strip()


def _get_kv(kv: dict, key: str) -> str:
    # 1. Exact match
    val = kv.get(key, "").strip()
    if val:
        return val

    # 2. Case-insensitive exact match
    for k, v in kv.items():
        if k.lower().strip() == key.lower().strip():
            return v.strip() or "—"

    # 3. Diacritic-normalized match (handles "Vị Tri" vs "Vị trí", "MODEL" vs "Model")
    norm_key = _normalize(key)
    for k, v in kv.items():
        if _normalize(k) == norm_key:
            return v.strip() or "—"

    # 4. Alias fallback
    aliases = _KEY_ALIASES.get(key, [])
    for k, v in kv.items():
        if _normalize(k) in aliases:
            return v.strip() or "—"

    return "—"


# ─────────────────────────────────────────────
# Persistent User Settings  (workshops.json)
# Key layout:
#   "<user_id>"       → workshop name  (legacy flat string)
#   "loc_<user_id>"   → location / Vị trí value
# ─────────────────────────────────────────────
WORKSHOP_FILE = "workshops.json"

def _load_settings() -> dict:
    if os.path.exists(WORKSHOP_FILE):
        try:
            with open(WORKSHOP_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error reading {WORKSHOP_FILE}: {e}")
    return {}

def _save_settings(data: dict):
    try:
        with open(WORKSHOP_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.error(f"Error writing to {WORKSHOP_FILE}: {e}")

# ── Workshop (Xưởng) ──────────────────────────
def save_user_workshop(user_id: int, workshop: str, context: Optional[ContextTypes.DEFAULT_TYPE] = None):
    data = _load_settings()
    data[str(user_id)] = workshop
    _save_settings(data)
    if context is not None:
        context.user_data["workshop"] = workshop

def get_user_workshop(user_id: int) -> str:
    return _load_settings().get(str(user_id), "")

# ── Location (Vị trí) ─────────────────────────
def save_user_location(user_id: int, location: str, context: Optional[ContextTypes.DEFAULT_TYPE] = None):
    data = _load_settings()
    data[f"loc_{user_id}"] = location
    _save_settings(data)
    if context is not None:
        context.user_data["current_position"] = location

def get_user_location(user_id: int) -> str:
    return _load_settings().get(f"loc_{user_id}", "")

# ── Configuration flow settings helper ───────
def save_user_config(user_id: int, factory: str, workshop: str, position: str, context: ContextTypes.DEFAULT_TYPE):
    data = _load_settings()
    data[f"fac_{user_id}"] = factory
    data[str(user_id)] = workshop
    data[f"loc_{user_id}"] = position
    _save_settings(data)
    
    context.user_data["factory"] = factory
    context.user_data["workshop"] = workshop
    context.user_data["current_position"] = position

def load_user_config(user_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Loads configured factory, workshop and location into context.user_data if missing."""
    if "workshop" not in context.user_data or "current_position" not in context.user_data:
        data = _load_settings()
        context.user_data["factory"] = data.get(f"fac_{user_id}", "")
        context.user_data["workshop"] = data.get(str(user_id), "")
        context.user_data["current_position"] = data.get(f"loc_{user_id}", "")


# ─────────────────────────────────────────────
# Helper: format OCR result as readable message
# ─────────────────────────────────────────────
def _format_result(kv: dict, markdown_text: str, engine_name: str, processing_time) -> str:
    msg  = f"📋 *Kết quả OCR* — _{engine_name}_\n"
    msg += f"⏱ Thời gian xử lý: `{processing_time}s`\n\n"
    for label, key in FIELDS:
        msg += f"{label}: *{_get_kv(kv, key)}*\n"
    # if markdown_text:
    #     snippet = markdown_text[:600]
    #     if len(markdown_text) > 600:
    #         snippet += "\n... (truncated)"
    #     msg += f"\n📝 *Văn bản gốc:*\n```\n{snippet}\n```"
    return msg


# ─────────────────────────────────────────────
# /ws command redirect
# ─────────────────────────────────────────────
async def ws_redirect_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🏭 Lệnh <code>/ws</code> đã được thay thế. Vui lòng sử dụng lệnh <code>/x</code> để thiết lập cấu hình Xưởng chính, Tổ/Xưởng và Vị trí thông qua màn hình lựa chọn.",
        parse_mode="HTML"
    )

# ─────────────────────────────────────────────
# /col or /l command — set persistent Vị trí
# ─────────────────────────────────────────────
async def set_location_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    match = re.match(r'^/(?:col|l)(?:\s+(.+))?$', text, re.IGNORECASE)

    location_value = match.group(1).strip() if (match and match.group(1)) else ""
    user_id = update.effective_user.id

    if location_value:
        save_user_location(user_id, location_value, context)
        safe_val = html.escape(location_value)
        await update.message.reply_text(
            f"📍 Đã thiết lập vị trí mặc định là: <b>{safe_val}</b>\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên vị trí khi OCR kết thúc.",
            parse_mode="HTML",
        )
        return ConversationHandler.END
    else:
        current_location = get_user_location(user_id)
        if current_location:
            safe_curr = html.escape(current_location)
            await update.message.reply_text(
                f"📍 Vị trí mặc định hiện tại của bạn: <b>{safe_curr}</b>\n\n"
                "Vui lòng nhập vị trí mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="HTML",
            )
        else:
            await update.message.reply_text(
                "📍 Bạn chưa thiết lập vị trí mặc định.\n\n"
                "Vui lòng nhập vị trí mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="HTML",
            )
        context.user_data["waiting_for_location"] = True
        return ConversationHandler.END


# ─────────────────────────────────────────────
# Location Adjustment Buttons Helpers & Callback
# ─────────────────────────────────────────────
def get_adjacent_positions(position: str) -> tuple[Optional[str], str, Optional[str]]:
    """
    Given a position string, returns (decrement_pos, keep_pos, increment_pos).
    If it cannot be incremented/decremented (e.g., does not end in a number),
    returns (None, keep_pos, None).
    """
    if not position:
        return None, "", None
    
    match = re.match(r"^([A-Za-z]+)\s*(\d+)$", position.strip())
    if not match:
        return None, position, None
        
    prefix = match.group(1).upper()
    num_str = match.group(2)
    num = int(num_str)
    
    dec_val = num - 1
    dec_pos = f"{prefix}{dec_val}" if dec_val > 0 else None
    inc_pos = f"{prefix}{num + 1}"
    
    return dec_pos, f"{prefix}{num}", inc_pos

def get_position_adjustment_keyboard(position: str) -> Optional[InlineKeyboardMarkup]:
    dec_pos, keep_pos, inc_pos = get_adjacent_positions(position)
    if not keep_pos:
        return None
        
    buttons = []
    if dec_pos:
        buttons.append(InlineKeyboardButton(f"⬅️ {dec_pos}", callback_data=f"pos_set:{dec_pos}"))
    buttons.append(InlineKeyboardButton(f"🆗 Giữ {keep_pos}", callback_data=f"pos_set:{keep_pos}"))
    if inc_pos:
        buttons.append(InlineKeyboardButton(f"➡️ {inc_pos}", callback_data=f"pos_set:{inc_pos}"))
        
    return InlineKeyboardMarkup([buttons])

async def handle_position_adjustment_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    chosen_pos = query.data.split(":", 1)[1]
    user_id = query.from_user.id
    
    save_user_location(user_id, chosen_pos, context)
    
    original_text = query.message.text
    
    success_header = "✅ <b>Đã lưu thông tin vào Google Sheets!</b>"
    if "Thông tin đã xác nhận" in original_text:
        success_header = "✅ <b>Thông tin đã xác nhận và lưu vào Google Sheets!</b>"
    elif "Dữ liệu đã sửa" in original_text:
        success_header = "✅ <b>Dữ liệu đã sửa được lưu thành công vào Google Sheets!</b>"
    elif "Đã cập nhật thành công mã MMTB" in original_text:
        match = re.search(r"mã MMTB (\S+)", original_text)
        if match:
            success_header = f"✅ <b>Đã cập nhật thành công mã MMTB <code>{html.escape(match.group(1))}</code> vào Google Sheets!</b>"
        else:
            success_header = "✅ <b>Đã cập nhật thành công mã MMTB vào Google Sheets!</b>"
            
    suffix = "Gửi ảnh mới hoặc dùng /f để tiếp tục." if "/f" in original_text else "Gửi ảnh mới để tiếp tục."
    
    safe_val = html.escape(chosen_pos)
    await query.edit_message_text(
        text=(
            f"{success_header}\n\n"
            f"📍 <b>Đã thiết lập vị trí mặc định mới:</b> <code>{safe_val}</code>\n\n"
            f"{suffix}"
        ),
        parse_mode="HTML"
    )


# ─────────────────────────────────────────────
# Structured Selection Flow keyboard builders
# ─────────────────────────────────────────────
from telegram import InlineKeyboardMarkup

def get_factory_keyboard() -> InlineKeyboardMarkup:
    """Generate Inline Keyboard for Factory selection."""
    buttons = []
    factories = list(FACTORY_STRUCTURE.keys())
    row = []
    for f in factories:
        row.append(InlineKeyboardButton(f, callback_data=f"cf_f:{f}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)

def get_workshop_keyboard(factory: str) -> InlineKeyboardMarkup:
    """Generate Inline Keyboard for Workshop selection dynamically based on Factory."""
    workshops = FACTORY_STRUCTURE.get(factory, [])
    buttons = []
    row = []
    for w in workshops:
        row.append(InlineKeyboardButton(w, callback_data=f"cf_w:{w}"))
        if len(row) == 2:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)

def get_letter_keyboard() -> InlineKeyboardMarkup:
    """Generate Inline Keyboard for Column Letter selection."""
    buttons = [
        [InlineKeyboardButton("A", callback_data="cf_l:A"), InlineKeyboardButton("B", callback_data="cf_l:B")],
        [InlineKeyboardButton("C", callback_data="cf_l:C"), InlineKeyboardButton("D", callback_data="cf_l:D")],
        [InlineKeyboardButton("E", callback_data="cf_l:E"), InlineKeyboardButton("F", callback_data="cf_l:F")],
    ]
    return InlineKeyboardMarkup(buttons)

def get_number_keyboard() -> InlineKeyboardMarkup:
    """Generate 5x5 Inline Keyboard for Column Number selection + back button."""
    buttons = []
    for r in range(5):
        row = []
        for c in range(1, 6):
            num = r * 5 + c
            row.append(InlineKeyboardButton(str(num), callback_data=f"cf_n:{num}"))
        buttons.append(row)
    buttons.append([InlineKeyboardButton("⬅️ Chọn lại chữ cái", callback_data="cf_back")])
    return InlineKeyboardMarkup(buttons)


# ─────────────────────────────────────────────
# Structured Selection Flow handler callbacks
# ─────────────────────────────────────────────
async def start_config_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /x configuration flow."""
    context.user_data.pop("temp_factory", None)
    context.user_data.pop("temp_workshop", None)
    context.user_data.pop("temp_letter", None)

    await update.message.reply_text(
        "🏭 <b>Chọn Xưởng chính (Factory):</b>",
        reply_markup=get_factory_keyboard(),
        parse_mode="HTML"
    )
    return CFG_SELECT_FACTORY

async def handle_cfg_factory_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores Factory choice and prompts Workshop choice."""
    query = update.callback_query
    await query.answer()

    factory = query.data.split(":", 1)[1]
    context.user_data["temp_factory"] = factory
    logging.info(f"[CONFIG FLOW] User {query.from_user.id} selected Factory: {factory}")

    await query.edit_message_text(
        text=f"🏭 Xưởng chính: <b>{html.escape(factory)}</b>\n\n🏢 <b>Chọn Tổ/Xưởng:</b>",
        reply_markup=get_workshop_keyboard(factory),
        parse_mode="HTML"
    )
    return CFG_SELECT_WORKSHOP

async def handle_cfg_workshop_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores Workshop choice and prompts Column Letter choice."""
    query = update.callback_query
    await query.answer()

    workshop = query.data.split(":", 1)[1]
    context.user_data["temp_workshop"] = workshop
    logging.info(f"[CONFIG FLOW] User {query.from_user.id} selected Workshop: {workshop}")

    factory = context.user_data["temp_factory"]

    await query.edit_message_text(
        text=f"🏭 Xưởng chính: <b>{html.escape(factory)}</b>\n🏢 Tổ/Xưởng: <b>{html.escape(workshop)}</b>\n\n🔠 <b>Chọn chữ cái cho Cột:</b>",
        reply_markup=get_letter_keyboard(),
        parse_mode="HTML"
    )
    return CFG_SELECT_LETTER

async def handle_cfg_letter_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores Column Letter choice and prompts Column Number choice."""
    query = update.callback_query
    await query.answer()

    letter = query.data.split(":", 1)[1]
    context.user_data["temp_letter"] = letter
    logging.info(f"[CONFIG FLOW] User {query.from_user.id} selected Column Letter: {letter}")

    factory = context.user_data["temp_factory"]
    workshop = context.user_data["temp_workshop"]

    await query.edit_message_text(
        text=f"🏭 Xưởng chính: <b>{html.escape(factory)}</b>\n🏢 Tổ/Xưởng: <b>{html.escape(workshop)}</b>\n🔠 Chữ cái cột: <b>{html.escape(letter)}</b>\n\n🔢 <b>Chọn số thứ tự cho Cột:</b>",
        reply_markup=get_number_keyboard(),
        parse_mode="HTML"
    )
    return CFG_SELECT_NUMBER

async def handle_cfg_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles back action from Number selection back to Letter selection."""
    query = update.callback_query
    await query.answer()

    factory = context.user_data["temp_factory"]
    workshop = context.user_data["temp_workshop"]
    context.user_data.pop("temp_letter", None)
    logging.info(f"[CONFIG FLOW] User {query.from_user.id} clicked back to Letter selection")

    await query.edit_message_text(
        text=f"🏭 Xưởng chính: <b>{html.escape(factory)}</b>\n🏢 Tổ/Xưởng: <b>{html.escape(workshop)}</b>\n\n🔠 <b>Chọn chữ cái cho Cột:</b>",
        reply_markup=get_letter_keyboard(),
        parse_mode="HTML"
    )
    return CFG_SELECT_LETTER

async def handle_cfg_number_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Stores Column Number choice, confirms and saves the configuration."""
    query = update.callback_query
    await query.answer()

    number = query.data.split(":", 1)[1]
    letter = context.user_data["temp_letter"]
    current_position = f"{letter}{number}"
    
    factory = context.user_data.pop("temp_factory")
    workshop = context.user_data.pop("temp_workshop")
    context.user_data.pop("temp_letter", None)

    user_id = query.from_user.id
    save_user_config(user_id, factory, workshop, current_position, context)
    logging.info(f"[CONFIG FLOW] User {user_id} configured factory={factory}, workshop={workshop}, position={current_position}")

    await query.edit_message_text(
        text=(
            "✅ <b>Đã cấu hình thành công!</b>\n\n"
            f"🏭 <b>Xưởng chính:</b> {html.escape(factory)}\n"
            f"🏢 <b>Tổ/Xưởng:</b> {html.escape(workshop)}\n"
            f"📍 <b>Vị trí cột:</b> {html.escape(current_position)}\n\n"
            "📸 Hãy gửi hình ảnh để xử lý OCR."
        ),
        parse_mode="HTML"
    )
    return ConversationHandler.END

async def cancel_config(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancels the /x config flow."""
    context.user_data.pop("temp_factory", None)
    context.user_data.pop("temp_workshop", None)
    context.user_data.pop("temp_letter", None)
    await update.message.reply_text("🚫 Đã huỷ cấu hình. Thiết lập cũ vẫn được giữ nguyên.")
    return ConversationHandler.END


# ─────────────────────────────────────────────
# /start command
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    user_id = update.effective_user.id
    load_user_config(user_id, context)
    context.user_data["bot_state"] = BotState.NORMAL
    
    current_factory = html.escape(context.user_data.get("factory") or "Chưa thiết lập ❌")
    current_workshop = html.escape(context.user_data.get("workshop") or "Chưa thiết lập ❌")
    current_position = html.escape(context.user_data.get("current_position") or "Chưa thiết lập ❌")
    
    await show_main_menu(
        update.message,
        "👋 <b>Chào mừng đến với Equipment OCR Bot!</b>\n\n"
        "⚙️ <b>Cấu hình mặc định hiện tại của bạn:</b>\n"
        f"🏭 Xưởng chính: <b>{current_factory}</b>\n"
        f"🏢 Tổ/Xưởng: <b>{current_workshop}</b>\n"
        f"📍 Vị trí cột: <b>{current_position}</b>\n"
        "<i>(Sử dụng lệnh <code>/x</code> để thay đổi xưởng/vị trí hoặc <code>/l [vị_trí]</code> để đổi nhanh vị trí cột)</i>\n\n"
        "📋 <b>Hướng dẫn sử dụng:</b>\n"
        "📷 <b>Đọc 1 hình</b> — Xử lý OCR từng ảnh riêng lẻ\n"
        "📚 <b>Đọc nhiều hình</b> — Thu thập nhiều ảnh rồi đọc cùng lúc\n\n"
        "<i>Sử dụng lệnh <code>/x</code> để thiết lập Xưởng & Vị trí trước khi bắt đầu.</i>",
    )
    return ConversationHandler.END


# ─────────────────────────────────────────────
# /help command
# ─────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 <b>Hướng dẫn sử dụng</b>\n\n"
        "1. Gửi ảnh nhãn máy/thiết bị vào chat\n"
        "2. Chọn engine OCR (Paddle hoặc Gemini)\n"
        "3. Kiểm tra kết quả và xác nhận hoặc sửa\n"
        "4. Dữ liệu được lưu vào Google Sheets\n\n"
        "<b>Lệnh:</b>\n"
        "/start  — Bắt đầu lại\n"
        "/cancel — Huỷ thao tác hiện tại\n"
        "/x      — Màn hình cấu hình Xưởng & Vị trí\n"
        "/l [vị_trí]   — Thiết lập vị trí cột nhanh\n"
        "/f [mã_mmtb]   — Tìm MMTB có sẵn\n"
        "/n      — Thêm thiết bị thủ công\n"
        "/help   — Xem hướng dẫn này",
        parse_mode="HTML",
    )
    return ConversationHandler.END


# ─────────────────────────────────────────────
# /cancel command  (available at any state)
# ─────────────────────────────────────────────
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    await update.message.reply_text(
        "🚫 Đã huỷ thao tác. Gửi ảnh mới để bắt đầu lại."
    )
    return ConversationHandler.END


# ═════════════════════════════════════════════
# STEP 1 — Photo received → ask engine
# ═════════════════════════════════════════════
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]          # largest available resolution
    context.user_data["pending_file_id"] = photo.file_id

    keyboard = [
        [InlineKeyboardButton("🔵 Paddle OCR", callback_data=CB_PADDLE)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL}", callback_data=CB_GEMINI_1)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_2}", callback_data=CB_GEMINI_2)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_3}", callback_data=CB_GEMINI_3)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_4}", callback_data=CB_GEMINI_4)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_5}", callback_data=CB_GEMINI_5)]
    ]
    await update.message.reply_text(
        "🖼️ Ảnh đã nhận! Chọn engine OCR để xử lý:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )
    return SELECT_ENGINE


# ═════════════════════════════════════════════
# STEP 2 — Engine chosen → run OCR → show result + confirm buttons
# ═════════════════════════════════════════════
async def handle_engine_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    engine      = query.data
    file_id     = context.user_data.get("pending_file_id")
    if engine == CB_PADDLE:
        engine_name = "Paddle OCR"
        endpoint    = PADDLE_ENDPOINT
    else:
        model_index = 1
        model_name = GEMINI_MODEL
        if engine == CB_GEMINI_2:
            model_index = 2
            model_name = GEMINI_MODEL_2
        elif engine == CB_GEMINI_3:
            model_index = 3
            model_name = GEMINI_MODEL_3
        elif engine == CB_GEMINI_4:
            model_index = 4
            model_name = GEMINI_MODEL_4
        elif engine == CB_GEMINI_5:
            model_index = 5
            model_name = GEMINI_MODEL_5
        engine_name = f"Gemini {model_name}"
        endpoint    = f"{GEMINI_ENDPOINT}?model_index={model_index}"

    context.user_data["engine_name"] = engine_name

    await query.edit_message_text(
        f"⏳ Đang xử lý với *{engine_name}*...",
        parse_mode="Markdown",
    )

    try:
        # ── Download image from Telegram ──────────────
        telegram_file = await context.bot.get_file(file_id)
        image_bytes   = bytes(await telegram_file.download_as_bytearray())
        logging.info(f"Downloaded image ({len(image_bytes)} bytes) → {endpoint}")

        # ── POST to FastAPI OCR endpoint (async, non-blocking) ─────────
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                endpoint,
                files=[("files", ("image.jpg", image_bytes, "image/jpeg"))],
            )
        logging.info(f"API status: {response.status_code}")

        if response.status_code != 200:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=(
                    f"❌ API lỗi `{response.status_code}`:\n"
                    f"```\n{response.text[:400]}\n```"
                ),
                parse_mode="Markdown",
            )
            return ConversationHandler.END

        data            = response.json()
        results         = data.get("results", [])
        processing_time = data.get("processing_time", "?")

        if not results:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="⚠️ Không tìm thấy kết quả OCR trong phản hồi.",
            )
            return ConversationHandler.END

        # Store OCR result in user_data for the correction loop
        result        = results[0]
        kv            = dict(result.get("key_value", {}))
        markdown_text = result.get("markdown", "")

        # Load user configuration defaults
        user_id = update.effective_user.id
        load_user_config(user_id, context)

        # ── Apply Persistent Workshop Overwrite ───────
        ws = context.user_data.get("workshop") or get_user_workshop(user_id)
        if ws:
            keys_to_remove = []
            for k in kv:
                if k.lower().strip() in ["xưởng", "xuong", "xuong san xuat", "nha may", "xương"]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del kv[k]
            kv["Xưởng"] = ws
            logging.info(f"Overwrote workshop with value: {ws} for user {user_id}")

        # ── Apply Persistent Location Overwrite ────────
        pos = context.user_data.get("current_position") or get_user_location(user_id)
        if pos:
            keys_to_remove = []
            for k in kv:
                if k.lower().strip() in ["vị trí", "vi tri", "vitri", "vi tri ", "tri"]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del kv[k]
            kv["Vị trí"] = pos
            logging.info(f"Overwrote location with value: {pos} for user {user_id}")

        context.user_data["kv"]              = kv
        context.user_data["markdown_text"]   = markdown_text
        context.user_data["processing_time"] = processing_time

        # ── Send formatted summary ─────────────────────
        formatted   = _format_result(kv, markdown_text, engine_name, processing_time)
        pretty_json = json.dumps(result, ensure_ascii=False, indent=2)
        if len(pretty_json) > 3000:
            pretty_json = pretty_json[:3000] + "\n... (truncated)"

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=formatted,
            parse_mode="Markdown",
        )
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"🗂️ *Raw JSON:*\n```json\n{pretty_json}\n```",
            parse_mode="Markdown",
        )

        status_kb = [
            [
                InlineKeyboardButton("🟢 Đang hoạt động", callback_data="status:active"),
                InlineKeyboardButton("🔴 Ngưng hoạt động", callback_data="status:inactive"),
            ],
            [
                InlineKeyboardButton("⚠️ Đã thanh lý", callback_data="status:disposed"),
                InlineKeyboardButton("🏷️ Chưa dán tem", callback_data="status:no_label"),
            ],
        ]
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="⚡ *Chọn trạng thái thiết bị:*",
            reply_markup=InlineKeyboardMarkup(status_kb),
            parse_mode="Markdown",
        )
        return SELECT_STATUS

    except httpx.TimeoutException:
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="❌ API timeout. Hãy thử lại sau.",
        )
        return ConversationHandler.END
    except Exception as e:
        logging.exception(e)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"❌ Lỗi không xác định:\n`{str(e)}`",
            parse_mode="Markdown",
        )
        return ConversationHandler.END


# ═════════════════════════════════════════════
# STEP 2.5 — Status Choice
# ═════════════════════════════════════════════
async def handle_status_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    status_data = query.data
    status_map = {
        "status:active":   "Đang hoạt động",
        "status:inactive": "Ngưng hoạt động",
        "status:disposed": "Đã thanh lý",
        "status:no_label": "Chưa dán tem",
    }
    chosen_status = status_map.get(status_data, "—")
    
    if "kv" not in context.user_data:
        context.user_data["kv"] = {}
    context.user_data["kv"]["status"] = chosen_status

    confirm_kb = [[
        InlineKeyboardButton("✅ Đúng, lưu lại", callback_data=CB_CONFIRM_YES),
        InlineKeyboardButton("✏️ Sai, sửa lại",  callback_data=CB_CONFIRM_EDIT),
    ]]

    kv = context.user_data["kv"]
    markdown_text = context.user_data.get("markdown_text", "")
    engine_name = context.user_data.get("engine_name", "OCR")
    processing_time = context.user_data.get("processing_time", "?")

    formatted = _format_result(kv, markdown_text, engine_name, processing_time)

    await query.edit_message_text(
        text=f"✔️ Đã chọn trạng thái: *{chosen_status}*",
        parse_mode="Markdown",
    )

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=formatted,
        parse_mode="Markdown",
    )

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="❓ *Thông tin trên có chính xác không?*",
        reply_markup=InlineKeyboardMarkup(confirm_kb),
        parse_mode="Markdown",
    )
    return CONFIRM_RESULT


# ─────────────────────────────────────────────
# OCR validation & Save helpers
# ─────────────────────────────────────────────
async def _save_ocr_data(query, context, source: str, force_dup: bool = False):
    # Guard: prevent double-save if handler is somehow re-entered
    if context.user_data.get("sheets_saved"):
        await query.edit_message_text(
            "✅ Dữ liệu đã được lưu rồi. Gửi ảnh mới để tiếp tục."
        )
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(
        "💾 *Đang lưu dữ liệu vào Google Sheets...*",
        parse_mode="Markdown",
    )
    logging.info(f"[BOT] Saving to Google Sheets triggered from {source}")
    try:
        kv         = context.user_data["kv"]
        ocr_result = OCRResult(
            markdown=context.user_data.get("markdown_text", ""),
            key_value=kv,
        )
        await sheet_service.append_results([ocr_result], source, force=force_dup)
        context.user_data["sheets_saved"] = True
        logging.info("Machine saved successfully")
        
        user_id = query.from_user.id
        current_pos = context.user_data.get("current_position") or get_user_location(user_id)
        reply_markup = get_position_adjustment_keyboard(current_pos) if current_pos else None

        if force_dup:
            msg_text = "✅ Đã lưu thông tin thiết bị thành công.\n\n(Lưu ý: Mã MMTB này đã tồn tại trong Google Sheet.)"
        else:
            msg_text = "✅ Đã lưu thông tin thiết bị thành công."

        msg_text += (
            "\n\n💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
            "Gửi ảnh mới để tiếp tục."
        )

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=msg_text,
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
    except Exception as e:
        logging.exception(e)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"⚠️ Lưu vào Google Sheets thất bại:\n`{str(e)}`",
            parse_mode="Markdown",
        )
    context.user_data.clear()
    return ConversationHandler.END


async def _do_mmtb_validation_check(query, context, source: str) -> Optional[int]:
    """
    Checks if Mã MMTB is valid.
    If not valid, edits message to show warning and returns MMTB_WARN.
    Otherwise returns None.
    """
    kv = context.user_data.get("kv", {})
    mmtb = _get_kv(kv, "Mã MMTB").strip()
    
    if validate_mmtb(mmtb):
        return None
        
    user_id = query.from_user.id
    logging.warning(
        f"[MMTB_VALIDATION] Invalid code '{mmtb}' — source={source} user_id={user_id}"
    )
    
    context.user_data["mmtb_warn_source"] = source
    
    warning_text = (
        f"⚠️ <b>Mã MMTB có thể không hợp lệ:</b>\n\n"
        f"Giá trị hiện tại: <code>{html.escape(mmtb) if mmtb else '(Trống)'}</code>\n"
        f"Định dạng mong đợi: <code>B########</code> (Ví dụ: <code>B22400711</code>)\n\n"
        f"Bạn muốn làm gì?"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("✏️ Sửa mã MMTB", callback_data=CB_MMTB_EDIT),
            InlineKeyboardButton("⏭️ Vẫn lưu", callback_data=CB_MMTB_FORCE_SAVE),
        ]
    ]
    
    await query.edit_message_text(
        text=warning_text,
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode="HTML",
    )
    return MMTB_WARN


async def handle_mmtb_warn_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    source = context.user_data.get("mmtb_warn_source", "TELEGRAM_CONFIRM")
    
    if data == CB_MMTB_FORCE_SAVE:
        user_id = query.from_user.id
        kv = context.user_data.get("kv", {})
        mmtb = _get_kv(kv, "Mã MMTB")
        logging.warning(
            f"[MMTB_VALIDATION] User force-saved invalid code '{mmtb}' — source={source} user_id={user_id}"
        )
        return await _run_duplication_check_and_save(query, context, source)
        
    elif data == CB_MMTB_EDIT:
        context.user_data["field_index"] = 1
        context.user_data["mmtb_single_edit"] = True
        
        await query.edit_message_text(
            "✏️ *Sửa mã MMTB...*\n\n"
            "Nhập giá trị mới hoặc chọn:",
            parse_mode="Markdown",
        )
        return await _ask_next_field(context, query.message.chat_id)


async def _run_duplication_check_and_save(query, context, source: str):
    kv = context.user_data.get("kv", {})
    machine_code = _get_kv(kv, "Mã MMTB").strip()

    logging.info(f"User requested save for machine code: {machine_code}")

    # Immediately reply with the loading message
    await query.edit_message_text(
        "⏳ Xin vui lòng đợi, để mình kiểm tra xem thông tin có bị trùng lặp không nhé!"
    )
    logging.info("Duplication check started")

    try:
        is_duplicate = await duplicate_service.check_duplication_code(machine_code)
        if not is_duplicate:
            logging.info("Duplicate not found")
            return await _save_ocr_data(query, context, source)
        else:
            logging.info("Duplicate found")
            warning_text = (
                "⚠️ Mình đã tìm thấy thiết bị có cùng Mã MMTB trong Google Sheet.\n\n"
                "Điều này có thể tạo ra dữ liệu trùng lặp.\n\n"
                "Bạn vẫn muốn lưu thiết bị này chứ?"
            )
            keyboard = [
                [
                    InlineKeyboardButton("✅ Vẫn lưu", callback_data=CB_DUP_FORCE_SAVE),
                    InlineKeyboardButton("❌ Hủy", callback_data=CB_DUP_CANCEL),
                ]
            ]
            context.user_data["dup_warn_source"] = source
            
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=warning_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
            )
            return DUP_WARN

    except Exception as e:
        logging.exception(f"Error checking duplicate equipment code: {machine_code}")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=(
                "❌ Không thể kiểm tra dữ liệu trên Google Sheet.\n\n"
                "Vui lòng thử lại sau."
            )
        )
        context.user_data.clear()
        return ConversationHandler.END


async def handle_dup_warn_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    data = query.data
    source = context.user_data.get("dup_warn_source", "TELEGRAM_CONFIRM")

    if data == CB_DUP_FORCE_SAVE:
        logging.info("User selected Force Save")
        return await _save_ocr_data(query, context, source, force_dup=True)

    elif data == CB_DUP_CANCEL:
        logging.info("User cancelled save")
        await query.edit_message_text(
            "❌ Đã hủy thao tác lưu.\n\n"
            "Không có dữ liệu nào được thêm vào Google Sheet."
        )
        context.user_data.clear()
        return ConversationHandler.END


# ═════════════════════════════════════════════
# STEP 3 — Confirmation response
# ═════════════════════════════════════════════
async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_CONFIRM_YES:
        # Check validation
        warn_state = await _do_mmtb_validation_check(query, context, "TELEGRAM_CONFIRM")
        if warn_state is not None:
            return warn_state

        return await _run_duplication_check_and_save(query, context, "TELEGRAM_CONFIRM")

    # ── CB_CONFIRM_EDIT: start field-by-field correction ──
    await query.edit_message_text(
        "✏️ *Bắt đầu sửa thông tin...*\n\n"
        "Nhập giá trị mới hoặc bấm nút *Giữ nguyên* để giữ giá trị cũ.",
        parse_mode="Markdown",
    )
    context.user_data["field_index"] = 0
    return await _ask_next_field(context, query.message.chat_id)


# ─────────────────────────────────────────────
# Helper: send prompt for the current field
# ─────────────────────────────────────────────
async def _ask_next_field(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    idx = context.user_data["field_index"]
    kv  = context.user_data["kv"]

    if idx >= len(FIELDS):
        # All fields processed → show corrected summary
        return await _show_correction_summary(context, chat_id)

    label, key = FIELDS[idx]
    current    = _get_kv(kv, key)
    progress   = f"{idx + 1}/{len(FIELDS)}"

    if key == "status":
        keyboard = [
            [
                InlineKeyboardButton("🟢 Đang hoạt động", callback_data="status:active"),
                InlineKeyboardButton("🔴 Ngưng hoạt động", callback_data="status:inactive"),
            ],
            [
                InlineKeyboardButton("⚠️ Đã thanh lý", callback_data="status:disposed"),
                InlineKeyboardButton("🏷️ Chưa dán tem", callback_data="status:no_label"),
            ],
            [
                InlineKeyboardButton("↩️ Giữ nguyên", callback_data="status:keep"),
                InlineKeyboardButton("⏭️ Bỏ qua", callback_data="edit:skip"),
            ],
        ]
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"📝 *({progress}) {label}*\n\n"
                f"Hiện tại: `{current}`\n\n"
                "Chọn trạng thái mới:"
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
    else:
        keyboard = [[
            InlineKeyboardButton("↩️ Giữ nguyên", callback_data="edit:keep"),
            InlineKeyboardButton("⏭️ Bỏ qua", callback_data="edit:skip"),
        ]]
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"📝 *({progress}) {label}*\n\n"
                f"Hiện tại: `{current}`\n\n"
                "Nhập giá trị mới, hoặc bấm nút để giữ nguyên:"
            ),
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown",
        )
    return EDIT_FIELD


# ═════════════════════════════════════════════
# STEP 4 — Receive text input for each field
# ═════════════════════════════════════════════
async def handle_field_input(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_input = update.message.text.strip().upper()
    idx        = context.user_data["field_index"]
    kv         = context.user_data["kv"]
    label, key = FIELDS[idx]

    # Flow 1: manual-add Validation Guard
    if key == "Mã MMTB" and context.user_data.get("is_manual_add"):
        target_val = _get_kv(kv, key) if user_input == "-" else user_input
        if not validate_mmtb(target_val):
            logging.warning(
                f"[MMTB_VALIDATION] Invalid code '{target_val}' — source=manual_add user_id={update.effective_user.id}"
            )
            keyboard = [[InlineKeyboardButton("⏭️ Bỏ qua", callback_data="edit:skip")]]
            await update.message.reply_text(
                "❌ <b>Mã MMTB không hợp lệ.</b>\n\n"
                "Định dạng hợp lệ: <code>B########</code> (chữ B in hoa theo sau bởi đúng 8 chữ số)\n"
                "Ví dụ: <code>B22400711</code>\n\n"
                "Vui lòng nhập lại hoặc chọn:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML",
            )
            return EDIT_FIELD

    if user_input == "-":
        current = _get_kv(kv, key)
        await update.message.reply_text(
            f"↩️ Giữ nguyên *{label}*: `{current}`",
            parse_mode="Markdown",
        )
    else:
        kv[key] = user_input
        context.user_data["kv"] = kv
        await update.message.reply_text(
            f"✔️ Đã cập nhật *{label}*: `{user_input}`",
            parse_mode="Markdown",
        )

    if context.user_data.get("mmtb_single_edit"):
        context.user_data.pop("mmtb_single_edit", None)
        return await _show_correction_summary(context, update.message.chat_id)

    # Advance to next field
    context.user_data["field_index"] = idx + 1
    return await _ask_next_field(context, update.message.chat_id)


async def handle_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    status_data = query.data
    status_map = {
        "status:active":   "Đang hoạt động",
        "status:inactive": "Ngưng hoạt động",
        "status:disposed": "Đã thanh lý",
        "status:no_label": "Chưa dán tem",
    }
    
    idx = context.user_data["field_index"]
    kv = context.user_data["kv"]
    label, key = FIELDS[idx]

    if status_data == "status:keep":
        chosen_status = _get_kv(kv, key)
        await query.edit_message_text(
            text=f"↩️ Giữ nguyên *{label}*: `{chosen_status}`",
            parse_mode="Markdown",
        )
    else:
        chosen_status = status_map.get(status_data, "—")
        kv[key] = chosen_status
        context.user_data["kv"] = kv
        await query.edit_message_text(
            text=f"✔️ Đã cập nhật *{label}*: `{chosen_status}`",
            parse_mode="Markdown",
        )

    # Advance to next field
    context.user_data["field_index"] = idx + 1
    return await _ask_next_field(context, query.message.chat_id)


async def handle_edit_keep_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    idx = context.user_data["field_index"]
    kv = context.user_data["kv"]
    label, key = FIELDS[idx]

    current = _get_kv(kv, key)

    # Flow 1: manual-add Validation Guard
    if key == "Mã MMTB" and context.user_data.get("is_manual_add"):
        if not validate_mmtb(current):
            logging.warning(
                f"[MMTB_VALIDATION] Invalid code '{current}' kept in manual_add — user_id={query.from_user.id}"
            )
            keyboard = [[InlineKeyboardButton("⏭️ Bỏ qua", callback_data="edit:skip")]]
            await query.edit_message_text(
                text="❌ <b>Mã MMTB không hợp lệ.</b>\n\n"
                     "Định dạng hợp lệ: <code>B########</code> (chữ B in hoa theo sau bởi đúng 8 chữ số)\n"
                     "Ví dụ: <code>B22400711</code>\n\n"
                     "Vui lòng nhập lại hoặc chọn:",
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="HTML",
            )
            return EDIT_FIELD

    await query.edit_message_text(
        text=f"↩️ Giữ nguyên *{label}*: `{current}`",
        parse_mode="Markdown",
    )

    if context.user_data.get("mmtb_single_edit"):
        context.user_data.pop("mmtb_single_edit", None)
        return await _show_correction_summary(context, query.message.chat_id)

    # Advance to next field
    context.user_data["field_index"] = idx + 1
    return await _ask_next_field(context, query.message.chat_id)


async def handle_edit_skip_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """⏭️ Bỏ qua handler: keep value in /f, blank (or CHƯA CÓ MÃ) in /n."""
    query = update.callback_query
    await query.answer()

    idx = context.user_data["field_index"]
    kv  = context.user_data["kv"]
    label, key = FIELDS[idx]

    is_manual = context.user_data.get("is_manual_add", False)

    if is_manual:
        # Manual-add flow: Mã MMTB → CHƯA CÓ MÃ; other fields → blank
        if key == "Mã MMTB":
            kv[key] = "CHƯA CÓ MÃ"
            await query.edit_message_text(
                text=f"⏭️ Bỏ qua *{label}* → sẽ lưu là `CHƯA CÓ MÃ`",
                parse_mode="Markdown",
            )
        else:
            kv[key] = ""
            await query.edit_message_text(
                text=f"⏭️ Đã bỏ qua *{label}*",
                parse_mode="Markdown",
            )
        context.user_data["kv"] = kv
    else:
        # /f re-edit flow or warning re-edit flow: skip = keep existing value unchanged
        current = _get_kv(kv, key)
        await query.edit_message_text(
            text=f"⏭️ Bỏ qua — giữ nguyên *{label}*: `{current}`",
            parse_mode="Markdown",
        )

    if context.user_data.get("mmtb_single_edit"):
        context.user_data.pop("mmtb_single_edit", None)
        return await _show_correction_summary(context, query.message.chat_id)

    # Advance to next field
    context.user_data["field_index"] = idx + 1
    return await _ask_next_field(context, query.message.chat_id)


# ─────────────────────────────────────────────
# Helper: show corrected summary + save / edit-again buttons
# ─────────────────────────────────────────────
async def _show_correction_summary(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    kv            = context.user_data["kv"]
    markdown_text = context.user_data.get("markdown_text", "")
    engine_name   = context.user_data.get("engine_name", "OCR")
    proc_time     = context.user_data.get("processing_time", "?")

    formatted = "✏️ *Thông tin đã sửa:*\n\n" + _format_result(
        kv, markdown_text, engine_name, proc_time
    )

    # Route save/edit buttons to the correct callback prefix depending on flow
    is_find_edit  = context.user_data.get("is_find_edit", False)
    is_manual_add = context.user_data.get("is_manual_add", False)
    if is_find_edit:
        save_cb, edit_cb = CB_FIND_SAVE, CB_FIND_EDIT_AGAIN
    elif is_manual_add:
        save_cb, edit_cb = CB_MANUAL_SAVE, CB_MANUAL_EDIT
    else:
        save_cb, edit_cb = CB_SAVE_CONFIRM, CB_SAVE_EDIT

    save_kb = [[
        InlineKeyboardButton("💾 Xác nhận & Lưu", callback_data=save_cb),
        InlineKeyboardButton("🔁 Sửa lại từ đầu",  callback_data=edit_cb),
    ]]

    await context.bot.send_message(
        chat_id=chat_id,
        text=formatted,
        reply_markup=InlineKeyboardMarkup(save_kb),
        parse_mode="Markdown",
    )
    return CONFIRM_SAVE


# ═════════════════════════════════════════════
# STEP 5 — Final save confirmation
# ═════════════════════════════════════════════
async def handle_save_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_SAVE_EDIT:
        # Restart field loop from the beginning
        context.user_data["field_index"] = 0
        await query.edit_message_text(
            "🔁 *Sửa lại từ đầu...*\n\nGửi `-` nếu muốn giữ nguyên.",
            parse_mode="Markdown",
        )
        return await _ask_next_field(context, query.message.chat_id)

    if query.data == CB_SAVE_CONFIRM:
        # Check validation
        warn_state = await _do_mmtb_validation_check(query, context, "TELEGRAM_CORRECTED_CONFIRM")
        if warn_state is not None:
            return warn_state

        return await _run_duplication_check_and_save(query, context, "TELEGRAM_CORRECTED_CONFIRM")


# ═════════════════════════════════════════════
# /n command — manual equipment add flow
# ═════════════════════════════════════════════
async def new_manual_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /n. Starts a manual field-by-field equipment add flow."""
    user_id = update.effective_user.id
    load_user_config(user_id, context)

    # Pre-fill kv with the user's saved workshop and location as defaults
    kv: dict = {}
    ws  = context.user_data.get("workshop") or get_user_workshop(user_id)
    pos = context.user_data.get("current_position") or get_user_location(user_id)
    if ws:
        kv["Xưởng"] = ws
    if pos:
        kv["Vị trí"] = pos

    context.user_data["kv"]              = kv
    context.user_data["markdown_text"]   = ""
    context.user_data["engine_name"]     = "Nhập tay"
    context.user_data["processing_time"] = "—"
    context.user_data["field_index"]     = 0
    context.user_data["is_manual_add"]   = True

    await update.message.reply_text(
        "✍️ <b>Thêm thiết bị thủ công</b>\n\n"
        "Nhập thông tin cho từng trường. Bấm <b>↩️ Giữ nguyên</b> để giữ giá trị hiện tại, "
        "hoặc <b>⏭️ Bỏ qua</b> để bỏ qua trường đó.\n\n"
        "💡 Nếu bỏ qua <b>Mã MMTB</b>, hệ thống sẽ ghi <code>CHƯA CÓ MÃ</code>.",
        parse_mode="HTML",
    )
    return await _ask_next_field(context, update.message.chat_id)


# ─────────────────────────────────────────────
# /n flow — final save confirmation
# ─────────────────────────────────────────────
async def handle_manual_save_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_MANUAL_EDIT:
        # Restart field loop from the beginning
        context.user_data["field_index"] = 0
        await query.edit_message_text(
            "🔁 *Sửa lại từ đầu...*\n\nGửi `-` nếu muốn giữ nguyên.",
            parse_mode="Markdown",
        )
        return await _ask_next_field(context, query.message.chat_id)

    # ── CB_MANUAL_SAVE: append new row to Google Sheets ──
    if context.user_data.get("sheets_saved"):
        await query.edit_message_text(
            "✅ Dữ liệu đã được lưu rồi. Gửi ảnh hoặc dùng /n để tiếp tục."
        )
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(
        "💾 *Đang lưu dữ liệu vào Google Sheets...*",
        parse_mode="Markdown",
    )
    logging.info("[BOT] Saving to Google Sheets triggered from TELEGRAM_MANUAL step")

    try:
        kv = context.user_data["kv"]
        ocr_result = OCRResult(
            markdown=context.user_data.get("markdown_text", ""),
            key_value=kv,
        )
        await sheet_service.append_results([ocr_result], "TELEGRAM_MANUAL")
        context.user_data["sheets_saved"] = True
        logging.info("Machine saved successfully")

        user_id = query.from_user.id
        current_pos = context.user_data.get("current_position") or get_user_location(user_id)
        reply_markup = get_position_adjustment_keyboard(current_pos) if current_pos else None

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ <b>Đã thêm thiết bị mới vào Google Sheets!</b>\n\n"
                 "💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
                 "Gửi ảnh hoặc dùng /n để thêm thiết bị tiếp theo.",
            reply_markup=reply_markup,
            parse_mode="HTML",
        )
    except Exception as e:
        logging.exception(e)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"⚠️ Lưu vào Google Sheets thất bại:\n`{str(e)}`",
            parse_mode="Markdown",
        )

    context.user_data.clear()
    return ConversationHandler.END


# ═════════════════════════════════════════════
# /f command — look up Mã MMTB in Google Sheets
# ═════════════════════════════════════════════
async def find_mmtb_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Entry point for /f <mã>. Searches the sheet and offers an edit option."""
    text  = update.message.text.strip()
    match = re.match(r'^/f(?:\s+(.+))?$', text, re.IGNORECASE)
    mmtb_code = match.group(1).strip() if (match and match.group(1)) else ""

    if not mmtb_code:
        await update.message.reply_text(
            "🔍 Vui lòng nhập mã MMTB cần tìm.\n"
            "Ví dụ: `/f B001`",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    await update.message.reply_text(
        f"🔍 Đang tìm kiếm mã MMTB: `{mmtb_code}`...",
        parse_mode="Markdown",
    )

    try:
        result = await sheet_service.lookup_row_by_mmtb(mmtb_code)
    except Exception as e:
        logging.exception(e)
        await update.message.reply_text(
            f"❌ Lỗi khi tìm kiếm:\n`{str(e)}`",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    if not result:
        await update.message.reply_text(
            f"⚠️ Không tìm thấy mã MMTB `{mmtb_code}` trong Google Sheets.\n"
            "Kiểm tra lại mã hoặc thêm thiết bị bằng cách gửi ảnh.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END

    kv        = result["kv"]
    row_index = result["row_index"]

    # Stash for the edit flow
    context.user_data["find_kv"]        = kv
    context.user_data["find_row_index"] = row_index
    context.user_data["find_mmtb_code"] = mmtb_code

    # Build display summary
    msg = f"✅ *Tìm thấy — Mã MMTB: `{mmtb_code}`*\n\n"
    for label, key in FIELDS:
        msg += f"{label}: *{_get_kv(kv, key)}*\n"
    msg += "\n❓ *Bạn có muốn chỉnh sửa thông tin này không?*"

    edit_kb = [[
        InlineKeyboardButton("✏️ Sửa thông tin", callback_data=CB_FIND_EDIT_YES),
        InlineKeyboardButton("❌ Không, đóng",   callback_data=CB_FIND_EDIT_NO),
    ]]

    await update.message.reply_text(
        msg,
        reply_markup=InlineKeyboardMarkup(edit_kb),
        parse_mode="Markdown",
    )
    return FIND_ASK_EDIT


# ─────────────────────────────────────────────
# /f flow — user answers Yes / No to edit prompt
# ─────────────────────────────────────────────
async def handle_find_edit_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_FIND_EDIT_NO:
        await query.edit_message_text(
            "❌ Đã đóng. Gửi ảnh hoặc dùng /f để tìm kiếm tiếp."
        )
        context.user_data.clear()
        return ConversationHandler.END

    # ── CB_FIND_EDIT_YES: enter the shared field-edit loop ────────────
    context.user_data["kv"]             = dict(context.user_data.get("find_kv", {}))
    context.user_data["markdown_text"]  = ""
    context.user_data["engine_name"]    = "Chỉnh sửa"
    context.user_data["processing_time"] = "—"
    context.user_data["field_index"]    = 0
    context.user_data["is_find_edit"]   = True   # signals save → update, not append

    await query.edit_message_text(
        "✏️ *Bắt đầu sửa thông tin...*\n\n"
        "Nhập giá trị mới hoặc bấm nút *Giữ nguyên* để giữ giá trị cũ.",
        parse_mode="Markdown",
    )
    return await _ask_next_field(context, query.message.chat_id)


# ─────────────────────────────────────────────
# /f flow — final save: update the existing row
# ─────────────────────────────────────────────
async def handle_find_save_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_FIND_EDIT_AGAIN:
        # Restart the field loop from the beginning
        context.user_data["field_index"] = 0
        await query.edit_message_text(
            "🔁 *Sửa lại từ đầu...*\n\nGửi `-` nếu muốn giữ nguyên.",
            parse_mode="Markdown",
        )
        return await _ask_next_field(context, query.message.chat_id)

    # ── CB_FIND_SAVE: write corrected data back to the same sheet row ──
    if context.user_data.get("sheets_saved"):
        await query.edit_message_text(
            "✅ Dữ liệu đã được cập nhật rồi. Gửi ảnh mới hoặc dùng /f để tiếp tục."
        )
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(
        "💾 *Đang cập nhật dữ liệu vào Google Sheets...*",
        parse_mode="Markdown",
    )

    kv        = context.user_data["kv"]
    row_index = context.user_data.get("find_row_index")
    mmtb_code = context.user_data.get("find_mmtb_code", "")

    logging.info(
        f"[BOT] Updating sheet row {row_index} for Mã MMTB '{mmtb_code}' "
        "from FIND_UPDATE step"
    )

    try:
        ocr_result = OCRResult(
            markdown=context.user_data.get("markdown_text", ""),
            key_value=kv,
        )
        from app.helpers import extract_row_data
        row_data = extract_row_data(ocr_result.key_value)

        success = await sheet_service.update_row_in_sheet(row_index, row_data, "FIND_UPDATE")

        if success:
            context.user_data["sheets_saved"] = True
            
            user_id = query.from_user.id
            current_pos = context.user_data.get("current_position") or get_user_location(user_id)
            reply_markup = get_position_adjustment_keyboard(current_pos) if current_pos else None

            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=(
                    f"✅ <b>Đã cập nhật thành công mã MMTB <code>{html.escape(mmtb_code)}</code> vào Google Sheets!</b>\n\n"
                    "💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
                    "Gửi ảnh mới hoặc dùng /f để tiếp tục."
                ),
                reply_markup=reply_markup,
                parse_mode="HTML",
            )
        else:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="⚠️ Cập nhật Google Sheets thất bại. Vui lòng thử lại.",
            )
    except Exception as e:
        logging.exception(e)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=f"⚠️ Lỗi khi cập nhật:\n`{str(e)}`",
            parse_mode="Markdown",
        )

    context.user_data.clear()
    return ConversationHandler.END


# ═════════════════════════════════════════════
# MULTI-IMAGE FLOW HANDLERS
# ═════════════════════════════════════════════

# ─────────────────────────────────────────────
# Helper: build OCR engine inline keyboard (reused from single-image flow)
# ─────────────────────────────────────────────
def _multi_engine_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔵 Paddle OCR",              callback_data=CB_MULTI_PADDLE)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL}",   callback_data=CB_MULTI_GEMINI_1)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_2}", callback_data=CB_MULTI_GEMINI_2)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_3}", callback_data=CB_MULTI_GEMINI_3)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_4}", callback_data=CB_MULTI_GEMINI_4)],
        [InlineKeyboardButton(f"✨ Gemini {GEMINI_MODEL_5}", callback_data=CB_MULTI_GEMINI_5)],
    ])


# ─────────────────────────────────────────────
# Helper: apply workshop/location overrides to a kv dict
# (mirrors the logic in handle_engine_choice)
# ─────────────────────────────────────────────
def _apply_context_overrides(kv: dict, context: ContextTypes.DEFAULT_TYPE, user_id: int) -> dict:
    ws = context.user_data.get("workshop") or get_user_workshop(user_id)
    if ws:
        keys_to_remove = [k for k in kv if k.lower().strip() in
                          ["xưởng", "xuong", "xuong san xuat", "nha may", "xương"]]
        for k in keys_to_remove:
            del kv[k]
        kv["Xưởng"] = ws

    pos = context.user_data.get("current_position") or get_user_location(user_id)
    if pos:
        keys_to_remove = [k for k in kv if k.lower().strip() in
                          ["vị trí", "vi tri", "vitri", "vi tri ", "tri"]]
        for k in keys_to_remove:
            del kv[k]
        kv["Vị trí"] = pos

    return kv


# ─────────────────────────────────────────────
# Helper: OCR a single image against an endpoint (async)
# ─────────────────────────────────────────────
async def _ocr_one_image(bot, file_id: str, endpoint: str, index: int) -> dict:
    """
    Downloads a Telegram photo and POSTs it to the OCR endpoint.
    Returns a dict with keys: file_id, kv, markdown, error.
    """
    try:
        telegram_file = await bot.get_file(file_id)
        image_bytes = bytes(await telegram_file.download_as_bytearray())
        logging.info(f"[MULTI] Image {index}: downloaded {len(image_bytes)} bytes → {endpoint}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                endpoint,
                files=[("files", ("image.jpg", image_bytes, "image/jpeg"))],
            )

        if response.status_code != 200:
            logging.error(f"[MULTI] Image {index}: API error {response.status_code}")
            return {"file_id": file_id, "kv": {}, "markdown": "",
                    "error": f"API lỗi {response.status_code}"}

        data = response.json()
        results = data.get("results", [])
        if not results:
            return {"file_id": file_id, "kv": {}, "markdown": "",
                    "error": "Không tìm thấy kết quả OCR"}

        result = results[0]
        kv = dict(result.get("key_value", {}))
        markdown = result.get("markdown", "")
        return {"file_id": file_id, "kv": kv, "markdown": markdown, "error": None}

    except httpx.TimeoutException:
        logging.error(f"[MULTI] Image {index}: timeout")
        return {"file_id": file_id, "kv": {}, "markdown": "", "error": "Timeout"}
    except Exception as e:
        logging.exception(f"[MULTI] Image {index}: unexpected error")
        return {"file_id": file_id, "kv": {}, "markdown": "", "error": str(e)}


# ─────────────────────────────────────────────
# Helper: format a multi-image OCR result list as a message
# ─────────────────────────────────────────────
def _format_multi_results(classified: list) -> str:
    """classified is a list of dicts with keys: kv, valid, duplicate, within_batch_dup, error."""
    valid_count   = sum(1 for m in classified if m["valid"] and not m["duplicate"] and not m["within_batch_dup"])
    dup_count     = sum(1 for m in classified if m["duplicate"] or m["within_batch_dup"])
    invalid_count = sum(1 for m in classified if not m["valid"])

    msg  = "📋 <b>Kết quả đọc OCR</b>\n\n"
    msg += f"Đã nhận diện: <b>{len(classified)} thiết bị</b>\n"
    msg += f"✅ Có thể lưu: <b>{valid_count}</b>\n"
    msg += f"⚠️ Đã tồn tại / trùng lặp: <b>{dup_count}</b>\n"
    msg += f"❌ Không hợp lệ: <b>{invalid_count}</b>\n\n"

    for i, m in enumerate(classified, 1):
        kv = m["kv"]
        badge = (
            "✅" if (m["valid"] and not m["duplicate"] and not m["within_batch_dup"])
            else "⚠️" if (m["duplicate"] or m["within_batch_dup"])
            else "❌"
        )
        msg += f"{badge} <b>{i}. {html.escape(_get_kv(kv, 'machine_name') or '(Không rõ tên)')}</b>\n"
        msg += f"   🔢 Mã MMTB: <code>{html.escape(_get_kv(kv, 'Mã MMTB'))}</code>\n"
        msg += f"   📦 Model: {html.escape(_get_kv(kv, 'Model'))}\n"
        msg += f"   🏭 Xưởng: {html.escape(_get_kv(kv, 'Xưởng'))}\n"
        msg += f"   📍 Vị trí: {html.escape(_get_kv(kv, 'Vị trí'))}\n"
        msg += f"   ⚡ Trạng thái: {html.escape(_get_kv(kv, 'status'))}\n"
        if m.get("error"):
            msg += f"   ⚠️ Lỗi OCR: {html.escape(m['error'])}\n"
        if m["within_batch_dup"]:
            msg += f"   ⚠️ <i>Trùng lặp trong lô hiện tại</i>\n"
        elif m["duplicate"]:
            msg += f"   ⚠️ <i>Đã tồn tại trong Google Sheets</i>\n"
        elif not m["valid"]:
            msg += f"   ❌ <i>Mã MMTB không hợp lệ — sẽ không được lưu</i>\n"
        msg += "\n"

    return msg.strip()


# ─────────────────────────────────────────────
# MULTI-IMAGE: "📷 Đọc 1 hình" — enter single-image mode
# ─────────────────────────────────────────────
async def handle_main_menu_single(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User selected 📷 Đọc 1 hình from the main menu."""
    # Clear any lingering multi-image state
    for key in ["bot_state", "multi_images", "multi_selected_engine",
                "multi_ocr_results", "multi_classified"]:
        context.user_data.pop(key, None)
    context.user_data["bot_state"] = BotState.NORMAL

    user_id = update.effective_user.id
    load_user_config(user_id, context)

    await show_main_menu(
        update.message,
        "📷 <b>Chế độ đọc 1 hình</b>\n\n"
        "Hãy gửi ảnh nhãn thiết bị vào chat để bắt đầu OCR."
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: "📚 Đọc nhiều hình" — enter multi-image mode
# ─────────────────────────────────────────────
async def handle_enter_multi_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User selected 📚 Đọc nhiều hình from the main menu."""
    user_id = update.effective_user.id
    load_user_config(user_id, context)

    # Initialise multi-image session
    context.user_data["bot_state"]   = BotState.MULTI_IMAGE_COLLECTING
    context.user_data["multi_images"] = []
    context.user_data.pop("multi_selected_engine", None)
    context.user_data.pop("multi_ocr_results", None)
    context.user_data.pop("multi_classified", None)

    await show_multi_image_menu(
        update.message,
        "📚 <b>Chế độ đọc nhiều hình</b>\n\n"
        "Hãy gửi các ảnh nhãn thiết bị vào chat.\n"
        "Sau khi gửi đủ ảnh, nhấn <b>▶️ Tiến hành đọc</b> để bắt đầu OCR.\n\n"
        "Nhấn <b>↩️ Trở về</b> để hủy và quay lại menu chính."
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: photo received during collecting state
# (Registered at group=-1 so it runs before conv_handler)
# ─────────────────────────────────────────────
async def handle_multi_image_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Collect photos while in MULTI_IMAGE_COLLECTING state."""
    if context.user_data.get("bot_state") != BotState.MULTI_IMAGE_COLLECTING:
        # Not in multi-collect mode — let conv_handler take over
        return

    photo = update.message.photo[-1]
    images: list = context.user_data.setdefault("multi_images", [])
    images.append(photo.file_id)
    count = len(images)

    logging.info(f"[MULTI] User {update.effective_user.id} uploaded image #{count}: {photo.file_id}")

    await update.message.reply_text(
        f"📷 Đã nhận <b>{count}</b> hình.\n"
        "Tiếp tục gửi ảnh hoặc nhấn <b>▶️ Tiến hành đọc</b> khi đã đủ.",
        parse_mode="HTML",
    )
    # Mark update as handled so conv_handler does NOT also process this photo
    raise ApplicationHandlerStop


# ─────────────────────────────────────────────
# MULTI-IMAGE: "↩️ Trở về" — cancel and return to main menu
# ─────────────────────────────────────────────
async def handle_multi_go_back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel multi-image mode and return to main menu."""
    bot_state = context.user_data.get("bot_state")
    if bot_state not in (
        BotState.MULTI_IMAGE_COLLECTING,
        BotState.MULTI_IMAGE_SELECTING_MODEL,
        BotState.MULTI_IMAGE_PROCESSING,
        BotState.MULTI_IMAGE_REVIEW,
        BotState.MULTI_IMAGE_SAVING,
    ):
        # Not in a multi-image state — ignore or show main menu
        context.user_data["bot_state"] = BotState.NORMAL
        await show_main_menu(update.message, "📋 Menu chính:")
        return

    # Clear all multi-image temporary state
    for key in ["multi_images", "multi_selected_engine",
                "multi_ocr_results", "multi_classified"]:
        context.user_data.pop(key, None)
    context.user_data["bot_state"] = BotState.NORMAL

    logging.info(f"[MULTI] User {update.effective_user.id} returned to main menu")

    await show_main_menu(
        update.message,
        "↩️ Đã hủy và quay về menu chính.\n\n"
        "Chọn chức năng bên dưới:"
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: "▶️ Tiến hành đọc" — verify images, show model selector
# ─────────────────────────────────────────────
async def handle_multi_proceed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """User pressed Tiến hành đọc — validate image count then show model picker."""
    bot_state = context.user_data.get("bot_state")
    if bot_state != BotState.MULTI_IMAGE_COLLECTING:
        # Not in collecting state — silently ignore
        return

    images = context.user_data.get("multi_images", [])
    if not images:
        await update.message.reply_text(
            "⚠️ Vui lòng gửi ít nhất một hình trước khi tiến hành đọc."
        )
        return

    context.user_data["bot_state"] = BotState.MULTI_IMAGE_SELECTING_MODEL
    logging.info(f"[MULTI] User {update.effective_user.id} proceeding with {len(images)} image(s)")

    # Hide bottom keyboard, then show inline model selector
    await hide_bottom_menu(
        update.message,
        f"🤖 Đã nhận <b>{len(images)}</b> hình.\n\n"
        "Vui lòng chọn model OCR:",
    )
    await update.message.reply_text(
        "Chọn model:",
        reply_markup=_multi_engine_keyboard(),
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: OCR model selected → run OCR concurrently
# ─────────────────────────────────────────────
async def handle_multi_engine_choice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle multi-image OCR model callback → run OCR on all images concurrently."""
    query = update.callback_query
    await query.answer()

    if context.user_data.get("bot_state") != BotState.MULTI_IMAGE_SELECTING_MODEL:
        await query.edit_message_text("⚠️ Trạng thái không hợp lệ. Vui lòng bắt đầu lại.")
        return

    engine = query.data  # e.g. "multi_engine:gemini2"
    if engine == CB_MULTI_PADDLE:
        engine_name = "Paddle OCR"
        endpoint    = PADDLE_ENDPOINT
    else:
        model_map = {
            CB_MULTI_GEMINI_1: (GEMINI_MODEL,   1),
            CB_MULTI_GEMINI_2: (GEMINI_MODEL_2, 2),
            CB_MULTI_GEMINI_3: (GEMINI_MODEL_3, 3),
            CB_MULTI_GEMINI_4: (GEMINI_MODEL_4, 4),
            CB_MULTI_GEMINI_5: (GEMINI_MODEL_5, 5),
        }
        model_name, model_index = model_map.get(engine, (GEMINI_MODEL, 1))
        engine_name = f"Gemini {model_name}"
        endpoint    = f"{GEMINI_ENDPOINT}?model_index={model_index}"

    context.user_data["multi_selected_engine"] = engine_name
    context.user_data["bot_state"] = BotState.MULTI_IMAGE_PROCESSING

    images: list = context.user_data.get("multi_images", [])
    user_id = query.from_user.id

    # Show a single progress message (edit it as we go)
    await query.edit_message_text(
        f"⏳ Đang đọc <b>{len(images)}</b> hình ảnh với <b>{html.escape(engine_name)}</b>...\n\n"
        "Vui lòng đợi trong giây lát.",
        parse_mode="HTML",
    )

    logging.info(f"[MULTI] Starting concurrent OCR for {len(images)} images | engine={engine_name}")

    # ── Concurrent OCR (timed) ───────────────────────────────────────────────
    import time as _time
    tasks = [
        _ocr_one_image(context.bot, file_id, endpoint, i + 1)
        for i, file_id in enumerate(images)
    ]
    _ocr_start = _time.perf_counter()
    try:
        raw_results = await asyncio.gather(*tasks)
        _ocr_elapsed = round(_time.perf_counter() - _ocr_start, 1)
        logging.info(f"[MULTI] OCR finished in {_ocr_elapsed}s for {len(images)} images")
    except Exception:
        _ocr_elapsed = round(_time.perf_counter() - _ocr_start, 1)
        logging.exception(f"[MULTI] Unexpected error during asyncio.gather after {_ocr_elapsed}s")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="❌ Không thể đọc hình ảnh.\n\nVui lòng thử lại."
        )
        context.user_data["bot_state"] = BotState.MULTI_IMAGE_COLLECTING
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Quay lại thu thập ảnh:",
            reply_markup=MULTI_IMAGE_KEYBOARD,
        )
        return

    # ── Apply overrides & classification ────────────────────────────────────
    load_user_config(user_id, context)
    classified = []
    seen_codes: set[str] = set()

    for raw in raw_results:
        kv = raw["kv"]
        error = raw["error"]

        if error or not kv:
            classified.append({
                "kv": kv, "markdown": raw.get("markdown", ""),
                "valid": False, "duplicate": False, "within_batch_dup": False,
                "error": error or "OCR trả về rỗng",
            })
            continue

        # Apply workshop/location from session
        kv = _apply_context_overrides(kv, context, user_id)
        # Hardcode status for multi-image flow
        kv["status"] = "Đang hoạt động"

        mmtb = _get_kv(kv, "Mã MMTB").strip()
        is_valid = (
            validate_mmtb(mmtb)
            and mmtb not in ("", "—", "CHƯA CÓ MÃ")
        )

        # Within-batch duplicate check
        within_batch_dup = False
        if is_valid:
            norm_code = mmtb.upper()
            if norm_code in seen_codes:
                within_batch_dup = True
            else:
                seen_codes.add(norm_code)

        classified.append({
            "kv": kv, "markdown": raw.get("markdown", ""),
            "valid": is_valid,
            "duplicate": False,       # Google Sheet check happens next
            "within_batch_dup": within_batch_dup,
            "error": None,
        })

    # ── Google Sheet duplication check (sequential to avoid rate-limits) ─────
    try:
        for m in classified:
            if not m["valid"] or m["within_batch_dup"]:
                continue
            mmtb = _get_kv(m["kv"], "Mã MMTB").strip()
            m["duplicate"] = await duplicate_service.check_duplication_code(mmtb)
    except Exception as e:
        logging.exception("[MULTI] Error during duplication check")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="❌ Không thể kiểm tra dữ liệu trên Excel.\n\nVui lòng thử lại sau."
        )
        context.user_data["bot_state"] = BotState.MULTI_IMAGE_COLLECTING
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Quay lại thu thập ảnh:",
            reply_markup=MULTI_IMAGE_KEYBOARD,
        )
        return

    context.user_data["multi_classified"] = classified
    context.user_data["bot_state"] = BotState.MULTI_IMAGE_REVIEW

    # ── Show results ─────────────────────────────────────────────────────────
    result_text = _format_multi_results(classified, _ocr_elapsed)

    valid_saveable = sum(
        1 for m in classified
        if m["valid"] and not m["duplicate"] and not m["within_batch_dup"]
    )

    # Edit the progress message to show summary
    try:
        await query.edit_message_text(
            result_text,
            parse_mode="HTML",
        )
    except Exception:
        # If edit fails (e.g. message too old), send a new one
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=result_text,
            parse_mode="HTML",
        )

    # Confirmation message with inline buttons (keyboard stays hidden)
    if valid_saveable > 0:
        confirm_text = (
            f"Bạn có muốn lưu <b>{valid_saveable}</b> thiết bị hợp lệ vào Google Sheets không?"
        )
    else:
        confirm_text = (
            "⚠️ Không có thiết bị hợp lệ nào để lưu.\n"
            "Tất cả đều không hợp lệ hoặc đã tồn tại."
        )

    save_kb = [
        [InlineKeyboardButton("💾 Lưu dữ liệu máy", callback_data=CB_MULTI_SAVE)],
        [InlineKeyboardButton("❌ Hủy",              callback_data=CB_MULTI_CANCEL)],
    ]
    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text=confirm_text,
        reply_markup=InlineKeyboardMarkup(save_kb),
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: "💾 Lưu dữ liệu máy" — save valid machines
# ─────────────────────────────────────────────
async def handle_multi_save_confirm(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if context.user_data.get("bot_state") != BotState.MULTI_IMAGE_REVIEW:
        await query.edit_message_text("⚠️ Trạng thái không hợp lệ. Vui lòng bắt đầu lại.")
        return

    context.user_data["bot_state"] = BotState.MULTI_IMAGE_SAVING
    classified: list = context.user_data.get("multi_classified", [])

    valid_machines = [
        m for m in classified
        if m["valid"] and not m["duplicate"] and not m["within_batch_dup"]
    ]

    if not valid_machines:
        await query.edit_message_text(
            "⚠️ Không có thiết bị hợp lệ nào để lưu.\n"
            "Quay lại thu thập ảnh:",
        )
        context.user_data["bot_state"] = BotState.MULTI_IMAGE_COLLECTING
        context.user_data["multi_images"] = []
        context.user_data.pop("multi_classified", None)
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="Sẵn sàng nhận ảnh mới:",
            reply_markup=MULTI_IMAGE_KEYBOARD,
        )
        return

    await query.edit_message_text(
        f"💾 Đang lưu <b>{len(valid_machines)}</b> thiết bị vào Google Sheets...",
        parse_mode="HTML",
    )
    logging.info(f"[MULTI] Saving {len(valid_machines)} valid machines")

    try:
        ocr_results = [
            OCRResult(markdown=m["markdown"], key_value=m["kv"])
            for m in valid_machines
        ]
        await sheet_service.append_results(ocr_results, "TELEGRAM_MULTI", force=False)
        logging.info(f"[MULTI] Saved {len(valid_machines)} machines successfully")

        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text=(
                f"✅ <b>Đã lưu thành công {len(valid_machines)} thiết bị vào Google Sheets!</b>\n\n"
                "Bạn có thể tiếp tục gửi ảnh để đọc lô tiếp theo."
            ),
            parse_mode="HTML",
        )
    except Exception as e:
        logging.exception("[MULTI] Failed to save to Google Sheets")
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="❌ Không thể lưu dữ liệu.\n\nVui lòng thử lại."
        )

    # ── Restore multi-image state for next batch ──────────────────────────
    context.user_data["bot_state"]   = BotState.MULTI_IMAGE_COLLECTING
    context.user_data["multi_images"] = []
    context.user_data.pop("multi_classified", None)
    context.user_data.pop("multi_selected_engine", None)

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="📚 Sẵn sàng nhận lô ảnh mới. Gửi ảnh hoặc nhấn <b>↩️ Trở về</b> để về menu chính.",
        reply_markup=MULTI_IMAGE_KEYBOARD,
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────
# MULTI-IMAGE: "❌ Hủy" — cancel review, return to collecting
# ─────────────────────────────────────────────
async def handle_multi_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    await query.edit_message_text(
        "❌ Đã hủy. Không có dữ liệu nào được lưu."
    )

    # Clear batch but stay in multi-image mode
    context.user_data["bot_state"]   = BotState.MULTI_IMAGE_COLLECTING
    context.user_data["multi_images"] = []
    context.user_data.pop("multi_classified", None)
    context.user_data.pop("multi_selected_engine", None)

    logging.info(f"[MULTI] User {query.from_user.id} cancelled review; returning to collecting")

    await context.bot.send_message(
        chat_id=query.message.chat_id,
        text="📚 Sẵn sàng nhận lô ảnh mới. Gửi ảnh hoặc nhấn <b>↩️ Trở về</b> để về menu chính.",
        reply_markup=MULTI_IMAGE_KEYBOARD,
        parse_mode="HTML",
    )


# ─────────────────────────────────────────────
# Fallback: unexpected text outside EDIT_FIELD state
# ─────────────────────────────────────────────
async def handle_unexpected_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("waiting_for_location"):
        location_value = update.message.text.strip()
        if not location_value:
            await update.message.reply_text("⚠️ Vị trí không được để trống. Vui lòng nhập lại:")
            return

        user_id = update.effective_user.id
        save_user_location(user_id, location_value, context)
        context.user_data["waiting_for_location"] = False
        safe_val = html.escape(location_value)
        await update.message.reply_text(
            f"📍 Đã thiết lập vị trí mặc định là: <b>{safe_val}</b>\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên vị trí khi OCR kết thúc.",
            parse_mode="HTML"
        )
        return

    await update.message.reply_text(
        "⚠️ Hãy gửi ảnh để bắt đầu, hoặc dùng /cancel để huỷ thao tác hiện tại."
    )


# Main
# ─────────────────────────────────────────────
config_conv_handler = ConversationHandler(
    entry_points=[
        CommandHandler("x", start_config_cmd),
        MessageHandler(filters.Regex(r'^/(x)\b'), start_config_cmd),
    ],
    states={
        CFG_SELECT_FACTORY: [
            CallbackQueryHandler(handle_cfg_factory_choice, pattern="^cf_f:"),
        ],
        CFG_SELECT_WORKSHOP: [
            CallbackQueryHandler(handle_cfg_workshop_choice, pattern="^cf_w:"),
        ],
        CFG_SELECT_LETTER: [
            CallbackQueryHandler(handle_cfg_letter_choice, pattern="^cf_l:"),
        ],
        CFG_SELECT_NUMBER: [
            CallbackQueryHandler(handle_cfg_number_choice, pattern="^cf_n:"),
            CallbackQueryHandler(handle_cfg_back, pattern="^cf_back$"),
        ],
    },
    fallbacks=[
        CommandHandler("cancel", cancel_config),
        CommandHandler("start", start),
    ],
    allow_reentry=True,
)

conv_handler = ConversationHandler(
    entry_points=[
        MessageHandler(filters.PHOTO, handle_photo),
        CommandHandler("f", find_mmtb_cmd),      # /f <mã> — find & edit flow
        CommandHandler("n", new_manual_cmd),      # /n — manual add flow
    ],
    states={
        SELECT_ENGINE: [
            CallbackQueryHandler(handle_engine_choice, pattern="^engine:"),
        ],
        SELECT_STATUS: [
            CallbackQueryHandler(handle_status_choice, pattern="^status:"),
        ],
        CONFIRM_RESULT: [
            CallbackQueryHandler(handle_confirmation, pattern="^confirm:"),
        ],
        EDIT_FIELD: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, handle_field_input),
            CallbackQueryHandler(handle_status_callback,    pattern="^status:"),
            CallbackQueryHandler(handle_edit_keep_callback, pattern="^edit:keep"),
            CallbackQueryHandler(handle_edit_skip_callback, pattern="^edit:skip"),
        ],
        CONFIRM_SAVE: [
            CallbackQueryHandler(handle_save_confirmation,        pattern="^save:"),
            CallbackQueryHandler(handle_find_save_confirmation,   pattern="^find_save:"),
            CallbackQueryHandler(handle_manual_save_confirmation, pattern="^manual_save:"),
        ],
        # ── /f find-and-edit states ───────────────────────────────────────
        FIND_ASK_EDIT: [
            CallbackQueryHandler(handle_find_edit_choice, pattern="^find:"),
        ],
        MMTB_WARN: [
            CallbackQueryHandler(handle_mmtb_warn_callback, pattern=f"^({CB_MMTB_FORCE_SAVE}|{CB_MMTB_EDIT})$"),
        ],
        DUP_WARN: [
            CallbackQueryHandler(handle_dup_warn_callback, pattern=f"^({CB_DUP_FORCE_SAVE}|{CB_DUP_CANCEL})$"),
        ],
    },
    fallbacks=[
        CommandHandler("cancel", cancel),
        CommandHandler("start",  start),
        MessageHandler(filters.PHOTO, handle_photo),  # allow sending a new photo mid-flow
    ],
    allow_reentry=True,
)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # ── Priority group -1: intercept multi-image photos BEFORE conv_handler ──
    app.add_handler(
        MessageHandler(filters.PHOTO, handle_multi_image_photo),
        group=-1,
    )

    # ── Priority group 0 (default) ────────────────────────────────────────────
    app.add_handler(CommandHandler("start",  start))
    app.add_handler(CommandHandler("help",   help_cmd))
    app.add_handler(CommandHandler("cancel", cancel))

    # ── Main menu ReplyKeyboard text buttons ──────────────────────────────────
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^📚 Đọc nhiều hình$'),
        handle_enter_multi_image,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^📷 Đọc 1 hình$'),
        handle_main_menu_single,
    ))

    # ── Multi-image flow text buttons ─────────────────────────────────────────
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^↩️ Trở về$'),
        handle_multi_go_back,
    ))
    app.add_handler(MessageHandler(
        filters.TEXT & filters.Regex(r'^▶️ Tiến hành đọc$'),
        handle_multi_proceed,
    ))

    # ── Multi-image inline callbacks ──────────────────────────────────────────
    app.add_handler(CallbackQueryHandler(handle_multi_engine_choice,  pattern="^multi_engine:"))
    app.add_handler(CallbackQueryHandler(handle_multi_save_confirm,    pattern=f"^{CB_MULTI_SAVE}$"))
    app.add_handler(CallbackQueryHandler(handle_multi_cancel,          pattern=f"^{CB_MULTI_CANCEL}$"))

    # ── Existing handlers (unchanged) ─────────────────────────────────────────
    app.add_handler(config_conv_handler)
    app.add_handler(MessageHandler(filters.Regex(r'^/ws\b'), ws_redirect_cmd))
    app.add_handler(MessageHandler(filters.Regex(r'^/(l)\b'), set_location_cmd))
    app.add_handler(conv_handler)
    app.add_handler(CallbackQueryHandler(handle_position_adjustment_callback, pattern="^pos_set:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unexpected_text))

    logging.info(f"Bot running | Paddle: {PADDLE_ENDPOINT} | Gemini: {GEMINI_ENDPOINT}")
    app.run_polling()
import os
import json
import logging
import asyncio
import httpx

from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

from app.helpers import append_results_to_sheet_sync, lookup_row_by_mmtb_sync, update_row_in_sheet_sync
from app.schemas import OCRResult
import unicodedata
import re
import html



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
SELECT_ENGINE, SELECT_STATUS, CONFIRM_RESULT, EDIT_FIELD, CONFIRM_SAVE, FIND_ASK_EDIT = range(6)

# Callback data constants
CB_PADDLE          = "engine:paddle"
CB_GEMINI          = "engine:gemini"
CB_CONFIRM_YES     = "confirm:yes"
CB_CONFIRM_EDIT    = "confirm:edit"
CB_SAVE_CONFIRM    = "save:confirm"
CB_SAVE_EDIT       = "save:edit_again"
# /f find-and-edit flow
CB_FIND_EDIT_YES   = "find:edit_yes"
CB_FIND_EDIT_NO    = "find:edit_no"
CB_FIND_SAVE       = "find_save:confirm"
CB_FIND_EDIT_AGAIN = "find_save:edit_again"

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
def save_user_workshop(user_id: int, workshop: str):
    data = _load_settings()
    data[str(user_id)] = workshop
    _save_settings(data)

def get_user_workshop(user_id: int) -> str:
    return _load_settings().get(str(user_id), "")

# ── Location (Vị trí) ─────────────────────────
def save_user_location(user_id: int, location: str):
    data = _load_settings()
    data[f"loc_{user_id}"] = location
    _save_settings(data)

def get_user_location(user_id: int) -> str:
    return _load_settings().get(f"loc_{user_id}", "")


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
# /ws command — set persistent Xưởng
# ─────────────────────────────────────────────
async def create_workshop_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    match = re.match(r'^/(?:x)(?:\s+(.+))?$', text, re.IGNORECASE)
    
    workshop_name = match.group(1).strip() if (match and match.group(1)) else ""
    user_id = update.effective_user.id
    
    if workshop_name:
        save_user_workshop(user_id, workshop_name)
        await update.message.reply_text(
            f"🏭 Đã thiết lập xưởng mặc định là: *{workshop_name}*\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên xưởng khi OCR kết thúc.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END
    else:
        current_workshop = get_user_workshop(user_id)
        if current_workshop:
            await update.message.reply_text(
                f"🏭 Xưởng mặc định hiện tại của bạn: *{current_workshop}*\n\n"
                "Vui lòng nhập tên xưởng mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "🏭 Bạn chưa thiết lập xưởng mặc định.\n\n"
                "Vui lòng nhập tên xưởng mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="Markdown",
            )
        context.user_data["waiting_for_workshop"] = True
        return ConversationHandler.END


# ─────────────────────────────────────────────
# /col command — set persistent Vị trí
# ─────────────────────────────────────────────
async def set_location_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    match = re.match(r'^/(?:col|l)(?:\s+(.+))?$', text, re.IGNORECASE)

    location_value = match.group(1).strip() if (match and match.group(1)) else ""
    user_id = update.effective_user.id

    if location_value:
        save_user_location(user_id, location_value)
        await update.message.reply_text(
            f"📍 Đã thiết lập vị trí mặc định là: *{location_value}*\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên vị trí khi OCR kết thúc.",
            parse_mode="Markdown",
        )
        return ConversationHandler.END
    else:
        current_location = get_user_location(user_id)
        if current_location:
            await update.message.reply_text(
                f"📍 Vị trí mặc định hiện tại của bạn: *{current_location}*\n\n"
                "Vui lòng nhập vị trí mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="Markdown",
            )
        else:
            await update.message.reply_text(
                "📍 Bạn chưa thiết lập vị trí mặc định.\n\n"
                "Vui lòng nhập vị trí mới muốn thiết lập (hoặc gửi /cancel để huỷ):",
                parse_mode="Markdown",
            )
        context.user_data["waiting_for_location"] = True
        return ConversationHandler.END


# ─────────────────────────────────────────────
# /start command
# ─────────────────────────────────────────────
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.clear()
    user_id = update.effective_user.id
    
    current_location = html.escape(get_user_location(user_id) or "Chưa thiết lập ❌")
    current_workshop = html.escape(get_user_workshop(user_id) or "Chưa thiết lập ❌")
    
    await update.message.reply_text(
        "👋 <b>Chào mừng đến với Equipment OCR Bot!</b>\n\n"
        "⚙️ <b>Cấu hình mặc định hiện tại của bạn:</b>\n"
        f"📍 Vị trí mặc định: <b>{current_location}</b>\n"
        f"🏭 Xưởng mặc định: <b>{current_workshop}</b>\n"
        "<i>(Sử dụng lệnh <code>/l [vị_trí]</code> và <code>/x [tên_xưởng]</code> để thay đổi)</i>\n\n"
        "📋 <b>Quy trình thêm thiết bị:</b>\n"
        "1️⃣ <b>Gửi ảnh</b> hoặc PDF nhãn máy/thiết bị vào chat\n"
        "2️⃣ <b>Chọn OCR Engine</b> (Paddle hoặc Gemini)\n"
        "3️⃣ <b>Kiểm tra thông tin</b> trích xuất (đặc biệt là Vị trí & Xưởng)\n"
        "4️⃣ <b>Xác nhận</b> hoặc <b>Sửa đổi</b> các trường nếu có sai lệch\n"
        "5️⃣ Dữ liệu sẽ tự động lưu vào Google Sheets\n\n"
        "📸 Hãy gửi ảnh hoặc PDF nhãn thiết bị để bắt đầu!",
        parse_mode="HTML",
    )
    return ConversationHandler.END


# ─────────────────────────────────────────────
# /help command
# ─────────────────────────────────────────────
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📖 *Hướng dẫn sử dụng*\n\n"
        "1. Gửi ảnh nhãn máy/thiết bị vào chat\n"
        "2. Chọn engine OCR (Paddle hoặc Gemini)\n"
        "3. Kiểm tra kết quả và xác nhận hoặc sửa\n"
        "4. Dữ liệu được lưu vào Google Sheets\n\n"
        "*Lệnh:*\n"
        "/start  — Bắt đầu lại\n"
        "/cancel — Huỷ thao tác hiện tại\n"
        "/x [tên_xưởng] — Thiết lập xưởng mặc định\n"
        "/l [vị_trí]   — Thiết lập vị trí mặc định\n"
        "/f [mã_mmtb]   — Tìm MMTB có sẵn\n"
        "/help   — Xem hướng dẫn này",
        parse_mode="Markdown",
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

    keyboard = [[
        InlineKeyboardButton("🔵 Paddle OCR", callback_data=CB_PADDLE),
        InlineKeyboardButton("✨ Gemini OCR", callback_data=CB_GEMINI),
    ]]
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
    engine_name = "Paddle OCR" if engine == CB_PADDLE else "Gemini OCR"
    endpoint    = PADDLE_ENDPOINT if engine == CB_PADDLE else GEMINI_ENDPOINT

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

        # ── Apply Persistent Workshop Overwrite ───────
        persistent_workshop = get_user_workshop(update.effective_user.id)
        if persistent_workshop:
            keys_to_remove = []
            for k in kv:
                if k.lower().strip() in ["xưởng", "xuong", "xuong san xuat", "nha may", "xương"]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del kv[k]
            kv["Xưởng"] = persistent_workshop
            logging.info(f"Overwrote workshop with persistent value: {persistent_workshop} for user {update.effective_user.id}")

        # ── Apply Persistent Location Overwrite ────────
        persistent_location = get_user_location(update.effective_user.id)
        if persistent_location:
            keys_to_remove = []
            for k in kv:
                if k.lower().strip() in ["vị trí", "vi tri", "vitri", "vi tri ", "tri"]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del kv[k]
            kv["Vị trí"] = persistent_location
            logging.info(f"Overwrote location with persistent value: {persistent_location} for user {update.effective_user.id}")

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

        status_kb = [[
            InlineKeyboardButton("🟢 Đang hoạt động", callback_data="status:active"),
            InlineKeyboardButton("🔴 Ngưng hoạt động", callback_data="status:inactive"),
            InlineKeyboardButton("⚠️ Đã thanh lý", callback_data="status:disposed"),
        ]]
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
        "status:active": "Đang hoạt động",
        "status:inactive": "Ngưng hoạt động",
        "status:disposed": "Đã thanh lý",
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


# ═════════════════════════════════════════════
# STEP 3 — Confirmation response
# ═════════════════════════════════════════════
async def handle_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == CB_CONFIRM_YES:
        # Guard: prevent double-save if handler is somehow re-entered
        if context.user_data.get("sheets_saved"):
            await query.edit_message_text(
                "✅ Dữ liệu đã được lưu rồi. Gửi ảnh mới để tiếp tục."
            )
            context.user_data.clear()
            return ConversationHandler.END

        # Save to Google Sheets — triggered ONLY by explicit user confirmation
        await query.edit_message_text(
            "💾 *Đang lưu dữ liệu vào Google Sheets...*",
            parse_mode="Markdown",
        )
        logging.info("[BOT] Saving to Google Sheets triggered from TELEGRAM_CONFIRM step")
        try:
            kv         = context.user_data["kv"]
            ocr_result = OCRResult(
                markdown=context.user_data.get("markdown_text", ""),
                key_value=kv,
            )
            await asyncio.to_thread(
                append_results_to_sheet_sync, [ocr_result], "TELEGRAM_CONFIRM"
            )
            context.user_data["sheets_saved"] = True
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ <b>Thông tin đã xác nhận và lưu vào Google Sheets!</b>\n\n"
                     "💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
                     "Gửi ảnh mới để tiếp tục.",
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
                InlineKeyboardButton("⚠️ Đã thanh lý", callback_data="status:disposed"),
            ],
            [
                InlineKeyboardButton("↩️ Giữ nguyên", callback_data="status:keep"),
            ]
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

    # Advance to next field
    context.user_data["field_index"] = idx + 1
    return await _ask_next_field(context, update.message.chat_id)


async def handle_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    status_data = query.data
    status_map = {
        "status:active": "Đang hoạt động",
        "status:inactive": "Ngưng hoạt động",
        "status:disposed": "Đã thanh lý",
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
    await query.edit_message_text(
        text=f"↩️ Giữ nguyên *{label}*: `{current}`",
        parse_mode="Markdown",
    )

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
    is_find_edit = context.user_data.get("is_find_edit", False)
    save_cb = CB_FIND_SAVE       if is_find_edit else CB_SAVE_CONFIRM
    edit_cb = CB_FIND_EDIT_AGAIN if is_find_edit else CB_SAVE_EDIT

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

    # ── CB_SAVE_CONFIRM: append corrected row to Google Sheets ──
    # Guard: prevent double-save if handler is somehow re-entered
    if context.user_data.get("sheets_saved"):
        await query.edit_message_text(
            "✅ Dữ liệu đã được lưu rồi. Gửi ảnh mới để tiếp tục."
        )
        context.user_data.clear()
        return ConversationHandler.END

    await query.edit_message_text(
        "💾 *Đang lưu dữ liệu đã sửa vào Google Sheets...*",
        parse_mode="Markdown",
    )
    logging.info("[BOT] Saving to Google Sheets triggered from TELEGRAM_CORRECTED_CONFIRM step")

    try:
        kv         = context.user_data["kv"]
        ocr_result = OCRResult(
            markdown=context.user_data.get("markdown_text", ""),
            key_value=kv,
        )
        # Run the synchronous gspread call in a thread to not block the event loop
        await asyncio.to_thread(
            append_results_to_sheet_sync, [ocr_result], "TELEGRAM_CORRECTED_CONFIRM"
        )
        context.user_data["sheets_saved"] = True
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="✅ <b>Dữ liệu đã sửa được lưu thành công vào Google Sheets!</b>\n\n"
                 "💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
                 "Gửi ảnh mới để tiếp tục.",
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
        result = await asyncio.to_thread(lookup_row_by_mmtb_sync, mmtb_code)
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

        success = await asyncio.to_thread(
            update_row_in_sheet_sync, row_index, row_data, "FIND_UPDATE"
        )

        if success:
            context.user_data["sheets_saved"] = True
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text=(
                    f"✅ <b>Đã cập nhật thành công mã MMTB <code>{html.escape(mmtb_code)}</code> vào Google Sheets!</b>\n\n"
                    "💡 <b>Nhắc nhở:</b> Đừng quên kiểm tra và cập nhật vị trí mặc định của bạn bằng lệnh <code>/l [vị_trí]</code> (nếu cần thiết) khi đổi khu vực nhé!\n\n"
                    "Gửi ảnh mới hoặc dùng /f để tiếp tục."
                ),
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


# ─────────────────────────────────────────────
# Fallback: unexpected text outside EDIT_FIELD state
# ─────────────────────────────────────────────
async def handle_unexpected_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get("waiting_for_workshop"):
        workshop_name = update.message.text.strip()
        if not workshop_name:
            await update.message.reply_text("⚠️ Tên xưởng không được để trống. Vui lòng nhập lại:")
            return
        
        user_id = update.effective_user.id
        save_user_workshop(user_id, workshop_name)
        context.user_data["waiting_for_workshop"] = False
        await update.message.reply_text(
            f"🏭 Đã thiết lập xưởng mặc định là: *{workshop_name}*\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên xưởng khi OCR kết thúc.",
            parse_mode="Markdown"
        )
        return

    if context.user_data.get("waiting_for_location"):
        location_value = update.message.text.strip()
        if not location_value:
            await update.message.reply_text("⚠️ Vị trí không được để trống. Vui lòng nhập lại:")
            return

        user_id = update.effective_user.id
        save_user_location(user_id, location_value)
        context.user_data["waiting_for_location"] = False
        await update.message.reply_text(
            f"📍 Đã thiết lập vị trí mặc định là: *{location_value}*\n"
            "Giá trị này sẽ được tự động áp dụng và ghi đè lên vị trí khi OCR kết thúc.",
            parse_mode="Markdown"
        )
        return

    await update.message.reply_text(
        "⚠️ Hãy gửi ảnh để bắt đầu, hoặc dùng /cancel để huỷ thao tác hiện tại."
    )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
conv_handler = ConversationHandler(
    entry_points=[
        MessageHandler(filters.PHOTO, handle_photo),
        CommandHandler("f", find_mmtb_cmd),      # /f <mã> — find & edit flow
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
            CallbackQueryHandler(handle_status_callback, pattern="^status:"),
            CallbackQueryHandler(handle_edit_keep_callback, pattern="^edit:keep"),
        ],
        CONFIRM_SAVE: [
            CallbackQueryHandler(handle_save_confirmation,      pattern="^save:"),
            CallbackQueryHandler(handle_find_save_confirmation, pattern="^find_save:"),
        ],
        # ── /f find-and-edit states ───────────────────────────────────────
        FIND_ASK_EDIT: [
            CallbackQueryHandler(handle_find_edit_choice, pattern="^find:"),
        ],
    },
    fallbacks=[
        CommandHandler("cancel", cancel),
        CommandHandler("start",  start),
        MessageHandler(filters.PHOTO, handle_photo),  # allow sending a new photo mid-flow
    ],
    allow_reentry=True,
)

app = ApplicationBuilder().token(BOT_TOKEN).build()

app.add_handler(CommandHandler("start",  start))
app.add_handler(CommandHandler("help",   help_cmd))
app.add_handler(CommandHandler("cancel", cancel))
app.add_handler(MessageHandler(filters.Regex(r'^/(x)\b'), create_workshop_cmd))
app.add_handler(MessageHandler(filters.Regex(r'^/(l)\b'), set_location_cmd))
app.add_handler(conv_handler)
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_unexpected_text))

print(f"🤖 Bot running | Paddle: {PADDLE_ENDPOINT} | Gemini: {GEMINI_ENDPOINT}")
app.run_polling()
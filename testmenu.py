from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
import os




# Persistent menu
MAIN_MENU = [
    ["🛍️ Sản phẩm", "💬 Hỗ trợ"],
    ["👛 Ví"],
]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = ReplyKeyboardMarkup(
        MAIN_MENU,
        resize_keyboard=True,
        is_persistent=True,
    )

    await update.message.reply_text(
        "👋 Xin chào!\n\nVui lòng chọn một chức năng:",
        reply_markup=keyboard,
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text

    if text == "🛍️ Sản phẩm":
        await update.message.reply_text(
            "🛍️ Bạn đã chọn Sản phẩm."
        )

    elif text == "💬 Hỗ trợ":
        await update.message.reply_text(
            "💬 Bạn đã chọn Hỗ trợ."
        )

    elif text == "👛 Ví":
        await update.message.reply_text(
            "👛 Bạn đã chọn Ví."
        )

    else:
        await update.message.reply_text(
            f"Bạn gửi: {text}"
        )


def main():
    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))

    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
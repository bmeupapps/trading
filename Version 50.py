import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.dates import DateFormatter
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.lib.units import inch
import webbrowser
import urllib.parse
import threading

# --- Load Sim Data ---
sim_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sim Data.txt")
def load_historical(symbol):
    with open(sim_data_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    for entry in data:
        if entry["symbol"].upper() == symbol.upper():
            return entry["companyName"], entry["price"], entry["historical"]
    return symbol, 100.0, []

# --- Inputs ---
stock_symbol = "AAPL"
hedge_symbol = "TSLA"
short_call = 220
long_call = 230
premium = 2.5
contract_size = 1
hedge_put_price = 260
expiration = "28/06/25"
delta = -0.45
hedge_delta = -0.50
target_price = 215.0
risk_reward = round((premium * 100) / ((long_call - short_call - premium) * 100), 1)

stock_name, _, stock_data = load_historical(stock_symbol)
hedge_name, _, hedge_data = load_historical(hedge_symbol)

def parse_dates(data):
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
    prices = [d["close"] for d in data]
    return np.array(dates), np.array(prices)

def simplify_xaxis(ax):
    ax.xaxis.set_major_formatter(DateFormatter('%m'))

# --- Chart Functions ---

# Chart 1: P&L Chart – scale factor 0.9 (width reduced by 10); height remains original.
def create_pl_chart():
    scale = 1.0
    fig, ax = plt.subplots(figsize=(6 * scale, 2.5 * scale))
    x = np.linspace(short_call - 20, long_call + 20, 500)
    y = np.where(
        x <= short_call,
        premium * 100,
        np.where(
            x < long_call,
            (premium - (x - short_call)) * 100,
            (premium - (long_call - short_call)) * 100
        )
    )
    ax.plot(x, y, linewidth=1.5, label="P&L")
    ax.axhline(premium * 100, color="green", linestyle="--", linewidth=2,
               label="Max Profit ($" + str(int(premium * 100)) + ")")
    ax.axhline((premium - (long_call - short_call)) * 100, color="red", linestyle="--", linewidth=2,
               label="Max Loss ($" + str(int((premium - (long_call - short_call)) * 100)) + ")")
    ax.axvline(short_call + premium, color="gray", linestyle="--", linewidth=2,
               label="Breakeven ($" + str(int(short_call + premium)) + ")")
    
    break_even = short_call + premium
    ax.fill_between(x, y, where=(x >= break_even), facecolor="green", alpha=0.3)
    ax.fill_between(x, y, where=(x < break_even), facecolor="red", alpha=0.3)
    
    ax.set_title(f"{stock_name} Bear Call Spread Report - P&L Chart", fontsize=10)
    ax.grid(True)
    ax.legend(fontsize=8)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf
    
# Hedge Stock Chart: Reduced by 10% (scale factor 0.90)
def create_hedge_chart(dates, prices):
    scale = 0.90
    fig, ax = plt.subplots(figsize=(6 * scale, 2.5 * scale))
    ax.plot(dates, prices, label='Hedge Price', linewidth=1.5)
    if len(prices) >= 20:
        window = 20
        ma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
        upper = ma + 2 * std
        lower = ma - 2 * std
        valid_dates = dates[window-1:]
        ax.plot(valid_dates, upper, linestyle="--", color="blue", label="Upper Bollinger")
        ax.plot(valid_dates, lower, linestyle="--", color="orange", label="Lower Bollinger")
    ax.axhline(hedge_put_price, color='red', linestyle="--", linewidth=2,
               label="Hedge Put ($" + str(hedge_put_price) + ")")
    ax.text(0.5, 1.05, "Hedge Company: " + hedge_name, transform=ax.transAxes, ha='center', fontsize=9)
    simplify_xaxis(ax)
    ax.grid(True)
    ax.legend(fontsize=7)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

# Chart 2: Bollinger Bands Chart for AAPL – scale factor 0.8 (width reduced by 20)
def create_bollinger_chart(dates, prices):
    scale = 0.8
    fig, ax = plt.subplots(figsize=(6 * scale, 2.5))  # height unchanged
    window = 20
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
    upper = ma + 2 * std
    lower = ma - 2 * std
    valid_dates = dates[window-1:]
    ax.plot(dates, prices, label='Price', linewidth=1.2)
    ax.plot(valid_dates, upper, linestyle="--", label='Upper Band')
    ax.plot(valid_dates, lower, linestyle="--", label='Lower Band')
    simplify_xaxis(ax)
    ax.set_title(f"{stock_name} Bear Call Spread Report - Bollinger Bands", fontsize=9)
    ax.grid(True)
    ax.legend(fontsize=8)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Chart 3: Price Chart for AAPL – scale factor 0.9 (width reduced by 10)
def create_price_chart(dates, prices):
    scale = 0.9
    fig, ax = plt.subplots(figsize=(6 * scale, 2.5))  # height unchanged
    ax.plot(dates, prices, label='Price', color='black')
    simplify_xaxis(ax)
    ax.set_title(f"{stock_name} Bear Call Spread Report - Price Chart", fontsize=9)
    ax.grid(True)
    ax.legend(fontsize=8)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

# Chart 4: MACD Chart for AAPL – scale factor 0.8 (width reduced by 20)
# Restoring arrow annotations for convergence/divergence.
def create_macd_chart(dates, prices):
    scale = 0.8
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6 * scale, 5 * scale), sharex=True)
    exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')
    exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')
    macd = exp12[-len(exp26):] - exp26
    signal = np.convolve(macd, np.ones(9)/9, mode='valid')
    offset = len(prices) - len(signal)
    macd_dates = dates[offset:]
    macd_values = macd[-len(signal):]
    ax1.plot(macd_dates, prices[offset:], label='Price', color='black')
    ax1.set_title("Price", fontsize=9)
    simplify_xaxis(ax1)
    ax1.grid(True)
    ax2.plot(macd_dates, macd_values, label='MACD', color='blue', linewidth=1.5)
    ax2.plot(macd_dates, signal, label='Signal', color='red', linestyle="--", linewidth=1.5)
    ax2.set_title("MACD", fontsize=9)
    simplify_xaxis(ax2)
    ax2.grid(True)
    ax2.legend(fontsize=8)
    # Restore arrow annotations for convergence/divergence
    for i in range(1, len(signal)):
        diff_prev = macd_values[i-1] - signal[i-1]
        diff_curr = macd_values[i] - signal[i]
        if diff_prev * diff_curr < 0:
            color = "green" if diff_curr > 0 else "red"
            ax2.annotate('', xy=(macd_dates[i], macd_values[i]),
                         xytext=(macd_dates[i-1], macd_values[i-1]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax1.annotate('', xy=(macd_dates[i], prices[offset:][i]),
                         xytext=(macd_dates[i-1], prices[offset:][i-1]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2))
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    # Set overall title for MACD chart
    # We create a new image with title overlay.
    return buf

# --- Table Function (font 14; row spacing increased by 10%)
def draw_trade_table(c, width, height):
    y_start = height * 0.9
    c.setFont("Helvetica-Bold", 14)
    table = {
        "Stock": stock_symbol, "Delta": f"{delta:.2f}", "Premium": f"${premium:.2f}",
        "Target Price": f"${target_price:.2f}", "Short Call": f"${short_call:.2f}", "Long Call": f"${long_call:.2f}",
        "Contract Size": contract_size, "Expiration": expiration, "Risk/Reward": risk_reward,
        "Hedge Stock": hedge_symbol, "Hedge Delta": f"{hedge_delta:.2f}", "Hedge Put": f"${hedge_put_price:.2f}"
    }
    keys = list(table.keys())
    values = list(table.values())
    row_height = 17
    col_width = width / 3
    for row in range(5):
        for col in range(3):
            idx = row * 3 + col
            if idx < len(keys):
                c.drawString(col * col_width + 10, y_start - row * row_height, f"{keys[idx]}: {values[idx]}")

# --- Generate PDF Layout ---
def generate_pdf():
    filename = "Version 46_5.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Page 1: Title and table in top 30%; then P&L and Hedge charts in lower 70%
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height * 0.95, f"{stock_symbol} - {stock_name} - Bear Call Spread Report")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, height * 0.92, f"Hedging Stock: {hedge_symbol} - {hedge_name}")
    draw_trade_table(c, width, height)
    stock_dates, stock_prices = parse_dates(stock_data)
    hedge_dates, hedge_prices = parse_dates(hedge_data)
    chart_height = height * 0.35
    c.drawImage(ImageReader(create_pl_chart()), 0, 0, width=width, height=chart_height)
    c.drawImage(ImageReader(create_hedge_chart(hedge_dates, hedge_prices)), 0, chart_height, width=width, height=chart_height)
    c.showPage()

    # Page 2: Technical Indicators arranged vertically:
    # Bollinger Bands (40%), Price Chart (30%), MACD Chart (30%)
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 10, "Stock Technical Indicators")
    c.drawString(10, height - 30, "Bollinger Bands:")
    c.drawImage(ImageReader(create_bollinger_chart(stock_dates, stock_prices)), 0, height * 0.6, width=width, height=height * 0.4)
    c.drawString(10, height * 0.6 - 20, "Price Chart:")
    c.drawImage(ImageReader(create_price_chart(stock_dates, stock_prices)), 0, height * 0.3, width=width, height=height * 0.3)
    c.drawString(10, height * 0.3 - 20, "MACD:")
    c.drawImage(ImageReader(create_macd_chart(stock_dates, stock_prices)), 0, 0, width=width, height=height * 0.3)
    
    c.save()
    return filename

# --- MAIN EXECUTION ---
pdf_file = generate_pdf()
pdf_path = os.path.abspath(pdf_file)
encoded_pdf_path = urllib.parse.quote(pdf_path)
print("Generated PDF:", pdf_path)

def open_pdf():
    pdf_url = "file://" + encoded_pdf_path
    print("Opening PDF URL:", pdf_url)
    webbrowser.open(pdf_url)

# --- Backup and Commit Section ---
try:
    original_script = os.path.basename(__file__)
except NameError:
    original_script = "Version 46_5.py"
backup_script = "Version 46_5 Backup.py"
shutil.copyfile(original_script, backup_script)
if os.path.exists(backup_script):
    print("Backup file created:", backup_script)
else:
    print("ERROR: Backup file was not created!")

# --- Working Copy Config ---
secret_key = "ODE123456"  # Replace with your actual key
repo = "trading"
branch = "main"
commit_message = "Version 46_5"
filename_to_commit = backup_script

# --- URL Encoding for Working Copy ---
encoded_msg = urllib.parse.quote(commit_message, safe='')
encoded_file = urllib.parse.quote(filename_to_commit, safe='')
wc_url = ("working-copy://x-callback-url/commit?" +
          "key=" + secret_key +
          "&repo=" + repo +
          "&branch=" + branch +
          "&message=" + encoded_msg +
          "&paths%5B%5D=" + encoded_file +
          "&add=true")
print("Final Working Copy Commit URL:")
print(wc_url)

def open_wc_url():
    if wc_url.startswith("working-copy://"):
        print("Committing script via Working Copy...")
        print("Commit URL:", wc_url)
        webbrowser.open(wc_url)
    else:
        print("ERROR: Invalid Working Copy URL.")

# --- Timed Launch: Trigger WC sync then open PDF ---
threading.Timer(1.0, open_wc_url).start()
threading.Timer(3.0, open_pdf).start()

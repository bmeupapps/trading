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
short_call = 216
long_call = 220
premium = 2.5
contract_size = 1
hedge_put_price = 250
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
    max_profit = premium * 100
    max_loss = (premium - (long_call - short_call)) * 100
    breakeven = short_call + premium
    ax.axhline(max_profit, color="green", linestyle="--", linewidth=2,
               label="Max Profit ($" + str(int(max_profit)) + ")")
    ax.axhline(max_loss, color="red", linestyle="--", linewidth=2,
               label="Max Loss ($" + str(int(max_loss)) + ")")
    ax.axvline(breakeven, color="gray", linestyle="--", linewidth=2,
               label="Breakeven ($" + str(int(breakeven)) + ")")
    ax.fill_between(x, y, where=(x >= breakeven), facecolor="green", alpha=0.3)
    ax.fill_between(x, y, where=(x < breakeven), facecolor="red", alpha=0.3)
    y_values = [0, max_profit, max_loss]
    y_min = min(y_values)
    y_max = max(y_values)
    padding = (y_max - y_min) * 0.1 if y_max != y_min else 10
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.set_title(f"{stock_name} Bear Call Spread Report - P&L Chart", fontsize=10)
    ax.grid(True)
    ax.legend(fontsize=8)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_hedge_chart(dates, prices):
    scale = 0.90
    fig, ax = plt.subplots(figsize=(6 * scale, 2.5 * scale))
    ax.plot(dates, prices, label='Hedge Price', linewidth=1.5)
    y_min = min(prices)
    y_max = max(prices)
    if len(prices) >= 20:
        window = 20
        ma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
        upper = ma + 2 * std
        lower = ma - 2 * std
        valid_dates = dates[window-1:]
        ax.plot(valid_dates, upper, linestyle="--", color="blue", label="Upper Bollinger")
        ax.plot(valid_dates, lower, linestyle="--", color="orange", label="Lower Bollinger")
        y_min = min(y_min, min(lower))
        y_max = max(y_max, max(upper))
    ax.axhline(hedge_put_price, color='red', linestyle="--", linewidth=2,
               label="Hedge Put ($" + str(hedge_put_price) + ")")
    y_min = min(y_min, hedge_put_price)
    y_max = max(y_max, hedge_put_price)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - padding, y_max + padding)
    ax.text(0.5, 1.05, "Hedge Company: " + hedge_name, transform=ax.transAxes, ha='center', fontsize=9)
    simplify_xaxis(ax)
    ax.grid(True)
    ax.legend(fontsize=7)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf
    
def create_bollinger_chart(dates, prices):
    fig, ax = plt.subplots(figsize=(8.27, 5.85))  # Half A4 portrait height

    ax.plot(dates, prices, label='Price', linewidth=1.5)
    all_y = list(prices) + [short_call, long_call]

    if len(prices) >= 20:
        window = 20
        ma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
        upper = ma + 2 * std
        lower = ma - 2 * std

        min_len = min(len(ma), len(std), len(upper), len(lower), len(dates) - (window - 1))
        valid_dates = dates[window - 1:window - 1 + min_len]
        ax.plot(valid_dates, upper[:min_len], linestyle="--", color="blue", label="Upper Bollinger")
        ax.plot(valid_dates, lower[:min_len], linestyle="--", color="orange", label="Lower Bollinger")
        all_y += list(upper[:min_len]) + list(lower[:min_len])

    ax.axhline(short_call, color='red', linestyle='--', linewidth=2, label=f'Short Call (${short_call})')
    ax.axhline(long_call, color='green', linestyle='--', linewidth=2, label=f'Long Call (${long_call})')
    padding = (max(all_y) - min(all_y)) * 0.1
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_title("Bollinger Bands")
    simplify_xaxis(ax)
    ax.grid(True)
    ax.legend(fontsize=7)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_macd_chart(dates, prices):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.27, 5.85),  # Half A4 portrait
        sharex=True,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    if len(prices) < 35:
        raise ValueError("Not enough price data for MACD chart")

    exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')
    exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')
    macd_raw = exp12[-len(exp26):] - exp26
    signal = np.convolve(macd_raw, np.ones(9)/9, mode='valid')

    macd_values = macd_raw[-len(signal):]
    signal_values = signal
    macd_dates = dates[-len(signal):]
    price_values = prices[-len(signal):]

    # --- Top panel: Price + Call Lines + Markers ---
    ax1.plot(macd_dates, price_values, label='Price', color='black', linewidth=1.5)
    ax1.axhline(short_call, color='red', linestyle='--', linewidth=2, label=f'Short Call (${short_call})')
    ax1.axhline(long_call, color='green', linestyle='--', linewidth=2, label=f'Long Call (${long_call})')

    for i in range(1, len(macd_values)):
        if macd_values[i] > signal_values[i] and macd_values[i - 1] <= signal_values[i - 1]:
            ax1.plot(macd_dates[i-1:i+1], [price_values[i]]*2, color='green', linewidth=3)
        elif macd_values[i] < signal_values[i] and macd_values[i - 1] >= signal_values[i - 1]:
            ax1.plot(macd_dates[i-1:i+1], [price_values[i]]*2, color='red', linewidth=3)

    all_price_y = list(price_values) + [short_call, long_call]
    padding = (max(all_price_y) - min(all_price_y)) * 0.1
    ax1.set_ylim(min(all_price_y) - padding, max(all_price_y) + padding)

    # --- Bottom panel: MACD + Signal + Markers (no call lines) ---
    ax2.plot(macd_dates, macd_values, label='MACD', color='blue', linewidth=1.5)
    ax2.plot(macd_dates, signal_values, label='Signal', color='red', linestyle="--", linewidth=1.5)

    for i in range(1, len(macd_values)):
        if macd_values[i] > signal_values[i] and macd_values[i - 1] <= signal_values[i - 1]:
            ax2.plot(macd_dates[i-1:i+1], [macd_values[i]]*2, color='green', linewidth=3)
        elif macd_values[i] < signal_values[i] and macd_values[i - 1] >= signal_values[i - 1]:
            ax2.plot(macd_dates[i-1:i+1], [macd_values[i]]*2, color='red', linewidth=3)

    all_macd_y = list(macd_values) + list(signal_values)
    padding = (max(all_macd_y) - min(all_macd_y)) * 0.1 if max(all_macd_y) > min(all_macd_y) else 1
    ax2.set_ylim(min(all_macd_y) - padding, max(all_macd_y) + padding)

    ax1.set_title(f"{stock_name} Bear Call Spread Report - MACD Chart", fontsize=9)
    ax1.grid(True)
    ax2.grid(True)
    ax2.legend(fontsize=8)
    simplify_xaxis(ax2)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf


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

def generate_pdf():
    filename = "Version 51.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
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
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 10, "Stock Technical Indicators")
    c.drawString(10, height - 30, "Bollinger Bands:")
    c.drawImage(ImageReader(create_bollinger_chart(stock_dates, stock_prices)), 0, height * 0.5, width=width, height=height * 0.5)
    c.drawString(10, height * 0.5 - 20, "MACD:")
    c.drawImage(ImageReader(create_macd_chart(stock_dates, stock_prices)), 0, 0, width=width, height=height * 0.5)
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
    original_script = "Version 51.py"
backup_script = "Version 51 Backup.py"
shutil.copyfile(original_script, backup_script)
if os.path.exists(backup_script):
    print("Backup file created:", backup_script)
else:
    print("ERROR: Backup file was not created!")

# --- Working Copy Config ---
secret_key = "ODE123456"  # Replace with your actual key
repo = "trading"
branch = "main"
commit_message = "Version 51"
filename_to_commit = backup_script

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
        webbrowser.open(wc_url)

threading.Timer(1.0, open_wc_url).start()
threading.Timer(3.0, open_pdf).start()

# Bear Call Spread Report – Version 46 (Sim Data + Charts + Working Copy Commit for Version 46.py)

import os
import json
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

# Load Sim Data
sim_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sim Data.txt")

def load_historical(symbol):
    with open(sim_data_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)
    for entry in data:
        if entry["symbol"].upper() == symbol.upper():
            return entry["companyName"], entry["price"], entry["historical"]
    return symbol, 100.0, []

# Inputs
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
risk_reward = round((premium * 100 * contract_size) / ((long_call - short_call - premium) * 100 * contract_size), 1)

stock_name, _, stock_data = load_historical(stock_symbol)
hedge_name, _, hedge_data = load_historical(hedge_symbol)

def parse_dates(data):
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
    prices = [d["close"] for d in data]
    return np.array(dates), np.array(prices)

def simplify_xaxis(ax):
    ax.xaxis.set_major_formatter(DateFormatter('%m'))

def create_pl_chart():
    x = np.linspace(short_call - 20, long_call + 20, 500)
    y = np.where(
        x <= short_call,
        premium * 100 * contract_size,
        np.where(
            x < long_call,
            (premium - (x - short_call)) * 100 * contract_size,
            (premium - (long_call - short_call)) * 100 * contract_size
        )
    )
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(x, y, linewidth=1.5, label="P&L")
    ax.axhline(premium * 100 * contract_size, color="green", linestyle="--", linewidth=2,
               label="Max Profit ($" + str(premium * 100 * contract_size) + ")")
    ax.axhline((premium - (long_call - short_call)) * 100 * contract_size, color="red", linestyle="--", linewidth=2,
               label="Max Loss ($" + str((premium - (long_call - short_call)) * 100 * contract_size) + ")")
    ax.axvline(short_call + premium, color="gray", linestyle="--", linewidth=2,
               label="Breakeven ($" + str(short_call + premium) + ")")
    ax.set_title("P&L Chart")
    ax.grid(True)
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_hedge_chart(dates, prices):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(dates, prices, label='Hedge Price', linewidth=1.5)
    ax.axhline(hedge_put_price, color='red', linestyle="--", linewidth=2,
               label="Hedge Put ${}".format(hedge_put_price))
    ax.text(0.5, 1.05, "Hedge Company: {}".format(hedge_name), transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center', fontsize=10, color='black')
    simplify_xaxis(ax)
    ax.grid(True)
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_stock_chart(dates, prices):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(dates, prices, label='Stock Price', linewidth=1.5)
    ax.axhline(short_call, color='red', linestyle="--", linewidth=2, label="Short Call ${}".format(short_call))
    ax.axhline(long_call, color='green', linestyle="--", linewidth=2, label="Long Call ${}".format(long_call))
    simplify_xaxis(ax)
    ax.set_title("Stock with Call Strikes")
    ax.grid(True)
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_bollinger_chart(dates, prices):
    window = 20
    ma = np.convolve(prices, np.ones(window)/window, mode='valid')
    std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
    upper = ma + 2 * std
    lower = ma - 2 * std
    valid_dates = dates[window-1:]
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(dates, prices, label='Price', linewidth=1.2)
    ax.plot(valid_dates, upper, linestyle="--", label='Upper Band')
    ax.plot(valid_dates, lower, linestyle="--", label='Lower Band')
    simplify_xaxis(ax)
    ax.set_title("Bollinger Bands")
    ax.grid(True)
    ax.legend()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_macd_split_chart(dates, prices):
    exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')
    exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')
    macd = exp12[-len(exp26):] - exp26
    signal = np.convolve(macd, np.ones(9)/9, mode='valid')
    offset = len(prices) - len(signal)
    macd_dates = dates[offset:]
    prices_trim = prices[offset:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
    ax1.plot(macd_dates, prices_trim, label='Price', color='black')
    ax1.set_title("Price")
    simplify_xaxis(ax1)
    ax1.grid(True)

    ax2.plot(macd_dates, macd[-len(signal):], label='MACD', color='blue', linewidth=1.5)
    ax2.plot(macd_dates, signal, label='Signal', color='red', linestyle="--", linewidth=1.5)
    ax2.set_title("MACD")
    simplify_xaxis(ax2)
    ax2.grid(True)
    ax2.legend()

    for i in range(1, len(signal)):
        diff_prev = macd[-len(signal):][i-1] - signal[i-1]
        diff_curr = macd[-len(signal):][i] - signal[i]
        if diff_prev * diff_curr < 0:
            color = "green" if diff_curr > 0 else "red"
            ax2.annotate('', xy=(macd_dates[i], macd[-len(signal):][i]),
                         xytext=(macd_dates[i-1], macd[-len(signal):][i-1]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2))
            ax1.annotate('', xy=(macd_dates[i], prices_trim[i]),
                         xytext=(macd_dates[i-1], prices_trim[i-1]),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2))

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def draw_trade_table(c, width, height):
    table = {
        "Stock": stock_symbol,
        "Delta": "{:.2f}".format(delta),
        "Premium": "${:.2f}".format(premium),
        "Target Price": "${:.2f}".format(target_price),
        "Short Call": "${:.2f}".format(short_call),
        "Long Call": "${:.2f}".format(long_call),
        "Contract Size": contract_size,
        "Expiration": expiration,
        "Risk/Reward": risk_reward,
        "Hedge Stock": hedge_symbol,
        "Hedge Delta": "{:.2f}".format(hedge_delta),
        "Hedge Put": "${:.2f}".format(hedge_put_price)
    }
    keys = list(table.keys())
    values = list(table.values())
    row_height = 20
    col_width = width / 3
    y_start = height - 130
    c.setFont("Helvetica-Bold", 10)
    for row in range(5):
        for col in range(3):
            idx = row * 3 + col
            if idx < len(keys):
                c.drawString(col * col_width + 50, y_start - row * row_height, "{}: {}".format(keys[idx], values[idx]))

def generate_pdf():
    filename = "Version 46.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 50, "{} - {} - Bear Call Spread Report".format(stock_symbol, stock_name))
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, height - 75, "Hedging Stock: {} - {}".format(hedge_symbol, hedge_name))
    draw_trade_table(c, width, height)

    stock_dates, stock_prices = parse_dates(stock_data)
    hedge_dates, hedge_prices = parse_dates(hedge_data)

    c.drawImage(ImageReader(create_pl_chart()), 60, height - 390, width=5.5 * inch, height=2.5 * inch)
    c.drawImage(ImageReader(create_hedge_chart(hedge_dates, hedge_prices)), 60, height - 620, width=5.5 * inch, height=2.5 * inch)
    c.showPage()

    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 50, "Stock Technical Indicators")
    c.drawString(60, height - 100, "Bollinger Bands:")
    c.drawImage(ImageReader(create_bollinger_chart(stock_dates, stock_prices)), 60, height - 360, width=5.5 * inch, height=2.5 * inch)
    c.drawString(60, height - 390, "MACD:")
    c.drawImage(ImageReader(create_macd_split_chart(stock_dates, stock_prices)), 60, height - 730, width=5.5 * inch, height=5 * inch)

    c.save()
    return filename

# === MAIN EXECUTION ===
pdf_file = generate_pdf()
webbrowser.open("file://" + os.path.abspath(pdf_file))

# === WORKING COPY COMMIT FOR VERSION 46.PY ===
secret_key = "TRADR0662"  # <-- Replace this with your actual Working Copy key
repo = "trading"
branch = "main"
filename = "Version 46.py"
commit_message = "Version 46"

if os.path.exists(filename):
    encoded_msg = urllib.parse.quote(commit_message, safe='')
    encoded_file = urllib.parse.quote(filename, safe='')
    url = (
        f"working-copy://x-callback-url/commit?"
        f"key={secret_key}&repo={repo}&branch={branch}"
        f"&message={encoded_msg}&paths[]={encoded_file}&add=true"
    )
    print("Committing script via Working Copy:\n", url)
    webbrowser.open(url)
else:
    print(f"ERROR: '{filename}' not found. Check your file path.")

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

# Load Sim Data
sim_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sim Data.txt")


def load_historical(symbol):
    with open(sim_data_path, "r") as f:
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
    ax.xaxis.set_major_formatter(DateFormatter('%m'))  # Show month number (1–12)

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
               label=f'Hedge Put ${hedge_put_price}')
    # Place Hedge Company name above the chart
    ax.text(0.5, 1.05, f'Hedge Company: {hedge_name}', transform=ax.transAxes,
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
    ax.axhline(short_call, color='red', linestyle="--", linewidth=2, label=f'Short Call ${short_call}')
    ax.axhline(long_call, color='green', linestyle="--", linewidth=2, label=f'Long Call ${long_call}')
    simplify_xaxis(ax)
    ax.set_title("Stock with Call Strikes ")
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
    
    # Annotate divergence/convergence events:
    # Detect crossing events where MACD crosses Signal and add small angled arrows.
    for i in range(1, len(signal)):
        diff_prev = macd[-len(signal):][i-1] - signal[i-1]
        diff_curr = macd[-len(signal):][i] - signal[i]
        if diff_prev * diff_curr < 0:  # A crossing is detected
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
        "Stock": stock_symbol, "Delta": f"{delta:.2f}", "Premium": f"${premium:.2f}",
        "Target Price": f"${target_price:.2f}", "Short Call": f"${short_call:.2f}", "Long Call": f"${long_call:.2f}",
        "Contract Size": contract_size, "Expiration": expiration, "Risk/Reward": risk_reward,
        "Hedge Stock": hedge_symbol, "Hedge Delta": f"{hedge_delta:.2f}", "Hedge Put": f"${hedge_put_price:.2f}"
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
                c.drawString(col * col_width + 50, y_start - row * row_height, f"{keys[idx]}: {values[idx]}")

def generate_pdf():
    filename = f"Bear_Call_Spread_V36_5_{datetime.today().date()}.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # Page 1
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height - 50, f"{stock_symbol} - {stock_name} - Bear Call Spread Report")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, height - 75, f"Hedging Stock: {hedge_symbol} - {hedge_name}")
    draw_trade_table(c, width, height)

    stock_dates, stock_prices = parse_dates(stock_data)
    hedge_dates, hedge_prices = parse_dates(hedge_data)

    c.drawImage(ImageReader(create_pl_chart()), 60, height - 390, width=5.5 * inch, height=2.5 * inch)  # P&L chart
    c.drawImage(ImageReader(create_hedge_chart(hedge_dates, hedge_prices)), 60, height - 620, width=5.5 * inch, height=2.5 * inch)
    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width/2, height - 50, "Stock Technical Indicators")
    c.drawString(60, height - 100, "Bollinger Bands:")
    c.drawImage(ImageReader(create_bollinger_chart(stock_dates, stock_prices)), 60, height - 360, width=5.5 * inch, height=2.5 * inch)
    c.drawString(60, height - 390, "MACD:")
    c.drawImage(ImageReader(create_macd_split_chart(stock_dates, stock_prices)), 60, height - 730, width=5.5 * inch, height=5 * inch)

    c.save()
    webbrowser.open("file://" + os.path.abspath(filename))

generate_pdf()

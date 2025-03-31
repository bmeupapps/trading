import os
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import urllib.parse
import re
import ssl
import webbrowser
import threading
from datetime import datetime
from matplotlib.dates import DateFormatter
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader

# --- Load Sim Data ---
sim_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Sim Data.txt")

def load_historical(symbol):
    try:
        fmp_key = "i5nShJm6WKlPcM5h5iKlSaTY0ThnH8xA"
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?serietype=line&timeseries=365&apikey={fmp_key}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.load(response)
        if "historical" in data and len(data["historical"]) > 0:
            company_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={fmp_key}"
            with urllib.request.urlopen(company_url, timeout=10) as response:
                profile = json.load(response)
                company_name = profile[0]["companyName"] if profile and "companyName" in profile[0] else symbol
            return company_name, data["historical"][0]["close"], data["historical"]
    except Exception as e:
        print(f"FMP fetch failed for {symbol}: {e}")
    try:
        with open(sim_data_path, "r", encoding="utf-8", errors="ignore") as f:
            data = json.load(f)
        for entry in data:
            if entry["symbol"].upper() == symbol.upper():
                return entry["companyName"], entry["price"], entry["historical"]
    except Exception as e:
        print(f"Sim Data fallback error: {e}")
    return symbol, 100.0, []

def fetch_beta(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/key-statistics"
        headers = {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, context=ssl._create_unverified_context(), timeout=10) as response:
            html = response.read().decode('utf-8')
        match = re.search(r'Beta \(5Y Monthly\)</span></td><td[^>]*><span[^>]*>([^<]+)</span>', html)
        if match:
            return match.group(1).strip()
    except Exception as e:
        print(f"Beta fetch error for {symbol}: {e}")
    return "N/A"

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

# --- Derived values ---
risk_reward = round((premium * 100) / ((long_call - short_call - premium) * 100), 1)
stock_name, _, stock_data = load_historical(stock_symbol)
hedge_name, _, hedge_data = load_historical(hedge_symbol)
stock_beta = fetch_beta(stock_symbol)
hedge_beta = fetch_beta(hedge_symbol)

def parse_dates(data):
    dates = [datetime.strptime(d["date"], "%Y-%m-%d") for d in data]
    prices = [d["close"] for d in data]
    return np.array(dates), np.array(prices)

def simplify_xaxis(ax):
    ax.xaxis.set_major_formatter(DateFormatter('%m'))

def create_pl_chart():
    fig, ax = plt.subplots(figsize=(6, 2.5))
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
    y_min = min(0, max_loss)
    y_max = max(0, max_profit)
    padding = (y_max - y_min) * 0.1
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
    fig, ax = plt.subplots(figsize=(6 * 0.9, 2.5))
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
               label=f"Hedge Put (${hedge_put_price})")
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
    fig, ax = plt.subplots(figsize=(8.27, 5.85))
    ax.plot(dates, prices, label='Price', linewidth=1.5)
    all_y = list(prices) + [short_call, long_call]
    if len(prices) >= 20:
        window = 20
        ma = np.convolve(prices, np.ones(window)/window, mode='valid')
        std = np.array([np.std(prices[i-window:i]) for i in range(window, len(prices)+1)])
        upper = ma + 2 * std
        lower = ma - 2 * std
        valid_dates = dates[window-1:]
        ax.plot(valid_dates, upper, linestyle="--", color="blue", label="Upper Bollinger")
        ax.plot(valid_dates, lower, linestyle="--", color="orange", label="Lower Bollinger")
        all_y += list(upper) + list(lower)
    ax.axhline(short_call, color='red', linestyle='--', linewidth=2, label=f'Short Call (${short_call})')
    ax.axhline(long_call, color='green', linestyle='--', linewidth=2, label=f'Long Call (${long_call})')
    padding = (max(all_y) - min(all_y)) * 0.1
    ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
    ax.set_title(f"{stock_name} Bollinger Bands")
    ax.text(0.01, 0.97, "Bands 20-day MA - 2 Standard Deviations", transform=ax.transAxes,
            ha='left', va='top', fontsize=8, style='italic')
    simplify_xaxis(ax)
    ax.grid(True)
    ax.legend(fontsize=7)
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def create_macd_chart(dates, prices):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 5.85), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Calculate EMAs
    if len(prices) < 35:
        raise ValueError("Not enough data for MACD")

    exp12 = np.convolve(prices, np.ones(12)/12, mode='valid')
    exp26 = np.convolve(prices, np.ones(26)/26, mode='valid')
    macd = exp12[-len(exp26):] - exp26
    signal = np.convolve(macd, np.ones(9)/9, mode='valid')

    # Align all arrays
    aligned_len = len(signal)
    macd_values = macd[-aligned_len:]
    signal_values = signal
    price_values = prices[-aligned_len:]
    macd_dates = dates[-aligned_len:]

    # --- Top Panel: Price + Call Lines + Markers ---
    ax1.plot(macd_dates, price_values, label='Price', color='black', linewidth=1.5)
    ax1.axhline(short_call, color='red', linestyle='--', linewidth=2, label=f'Short Call (${short_call})')
    ax1.axhline(long_call, color='green', linestyle='--', linewidth=2, label=f'Long Call (${long_call})')

    for i in range(2, aligned_len - 1):
        if macd_values[i] > signal_values[i] and macd_values[i - 1] <= signal_values[i - 1]:
            # Divergence
            ax1.plot(macd_dates[i-2:i+1], [price_values[i]] * 3, color='green', linewidth=4)
            ax2.plot(macd_dates[i-2:i+1], [macd_values[i]] * 3, color='green', linewidth=4)
        elif macd_values[i] < signal_values[i] and macd_values[i - 1] >= signal_values[i - 1]:
            # Convergence
            ax1.plot(macd_dates[i-2:i+1], [price_values[i]] * 3, color='red', linewidth=4)
            ax2.plot(macd_dates[i-2:i+1], [macd_values[i]] * 3, color='red', linewidth=4)

    # --- Bottom Panel: MACD + Signal ---
    ax2.plot(macd_dates, macd_values, label='MACD', color='blue', linewidth=1.5)
    ax2.plot(macd_dates, signal_values, label='Signal', color='red', linestyle='--', linewidth=1.5)

    # Auto-scale
    price_padding = (max(price_values) - min(price_values)) * 0.1
    ax1.set_ylim(min(price_values) - price_padding, max(price_values) + price_padding)

    macd_range = max(max(macd_values), max(signal_values)) - min(min(macd_values), min(signal_values))
    macd_padding = macd_range * 0.1 if macd_range > 0 else 1
    ax2.set_ylim(min(min(macd_values), min(signal_values)) - macd_padding,
                 max(max(macd_values), max(signal_values)) + macd_padding)

    ax1.set_title(f"{stock_name} Bear Call Spread Report - MACD Chart", fontsize=9)
    ax1.grid(True)
    ax2.grid(True)
    ax2.legend(fontsize=8)
    simplify_xaxis(ax2)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf

def draw_trade_table(c, width, height):
    y_start = height * 0.9 - 10
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
    for row in range(4):
        for col in range(3):
            idx = row * 3 + col
            if idx < len(keys):
                c.drawString(col * col_width + 10, y_start - row * row_height, f"{keys[idx]}: {values[idx]}")
    hedge_exposure = 100 * (hedge_put_price + premium)
    call_risk = (long_call - short_call - premium) * 100
    delta_exposure = abs(delta) * 100
    c.drawString(10, y_start - 4 * row_height, f"Hedge Exposure: ${hedge_exposure:,.0f}")
    c.drawString(col_width + 10, y_start - 4 * row_height, f"Call Spread Risk: ${call_risk:,.0f}")
    c.drawString(col_width * 2 + 10, y_start - 4 * row_height, f"Net Prem ∆/$1: ±${delta_exposure:,.0f}")
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(col_width * 2 + 10, y_start - (4 * row_height + 12), "* ignoring theta")
    c.setFont("Helvetica-Bold", 14)
    y_beta = y_start - 5 * row_height
    c.drawString(10, y_beta, f"{stock_symbol} {stock_name}  Beta: {stock_beta}")
    c.drawString(10, y_beta - row_height, f"{hedge_symbol} {hedge_name}  Beta: {hedge_beta}")

def generate_pdf():
    filename = "Version 52.1.pdf"
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height * 0.95, f"{stock_symbol} - {stock_name} - Bear Call Spread Report")
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width / 2, height * 0.92, f"Hedging Stock: {hedge_symbol} - {hedge_name}")
    draw_trade_table(c, width, height)
    stock_dates, stock_prices = parse_dates(stock_data)
    hedge_dates, hedge_prices = parse_dates(hedge_data)
    chart_height = height * 0.35
    c.drawImage(ImageReader(create_pl_chart()), 0, 0, width=width, height=chart_height)
    c.drawImage(ImageReader(create_hedge_chart(hedge_dates, hedge_prices)), 0, chart_height, width=width, height=chart_height)
    c.showPage()
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(width / 2, height - 10, "Stock Technical Indicators")
    c.drawString(10, height - 30, "Bollinger Bands:")
    c.drawImage(ImageReader(create_bollinger_chart(stock_dates, stock_prices)), 0, height * 0.5, width=width, height=height * 0.5)
    c.drawString(10, height * 0.5 - 20, "MACD:")
    c.drawImage(ImageReader(create_macd_chart(stock_dates, stock_prices)), 0, 0, width=width, height=height * 0.5)
    c.save()
    return filename

# --- Execution ---
pdf_file = generate_pdf()
pdf_path = os.path.abspath(pdf_file)
encoded_pdf_path = urllib.parse.quote(pdf_path)

def open_pdf():
    webbrowser.open("file://" + encoded_pdf_path)

# Backup and Commit
try:
    original_script = os.path.basename(__file__)
except NameError:
    original_script = "Version 52.1.py"
backup_script = "Version 52.1 Backup.py"
shutil.copyfile(original_script, backup_script)

# Working Copy Commit
secret_key = "ODE123456"
repo = "trading"
branch = "main"
commit_message = "Version 52.1"
encoded_msg = urllib.parse.quote(commit_message, safe='')
encoded_file = urllib.parse.quote(backup_script, safe='')
wc_url = (
    f"working-copy://x-callback-url/commit?key={secret_key}&repo={repo}"
    f"&branch={branch}&message={encoded_msg}&paths%5B%5D={encoded_file}&add=true"
)

def open_wc_url():
    print("Committing via Working Copy...")
    webbrowser.open(wc_url)

threading.Timer(1.0, open_wc_url).start()
threading.Timer(3.0, open_pdf).start()



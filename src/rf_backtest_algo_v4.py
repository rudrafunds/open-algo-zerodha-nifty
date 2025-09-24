from kiteconnect import KiteConnect
import json
import pandas as pd
import numpy as np
import datetime
import logging
import sys
from scipy.signal import argrelextrema
import os
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from io import BytesIO

#Rf imports
from rf_patterns import detect_candle_patterns, detect_chart_patterns, calculate_emas

# process_candle
# Trades from 9:20 to 2:00PM

# Logger
class ColoredFormatter(logging.Formatter):
    GREEN = '\033[92m'  # Color for BUY
    RED = '\033[91m'    # Color for SELL
    YELLOW = '\033[93m'
    RESET = '\033[0m'   # Reset color
    def format(self, record):
        message = super().format(record)
        # Apply colors separately for BUY and SELL
        message = message.replace("True", f"{self.GREEN}True{self.RESET}")
        message = message.replace("CE", f"{self.GREEN}CE{self.RESET}")
        message = message.replace("high", f"{self.GREEN}high{self.RESET}")
        message = message.replace("Target-TSL", f"{self.GREEN}Target-TSL{self.RESET}")
        message = message.replace("Target-Hit", f"{self.GREEN}Target-Hit{self.RESET}")
        message = message.replace("BreakEven-SL", f"{self.GREEN}BreakEven-SL{self.RESET}")
        message = message.replace("Profit", f"{self.GREEN}Profit{self.RESET}")
        message = message.replace("breakout", f"{self.GREEN}breakout{self.RESET}")
        message = message.replace("bullish", f"{self.GREEN}bullish{self.RESET}")
        message = message.replace("Gap-Up", f"{self.GREEN}Gap-Up{self.RESET}")
        message = message.replace("Sustaining", f"{self.GREEN}Sustaining{self.RESET}")

        message = message.replace("False", f"{self.RED}False{self.RESET}")
        message = message.replace("PE", f"{self.RED}PE{self.RESET}")
        message = message.replace("low", f"{self.RED}low{self.RESET}")
        message = message.replace("SL-Hit", f"{self.RED}SL-Hit{self.RESET}")
        message = message.replace("SL", f"{self.RED}SL{self.RESET}")
        message = message.replace("Loss", f"{self.RED}Loss{self.RESET}")
        message = message.replace("breakdown", f"{self.RED}breakdown{self.RESET}")
        message = message.replace("bearish", f"{self.RED}bearish{self.RESET}")
        message = message.replace("Gap-Down", f"{self.RED}Gap-Down{self.RESET}")
        message = message.replace("Reversing", f"{self.RED}Reversing{self.RESET}")

        message = message.replace("system", f"{self.YELLOW}system{self.RESET}")
        return message
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter('%(message)s'))
logger.addHandler(console_handler)

# Configuration settings
global_config = None

# Kite Connect
global_kite = None

# Strategy Parameters
nifty_token = 256265
lot_size = 525
target_points = 25
max_sl_points = 25
ema_period = 20
target_premium = 100
sl_cooling_period = 4
target_cooling_period = 4
breakeven_cooling_period = 4
loss_threshold = -22500 # 10,000
profit_threshold = 25000 # 18,000
profit_lock_active = False
loss_lock_active = False
profit_consecutive_sl_hits = 0
profit_lock_max_sl_hit = 2

# Global variables
ohlc_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
daily_levels = {}
opening_range = {}
day_range = {'high': None, 'low': None}
pattern_lookback = 20  # Candles for pattern detection
last_exit_time = None
last_exit_reason = None
signal_message = ""

# Backtest variables
virtual_profit = 0
virtual_cumulative_profit = 0
trade_log = []
current_position = None

# Copying
gap_threshold = 0.25
pivot_reversal_threshold = 0.15  # 15% penetration into pivot zone
pivot_confirmation_candles = 3   # Number of candles to confirm reversal
reversal_confirmation_candles = 2  # Number of candles to confirm reversal
min_reversal_distance = 20  # Minimum points from day high/low to consider

nse_holidays = [
    "2024-01-26",  # Republic Day
    "2024-01-22",  # Negotiable Instrument Act
    "2024-02-19",  # Chatrapati Shivaji Maharaj Jayanti
    "2024-03-08",  # Maha Shivaratri
    "2024-03-25",  # Holi
    "2024-03-29",  # Good Friday
    "2024-04-11",  # Eid-Ul-Fitr (Ramzan Eid)
    "2024-04-17",  # Ram Navami
    "2024-05-01",  # Maharashtra Day
    "2024-05-20",  # Mumbai Elections
    "2024-06-17",  # Bakri Eid
    "2024-07-17",  # Muharram
    "2024-08-15",  # Independence Day
    "2024-10-02",  # Mahatma Gandhi Jayanti
    "2024-10-13",  # Dasara
    "2024-11-01",  # Diwali-Laxmi Pujan
    "2024-11-02",  # Diwali-Balipratipada
    "2024-11-15",  # Guru Nanak Jayanti
    "2024-11-20",  # Maharashtra Elections
    "2024-12-25"   # Christmas
]

# Define colors
GREEN = colors.HexColor("#388E3C")  # Success
RED = colors.HexColor("#D32F2F")    # Danger
YELLOW = colors.HexColor("#FBC02D") # Warning
BLACK = colors.HexColor("#212121")  # Default text color

def create_report_directory(start_date, end_date):
    """Create nested folder structure for reports based on start and end dates."""
    # Convert start and end dates to string format (e.g., "2025-01-01_to_2025-02-17")
    date_range_str = f"{start_date.strftime('%Y%m%d')}_to_{end_date.strftime('%Y%m%d')}"
    
    # Main reports folder
    main_dir = "backtest_reports"
    os.makedirs(main_dir, exist_ok=True)
    
    # Session-specific subfolder with date range in folder name
    session_dir = os.path.join(main_dir, date_range_str)
    os.makedirs(session_dir, exist_ok=True)
    
    return session_dir

def save_trades_to_csv(trades, report_dir, test_date):
    if not trade_log:
        print("No trades to save")
        return
    
    # Define CSV columns
    fields = ['timestamp', 'type', 'entry_timestamp', 'entry', 'sl', 'exit_timestamp', 'besl', 'tsl', 'exit', 'reason', 'profit', 'signal_message']
    
    filename = f"trades_{test_date.strftime('%Y%m%d')}.csv"
    filepath = os.path.join(report_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trade_log)
    print(f"Trade log saved to {filepath}")

# Function to apply color formatting
def colorize_text(value):
    value = str(value)  # Ensure it's a string    
    if value.startswith('+') or any(word in value for word in ["True", "Profit", "CE", "bullish", "breakout", "Gap-Up", "Sustaining", "high", "Target-TSL", "BreakEven-SL"]):
        return "#388E3C"  # Green
    elif value.startswith('-') or any(word in value for word in ["False", "Loss", "PE", "bearish", "breakdown", "SL-Hit", "Gap-Down", "Reversing"]):
        return "#D32F2F"  # Red
    elif "system" in value:
        return "#FBC02D"  # Yellow
    else:
        return "#212121"  # Default (Black)

def save_trades_to_pdf(trade_log, report_dir, test_date):
    if not trade_log:
        print("No trades to save")
        return

    filename = f"report_{test_date.strftime('%Y%m%d')}.pdf"
    filepath = os.path.join(report_dir, filename)

    # Create a BytesIO buffer to hold the PDF content
    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)

    #doc = SimpleDocTemplate(filepath, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Material UI-inspired color palette
    primary_color = colors.HexColor("#6200EE")  # Deep Purple 500
    secondary_color = colors.HexColor("#03DAC6")  # Teal 200
    background_color = colors.HexColor("#F5F5F5")  # Light Grey 100
    text_color = colors.HexColor("#212121")  # Dark Grey 900
    table_header_color = colors.HexColor("#3700B3")  # Darker Purple
    table_alt_row_color = colors.HexColor("#EDE7F6")  # Light Purple
    border_color = colors.HexColor("#BDBDBD")  # Grey 400

    # Title Styling
    title_style = styles["Title"]
    title_style.textColor = primary_color
    title_style.fontSize = 18

    elements.append(Paragraph(f"Trade Analysis Report - {test_date.strftime('%d-%m-%Y')}", title_style))
    elements.append(Spacer(1, 10))  # Add some space

    # Summary Section
    elements.append(Paragraph("<b>Summary Statistics:</b>", styles['Heading2']))
    profits = [t['profit'] for t in trade_log]
    
    summary_data = [
        ['Total Trades', len(trade_log)],
        ['Total Profit', f"{'+' if sum(profits) > 0 else ''}{sum(profits):.2f}"],
        ['Winning Trades', len([p for p in profits if p > 0])],
        ['Losing Trades', len([p for p in profits if p < 0])],
        ['Win Rate', f"{len([p for p in profits if p > 0])/len(profits):.2%}"],
        ['Max Profit', f"{'+' if max(profits) > 0 else ''}{max(profits) if profits else 0:.2f}"],
        ['Max Loss', f"{min(profits) if profits else 0:.2f}"],
        ['Quantity', lot_size],
        ['Max SL Points', max_sl_points],
        ['Max Target Points', target_points],
        ['Cooling SL min', sl_cooling_period],
        ['Cooling BreakEven min', breakeven_cooling_period],
        ['Cooling Target min', target_cooling_period],
        ['Per Trade Profit', f"+{lot_size*target_points}"],
        ['Per Trade Loss', f"-{lot_size*max_sl_points}"],
        ['Profit Threshold', f"+{profit_threshold}"],
        ['Loss Threshold', f"-{loss_threshold}"]
    ]

    # Apply colorization to summary data
    formatted_summary_data = [
        [Paragraph(f'<font color="{colorize_text(str(cell))}">{cell}</font>', styles['Normal']) for cell in row]
        for row in summary_data
    ]

    summary_table = Table(formatted_summary_data, colWidths=[2 * inch, 1.5 * inch])
    summary_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, -1), background_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), text_color),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
    ]))

    elements.append(summary_table)
    elements.append(Spacer(1, 15))  # Add some space

    # Trades Section
    elements.append(Paragraph("<b>Trades:</b>", styles['Heading2']))
    elements.append(Spacer(1, 5))  # Space before table

    # Table Headers
    table_data = [['No.', 'Date', 'Type', 'Entry', 'SL', 'Time', 'BESL', 'TSL', 'Exit', 'P&L', 'Reason']]
    for i, trade in enumerate(trade_log, start=1):
        background_color = colors.green if trade['profit'] > 0 else colors.red if trade['profit'] < 0 else colors.whitesmoke
        text_color = colors.white if background_color == colors.red else colors.black  # Use white text on red, black on green/neutral
        trade_row = [
            str(i),  # Numbering each row
            trade['timestamp'].strftime('%H:%M'),
            trade['type'],
            f"{trade['entry']:.2f}",
            trade['sl'],
            trade['exit_timestamp'],
            trade['besl'],
            trade['tsl'],
            f"{trade['exit']:.2f}",
            f"{'+' if trade['profit'] > 0 else ''}{trade['profit']:.0f}",
            trade['reason']
        ]
        formatted_row = [
            Paragraph(f'<font color="{text_color}">{cell}</font>', styles['Normal']) for cell in trade_row
        ]
        table_data.append(formatted_row)   

    # Create table with enhanced styling
    trade_table_col_widths = [
        0.4 * inch,  # No.
        0.6 * inch,  # Date
        0.4 * inch,  # Type
        0.8 * inch,  # Entry
        0.8 * inch,  # SL
        0.6 * inch,  # Time
        0.8 * inch,  # BESL
        0.8 * inch,  # TSL
        0.8 * inch,  # Exit
        0.8 * inch,  # P&L
        1.2 * inch   # Reason
    ]
    table = Table(table_data, colWidths=trade_table_col_widths)

    # Define the background colors for rows based on profit
    row_styles = []
    for i, trade in enumerate(trade_log, start=1):
        background_color = colors.green if trade['profit'] > 0 else colors.red if trade['profit'] < 0 else colors.whitesmoke
        row_styles.append(('BACKGROUND', (0, i), (-1, i), background_color))

    # ✅ Apply table styling
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),  # Header color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        #('BACKGROUND', (0, 1), (-1, -1), background_color),  # Alternating row colors
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6)
    ] + row_styles ))
    elements.append(table)

    # Create a separate table for Signal Messages
    signal_table_data = [['No.', 'Timestamp', 'Signal Messages']]  # Header row

    for i, trade in enumerate(trade_log, start=1):
        signal_table_data.append([
            i,  # Numbering each row
            trade['timestamp'].strftime('%H:%M'),  # Only timestamp
            f"📢 {trade['signal_message']}"  # Signal Message
        ])
    
    # Define column widths
    # Adjusted column widths for Signal Messages table
    signal_table_col_widths = [
        0.5 * inch,  # No.
        1.2 * inch,  # Timestamp
        6.3 * inch   # Signal Message (spans most of the width)
    ]

    # Create Signal Message Table
    signal_table = Table(signal_table_data, colWidths=signal_table_col_widths)

    # Apply styling for Signal Message Table
    signal_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), primary_color),  # Header color
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),  # Header text color
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 0.5, border_color),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
    ]))

    # Add Signal Messages Table to the PDF
    elements.append(Paragraph("<br/><b>Signal Messages:</b>", styles['Heading2']))
    elements.append(signal_table)

    doc.build(elements)

    # Encode the PDF content
    pdf_buffer.seek(0)
    pdf_content = pdf_buffer.getvalue()

    # Save the encoded PDF content to a file
    with open(filepath, 'wb') as f:
        f.write(pdf_content)

    print(f"📄 PDF report saved to {filepath}")

# Function to get business days excluding holidays and weekends
def get_trading_days(start_date, end_date):
    # Convert holiday list to datetime
    holidays = pd.to_datetime(nse_holidays)
    # Generate all business days between start and end date
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    # Exclude weekends and holidays
    trading_days = [date for date in dates if date not in holidays]    
    return trading_days

def backtest(start_date, end_date):
    global ohlc_data, daily_levels, opening_range, day_range, virtual_profit, current_position, trade_log, virtual_cumulative_profit, loss_lock_active, profit_lock_active, profit_consecutive_sl_hits
    
    start_time = datetime.datetime.now()
    report_dir = create_report_directory(start_date, end_date)
    
    # Get all trading days excluding weekends and NSE holidays
    holidays = pd.to_datetime(nse_holidays)
    trading_days = get_trading_days(start_date, end_date)
    
    for test_date in trading_days:
        print(test_date.strftime('%Y-%m-%d'))
        # Reset daily variables
        virtual_profit = 0
        current_position = None
        trade_log = []
        day_range = {'high': None, 'low': None}
        opening_range = {}
        daily_levels = calculate_daily_levels(test_date)
        loss_lock_active = False
        profit_lock_active = False
        profit_consecutive_sl_hits = 0
        
        # Fetch historical data for test date
        from_date = test_date.replace(hour=9, minute=15)
        to_date = test_date.replace(hour=15, minute=30)
        
        # Get 5-minute candles for the day
        candles = global_kite.historical_data(nifty_token, from_date, to_date, "5minute")
        if not candles:
            continue
            
        df = pd.DataFrame(candles)
        df.rename(columns={'date': 'timestamp'}, inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Process each candle in sequence
        for idx, candle in df.iterrows():
            if (loss_lock_active):
                #print(f"Loss Threshold reached: Done for the day")
                break
            elif (profit_lock_active):
                if profit_consecutive_sl_hits >=  profit_lock_max_sl_hit:
                    #print(f"Profit Threshold reached: Done for the day")
                    break
            process_candle(candle, test_date)
        
        # Print daily summary
        print(f"\n\n=== Daily Report [{test_date.strftime('%Y-%m-%d')}] ===")
        print(f"Virtual Profit: {virtual_profit:.2f}")
        virtual_cumulative_profit += virtual_profit
        print(f"Total Trades: {len(trade_log)}")
        if trade_log:
            wins = len([t for t in trade_log if t['profit'] > 0])
            print(f"Win Rate: {wins/len(trade_log):.2%}")
            print(f"Max Profit: {max(t['profit'] for t in trade_log):.2f}")
            print(f"Max Loss: {min(t['profit'] for t in trade_log):.2f}")
        #print("=== Trade Log ===")
        #for trade in trade_log:
        #    logging.info(f"{trade['timestamp']} | {trade['type']} | Entry: {trade['entry']} | SL: {trade['sl']} | {trade['exit_timestamp']} | BESL: {trade['besl']} | TSL: {trade['tsl']} | Exit: {trade['exit']} | Reason: {trade['reason']} | P&L: {trade['profit']:.2f}")
    
        # Save daily reports
        #save_trades_to_csv(trade_log, report_dir, test_date)
        save_trades_to_pdf(trade_log, report_dir, test_date)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f'\nVirtual Cumulative Profit: {virtual_cumulative_profit} for from: {start_date} to: {end_date}')
    logging.info(f"Backtest: in {duration.total_seconds():.2f} seconds")

def process_candle(candle, current_date):
    global current_position, ohlc_data, day_range, opening_range
    
    # Update OHLC data
    ohlc_data = pd.concat([ohlc_data, pd.DataFrame([candle])], ignore_index=True)
    
    # Update day range
    if day_range['high'] is None:
        day_range['high'] = candle['high']
        day_range['low'] = candle['low']
    else:
        day_range['high'] = max(day_range['high'], candle['high'])
        day_range['low'] = min(day_range['low'], candle['low'])
    
    # Set opening range (first candle)
    if candle['timestamp'].time() == datetime.time(9,15):
        opening_range['high'] = candle['high']
        opening_range['low'] = candle['low']
    
    # Calculate EMAs
    ohlc_data = calculate_emas(ohlc_data)
    
    # Check if current time is before 9:20 #Fixme
    current_time = candle['timestamp'].time()
    if current_time < datetime.time(9, 20) or current_time > datetime.time(14, 00):
        return  # Skip trading logic for candles before 9:20

    # Check entry conditions
    current_price = candle['close']
    if current_position is None:
        signal = entry_conditions(current_price)
        if signal:
            execute_virtual_trade(signal, candle)

    # Manage existing position
    if current_position:
        manage_virtual_position(candle)

def execute_virtual_trade(signal_type, candle):
    global current_position
    
    # Simulate getting option instrument
    entry_price = candle['close']

    current_position = {
        'type': signal_type,
        'entry_price': entry_price,
        'sl': entry_price - max_sl_points,
        'target': entry_price + target_points,
        'besl': 0,
        'tsl': 0,
        'entry_time': candle['timestamp'],
        'quantity': lot_size,
        'signal_message': signal_message
    }

    if signal_type == 'PE':
        current_position['sl'] = entry_price + max_sl_points
        current_position['target'] = entry_price - target_points
    
    #logging.info(f"\nEntry | {candle['timestamp']} | Type: {signal_type} | Entry Price: {entry_price} ")

def manage_virtual_position(candle):
    global current_position, virtual_profit, trade_log, last_exit_time, last_exit_reason, loss_lock_active, profit_lock_active, profit_consecutive_sl_hits
    
    exit_reason = None
    exit_price = None
    current_price = candle['close']
    last_exit_time = None

    # Position monitoring message
    message = "\r"
    message += f"SL: {current_position['sl']} | "
    message += f"Entry: {current_position['entry_price']} | "
    message += f"BESL: {current_position['besl']} | "
    message += f"TSL: {current_position.get('tsl', 'N/A')} | "
    message += f"Current: {current_price} | "
    message += f"Target: {current_position['target']}   "
    #update_terminal(message)
    
    if current_position:
        position_type = current_position['type']
        entry_price = current_position['entry_price']
        
        # Common exit conditions
        if position_type == 'CE':  # Long position
            # Stop Loss trigger (price falls below SL)
            if current_price <= current_position['sl']:
                exit_price = current_position['sl']
                exit_reason = 'SL-Hit'
                last_exit_time = candle['timestamp']

                if (profit_lock_active):
                    profit_consecutive_sl_hits += 1
                    exit_reason += ' | Profit Threshold'
            # Break Even SL
            elif ((current_position['besl'] == 0) and (current_price >= (current_position['entry_price'] + 0.5 * (current_position['target'] - current_position['entry_price'])))):
                current_position['besl'] = current_position['entry_price']
            # Target hit (price rises above target)
            elif current_price >= current_position['target']:
                #exit_price = current_position['target']
                exit_reason = 'Target-Hit'
                # Move TSL up
                current_position['tsl'] = current_position['target'] - 10
                current_position['target'] += target_points                
            # Trailing SL (price falls below TSL)
            elif current_position.get('tsl') and current_price <= current_position['tsl']:
                exit_price = current_position['tsl']
                exit_reason = 'Target-TSL'
                last_exit_time = candle['timestamp']
            # BESL Trailing
            elif current_position.get('besl') and current_price <= current_position['besl']:
                exit_price = current_position['besl']
                exit_reason = 'BreakEven-SL'
                last_exit_time = candle['timestamp']

        elif position_type == 'PE':  # Short position
            # Stop Loss trigger (price rises above SL)
            if current_price >= current_position['sl']:
                exit_price = current_position['sl']
                exit_reason = 'SL-Hit'
                last_exit_time = candle['timestamp']

                if (profit_lock_active):
                    profit_consecutive_sl_hits += 1
                    exit_reason += ' | Profit Threshold'
            # Break Even SL
            elif ((current_position['besl'] == 0) and (current_price <= (current_position['entry_price'] + 0.5 * (current_position['entry_price'] - current_position['target'])))):
                current_position['besl'] = current_position['entry_price'] 
            # Target hit (price falls below target)
            elif current_price <= current_position['target']:
                #exit_price = current_position['target']
                exit_reason = 'Target-Hit'
                # Move TSL down
                current_position['tsl'] = current_position['target'] - 10
                current_position['target'] -= target_points                
            # Trailing SL (price rises above TSL)
            elif current_position.get('tsl') and current_price >= current_position['tsl']:
                exit_price = current_position['tsl']
                exit_reason = 'Target-TSL'
                last_exit_time = candle['timestamp']
            # BESL Trailing
            elif current_position.get('besl') and current_price >= current_position['besl']:
                exit_price = current_position['besl']
                exit_reason = 'BreakEven-SL'
                last_exit_time = candle['timestamp']

        # Execute exit if any condition met
        if exit_price:
            # Calculate P&L based on position type
            if position_type == 'CE':  # Long position
                profit = (exit_price - entry_price) * lot_size
            else:  # PE Short position
                profit = (entry_price - exit_price) * lot_size
            
            # Threshold
            virtual_profit += profit
            if (virtual_profit <= loss_threshold):
                loss_lock_active = True
                exit_reason += ' | Loss Threshold'
            elif (virtual_profit >= profit_threshold):
                profit_lock_active = True

            last_exit_reason = exit_reason
            trade_log.append({
                'timestamp': current_position['entry_time'],
                'type': position_type,
                'entry': entry_price,
                'sl': current_position['sl'],
                'besl': current_position['besl'],
                'tsl': current_position['tsl'],
                'exit': exit_price,
                'profit': profit,
                'reason': exit_reason,
                'entry_timestamp': current_position['entry_time'].strftime("%H:%M"),
                'exit_timestamp': candle['timestamp'].strftime("%H:%M"),
                'signal_message': current_position['signal_message']
            })
            
            #logging.info(f"\nExit | {candle['timestamp']} | {position_type} | "
            #             f"Price: {exit_price} | Reason: {exit_reason} | P&L: {profit:.2f}")
            current_position = None

def get_previous_trading_day(test_date, holidays):
    prev_day = test_date - pd.tseries.offsets.BDay(1)  # Move one day back
    
    # Check if the previous day is a weekend or holiday
    while prev_day.weekday() >= 5 or prev_day in holidays:  # 5 and 6 are Saturday and Sunday
        prev_day -= pd.Timedelta(days=1)  # Skip to the previous day
    
    return prev_day

# Modified helper functions
def calculate_daily_levels(test_date):
    """Calculate previous day's levels using historical data"""
    #prev_day = test_date - pd.tseries.offsets.BDay(1)
    holidays = pd.to_datetime(nse_holidays)
    prev_day = get_previous_trading_day(test_date, holidays)
    historical = global_kite.historical_data(nifty_token, prev_day, prev_day, "day")
    
    if not historical:
        return {}
    
    prev_day_data = historical[0]
    pivot = (prev_day_data['high'] + prev_day_data['low'] + prev_day_data['close']) / 3
    return {
        'prev_high': prev_day_data['high'],
        'prev_low': prev_day_data['low'],
        'prev_close': prev_day_data['close'],
        'pivot': pivot,
        's1': 2 * pivot - prev_day_data['high'],
        's2': pivot - (prev_day_data['high'] - prev_day_data['low']),
        'r1': 2 * pivot - prev_day_data['low'],
        'r2': pivot + (prev_day_data['high'] - prev_day_data['low'])
    }

def detect_pivot_reversal(current_price, pivot_points, ohlc):
    """Detect reversal signals at pivot levels with price action confirmation"""
    reversal_signals = {'CE': False, 'PE': False}
    
    # Calculate proximity to pivot levels
    distances = {
        level: abs(current_price - value)/value 
        for level, value in pivot_points.items()
    }
    nearest_level = min(distances, key=distances.get)
    
    # Recent price action analysis
    recent_lows = ohlc['low'].iloc[-pivot_confirmation_candles:].values
    recent_highs = ohlc['high'].iloc[-pivot_confirmation_candles:].values
    
    # Bullish Reversal at Support (S1/S2)
    if nearest_level in ['s1', 's2']:
        s_level = pivot_points[nearest_level]
        bullish_cond = (
            current_price >= s_level * (1 - pivot_reversal_threshold) and
            current_price <= s_level * (1 + pivot_reversal_threshold) and
            (recent_lows.min() <= s_level) and
            (ohlc['close'].iloc[-1] > ohlc['open'].iloc[-1])
            #(ohlc['volume'].iloc[-1] > ohlc['volume'].rolling(20).mean().iloc[-1])
        )
        if bullish_cond:
            reversal_signals['CE'] = True
    
    # Bearish Reversal at Resistance (R1/R2)
    elif nearest_level in ['r1', 'r2']:
        r_level = pivot_points[nearest_level]
        bearish_cond = (
            current_price >= r_level * (1 - pivot_reversal_threshold) and
            current_price <= r_level * (1 + pivot_reversal_threshold) and
            (recent_highs.max() >= r_level) and
            (ohlc['close'].iloc[-1] < ohlc['open'].iloc[-1])
            #(ohlc['volume'].iloc[-1] > ohlc['volume'].rolling(20).mean().iloc[-1])
        )
        if bearish_cond:
            reversal_signals['PE'] = True
    
    return reversal_signals

def detect_day_high_low_reversal(current_price):
    """Detect failed breakout and reversal patterns"""
    global day_range
    reversal_signals = {'CE': False, 'PE': False}
    today_high = day_range['high']
    today_low = day_range['low']
    
    # Bullish reversal (failed breakdown)
    if (current_price > today_low + min_reversal_distance and
        ohlc_data['close'].iloc[-1] > ohlc_data['open'].iloc[-1]):
        reversal_signals['CE'] = True
    # Bearish reversal (failed breakout)
    elif (current_price < today_high - min_reversal_distance and
        ohlc_data['close'].iloc[-1] < ohlc_data['open'].iloc[-1]):
        reversal_signals['PE'] = True
    
    return reversal_signals

def detect_gap(current_open, previous_close, threshold=0.25):
    """Detect gap-up or gap-down based on a threshold percentage."""
    if current_open > previous_close * (1 + threshold / 100):
        return 'Gap-Up'
    elif current_open < previous_close * (1 - threshold / 100):
        return 'Gap-Down'
    return 'No Gap'

def analyze_gap_sustainability(ohlc_data, current_price, threshold=0.25):
    """Analyze gap-up or gap-down sustainability using the current price."""
    first_candle = ohlc_data.iloc[0]
    current_open = first_candle['open']
    previous_close = daily_levels['prev_close']  # Previous day's close

    gap_status = detect_gap(current_open, previous_close, threshold)
    if gap_status == 'Gap-Up':
        if current_price > current_open:
            return 'Gap-Up Sustaining'
        else:
            return 'Gap-Up Reversing'
    elif gap_status == 'Gap-Down':
        if current_price < current_open:
            return 'Gap-Down Sustaining'
        else:
            return 'Gap-Down Reversing'
    return 'No Gap'

def update_day_range(tick):
    """Track current day's high/low in real-time"""
    global day_range
    if day_range['high'] is None:
        day_range['high'] = tick['last_price']
        day_range['low'] = tick['last_price']
    else:
        day_range['high'] = max(day_range['high'], tick['last_price'])
        day_range['low'] = min(day_range['low'], tick['last_price'])

def get_option_instrument(signal_type, target_premium=100):
    """Get option with premium closest to the target for the nearest expiry"""
    instruments = global_kite.instruments("NFO")
    nifty_ltp = global_kite.ltp(nifty_token)[str(nifty_token)]['last_price']

    # Filter by type and name
    filtered = [i for i in instruments
                if i['name'] == 'NIFTY'
                and i['instrument_type'] == ('CE' if signal_type == 'CE' else 'PE')]

    if not filtered:
        return None

    # Get nearest expiry
    expiries = sorted(list({i['expiry'] for i in filtered}))
    today = datetime.date.today()
    nearest_expiry = min([e for e in expiries if pd.to_datetime(e).date() >= today], default=None)

    if not nearest_expiry:
        return None

    # Filter by nearest expiry
    nearest_options = [i for i in filtered if i['expiry'] == nearest_expiry]

    # Get LTP for all nearest expiry options
    inst_ids = [i['instrument_token'] for i in nearest_options]
    quotes = global_kite.quote(inst_ids)

    # Find option with premium closest to target
    selected = None
    min_diff = float('inf')
    for inst in nearest_options:
        ltp = quotes[str(inst['instrument_token'])]['last_price']
        premium_diff = abs(ltp - target_premium)
        if premium_diff < min_diff:
            min_diff = premium_diff
            selected = inst
            selected['ltp'] = ltp

    return selected

def get_trend_from_ohlc(ohlc_data, current_price):
    """
    Determine trend (Uptrend, Downtrend, Sideways) using latest EMA values and current price.
    """
    if len(ohlc_data) < 2:  # Ensure there are at least 2 candles
        return "Not enough data"

    # Get latest EMA values
    latest_candle = ohlc_data.iloc[-1]
    prev_candle = ohlc_data.iloc[-2]

    ema_high = latest_candle['ema_high']
    ema_low = latest_candle['ema_low']
    prev_ema_high = prev_candle['ema_high']
    prev_ema_low = prev_candle['ema_low']

    # Determine trend based on EMA behavior & current price
    if ema_high > ema_low and current_price > ema_high and ema_high > prev_ema_high and ema_low > prev_ema_low:
        return "Uptrend"
    elif ema_high < ema_low and current_price < ema_low and ema_high < prev_ema_high and ema_low < prev_ema_low:
        return "Downtrend"
    else:
        return "Sideways"

def entry_conditions(current_price):
    """Check all entry conditions with pattern confirmation"""
    global last_exit_time, last_exit_reason, signal_message
    latest = ohlc_data.iloc[-1]

    gap_status = analyze_gap_sustainability(ohlc_data, current_price, gap_threshold)
    trend = get_trend_from_ohlc(ohlc_data, current_price)
    candle_patterns = detect_candle_patterns(ohlc_data)
    chart_patterns = detect_chart_patterns(ohlc_data, pattern_lookback)
    
    pivot_points = get_pivot_points()
    pivot_signals = detect_pivot_reversal(current_price, pivot_points, ohlc_data)
    day_high_low_signals = detect_day_high_low_reversal(current_price)
   
    message = f""
    message += f'\nLatest: {latest}'
    message += f"\nGap Status: {gap_status}"
    message += f'\nOpening Range 5m: {opening_range}'
    message += f"\nTrend: {trend}"
    message += f'\nCandle Patterns: {candle_patterns}'
    message += f'\nChart Patterns: {chart_patterns}'
    message += f"\nPivot Signals: {pivot_signals}"
    message += f"\nDay Range: {day_range}"
    message += f"\nDay Signals: {day_high_low_signals}"
    
    # Candle Patterns    
    # Neutral Candles
    neutral_candles = ['doji', 'spinning_top', 'marubozu', 'star']

    # 1-Candle Patterns
    bullish_1_candle = ['hammer', 'inverted_hammer', 'dragonfly_doji', 'bullish_spinning_top']
    bearish_1_candle = ['hanging_man', 'shooting_star', 'gravestone_doji', 'bearish_spinning_top']

    # 2-Candle Patterns
    bullish_2_candle = [
        'bullish_kicker', 'bullish_engulfing', 'bullish_harami',
        'piercing_line', 'tweezer_bottom'
    ]
    bearish_2_candle = [
        'bearish_kicker', 'bearish_engulfing', 'bearish_harami',
        'dark_cloud_cover', 'tweezer_top'
    ]

    # 3-Candle Patterns
    bullish_3_candle = [
        'morning_star', 'bullish_abandoned_baby', 'three_white_soldiers',
        'bullish_three_line_strike', 'morning_doji_star'
    ]
    bearish_3_candle = [
        'evening_star', 'bearish_abandoned_baby', 'three_black_crows',
        'bearish_three_line_strike', 'evening_doji_star'
    ]

    # Confirmation Patterns
    confirmation_bullish = ['three_inside_up', 'three_outside_up']
    confirmation_bearish = ['three_inside_down', 'three_outside_down']

    # Detect Candle Patterns
    detected_bullish_candle = any(p in candle_patterns for p in (
        bullish_1_candle + bullish_2_candle + bullish_3_candle + confirmation_bullish
    ))
    detected_bearish_candle = any(p in candle_patterns for p in (
        bearish_1_candle + bearish_2_candle + bearish_3_candle + confirmation_bearish
    ))

    # Detect Chart Patterns
    detected_bullish_chart_pattern = any(p in chart_patterns for p in [
        'inv_head_shoulders_break', 'double_bottom_break',
        'ascending_triangle', 'bull_flag', 'cup_handle'
    ])
    detected_bearish_chart_pattern = any(p in chart_patterns for p in [
        'head_shoulders_break', 'double_top_break',
        'descending_triangle', 'bear_flag'
    ])

    # Bullish entry (Call buy)
    opening_range_breakout = (current_price > opening_range['high'])
    previous_day_high_breakout = (current_price > daily_levels['prev_high'])
    pivot_point_breakout = (current_price > daily_levels['pivot'])
    
    bullish_gap = True
    if gap_status == 'Gap-Up Sustaining':
        bullish_gap = True
    elif gap_status == 'Gap-Up Reversing':
        bullish_gap = False

    bullish_reversal = pivot_signals['CE'] or day_high_low_signals['CE']
    bullish_cond = (opening_range_breakout or previous_day_high_breakout or pivot_point_breakout or (trend == 'Uptrend') and bullish_reversal) or (detected_bullish_candle or detected_bullish_chart_pattern)
    
    message += f'\nBullish Signal: {gap_status}'
    if opening_range_breakout:
        message += f'\nBullish Signal: opening_range_breakout'
    if previous_day_high_breakout:
        message += f'\nBullish Signal: opening_range_breakout'
    if pivot_point_breakout:
        message += f'\nBullish Signal: pivot_point_breakout'
    if detected_bullish_candle:
        message += f'\nBullish Signal: detected_bullish_candle'
    if detected_bullish_chart_pattern:
        message += f'\nBullish Signal: detected_bullish_pattern'
    if bullish_reversal:
        message += f'\nBullish Signal: bullish_reversal'
    if not bullish_cond:
        message += f'\nBullish Signal: Not found!'

    # Bearish entry (Put buy)
    opening_range_breakdown = (current_price < opening_range['low'])
    previous_day_low_breakdown = (current_price < daily_levels['prev_low'])
    pivot_point_breakdown = (current_price < daily_levels['pivot'])
    
    bearish_gap = True
    if gap_status == 'Gap-Down Sustaining':
        bearish_gap = True
    elif gap_status == 'Gap-Down Reversing':
        bearish_gap = False
    bearish_reversal = pivot_signals['PE'] or day_high_low_signals['PE']
    bearish_cond = (opening_range_breakdown or previous_day_low_breakdown or pivot_point_breakdown or (trend == 'Downtrend') or bearish_reversal) and (detected_bearish_candle or detected_bearish_chart_pattern)
    
    message += f'\nBearish Signal: {gap_status}'
    if opening_range_breakdown:
        message += f'\nBearish Signal: opening_range_breakdown'
    if previous_day_low_breakdown:
        message += f'\nBearish Signal: previous_day_low_breakdown'
    if pivot_point_breakdown:
        message += f'\nBearish Signal: pivot_point_breakdown'
    if detected_bearish_candle:
        message += f'\nBearish Signal: detected_bearish_candle'
    if detected_bearish_chart_pattern:
        message += f'\nBearish Signal: detected_bearish_pattern'
    if bearish_reversal:
        message += f'\nBearish Signal: bearish_reversal'
    if not bearish_cond:
        message += f'\nBearish Signal: Not found'

    # Cooldown period check using historical timestamps
    if last_exit_time:
        current_candle_time = latest['timestamp']
        if current_candle_time.tzinfo is not None:
            current_candle_time = current_candle_time.tz_convert(None)
        if last_exit_time.tzinfo is not None:
            last_exit_time = last_exit_time.tz_convert(None)
        time_since_last_exit =  current_candle_time - last_exit_time

        cooling_period = 5
        if "SL-Hit" in last_exit_reason or "SL-Hit | Profit Threshold" in last_exit_reason:
            cooling_period = sl_cooling_period
        elif last_exit_reason in ['Target-Hit', 'Target-TSL']:
            cooling_period = target_cooling_period
        elif last_exit_reason == 'BreakEven-Hit':
            cooling_period = breakeven_cooling_period

        cooling_duration = datetime.timedelta(minutes=cooling_period)        
        if time_since_last_exit < cooling_duration:
            remaining = cooling_duration - time_since_last_exit
            remaining_min = remaining.seconds // 60
            remaining_sec = remaining.seconds % 60
            
            status_msg = (f'Cooldown ({last_exit_reason}): '
                          f'{remaining_min}m {remaining_sec}s remaining')
            update_terminal(status_msg)
            return None
        else:
            message += "Cooldown period expired - Ready for new trades"
            #update_terminal("Cooldown period expired - Ready for new trades")

    '''
    # Enter only on 4th Minute
    now = datetime.datetime.now()
    if now.minute % 5 != 4:
        message += f'Waiting:  4th minute of the 5-minute candle to enter a new position...'
        update_termincal(message)
        return None   
    '''
    
    #logging.info(message)
    signal_message = message

    if bullish_cond:
        return 'CE'
    elif bearish_cond:
        return 'PE'
    return None

def get_opening_range():
    """Determine 5-minute opening range"""
    today_start = datetime.datetime.now().replace(hour=9, minute=15, second=0)
    historical = global_kite.historical_data(nifty_token, 
                                    today_start,
                                    datetime.datetime.now(),
                                    "5minute")
    if len(historical) > 0:
        opening_range['high'] = historical[0]['high']
        opening_range['low'] = historical[0]['low']

def get_pivot_points():
    return {
        's2': daily_levels['s2'],
        's1': daily_levels['s1'],
        'pp': daily_levels['pivot'],
        'r1': daily_levels['r1'],
        'r2': daily_levels['r2']
    }

def fetch_historical_ohlc(nifty_token):
    """Fetch past 5-minute OHLC data to backfill when script starts late."""
    global ohlc_data

    now = datetime.datetime.now()
    today_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # Fetch historical data from Kite API
    historical = global_kite.historical_data(nifty_token, today_start, now, "5minute")
    
    # Convert to DataFrame
    hist_df = pd.DataFrame(historical)
    if not hist_df.empty:
        hist_df.rename(columns={'date': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])  # Ensure proper datetime format
        ohlc_data = hist_df  # Set the fetched data as initial OHLC dataset

        # Update day range based on historical data
        day_range['high'] = hist_df['high'].max()
        day_range['low'] = hist_df['low'].min()

        logging.info(f"Backfilled OHLC data from {today_start} to {now}")
        logging.info(f"Updated day range: High = {day_range}")
    else:
        logging.info("No historical data found!")

def update_terminal(message):
    sys.stdout.write(f"\033[2K\033[1G{message}")
    sys.stdout.flush()

class ConfigLoader:
    """Handles configuration loading and validation"""
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = None

    def load(self):
        """Load and validate configuration"""
        try:
            with open(self.config_path) as f:
                self.config = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"Config file not found: {self.config_path}")
        except json.JSONDecodeError:
            raise RuntimeError(f"Invalid JSON in config file: {self.config_path}")

        # Validate kite section
        if 'kite' not in self.config:
            raise ValueError("Missing 'kite' section in config")
        
        # Validate backtest section
        if 'backtest' not in self.config:
            raise ValueError("Missing 'backtest' section in config")
        return self.config

class KiteManager:
    """Handles Kite Connect initialization and authentication"""
    def __init__(self, config):
        self.config = config
        self.kite = None

    def initialize(self):
        """Initialize Kite Connect connection"""
        try:
            # Access nested config values
            self.kite = KiteConnect(api_key=self.config['kite']['api_key'])
            self.kite.set_access_token(self.config['kite']['access_token'])
            return self.kite
        except Exception as e:
            raise RuntimeError(f"Kite connection failed: {str(e)}")

def main():
    """Main application entry point"""
    global global_config, global_kite
    # Load configurations
    try:
        config_loader = ConfigLoader()
        global_config = config_loader.load()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Config Error: {str(e)}")
        return

    # Initialize Kite Connect
    try:
        kite_manager = KiteManager(global_config)
        global_kite = kite_manager.initialize()
        logger.info("Kite Connect initialized successfully")

        logger.info("...started successfully")
        # Run backtest
        backtest(
            start_date=datetime.datetime.strptime(global_config['backtest']['start_date'], '%Y-%m-%d').date(),
            end_date=datetime.datetime.strptime(global_config['backtest']['end_date'], '%Y-%m-%d').date()
        )

    except Exception as e:
        logger.error(f"Kite Error: {str(e)}")

if __name__ == "__main__":
    try:
        logger.info("Starting...")
        main()
    except Exception as e:
        logger.error(f"Main: Error: {str(e)}")
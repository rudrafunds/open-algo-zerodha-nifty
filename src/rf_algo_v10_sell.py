from kiteconnect import KiteConnect, KiteTicker
import json
import pandas as pd
import numpy as np
import datetime
import time
import threading
import os
import logging
import sys
import atexit
import signal
import requests
import urllib.parse

# RF imports
from rf_patterns import detect_candle_patterns, detect_chart_patterns, calculate_emas
from rf_patterns import (
    CONST_CANDLES_NEUTRAL,
    CONST_CANDLES_1_BULLISH, CONST_CANDLES_2_BULLISH, CONST_CANDLES_3_BULLISH, CONST_CONFIRM_BULLISH,
    CONST_CANDLES_1_BEARISH, CONST_CANDLES_2_BEARISH, CONST_CANDLES_3_BEARISH, CONST_CONFIRM_BEARISH,
    CONST_CHART_BULLISH, CONST_CHART_BEARISH
)

# System starts here
global_system_init_time_start = time.time()

# Virtual
global_virtual_profit = None
global_virtual_positions_total = 0
global_virtual_positions_sl = 0
global_virtual_positions_target_tsl = 0

# Configuration settings
global_config = None

# Kite Connect
global_kite = None
global_kite_funds = 0

# Constants
CONST_LOGS_DIR="logs"
CONST_LOG_FILE_PREFIX="rf"
CONST_PATTERN_LOOKBACK = 20  # Candles for pattern detection
CONST_GAP_THRESHOLD = 0.25
CONST_PIVOT_REVERSAL_THRESHOLD = 0.005  # 0.5%
CONST_PIVOT_CONFIRMATION_CANDLES = 3     # Number of candles to analyze
CONST_REJECTION_MULTIPLIER = 2           # Wick-to-body ratio
CONST_MIN_REVERSAL_DISTANCE = 20  # Minimum points from day high/low to consider

CONST_GREEN = '\033[92m'
CONST_RED = '\033[91m'
CONST_YELLOW = '\033[93m'
CONST_RESET = '\033[0m'

CONST_SYMBOL_DICT = { # Symbols
    'SL-Hit': '🛡️',
    'TSL-Hit': '🎯',
    'BreakEven-Hit': '⚖️',
    'Shutdown': '🛑',
    'Buy': '✅',
    'Sell': '❌',
    'Target-Flag': '🔐⛳',
    'Loss-Flag': '🚩',
    'System-Start': '🌐',
    'Net-m2m': '💰',
    'Cut-Off': '⏰',
    'm2m': '💲',
    'green': '🟩',
    'yellow': '🟡',
    'red': '🟥',
    'Bull': '🐂',
    'CE': '🐂',
    'Bear': '🐻',
    'PE': '🐻',
    'Total': '➕',
    'Win': '🥇',
    'INR': '₹',
    'Up-Triangle': '🔼',
    'Down-Triangle': '🔻',
    'Goal': '🏆'
}

# Global variables
global_websocket_manager = None
global_tick_last_received_time = None
global_current_position = None
global_ohlc_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])
global_daily_levels = {}
global_opening_range = {}
global_day_range = {'high': None, 'low': None}
global_last_exit_time = None
global_last_exit_reason = None
global_profit_lock_active = False
global_profit_consecutive_sl_hits = 0
global_is_shutting_down = False

# Detect Signals
def detect_pivot_reversal(current_price, pivot_points, ohlc):
    """Detect reversal signals at pivot levels with price action and volume confirmation."""
    reversal_signals = {'CE': False, 'PE': False}
    
    # Get current candle details
    current_open = ohlc['open'].iloc[-1]
    current_close = ohlc['close'].iloc[-1]
    current_low = ohlc['low'].iloc[-1]
    current_high = ohlc['high'].iloc[-1]
    
    # Check all pivot points for potential reversals
    for level, pivot_value in pivot_points.items():
        # Calculate proximity to pivot level
        distance = abs(current_price - pivot_value) / pivot_value
        
        if distance <= CONST_PIVOT_REVERSAL_THRESHOLD:
            # Support Levels (S1, S2, P)
            if level in ['s1', 's2', 'pivot']:
                # Check if price has touched the support level recently
                recent_lows = ohlc['low'].iloc[-CONST_PIVOT_CONFIRMATION_CANDLES:].values
                if recent_lows.min() <= pivot_value:
                    # Calculate price rejection characteristics
                    lower_wick = min(current_open, current_close) - current_low
                    body_size = abs(current_close - current_open)
                    bullish_rejection = lower_wick > (body_size * CONST_REJECTION_MULTIPLIER)
                    
                    # Bullish reversal conditions
                    if all([
                        current_close > current_open,            # Bullish candle
                        bullish_rejection,                       # Price rejection
                        current_close > pivot_value              # Closing above support
                    ]):
                        reversal_signals['CE'] = True
            
            # Resistance Levels (R1, R2, P)
            elif level in ['r1', 'r2', 'pivot']:
                # Check if price has touched the resistance level recently
                recent_highs = ohlc['high'].iloc[-CONST_PIVOT_CONFIRMATION_CANDLES:].values
                if recent_highs.max() >= pivot_value:
                    # Calculate price rejection characteristics
                    upper_wick = current_high - max(current_open, current_close)
                    body_size = abs(current_close - current_open)
                    bearish_rejection = upper_wick > (body_size * CONST_REJECTION_MULTIPLIER)
                    
                    # Bearish reversal conditions
                    if all([
                        current_close < current_open,            # Bearish candle
                        bearish_rejection,                       # Price rejection
                        current_close < pivot_value              # Closing below resistance
                    ]):
                        reversal_signals['PE'] = True
    
    return reversal_signals

def detect_day_high_low_reversal(current_price):
    """Detect failed breakout and reversal patterns"""
    reversal_signals = {'CE': False, 'PE': False}
    today_high = global_day_range['high']
    today_low = global_day_range['low']
    
    # Bullish reversal (failed breakdown)
    if (current_price > today_low + CONST_MIN_REVERSAL_DISTANCE and
        global_ohlc_data['close'].iloc[-1] > global_ohlc_data['open'].iloc[-1]):
        reversal_signals['CE'] = True
    # Bearish reversal (failed breakout)
    elif (current_price < today_high - CONST_MIN_REVERSAL_DISTANCE and
        global_ohlc_data['close'].iloc[-1] < global_ohlc_data['open'].iloc[-1]):
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
    previous_close = global_daily_levels['prev_close']  # Previous day's close

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
    global global_day_range
    if global_day_range['high'] is None:
        global_day_range['high'] = tick['last_price']
        global_day_range['low'] = tick['last_price']
    else:
        global_day_range['high'] = max(global_day_range['high'], tick['last_price'])
        global_day_range['low'] = min(global_day_range['low'], tick['last_price'])

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

def get_opening_range():
    """Determine 5-minute opening range"""
    global global_opening_range

    today_start = datetime.datetime.now().replace(hour=9, minute=15, second=0)
    historical = global_kite.historical_data(global_config['system']['nifty_token'], 
                                    today_start,
                                    datetime.datetime.now(),
                                    "5minute")
    if len(historical) > 0:
        global_opening_range['high'] = historical[0]['high']
        global_opening_range['low'] = historical[0]['low']

def get_previous_trading_day(test_date):
    prev_day = test_date - pd.tseries.offsets.BDay(1)  # Move one day back

    # Get holidays
    holidays = pd.to_datetime(global_config['nse_holidays'])

    # Check if the previous day is a weekend or holiday
    while prev_day.weekday() >= 5 or prev_day in holidays:  # 5 and 6 are Saturday and Sunday
        prev_day -= pd.Timedelta(days=1)  # Skip to the previous day

    return prev_day

def calculate_daily_levels():
    """Calculate previous day's levels and pivot points"""
    global global_daily_levels

    # Get today's date
    today = datetime.date.today()

    # Get the previous trading day data
    previous_trading_day = get_previous_trading_day(today)
    historical = global_kite.historical_data(global_config['system']['nifty_token'], 
                                    previous_trading_day,
                                    today, 
                                    "day")
    if len(historical) >= 2:
        prev_day = historical[-2]
        pivot = (prev_day['high'] + prev_day['low'] + prev_day['close']) / 3
        s1 = 2 * pivot - prev_day['high']
        s2 = pivot - (prev_day['high'] - prev_day['low'])
        r1 = 2 * pivot - prev_day['low']
        r2 = pivot + (prev_day['high'] - prev_day['low'])
        global_daily_levels.update({
            'prev_high': prev_day['high'],
            'prev_low': prev_day['low'],
            'prev_close': prev_day['close'],
            'pivot': pivot,
            's1': s1,
            's2': s2,
            'r1': r1,
            'r2': r2
        })
        logging.info(f'System: Daily Levels - Ok')
        if global_config['system']['verbose']:
            logging.info(f"Levels: {global_daily_levels}")
    else:
        exit_system('Error to load daily levels')

# Utils
def telegram_notify(bot_message):
    if not global_config['system']['telegram']['enabled']:
        return None
    
    bot_token = global_config['system']['telegram']['bot_token']
    bot_chatID = global_config['system']['telegram']['bot_chat_id']
    if global_config['system']['virtual_env']:
        bot_chatID = global_config['system']['telegram']['bot_chat_id_dev']
    encoded_message = urllib.parse.quote(bot_message)
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=html&text=' + encoded_message
    response = requests.get(send_text)
    return response.json()

def format_value(value, compare_value=None):
    """Format the value with color based on its sign."""
    if (compare_value and value >= compare_value):
        return f"{CONST_GREEN}+{value:.2f}{CONST_RESET}"
    elif (compare_value and value <= compare_value):
        return f"{CONST_RED}-{value:.2f}{CONST_RESET}"
    elif value >= 0:
        return f"{CONST_GREEN}+{value:.2f}{CONST_RESET}"
    else:
        return f"{CONST_RED}{value:.2f}{CONST_RESET}"

def update_terminal(message):
    # Apply colors to specific keywords in the message        
    message = message.replace("PNL", f"{CONST_GREEN}PNL{CONST_RESET}")
    message = message.replace("CE", f"{CONST_GREEN}CE{CONST_RESET}")    
    
    message = message.replace("PE", f"{CONST_RED}PE{CONST_RESET}")
    message = message.replace("SL", f"{CONST_RED}SL{CONST_RESET}")    
    message = message.replace("BESL", f"{CONST_RED}BESL{CONST_RESET}")

    message = message.replace("Current", f"{CONST_YELLOW}Current{CONST_RESET}")
    message = message.replace("Target", f"{CONST_YELLOW}Target{CONST_RESET}")
    
    sys.stdout.write(f"\033[2K\033[1G{message}")
    sys.stdout.flush()

# Positions
def position_load_open():
    """Load open position from Kite"""
    global global_current_position
    positions = global_kite.positions()
    found_any = False
    # Loop through all positions
    for position in positions.get('net', []):
        if position['quantity'] < 0:
            # Found an open position
            found_any = True
            tradingsymbol = position['tradingsymbol']
            if global_config['system']['verbose']:
                logging.info(f"Position: Found kite open position - {tradingsymbol}")
            signal_type = ''
            if tradingsymbol.endswith('CE'):
                signal_type = 'CE'
            elif tradingsymbol.endswith('PE'):
                signal_type = 'PE'
        
            # Load into the memory
            global_current_position = {
                'type': signal_type,
                'instrument': position,
                'entry_price': position['buy_price'],
                'sl': position['last_price'] + (global_config['rules']['risk']['stop_loss']),
				'sl_25': 0,
                'target': position['last_price'] - (global_config['rules']['risk']['target']),
                'tsl': 0,
                'tsl_25': 0,
                'tsl_50': 0,
                'tsl_75': 0,
                'besl': 0,
                'besl_50': 0,
                'besl_75': 0,
                'quantity': position['quantity']
            }
            if global_config['system']['verbose']:
                logging.info(f"Position: Exists - {global_current_position}")
            # Done
            break

    logging.info(f"System: Open Kite positions - Ok")
    #if not found_any:
    #    logging.info("System: ...No open kite positions found")

def on_ticks(ws, ticks):
    global global_current_position, global_ohlc_data, global_tick_last_received_time, global_last_exit_reason
    global global_virtual_profit
    
    # Check for OHLC data gaps between ticks if misses due to network or WebSocket interuptions
    current_time = datetime.datetime.now()    
    if global_tick_last_received_time:
        time_diff = (current_time - global_tick_last_received_time).total_seconds()        
        # If gap larger than 1 minute (account for normal tick spacing)
        if time_diff > 60:
            logging.warning(f"WebSocket: Tick gap detected: {time_diff//60:.0f}m {time_diff%60:.0f}s")
            #websocket_manager._handle_data_gap(last_received_timestamp, current_time)

    # Retrieive Kite positions
    positions = global_kite.positions()
    net_mtm = sum([pos['m2m'] for pos in positions['net']])
    net_mtm = round(net_mtm, 2)

    # Process Ticks
    for tick in ticks:
        #print(tick)
        instrument_token = tick['instrument_token']
        current_price = tick['last_price']
        # Nifty Spot
        if instrument_token == global_config['system']['nifty_token']:
            # Update OHLC data
            update_ohlc(tick) 
            update_day_range(tick)           

            # Opening Range
            if datetime.datetime.now().time() < datetime.time(9,20) or not global_opening_range:
                get_opening_range()

            # Enter trade if no position open and find a signal
            if global_current_position is None:
                # No new trade after 2PM
                if datetime.datetime.now().time() > datetime.time(global_config['rules']['position']['cutoff_time_hh'], global_config['rules']['position']['cutoff_time_mm']):
                    exit_system(f"{CONST_SYMBOL_DICT['Cut-Off']} Cut-Off time over for new positions")
                
                # Search for any trade signals
                signal = position_entry_conditions(current_price)
                if signal:
                    logging.info(f"Position: Found {signal} signal")
                    position_open(signal, net_mtm)

        # Manage Open Position
        if global_current_position and instrument_token == global_current_position['instrument']['instrument_token']:
            running_mtm = global_current_position['quantity'] * (global_current_position['entry_price'] - current_price)

            # 2:30PM Cut-Off for open positions
            if datetime.datetime.now().time() > datetime.time(global_config['system']['cutoff_time_hh'], global_config['system']['cutoff_time_mm']):
                global_last_exit_reason = 'System-Cut-Off-Time'
                position_close(tick, current_price, running_mtm, net_mtm)
                exit_system(f"{CONST_SYMBOL_DICT['Cut-Off']} Cut-Off time for system shutdown")
            
            net_change = ((net_mtm+running_mtm)/global_kite_funds)*100
            net_change = round(net_change, 2)
            message = ""
            if not global_config['system']['only_progress_status']:   
                if not global_config['system']['termux']:
                    message += f"Net {format_value(net_mtm + running_mtm)}|"
                    message += f"% {format_value(net_change)}|"
                    message += f"P&L {format_value(running_mtm)}|"
                    if global_current_position['instrument'].get('strike') is not None:
                        message += f"{global_current_position['instrument'].get('strike')}-{global_current_position['type']}|"
                    else:
                        message += f"{global_current_position['type']}|"
                    
                    sl = global_current_position['sl']
                    if global_current_position['sl_25']:
                        sl = global_current_position['sl_25']
                    message += f"SL {sl:.2f}|"

                    message += f"Entry {global_current_position['entry_price']:.2f}|"
                    
                    besl = 0
                    if global_current_position['besl_75']:
                        besl = global_current_position['besl_75']
                    elif global_current_position['besl_50']:
                        besl = global_current_position['besl_50']
                    message += f"BESL {besl:.2f}|"

                    tsl = global_current_position['tsl']
                    if global_current_position['tsl_75']:
                        tsl = global_current_position['tsl_75']
                    elif global_current_position['tsl_50']:
                        tsl = global_current_position['tsl_50']
                    elif global_current_position['tsl_25']:
                        tsl = global_current_position['tsl_25']
                    message += f"TSL {tsl:.2f}|"
                    message += f"LTP {format_value(current_price, global_current_position['entry_price'])}|"
                    message += f"Target {global_current_position['target']:.2f}|"
                else:
                    net_change = ((running_mtm)/global_kite_funds)*100
                    net_change = round(net_change, 2)
                    message += f"P&L {format_value(running_mtm)}|"
                    message += f"% {format_value(net_change)}|"
                    message += f"LTP {format_value(current_price, global_current_position['entry_price'])}|"
                    message += f"T {global_current_position['target']:.2f}"

                update_terminal(message)
            else:
                # -- Start Progress Bar Code --
                entry_price = global_current_position['entry_price']
                if global_current_position['tsl']:
                    entry_price = global_current_position['tsl']
                target_price = global_current_position['target']
                sl_price = global_current_position['sl']
                current_px = current_price
                position_type_buy = False

                if position_type_buy:
                    if current_px >= entry_price:
                        direction = '▲T'
                        denominator = target_price - entry_price
                        progress = ((current_px - entry_price) / denominator) * 100 if denominator else 0
                    else:
                        direction = '▼SL'
                        denominator = entry_price - sl_price
                        progress = ((entry_price - current_px) / denominator) * 100 if denominator else 0
                else:
                    if current_px <= entry_price:
                        direction = '▼T'
                        denominator = entry_price - target_price
                        progress = ((entry_price - current_px) / denominator) * 100 if denominator else 0
                    else:
                        direction = '▲SL'
                        denominator = sl_price - entry_price
                        progress = ((current_px - entry_price) / denominator) * 100 if denominator else 0

                progress = max(0.0, min(100.0, progress))
                filled_length = int(progress // 10)
                empty = '░' * (10 - filled_length)

                if global_config['system']['termux']:
                    if direction in ('▲T', '▼T'):
                        filled = '🟩' * filled_length
                        bar = f'{format_value(current_price, entry_price)}|{filled}{empty}|'
                    else:
                        filled = '🟥' * filled_length
                        bar = f'{format_value(current_price, entry_price)}|{empty}{filled}|'

                    message += f"{format_value(running_mtm)}:{direction}{bar}"
                else:
                    if direction in ('▲T', '▼T'):
                        filled = '🟩' * filled_length
                        bar = f'{entry_price:.2f}|{format_value(current_price, entry_price)}|{filled}{empty}|{target_price:.2f}'
                    else:
                        filled = '🟥' * filled_length
                        bar = f'{sl_price:.2f}|{format_value(current_price, entry_price)}|{empty}{filled}|{entry_price:.2f}'

                    message += f"P&L {format_value(running_mtm)}|{progress:.1f}%: {direction} {bar}"
                # -- End Progress Bar --

                update_terminal(message)

            if (global_config['system']['virtual_env']):
                if global_virtual_profit is None:
                    global_virtual_profit = 0

            # Manage position
            manage_position(tick, net_mtm, running_mtm)

    # Last received tick timestamp
    global_tick_last_received_time = current_time

def fetch_historical_ohlc(nifty_token):
    """Fetch past 5-minute OHLC data to backfill when script starts late."""
    global global_ohlc_data, global_day_range

    now = datetime.datetime.now()
    today_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

    # Fetch historical data from Kite API
    historical = global_kite.historical_data(nifty_token, today_start, now, "5minute")
    
    # Convert to DataFrame
    hist_df = pd.DataFrame(historical)
    if not hist_df.empty:
        hist_df.rename(columns={'date': 'timestamp', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, inplace=True)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])  # Ensure proper datetime format
        global_ohlc_data = hist_df  # Set the fetched data as initial OHLC dataset

        # Update day range based on historical data
        global_day_range['high'] = hist_df['high'].max()
        global_day_range['low'] = hist_df['low'].min()

        logging.info(f"System: OHLC Backfilled - Ok")
    else:
        logging.info("System: ...No historical OHLC data found")

def update_ohlc(tick):
    """Maintain 5-minute OHLC data with EMAs using Kite ticker timestamps"""
    global global_ohlc_data
    
    # Use Kite's `exchange_timestamp` if available, otherwise fallback to system time
    tick_time = tick.get('exchange_timestamp', None)

    if tick_time is None:
        tick_time = datetime.datetime.now()  # Fallback if no exchange timestamp provided
    else:
        tick_time = pd.to_datetime(tick_time)  # Convert to datetime if it's not already

    # Align candle time to the nearest 5-minute mark
    bucket_minute = (tick_time.minute // 5) * 5
    candle_time = tick_time.replace(minute=bucket_minute, second=0, microsecond=0)

    # Ensure `global_ohlc_data` is initialized
    if global_ohlc_data is None or global_ohlc_data.empty:
        global_ohlc_data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close'])

    # Check if we need to start a new candle
    if global_ohlc_data.empty or global_ohlc_data.iloc[-1]['timestamp'] != candle_time:
        # Start a new candle
        new_row = {
            'timestamp': candle_time,
            'open': tick['last_price'],
            'high': tick['last_price'],
            'low': tick['last_price'],
            'close': tick['last_price']
        }
        
        new_row_df = pd.DataFrame([new_row])
        if not new_row_df.empty and not new_row_df.isna().all().all():
            if global_ohlc_data.empty:
                global_ohlc_data = new_row_df.copy()  # Direct assignment if global_ohlc_data is empty
            else:
                global_ohlc_data = pd.concat([global_ohlc_data.dropna(how="all"), new_row_df], ignore_index=True)
        #logging.info(f'Adding new 5-min candle: {global_ohlc_data}')

    else:
        # Update the existing candle
        global_ohlc_data.at[global_ohlc_data.index[-1], 'high'] = max(
            global_ohlc_data.iloc[-1]['high'], tick['last_price'])
        global_ohlc_data.at[global_ohlc_data.index[-1], 'low'] = min(
            global_ohlc_data.iloc[-1]['low'], tick['last_price'])
        global_ohlc_data.at[global_ohlc_data.index[-1], 'close'] = tick['last_price']
        #logging.info(f'Updating existing 5-min candle: {global_ohlc_data.iloc[-1]}')

    # Calculate EMAs after updating OHLC
    global_ohlc_data = calculate_emas(global_ohlc_data)

def position_entry_conditions(current_price):
    """Check all entry conditions with pattern confirmation"""
    global global_last_exit_time, global_last_exit_reason
    latest = global_ohlc_data.iloc[-1]

    gap_status = analyze_gap_sustainability(global_ohlc_data, current_price, CONST_GAP_THRESHOLD)
    trend = get_trend_from_ohlc(global_ohlc_data, current_price)
    candle_patterns = detect_candle_patterns(global_ohlc_data)
    chart_patterns = detect_chart_patterns(global_ohlc_data, CONST_PATTERN_LOOKBACK)
    
    pivot_points = {
        's2': global_daily_levels['s2'],
        's1': global_daily_levels['s1'],
        'pivot': global_daily_levels['pivot'],
        'r1': global_daily_levels['r1'],
        'r2': global_daily_levels['r2']
    }
    pivot_signals = detect_pivot_reversal(current_price, pivot_points, global_ohlc_data)
    day_high_low_signals = detect_day_high_low_reversal(current_price)
   
    message = f"\nLTP Price: {current_price}"
    message += f'\nLatest Candle: {latest}'
    message += f"\nGap Status: {gap_status}"
    message += f"\nTrend: {trend}"
    message += f'\nOpening Range 5m: {global_opening_range}'
    message += f"\nDay Range: {global_day_range}"
    message += f"\nPivot Signals: {pivot_signals}"
    message += f"\nDay Signals: {day_high_low_signals}"
    message += f'\nCandle Patterns: {candle_patterns}'
    message += f'\nChart Patterns: {chart_patterns}' 

    # Detect Candle Patterns and store matching elements
    matched_bullish_candle_patterns = [p for p in candle_patterns if p in (
        CONST_CANDLES_1_BULLISH + CONST_CANDLES_2_BULLISH + CONST_CANDLES_3_BULLISH + CONST_CONFIRM_BULLISH
    )]
    detected_bullish_candle = bool(matched_bullish_candle_patterns)
    matched_bearish_candle_patterns = [p for p in candle_patterns if p in (
        CONST_CANDLES_1_BEARISH + CONST_CANDLES_2_BEARISH + CONST_CANDLES_3_BEARISH + CONST_CONFIRM_BEARISH
    )]    
    detected_bearish_candle = bool(matched_bearish_candle_patterns)

    # Detect Chart Patterns and store matching elements
    matched_bullish_chart_patterns = [p for p in chart_patterns if p in CONST_CHART_BULLISH]
    detected_bullish_chart_pattern = bool(matched_bullish_chart_patterns)
    matched_bearish_chart_patterns = [p for p in chart_patterns if p in CONST_CHART_BEARISH]    
    detected_bearish_chart_pattern = bool(matched_bearish_chart_patterns)

    # Bullish entry (Put sell)
    opening_range_breakout = (current_price > global_opening_range['high'])
    previous_day_high_breakout = (current_price > global_daily_levels['prev_high'])
    pivot_point_breakout = (current_price > global_daily_levels['pivot'])    
    
    #bullish_reversal = pivot_signals['CE'] or day_high_low_signals['CE'] or any(p in latest_candle_patterns for p in ['hammer', 'morning_star'])
    bullish_gap = True
    if gap_status == 'Gap-Up Sustaining':
        bullish_gap = True
    elif gap_status == 'Gap-Up Reversing':
        bullish_gap = False

    bullish_reversal = pivot_signals['CE'] or day_high_low_signals['CE']
    bullish_cond = (opening_range_breakout or previous_day_high_breakout or pivot_point_breakout or (trend == 'Uptrend') or bullish_reversal) and (detected_bullish_candle or detected_bullish_chart_pattern)
    
    signals = []
    signals.append(gap_status)
    if opening_range_breakout:
        signals.append('opening_range_breakout')
    if previous_day_high_breakout:
        signals.append('previous_day_high_breakout')
    if pivot_point_breakout:        
        signals.append('pivot_point_breakout')
    if bullish_reversal:
        signals.append('bullish_reversal')
    message += f'\nBullish Signals: {signals}'
    if detected_bullish_candle:
        message += f"\nBullish Candles: {matched_bullish_candle_patterns}"
    if detected_bullish_chart_pattern:
        message += f"\nBullish Chart Patterns: {matched_bullish_chart_patterns}"
    if not bullish_cond:
        message += f'\nBullish: No Trade!'
    else:
        message += f'\nBullish: PE Trade'

    # Bearish entry (Call sell)
    opening_range_breakdown = (current_price < global_opening_range['low'])
    previous_day_low_breakdown = (current_price < global_daily_levels['prev_low'])
    pivot_point_breakdown = (current_price < global_daily_levels['pivot'])    
    
    #bearish_reversal = pivot_signals['PE'] or day_high_low_signals['PE'] or any(p in latest_candle_patterns for p in ['shooting_star', 'evening_star'])
    bearish_gap = True
    if gap_status == 'Gap-Down Sustaining':
        bearish_gap = True
    elif gap_status == 'Gap-Down Reversing':
        bearish_gap = False
    bearish_reversal = pivot_signals['PE'] or day_high_low_signals['PE']
    bearish_cond = (opening_range_breakdown or previous_day_low_breakdown or pivot_point_breakdown or (trend == 'Downtrend') or bearish_reversal) and (detected_bearish_candle or detected_bearish_chart_pattern)

    signals.clear()    
    signals.append(gap_status)
    if opening_range_breakdown:
        signals.append('opening_range_breakdown')
    if previous_day_low_breakdown:
        signals.append('previous_day_low_breakdown')
    if pivot_point_breakdown:
        signals.append('pivot_point_breakdown')
    if bearish_reversal:
        signals.append('bearish_reversal')
    message += f'\nBearish Signals: {signals}'
    if detected_bearish_candle:
        message += f"\nBearish Candles: {matched_bearish_candle_patterns}"
    if detected_bearish_chart_pattern:
        message += f"\nBearish Chart Patterns: {matched_bearish_chart_patterns}"
    if not bearish_cond:
        message += f'\nBearish: No Trade!'
    else:
        message += f'\nBearish: CE Trade\n'

    # Cooling period
    if global_last_exit_time:
        time_since_last_exit = datetime.datetime.now() - global_last_exit_time
        if global_last_exit_reason == 'SL-Hit':
            remaining_time = datetime.timedelta(minutes=global_config['rules']['cooldown_time']['stop_loss']) - time_since_last_exit
        elif global_last_exit_reason in ['Target-Hit', 'TSL-Hit']:
            remaining_time = datetime.timedelta(minutes=global_config['rules']['cooldown_time']['target']) - time_since_last_exit
        elif global_last_exit_reason in ['BreakEven-Hit']:
            remaining_time = datetime.timedelta(minutes=global_config['rules']['cooldown_time']['breakeven']) - time_since_last_exit
        if remaining_time.total_seconds() > 0:
            remaining_min = int(remaining_time.total_seconds() // 60)
            remaining_sec = int(remaining_time.total_seconds() % 60)
            update_terminal(f'Cooldown: Waiting {remaining_min}m {remaining_sec}s ...')
            return None
        else:
            message += "\nCooldown: Got over, Ready to trade!"
            global_last_exit_time = None
            global_last_exit_reason = None

    # Calculate the next 4th minute mark
    now = datetime.datetime.now()
    current_minute = now.minute
    current_second = now.second    
    current_candle_start = (current_minute // 5) * 5  # e.g., for 9:03 -> 9:00, for 9:06 -> 9:05    
    fourth_minute = current_candle_start + 4
    
    # Calculate the 5th minute of the current candle
    fifth_minute = current_candle_start + 5
    # Handle overflow (when fifth_minute >= 60)
    if fifth_minute >= 60:
        next_hour = (now.hour + 1) % 24  # Handle hour overflow at midnight
        fifth_minute_time = now.replace(hour=next_hour, minute=(fifth_minute % 60), second=0, microsecond=0)
    else:
        fifth_minute_time = now.replace(minute=fifth_minute, second=0, microsecond=0)
    if fourth_minute == current_minute:
        message += f"\nPosition: Ready 4th min of {fifth_minute_time.strftime('%H:%M:%S')} candle"
    else:
        remaining_time = (fourth_minute - current_minute) * 60 - current_second
        if remaining_time < 0:  # If we missed it, calculate for the next interval
            remaining_time += 5 * 60  # Move to the next 5-minute block
        remaining_min = remaining_time // 60
        remaining_sec = remaining_time % 60
        update_terminal(f"4th Candle: Waiting {remaining_min}m {remaining_sec}s ...")
        return None
    
    # Position entry signal ready!
    if global_config['system']['verbose']:
        logging.info(message)

    # Handle trade signals
    if bullish_cond:
        return 'PE'
    elif bearish_cond:
        return 'CE'
    else:
        update_terminal(f"Position: No Trade @{fifth_minute_time.strftime('%H:%M:%S')}")
        return None

def smart_decay_select_nifty_option(signal_type, target_premium=100, mask=25, margin_points=700):
    """Get option with premium in specified ranges, prioritizing lower range first"""
    instruments = global_kite.instruments("NFO")
    nifty_ltp = global_kite.ltp(global_config['system']['nifty_token'])[str(global_config['system']['nifty_token'])]['last_price']

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

    # Filter by nearest expiry and get LTPs
    nearest_options = [i for i in filtered if i['expiry'] == nearest_expiry]
    inst_ids = [i['instrument_token'] for i in nearest_options]
    quotes = global_kite.quote(inst_ids)
    
    # Add LTP to all instruments
    valid_options = []
    for inst in nearest_options:
        ltp = quotes.get(str(inst['instrument_token']), {}).get('last_price', None)
        if ltp and ltp > 0:  # Filter out invalid options
            inst['ltp'] = ltp
            valid_options.append(inst)

    if not valid_options:
        return None

    # Dynamic range calculation
    lower_range = (target_premium - mask, target_premium)  # e.g., 75-100
    upper_range = (target_premium, target_premium + mask)  # e.g., 100-125

    if global_config['system']['verbose']:
        logging.info(f"Premium: Smart selecting: {lower_range} to {upper_range}")

    # Find candidates in both ranges
    lower_candidates = [inst for inst in valid_options 
                       if lower_range[0] <= inst['ltp'] <= lower_range[1]]
    
    upper_candidates = [inst for inst in valid_options 
                       if upper_range[0] <= inst['ltp'] <= upper_range[1]]

    selected = None
    if lower_candidates:
        # In lower range: pick lowest premium in lower range
        selected = min(lower_candidates, key=lambda x: x['ltp'])
    elif upper_candidates:
        # In upper range: pick lowest premium in upper range
        selected = min(upper_candidates, key=lambda x: x['ltp'])
    else:
        # Fallback to closest premium overall if nothing in ranges
        valid = [inst for inst in valid_options]
        if valid:
            selected = min(valid, key=lambda x: abs(x['ltp'] - target_premium))

    logging.info(f"Premium: Selected - {selected['ltp']} from {selected['tradingsymbol']}")

    # Margin position logic (unchanged)
    selected_strike = selected['strike']
    margin_strike = selected_strike + (margin_points if signal_type == 'CE' else -margin_points)
    
    open_positions = global_kite.positions()['net']
    suffix = signal_type.upper()
    margin_position_exists = any(pos for pos in open_positions 
                               if pos['tradingsymbol'].startswith("NIFTY") 
                               and pos['quantity'] > 0
                               and pos['tradingsymbol'].endswith(suffix))
    
    margin_option = None
    if not margin_position_exists:
        margin_option = next((i for i in nearest_options if i['strike'] == margin_strike), None)
        if margin_option:
            margin_option['ltp'] = quotes.get(str(margin_option['instrument_token']), {}).get('last_price', None)

    return {
        "sell_instrument": selected,
        "buy_instrument": margin_option
    }

def position_open(signal_type, net_mtm):
    """Execute trade with premium-based strike selection"""
    global global_current_position, global_profit_lock_active, global_virtual_positions_total

    try:
        # MTM control and lot size decide
        lot_size = global_config['rules']['risk']['lot_size']
        profit_threshold = global_config['rules']['risk']['profit_threshold']
        loss_threshold = global_config['rules']['risk']['loss_threshold']

        if global_profit_lock_active:
            lot_size = global_config['rules']['risk']['profit_lock_lot_size']
            logging.info(f"Risk: Profit Lock: Lot size - {lot_size}, Consecutive SL hits - {global_profit_consecutive_sl_hits}")
        elif ((net_mtm + (profit_threshold * 0.1)) >= global_config['rules']['risk']['profit_threshold']):
            logging.info(f'Risk: MTM {format_value(net_mtm)} hit Profit threshold ({profit_threshold}) +10%')
            
            global_profit_lock_active = True
            lot_size = global_config['rules']['risk']['profit_lock_lot_size']

            # send telegram notification
            telegram_notify(f"{CONST_SYMBOL_DICT['Target-Flag']} Profit Target {net_mtm:.2f} hit. Locking...")
            logging.info(f"Risk: Profit Lock: Lot size - {lot_size}")
        elif ((net_mtm + (loss_threshold * 0.1)) <= global_config['rules']['risk']['loss_threshold']):
            logging.info(f'Risk: MTM {format_value(net_mtm)} hit Loss threshold ({loss_threshold}) -10%. Shutdown...')
            exit_system(f"{CONST_SYMBOL_DICT['Loss-Flag']} Loss Threshold {loss_threshold} hit")
 
        # Get suitable instrument
        nifty_option_premium = global_config['rules']['position']['nifty_option_premium']
        nifty_option_premium_mask = global_config['rules']['position']['nifty_option_premium_mask']
        nifty_option_margin = global_config['rules']['position']['nifty_option_margin']
        option_data = smart_decay_select_nifty_option(signal_type, nifty_option_premium, nifty_option_premium_mask, nifty_option_margin)
        
        if not option_data or not option_data["sell_instrument"]:
            logging.warning("Position: No suitable premium!")
            return

        sell_instrument = option_data["sell_instrument"]
        buy_instrument = option_data["buy_instrument"]

        logging.info(f"Position: Selling {sell_instrument['tradingsymbol']} @ {sell_instrument['ltp']}...")

        # If margin option is found and needed, buy it first but don't track it
        if buy_instrument:
            logging.info(f"Position: Buying Margin {buy_instrument['tradingsymbol']} @ {buy_instrument['ltp']}")

            if not global_config['system']['virtual_env']:
                order_id = global_kite.place_order(
                    tradingsymbol=buy_instrument['tradingsymbol'],
                    exchange=global_kite.EXCHANGE_NFO,
                    transaction_type=global_kite.TRANSACTION_TYPE_BUY,
                    quantity=lot_size,
                    order_type=global_kite.ORDER_TYPE_MARKET,
                    price=buy_instrument['ltp'],
                    product=global_kite.PRODUCT_MIS,
                    variety=global_kite.VARIETY_REGULAR
                )
                logging.info(f"Position: Margin Buy - Ok: Order ID: {order_id}")
            else:
                logging.info('Virtual: Margin Buy - Ok')

        # Now, place the SELL order for the main option
        if not global_config['system']['virtual_env']:
            order_id = global_kite.place_order(
                tradingsymbol=sell_instrument['tradingsymbol'],
                exchange=global_kite.EXCHANGE_NFO,
                transaction_type=global_kite.TRANSACTION_TYPE_SELL,
                quantity=lot_size,
                order_type=global_kite.ORDER_TYPE_LIMIT,
                price=sell_instrument['ltp'],
                product=global_kite.PRODUCT_MIS,
                variety=global_kite.VARIETY_REGULAR
            )
            logging.info(f"Position: Sell - Ok: Order ID: {order_id}")
        else:
            logging.info('Virtual: Sell - Ok')

        # Subscribe to updates
        global_websocket_manager.subscribe([sell_instrument['instrument_token']])
        global_current_position = {
            'type': signal_type,
            'instrument': sell_instrument,
            'entry_price': sell_instrument['ltp'],
            'sl': sell_instrument['ltp'] + global_config['rules']['risk']['stop_loss'],
            'sl_25': 0,
			'target': sell_instrument['ltp'] - global_config['rules']['risk']['target'],
            'tsl': 0,
            'tsl_25': 0,
            'tsl_50': 0,
            'tsl_75': 0,
            'besl': 0,
            'besl_50': 0,
            'besl_75': 0,
            'quantity': lot_size
        }

        # Update report
        global_config['persistent']['report']['positions']['total'] += 1
        global_config['persistent']['report']['positions'][signal_type] += 1

        # send telegram notification
        message = f"{CONST_SYMBOL_DICT['Buy']} {CONST_SYMBOL_DICT[signal_type]} Order Sell {sell_instrument['tradingsymbol']} {lot_size}Qty @ {sell_instrument['ltp']}"
        telegram_notify(message)

        if global_config['system']['verbose']:
            logging.info(f"Position: Current - {global_current_position}")

        if (global_config['system']['virtual_env']):
            global_virtual_positions_total += 1
            #logging.info(f"Virtual: Positions - Total: {global_virtual_positions_total}, Profit: {global_virtual_positions_target_tsl}, Loss: {global_virtual_positions_sl}")
            
    except Exception as e:
        logging.error(f"Position Error: {e}")

def manage_position(tick, net_mtm, position_mtm):
    """Manage open positions"""
    global global_current_position, global_last_exit_time, global_last_exit_reason
    global global_profit_lock_active, global_profit_consecutive_sl_hits
    global global_virtual_profit, global_virtual_positions_sl, global_virtual_positions_target_tsl
    
    target_points = global_config['rules']['risk']['target']
    trailing_treshold = global_config['rules']['risk']['trailing_threshold']
    sl_buffer = global_config['rules']['risk']['sl_buffer']
    trailing_buffer = global_config['rules']['risk']['lock_buffer']
    sl_trailing_enabled = global_config['rules']['risk']['sl_trailing_enable']

    current_price = tick['last_price']
    target = round(global_current_position['target'], 2)
    # Price <= 75% target
    if ((global_current_position['besl_75'] == 0) and (current_price <= (global_current_position['entry_price'] - (0.75 * target_points) + trailing_buffer))):
        global_current_position['besl_75'] = global_current_position['entry_price'] - (0.5 * target_points)
        logging.info(f"\nBreakEven: BESL_75 - {global_current_position['besl_75']} | {target}")
    # Price <= 50% target
    elif ((global_current_position['besl_50'] == 0) and (current_price <= (global_current_position['entry_price'] - (0.5 * target_points) + (trailing_buffer)))):
        global_current_position['besl_50'] = global_current_position['entry_price'] - global_config['rules']['risk']['besl_trailing']
        logging.info(f"\nBreakEven: BESL_50 - {global_current_position['besl_50']} | |{target}")
    # Price <= 25% target
    elif (sl_trailing_enabled and (global_current_position['sl_25'] == 0) and (current_price <= (global_current_position['entry_price'] - (0.25 * target_points)))):
        global_current_position['sl_25'] = global_current_position['sl'] - (0.25 * target_points)
        logging.info(f"\nRisk: SL_25 - {global_current_position['sl_25']} | {target}")
    # Price <= Target
    elif current_price <= (global_current_position['target'] + (trailing_buffer)):
        global_current_position['tsl'] = (global_current_position['target'] + trailing_treshold)
        global_current_position['target'] -= global_config['rules']['risk']['target']
        global_current_position['tsl_25'] = 0
        global_current_position['tsl_50'] = 0
        global_current_position['tsl_75'] = 0
        logging.info(f"\nTarget: TSL - {global_current_position['tsl']} | {target}")        
    # Price <= TSL & Price <= 75% of new Target
    elif global_current_position['tsl'] and (global_current_position.get('tsl_75') == 0) and (current_price <= (global_current_position['tsl'] + trailing_treshold - (0.75*target_points))):
        global_current_position['tsl_75'] = global_current_position['tsl'] - (0.75*target_points)
        logging.info(f"\nTrailing: TSL_75 - {global_current_position['tsl_75']} | {target}")
    # Price <= TSL & Price <= 50% of new Target
    elif global_current_position['tsl'] and (global_current_position.get('tsl_50')==0) and (current_price <= (global_current_position['tsl'] + trailing_treshold - (0.5*target_points))):
        global_current_position['tsl_50'] = global_current_position['tsl'] - (0.5*target_points)
        logging.info(f"\nTrailing: TSL_50 - {global_current_position['tsl_50']} | {target}")
    # Price <= TSL & Price <= 25% of new Target
    elif global_current_position['tsl'] and (global_current_position.get('tsl_25') == 0) and (current_price <= (global_current_position['tsl'] + trailing_treshold - (0.25*target_points))):
        global_current_position['tsl_25'] = global_current_position['tsl'] - (0.25*target_points)
        logging.info(f"\nTrailing: TSL_25 - {global_current_position['tsl_25']} | {target}")
    # Price >= TSL_75
    elif global_current_position.get('tsl_75') and current_price >= (global_current_position['tsl_75'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        global_last_exit_reason = 'TSL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['target'] += 1
        if global_profit_lock_active:
            # Reset profit lock consecutive sl
            global_profit_consecutive_sl_hits = 0
        global_last_exit_time = datetime.datetime.now()
        logging.info(f"\nTSL-Hit: TSL_75")
    # Price >= TSL_50
    elif global_current_position.get('tsl_50') and current_price >= (global_current_position['tsl_50'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        global_last_exit_reason = 'TSL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['target'] += 1
        if global_profit_lock_active:
            # Reset profit lock consecutive sl
            global_profit_consecutive_sl_hits = 0
        global_last_exit_time = datetime.datetime.now()
        logging.info(f"\nTSL-Hit: TSL_50")
    # Price >= TSL_25
    elif global_current_position.get('tsl_25') and current_price >= (global_current_position['tsl_25'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        global_last_exit_reason = 'TSL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['target'] += 1
        if global_profit_lock_active:
            # Reset profit lock consecutive sl
            global_profit_consecutive_sl_hits = 0
        global_last_exit_time = datetime.datetime.now()
        logging.info(f"\nTSL-Hit: TSL_25")
    # Price >= TSL
    elif global_current_position.get('tsl') and current_price >= (global_current_position['tsl'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        global_last_exit_reason = 'TSL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['target'] += 1
        if global_profit_lock_active:
            # Reset profit lock consecutive sl
            global_profit_consecutive_sl_hits = 0
        global_last_exit_time = datetime.datetime.now() 
    # Price >= BESL_75
    elif global_current_position.get('besl_75') and current_price >= (global_current_position['besl_75'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        logging.info(f"\nBreakEven-Hit: BESL_75")
        global_last_exit_reason = 'BreakEven-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['break_even'] += 1 
        global_last_exit_time = datetime.datetime.now()               
    # Price >= BESL_50
    elif global_current_position.get('besl_50') and current_price >= (global_current_position['besl_50'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_target_tsl += 1

        # Exit position
        logging.info(f"\nBreakEven-Hit: BESL_50")
        global_last_exit_reason = 'BreakEven-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['break_even'] += 1 
        global_last_exit_time = datetime.datetime.now()
    # Price >= SL_25
    elif global_current_position.get('sl_25') and (current_price >= (global_current_position['sl_25'] + sl_buffer)):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_sl += 1

        # Exit position
        global_last_exit_reason = 'SL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['stop_loss'] += 1 
        global_last_exit_time = datetime.datetime.now()        

        # Profit lock control
        if global_profit_lock_active:
            global_profit_consecutive_sl_hits += 1
        if global_profit_consecutive_sl_hits >= global_config['rules']['risk']['profit_lock_max_sl_hits']:
            exit_system(f"Profit Consecutive {global_profit_consecutive_sl_hits} SL-Hit")
        logging.info(f"\nSL-Hit: SL_25")
    elif current_price >= (global_current_position['sl'] + sl_buffer):
        if (global_config['system']['virtual_env']):
            global_virtual_positions_sl += 1

        # Exit position
        global_last_exit_reason = 'SL-Hit'
        position_close(tick, current_price, position_mtm, net_mtm)
        global_config['persistent']['report']['positions']['stop_loss'] += 1 
        global_last_exit_time = datetime.datetime.now()        

        # Profit lock control
        if global_profit_lock_active:
            global_profit_consecutive_sl_hits += 1
        if global_profit_consecutive_sl_hits >= global_config['rules']['risk']['profit_lock_max_sl_hits']:
            exit_system(f"Profit Consecutive {global_profit_consecutive_sl_hits} SL-Hit")

def position_close(tick, current_price=0, position_running_mtm=None, net_mtm=None):
    """Close current position"""
    global global_current_position, global_virtual_positions_total, global_virtual_profit, global_profit_lock_active
    
    try:
         # Check if position exists
        if not global_current_position:
            logging.warning("Position: No open position to close.")
            return
        
        message = f"\nPosition: Exiting: {global_current_position['instrument']['tradingsymbol']} @ {current_price} | {global_last_exit_reason}"
        running_change = (position_running_mtm/global_kite_funds)*100
        if position_running_mtm is not None:
            message += f": {position_running_mtm:.2f} | "
            message += f"{running_change:.2f} |"
        logging.info(message)
        if not global_config['system']['virtual_env']:
            # Kite Place Order     
            order_id = global_kite.place_order(
                tradingsymbol=global_current_position['instrument']['tradingsymbol'],
                exchange=global_kite.EXCHANGE_NFO,
                transaction_type=global_kite.TRANSACTION_TYPE_BUY,
                quantity=global_current_position['quantity'],
                order_type=global_kite.ORDER_TYPE_MARKET,
                product=global_kite.PRODUCT_MIS,
                variety=global_kite.VARIETY_REGULAR
            )
            logging.info(f"Position: Exit - Ok: Order ID - {order_id}")
        else:
            logging.info(f'Virtual: Exit - Ok: Order placed')

        # Unsubscribe from tick as no longer needed
        global_websocket_manager.unsubscribe([tick['instrument_token']])

        # send telegram notification
        message = f"{CONST_SYMBOL_DICT['Sell']} {CONST_SYMBOL_DICT[global_last_exit_reason]} Order Exit {global_current_position['instrument']['tradingsymbol']} @ {current_price} - {global_last_exit_reason}"
        if position_running_mtm is not None:
            pnl_symbol = CONST_SYMBOL_DICT['Up-Triangle'] if position_running_mtm >= 0 else CONST_SYMBOL_DICT['Down-Triangle']
            message += f" | {pnl_symbol} {position_running_mtm:.2f}"
            pnl_symbol = CONST_SYMBOL_DICT['Up-Triangle'] if (net_mtm+position_running_mtm) >= 0 else CONST_SYMBOL_DICT['Down-Triangle']
            message += f" | {CONST_SYMBOL_DICT['Net-m2m']} {pnl_symbol}{net_mtm+position_running_mtm:.2f}"
            net_change = ((net_mtm+position_running_mtm)/global_kite_funds)*100
            message += f" | {net_change:.2f}% |"

        telegram_notify(message)

        # Reset current position
        global_current_position = None

        if (global_config['system']['virtual_env']):
            global_virtual_profit += position_running_mtm
            logging.info(f"Virtual: Positions - Total: {global_virtual_positions_total}, Profit: {global_virtual_positions_target_tsl}, Loss: {global_virtual_positions_sl}")
            logging.info(f"Virtual P&L: {format_value(global_virtual_profit)}")

            # Control Virtual MTM
            if global_virtual_profit <= global_config['rules']['risk']['loss_threshold']:
                exit_system("Loss Threshold")
            elif global_virtual_profit >= global_config['rules']['risk']['profit_threshold']:
                if not global_profit_lock_active:
                    logging.info(f"\nVirtual: Risk: Profit-Treshold: {global_config['rules']['risk']['profit_threshold']} hit. Profit Locking...")
                    global_profit_lock_active = True

    except Exception as e:
        logging.error(f"Position: Exit Error: {e}")

def position_close_margin_all():
    """Close all margin options (buy positions for CE & PE)"""
    try:
        positions = global_kite.positions()
        margin_positions = []

        # Find all margin (buy) positions
        for position in positions.get('net', []):
            if position['quantity'] > 0:  # Only buy positions
                tradingsymbol = position['tradingsymbol']
                margin_positions.append({
                    'tradingsymbol': tradingsymbol,
                    'quantity': position['quantity']
                })
                if global_config['system']['verbose']:
                    logging.info(f"Position: Found margin {tradingsymbol}")

        # Close each margin position
        for pos in margin_positions:
            tradingsymbol = pos['tradingsymbol']
            quantity = pos['quantity']

            if not global_config['system']['virtual_env']:
                order_id = global_kite.place_order(
                    tradingsymbol=tradingsymbol,
                    exchange=global_kite.EXCHANGE_NFO,
                    transaction_type=global_kite.TRANSACTION_TYPE_SELL,
                    quantity=quantity,
                    order_type=global_kite.ORDER_TYPE_MARKET,
                    product=global_kite.PRODUCT_MIS,
                    variety=global_kite.VARIETY_REGULAR
                )
                if global_config['system']['verbose']:
                    logging.info(f"Position: Exit margin {tradingsymbol} | Order ID: {order_id}")
            else:
                logging.info(f"Virtual: Exit Margin Position: {tradingsymbol}")

        logging.info("System: Exit margin positions - Ok")

    except Exception as e:
        logging.error(f"Position: Margin exit Error: {e}")

# Core
# Web Socket Manager
class WebSocketManager:
    def __init__(self, api_key, access_token, on_ticks, get_subscription_tokens):
        self.api_key = api_key
        self.access_token = access_token
        self.on_ticks_callback = on_ticks
        self.get_subscription_tokens = get_subscription_tokens
        self.retry_delay = 5  # seconds
        self.is_connected = False
        self.is_reconnect = False
        
        self.ticker = KiteTicker(self.api_key, self.access_token)
        
        # Register event handlers
        self.ticker.on_connect = self._on_connect
        self.ticker.on_close = self._on_close
        self.ticker.on_ticks = self._on_ticks
        self.ticker.on_error = self._on_error

    def _on_connect(self, ws, response):
        """Handle websocket connection"""
        logging.info("System: WebSocket: Connection - Ok")
        self.is_connected = True
        self.is_reconnect = False
        
        # Subscribe to required instruments
        tokens = self.get_subscription_tokens()
        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)
            if global_config['system']['verbose']:
                logging.info(f"WebSocket: Subscribed to tokens - {tokens}")

    def _on_close(self, ws, code, reason):
        """Handle websocket closure"""
        logging.info(f"WebSocket: Connection closed. Code: {code}, Reason: {reason}")
        self.is_connected = False
        if not global_is_shutting_down:
            self._schedule_reconnect()

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks"""
        self.on_ticks_callback(ws, ticks)

    def _on_error(self, ws, code, reason):
        """Handle websocket errors"""
        logging.error(f"WebSocket: Error - Code: {code}, Reason: {reason}")
        self.is_connected = False
        if not global_is_shutting_down:
            self._schedule_reconnect()

    def _schedule_reconnect(self):
        """Schedule a reconnection attempt"""
        if not global_is_shutting_down:
            if not self.is_reconnect:
                self.is_reconnect = True
                logging.info(f"WebSocket: Reconnecting in {self.retry_delay} seconds...")
                threading.Timer(self.retry_delay, self.connect).start()
            else:
                logging.info(f"WebSocket: Reconnect is already in progress...")

    def connect(self):
        """Initiate websocket connection with retry mechanism"""
        try:
            self.ticker.connect(threaded=True)
        except Exception as e:
            logging.error(f"WebSocket: Error - Connection: {e}")
            self._schedule_reconnect()

    def disconnect(self, code=1000, reason="Normal Shutdown"):
        """Stop the websocket connection"""
        if self.is_connected:
            if global_config['system']['verbose']:
                logging.info("WebSocket: Stopping ticker...")            
            tokens = self.get_subscription_tokens()
            if tokens:
                try:
                    self.ticker.unsubscribe(tokens)
                    if global_config['system']['verbose']:
                        logging.info(f"WebSocket: Unsubscribed tokens - {tokens}")
                except Exception as e:
                    logging.error(f"WebSocket: Error - Unsubscribing: {e}")
            try:
                self.ticker.close(code, reason)           
                time.sleep(2)
                self.ticker.stop()
                logging.info("System: WebSocket: Disconnected - Ok")
            except Exception as e:
                logging.error(f"WebSocket: Error - Closing connection: {e}")

    def subscribe(self, tokens):
        """Subscribe to the given instrument tokens"""
        if self.is_connected:
            try:         
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
                if global_config['system']['verbose']:
                    logging.info(f"WebSocket: Subscribed to tokens - {tokens}")
            except Exception as e:
                logging.error(f"WebSocket: Error - Subscription: {e}")

    def unsubscribe(self, tokens):
        """Unsubscribe from the given instrument tokens"""
        if self.is_connected:
            try:         
                self.ticker.unsubscribe(tokens)
                if global_config['system']['verbose']:              
                    logging.info(f"WebSocket: Unsubscribed from tokens - {tokens}")
            except Exception as e:
                logging.error(f"WebSocket: Error - Unsubscribing: {e}")


    def resubscribe(self):
        """Resubscribe to current instruments"""
        if self.is_connected:
            tokens = self.get_subscription_tokens()
            self.ticker.subscribe(tokens)
            self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
            logging.info(f"WebSocket: Resubscribed to tokens - {tokens}")

def start_trading():
    global global_current_position, global_websocket_manager, global_kite_funds

    # Existing initialization logic
    if global_config['system']['virtual_env']:
        logging.info("System: Running in Virtual Environment")

    # Initialize system
    funds = global_kite.margins()
    global_kite_funds = funds['equity']['available']['opening_balance']
    calculate_daily_levels()
    position_load_open()
    fetch_historical_ohlc(global_config['system']['nifty_token'])

    # Define subscription tokens provider
    def get_subscription_tokens():
        tokens = [global_config['system']['nifty_token']]
        if global_current_position:
            tokens.append(global_current_position['instrument']['instrument_token'])
        return tokens

    # Initialize WebSocket manager
    global_websocket_manager = WebSocketManager(
        api_key=global_config['kite']['api_key'],
        access_token=global_config['kite']['access_token'],
        on_ticks=on_ticks,
        get_subscription_tokens=get_subscription_tokens
    )

    # Start websocket connection
    global_websocket_manager.connect()

    # System initialization done
    system_init_time_end = time.time()
    logging.info(f"System: init - Ok: {system_init_time_end-global_system_init_time_start:.6f}sec")

    # Main loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exit_system("User Terminated")

def exit_system(reason="", update_reports=True):
    """
    Gracefully exit the system after closing positions and stopping threads.
    """
    global global_is_shutting_down
    try:
        # Set shutdown flag to prevent further processing
        global_is_shutting_down = True
        message = f"{CONST_SYMBOL_DICT['Shutdown']} System: Shutdown: {reason} ..."
        logger.info(message)
        telegram_notify(message)

        # Close all margin positions
        position_close_margin_all()

        # Update reports in config
        if update_reports and global_config['persistent']['report']['enabled']:
            global_config['persistent']['report']['last_update_time'] = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
            total_positions = global_config['persistent']['report']['positions']['total']
            target_positions = global_config['persistent']['report']['positions']['target']
            global_config['persistent']['report']['positions']['win_percent'] = (round((target_positions / total_positions) * 100, 2) if total_positions > 0 else 0)

            global_config['persistent']['report']['opening_funds'] = global_kite_funds

            positions = global_kite.positions()
            net_mtm = sum([pos['m2m'] for pos in positions['net']])

            if global_config['system']['virtual_env']:
                net_mtm = global_virtual_profit
            net_mtm = round(net_mtm, 2)
            global_config['persistent']['report']['today_pnl'] = net_mtm

            net_change = (net_mtm/global_kite_funds)*100
            net_change = round(net_change, 2)
            global_config['persistent']['report']['today_change'] = net_change

            # Write to the file
            with open('config.json', 'w') as file:
                json.dump(global_config, file, indent=4)
            logging.info("System: Updated reports - Ok")
        
            report = f"Positions: {CONST_SYMBOL_DICT['Total']}: {global_config['persistent']['report']['positions']['total']} | "
            report += f"{CONST_SYMBOL_DICT['Win']}: {global_config['persistent']['report']['positions']['win_percent']} | "
            report += f"{CONST_SYMBOL_DICT['TSL-Hit']}: {global_config['persistent']['report']['positions']['target']} | "
            report += f"{CONST_SYMBOL_DICT['BreakEven-Hit']}: {global_config['persistent']['report']['positions']['break_even']} | "
            report += f"{CONST_SYMBOL_DICT['SL-Hit']}: {global_config['persistent']['report']['positions']['stop_loss']} | "
            report += f"{CONST_SYMBOL_DICT['Bull']}: {global_config['persistent']['report']['positions']['CE']} | "
            report += f"{CONST_SYMBOL_DICT['Bear']}: {global_config['persistent']['report']['positions']['PE']} |"

            # Report notify log and telegram
            logging.info(report)
            telegram_notify(report)

            # Goals
            current_funds = global_kite_funds + net_mtm  # current valuation
            goals = global_config['rules']['goal']
            goal_report = ""

            if current_funds < goals['short']:
                gap_pct = round(((goals['short'] - current_funds) / goals['short']) * 100, 2)
                goal_report = f"{CONST_SYMBOL_DICT['Goal']} Goal: {gap_pct}% - short"
            elif current_funds < goals['long']:
                gap_pct = round(((goals['long'] - current_funds) / goals['long']) * 100, 2)
                goal_report = f"{CONST_SYMBOL_DICT['Goal']} Goal: {gap_pct}% - long"
            elif current_funds < goals['decade']:
                gap_pct = round(((goals['decade'] - current_funds) / goals['decade']) * 100, 2)
                goal_report = f"{CONST_SYMBOL_DICT['Goal']} Goal: {gap_pct}% - decade"
            else:
                goal_report = f"{CONST_SYMBOL_DICT['Target-Flag']} Goal: 🌟 You have achieved the Decade Goal of ₹{goals['decade']:,}!"

            logging.info(goal_report)
            telegram_notify(goal_report)

            # Log and notify Today's pnl
            pnl_symbol = (
                f"{CONST_SYMBOL_DICT['green']}{CONST_SYMBOL_DICT['Up-Triangle']}"
                if net_mtm >= 0
                else f"{CONST_SYMBOL_DICT['red']}{CONST_SYMBOL_DICT['Down-Triangle']}"
            )
            today_pnl = f"P&L: {pnl_symbol}{CONST_SYMBOL_DICT['INR']} {net_mtm:.2f} | {net_change}% |"
            logging.info(today_pnl)
            telegram_notify(today_pnl)

        # Stop the ticker and ensure all threads are stopped
        if global_websocket_manager is not None:
            global_websocket_manager.disconnect()

        # Wait for threads to finish
        for thread in threading.enumerate():
            if thread.is_alive() and thread is not threading.current_thread():
                logging.info(f"System: Waiting for thread {thread.name} to finish...")
                thread.join(timeout=5)  # Wait for up to 5 seconds for the thread to finish
        logging.info("System: All threads stopped - Ok")

        logging.info("System: Shutdown - Ok")
        os._exit(0) # Exit with status code 0, indicating successful termination
    except SystemExit as e:
        logging.info(f"System: Exiting with code: {e}")
        raise  # Re-raise the exception to allow the script to exit
    except Exception as e:
        logging.error(f"System: Exception: {e}")
        os._exit(1)  # Exit with a non-zero status code to indicate an error

# Handle exit and sys signals
atexit.register(exit_system)
def signal_handler(sig, frame):
    exit_system(f"Signal {sig} received")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Logger
class ColoredLogger:
    class ColoredFormatter(logging.Formatter):
        # Define color codes
        CONST_GREEN = '\033[92m'
        CONST_RED = '\033[91m'
        CONST_YELLOW = '\033[93m'
        CONST_RESET = '\033[0m'

        def format(self, record):
            message = super().format(record)
            # Apply colors for positive/neutral terms
            green_terms = ["True", "CE", "Buy", "high", "TSL-Hit", "Target-Hit",
                           "BreakEven-Hit", "target", "Profit", "breakout", "bullish",
                           "Gap-Up", "Sustaining", "up", "hammer", "soldiers",
                           "success", "Success", "Uptrend", "s1", "s2", "Ready", "Ok"]
            for term in green_terms:
                message = message.replace(term, f"{self.CONST_GREEN}{term}{self.CONST_RESET}")

            # Apply colors for negative terms
            red_terms = ["False", "PE", "Sell", "low", "SL-Hit", "sl", "Loss", "Exit",
                         "breakdown", "bearish", "Gap-Down", "Reversing", "down", "crows",
                         "Error", "Downtrend", "Shutdown", "r1", "r2", "Exception"]
            for term in red_terms:
                message = message.replace(term, f"{self.CONST_RED}{term}{self.CONST_RESET}")

            # Apply colors for system terms
            yellow_terms = ["System", "Virtual", "Sideways", "No Trade", "pivot", "close"]
            for term in yellow_terms:
                message = message.replace(term, f"{self.CONST_YELLOW}{term}{self.CONST_RESET}")

            return message

    @classmethod
    def configure(cls):
        """Configures the root logger with colored formatting."""
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create and add new console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(cls.ColoredFormatter('%(message)s'))
        
        # Ensure the stream uses UTF-8 (for Windows)
        console_handler.stream.reconfigure(encoding="utf-8")  # Works in Python 3.9+

        logger.addHandler(console_handler)

        # Ensure logs directory exists
        os.makedirs(CONST_LOGS_DIR, exist_ok=True)

        # Generate log filename with today's date inside 'logs/' folder
        log_filename = os.path.join(CONST_LOGS_DIR, f"{CONST_LOG_FILE_PREFIX}_{datetime.datetime.now().strftime('%d_%m_%Y')}.log")

        # Create file handler (no rotation, new file each day)
        file_handler = logging.FileHandler(log_filename, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
 
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

        # Validate sections
        if 'kite' not in self.config:
            raise ValueError("Missing 'kite' section in config")
        if 'system' not in self.config:
            raise ValueError("Missing 'system' section in config") 
        if 'rules' not in self.config:
            raise ValueError("Missing 'rules' section in config")
        if 'persistent' not in self.config:
            raise ValueError("Missing 'persistent' section in config")  
        if 'nse_holidays' not in self.config:
            raise ValueError("Missing 'nse_holidays' section in config") 
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
            raise RuntimeError(f"Kite Connect Error: {str(e)}")

def main():
    """Main application entry point"""
    global global_config, global_kite
    # Load configurations
    try:
        config_loader = ConfigLoader()
        global_config = config_loader.load()
        message = f"{CONST_SYMBOL_DICT['System-Start']} System: Init Selling: {global_config['name']} - {global_config['version']}..."
        logger.info(message)

        # send telegram notification
        telegram_notify(message)
        logger.info("System: Config - Ok")
    except Exception as e:
        logger.error(f"Config Error: {str(e)}")
        return

    # Market time validations
    now = datetime.datetime.now()
    now_time = now.time()
    market_open = datetime.time(global_config['system']['market_open_time_hh'], global_config['system']['market_open_time_mm'])
    trade_start = datetime.time(global_config['system']['start_trading_time_hh'], global_config['system']['start_trading_time_mm'])
    market_close = datetime.time(global_config['system']['market_close_time_hh'], global_config['system']['market_close_time_hh'])

    if now_time < market_open or now_time > market_close:       
        exit_system("Market closed", False)        

    if now_time < trade_start:
        start_dt = datetime.datetime.combine(now.date(), trade_start)
        remaining_seconds = int((start_dt - now).total_seconds())
        logger.info(f"System: Market is open, waiting until {trade_start.strftime('%I:%M %p')} to start trading...")       

        while datetime.datetime.now().time() < trade_start:
            now = datetime.datetime.now()
            remaining_seconds = int((start_dt - now).total_seconds())
            minutes, seconds = divmod(remaining_seconds, 60)
            update_terminal(f"Start: Waiting {minutes:02d}m {seconds:02d}s...")
            time.sleep(1)
            
    # Initialize Kite Connect
    try:
        kite_manager = KiteManager(global_config)
        global_kite = kite_manager.initialize()
        logger.info("System: Kite Connect - Ok")

        # Start trading
        start_trading()
    except Exception as e:
        logger.error(f"Kite Error: {str(e)}")

if __name__ == "__main__":
    try:        
        ColoredLogger.configure()
        logger = logging.getLogger()
        main()
    except Exception as e:
        logger.error(f"Main: Error: {str(e)}")
        exit_system("Main() Exception")
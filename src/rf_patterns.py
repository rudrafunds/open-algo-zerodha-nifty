import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress

CONST_EMA_PERIOD = 20

# Candle Patterns    
# Neutral Candles
CONST_CANDLES_NEUTRAL = ['doji', 'spinning_top', 'marubozu', 'star']

# 1-Candle Patterns
CONST_CANDLES_1_BULLISH = ['hammer', 'inverted_hammer', 'dragonfly_doji', 'bullish_spinning_top']
CONST_CANDLES_1_BEARISH = ['hanging_man', 'shooting_star', 'gravestone_doji', 'bearish_spinning_top']

# 2-Candle Patterns
CONST_CANDLES_2_BULLISH = [
    'bullish_kicker', 'bullish_engulfing', 'bullish_harami',
    'piercing_line', 'tweezer_bottom'
]
CONST_CANDLES_2_BEARISH = [
    'bearish_kicker', 'bearish_engulfing', 'bearish_harami',
    'dark_cloud_cover', 'tweezer_top'
]

# 3-Candle Patterns
CONST_CANDLES_3_BULLISH = [
    'morning_star', 'bullish_abandoned_baby', 'three_white_soldiers',
    'bullish_three_line_strike', 'morning_doji_star'
]
CONST_CANDLES_3_BEARISH = [
    'evening_star', 'bearish_abandoned_baby', 'three_black_crows',
    'bearish_three_line_strike', 'evening_doji_star'
]

# Confirmation Patterns
CONST_CONFIRM_BULLISH = ['three_inside_up', 'three_outside_up']
CONST_CONFIRM_BEARISH = ['three_inside_down', 'three_outside_down']

# Chart Patterns
CONST_CHART_BULLISH = [
    'inv_head_shoulders_break', 'double_bottom_break',
    'ascending_triangle', 'bull_flag', 'cup_handle'
]
CONST_CHART_BEARISH = [
    'head_shoulders_break', 'double_top_break',
    'descending_triangle', 'bear_flag'
]
def detect_candle_patterns(df):
    """Detect all 1-3 candle patterns and confirmations"""
    patterns = []
    latest = df.iloc[-1]
    prev1 = df.iloc[-2] if len(df) >= 2 else None
    prev2 = df.iloc[-3] if len(df) >= 3 else None
    
    # Candle metrics
    body = abs(latest['close'] - latest['open'])
    total_range = latest['high'] - latest['low']
    upper_wick = latest['high'] - max(latest['open'], latest['close'])
    lower_wick = min(latest['open'], latest['close']) - latest['low']
    
    # 1-Candle Patterns
    # Doji variations
    if body <= total_range * 0.05:
        if upper_wick == lower_wick:
            patterns.append('doji')
        elif lower_wick > 2 * upper_wick:
            patterns.append('dragonfly_doji')
        elif upper_wick > 2 * lower_wick:
            patterns.append('gravestone_doji')
        else:
            patterns.append('long_legged_doji')
    
    # Spinning Tops
    if body < total_range * 0.5 and upper_wick > 0 and lower_wick > 0:
        if latest['close'] > latest['open']:
            patterns.append('bullish_spinning_top')
        else:
            patterns.append('bearish_spinning_top')
    
    # Hammer family
    if lower_wick >= 2 * body and body > 0:
        if latest['close'] > latest['open']:
            patterns.append('hammer')
        else:
            patterns.append('hanging_man')
    
    # Shooting Star/Inverted Hammer
    if upper_wick >= 2 * body and body > 0:
        if latest['close'] < latest['open']:
            patterns.append('shooting_star')
        else:
            patterns.append('inverted_hammer')
    
    # Marubozu
    if body == total_range:
        if latest['close'] > latest['open']:
            patterns.append('bullish_marubozu')
        else:
            patterns.append('bearish_marubozu')
    
    # 2-Candle Patterns
    if prev1 is not None and not prev1.empty and prev1.any():
        prev_body = abs(prev1['close'] - prev1['open'])
        
        # Engulfing
        if (body > prev_body and 
            ((latest['close'] > prev1['open'] and latest['open'] < prev1['close']) or 
            (latest['close'] < prev1['open'] and latest['open'] > prev1['close']))):
            if latest['close'] > latest['open']:
                patterns.append('bullish_engulfing')
            else:
                patterns.append('bearish_engulfing')
        
        # Harami
        if (body < prev_body and latest['high'] < prev1['high'] and latest['low'] > prev1['low']):
            if latest['close'] > latest['open']:
                patterns.append('bullish_harami')
            else:
                patterns.append('bearish_harami')
        
        # Tweezer
        if abs(latest['high'] - prev1['high']) < total_range * 0.05:
            patterns.append('tweezer_top' if latest['close'] < latest['open'] else 'tweezer_bottom')
        
        # Piercing Line/Dark Cloud
        if prev1['close'] < prev1['open'] and latest['close'] > latest['open']:
            if latest['open'] < prev1['low'] and latest['close'] > (prev1['open'] + prev1['close']) / 2:
                patterns.append('piercing_line')
        elif prev1['close'] > prev1['open'] and latest['close'] < latest['open']:
            if latest['open'] > prev1['high'] and latest['close'] < (prev1['open'] + prev1['close']) / 2:
                patterns.append('dark_cloud_cover')
        
        # Kicker Pattern
        if prev1['close'] < prev1['open'] and latest['open'] > prev1['high']:
            patterns.append('bullish_kicker')
        elif prev1['close'] > prev1['open'] and latest['open'] < prev1['low']:
            patterns.append('bearish_kicker')
    
    # 3-Candle Patterns
    if prev2 is not None:
        # Morning/Evening Star
        if prev2['close'] < prev2['open'] and abs(prev1['close'] - prev1['open']) < total_range * 0.1 and latest['close'] > prev2['close']:
            patterns.append('morning_star')
        elif prev2['close'] > prev2['open'] and abs(prev1['close'] - prev1['open']) < total_range * 0.1 and latest['close'] < prev2['close']:
            patterns.append('evening_star')
        
        # Three Soldiers
        if latest['close'] > prev1['close'] > prev2['close']:
            patterns.append('three_white_soldiers')
        elif latest['close'] < prev1['close'] < prev2['close']:
            patterns.append('three_black_crows')
        
        # Three Line Strike
        if latest['close'] > prev1['close'] > prev2['close']:
            patterns.append('bullish_three_line_strike')
        elif latest['close'] < prev1['close'] < prev2['close']:
            patterns.append('bearish_three_line_strike')
        
        # Abandoned Baby
        if prev2['close'] < prev2['open'] and prev1['high'] < prev2['low'] and latest['close'] > latest['open']:
            patterns.append('bullish_abandoned_baby')
        elif prev2['close'] > prev2['open'] and prev1['low'] > prev2['high'] and latest['close'] < latest['open']:
            patterns.append('bearish_abandoned_baby')
    
    # Confirmation Patterns
    if prev1 is not None and prev2 is not None:
        if latest['close'] > prev1['close'] > prev2['close']:
            patterns.append('three_inside_up')
        elif latest['close'] < prev1['close'] < prev2['close']:
            patterns.append('three_inside_down')
        if latest['close'] > prev1['close'] and prev1['close'] > prev2['close'] and latest['open'] < prev1['close']:
            patterns.append('three_outside_up')
        elif latest['close'] < prev1['close'] and prev1['close'] < prev2['close'] and latest['open'] > prev1['close']:
            patterns.append('three_outside_down')
    
    return patterns

def detect_chart_patterns(df, pattern_lookback):
    """Detect all major chart patterns in OHLC data."""
    patterns = []
    prices = df.iloc[-pattern_lookback:]  # Use `.iloc[]` to ensure index safety

    if prices.empty or len(prices) < 10:
        return []  # Not enough data to detect patterns

    # Head and Shoulders
    peaks = argrelextrema(prices['high'].values, np.greater, order=3)[0]
    if len(peaks) >= 3:
        left, head, right = peaks[-3:]
        high_left, high_head, high_right = prices['high'].iloc[left], prices['high'].iloc[head], prices['high'].iloc[right]
        
        if (high_head > high_left and high_head > high_right and 
            abs(high_left - high_right) < 0.01 * high_left):
            patterns.append('head_shoulders')

            # Ensure valid slice range
            low_values = prices['low'].iloc[min(left, right):max(left, right)]
            if not low_values.empty and prices['low'].iloc[-1] < low_values.min():
                patterns.append('head_shoulders_break')

    # Inverse Head and Shoulders
    troughs = argrelextrema(prices['low'].values, np.less, order=3)[0]
    if len(troughs) >= 3:
        left, head, right = troughs[-3:]
        low_left, low_head, low_right = prices['low'].iloc[left], prices['low'].iloc[head], prices['low'].iloc[right]

        if (low_head < low_left and low_head < low_right and 
            abs(low_left - low_right) < 0.01 * low_left):
            patterns.append('inv_head_shoulders')

            high_values = prices['high'].iloc[min(left, right):max(left, right)]
            if not high_values.empty and prices['high'].iloc[-1] > high_values.max():
                patterns.append('inv_head_shoulders_break')

    # Double Top
    if len(peaks) >= 2:
        p1, p2 = peaks[-2:]
        if abs(prices['high'].iloc[p1] - prices['high'].iloc[p2]) < 0.01 * prices['high'].iloc[p1]:
            patterns.append('double_top')

            low_values = prices['low'].iloc[p1:p2]
            if not low_values.empty and prices['low'].iloc[-1] < low_values.min():
                patterns.append('double_top_break')

    # Double Bottom
    if len(troughs) >= 2:
        t1, t2 = troughs[-2:]
        if abs(prices['low'].iloc[t1] - prices['low'].iloc[t2]) < 0.01 * prices['low'].iloc[t1]:
            patterns.append('double_bottom')

            high_values = prices['high'].iloc[t1:t2]
            if not high_values.empty and prices['high'].iloc[-1] > high_values.max():
                patterns.append('double_bottom_break')

    # Triangle Patterns
    if len(prices) > 10:
        high, low = prices['high'].values, prices['low'].values
        upper_trend = np.polyfit(np.arange(10), high[-10:], 1)[0]
        lower_trend = np.polyfit(np.arange(10), low[-10:], 1)[0]

        if upper_trend < 0 and lower_trend > 0:
            patterns.append('symmetrical_triangle')
        elif upper_trend < 0 and abs(lower_trend) < 0.1:
            patterns.append('descending_triangle')
        elif lower_trend > 0 and abs(upper_trend) < 0.1:
            patterns.append('ascending_triangle')

    # Flag/Pennant
    if len(prices) > 10:
        prev_range = prices['high'].iloc[-10:-5].max() - prices['low'].iloc[-10:-5].min()
        current_range = prices['high'].iloc[-5:].max() - prices['low'].iloc[-5:].min()
        
        if current_range < 0.5 * prev_range:
            if np.all(np.diff(prices['close'].iloc[-5:]) > 0):
                patterns.append('bull_flag')
            elif np.all(np.diff(prices['close'].iloc[-5:]) < 0):
                patterns.append('bear_flag')

    # Cup and Handle
    if len(prices) > 30:
        cup = prices.iloc[-30:-10]
        handle = prices.iloc[-10:]
        
        if (cup['low'].min() == cup['low'].iloc[15] and 
            handle['high'].max() < cup['high'].max() and 
            prices['close'].iloc[-1] > handle['high'].max()):
            patterns.append('cup_handle')

    return patterns

def detect_latest_candle_patterns(df):
    """Identify reversal candlestick patterns"""
    patterns = []
    latest = df.iloc[-1]
    
    # Calculate candle body and wicks
    body_size = abs(latest['close'] - latest['open'])
    upper_wick = latest['high'] - max(latest['close'], latest['open'])
    lower_wick = min(latest['close'], latest['open']) - latest['low']
    
    # Hammer/Inverse Hammer
    if lower_wick > body_size * 2 and upper_wick < body_size:
        patterns.append('hammer' if latest['close'] > latest['open'] else 'hanging_man')
    
    # Shooting Star
    if upper_wick > body_size * 2 and lower_wick < body_size:
        patterns.append('shooting_star')
    
    return patterns

def calculate_emas(df):
    """Calculate 20-period EMA for highs and lows"""
    df['ema_high'] = df['high'].ewm(span=CONST_EMA_PERIOD, adjust=False).mean()
    df['ema_low'] = df['low'].ewm(span=CONST_EMA_PERIOD, adjust=False).mean()
    return df


def v2_detect_chart_patterns(df, pattern_lookback=100):
    """Detect all major chart patterns in a single function"""
    CONST_CHART_PATTERNS_BULLISH = [
        'inv_head_shoulders', 'diamond_bottom', 'island_reversal_bottom',
        'rounding_bottom', 'double_bottom', 'triple_bottom', 'falling_wedge',
        'ascending_triangle', 'bull_flag', 'rectangle_bullish',
        'broadening_bottom', 'symmetrical_triangle_bullish', 'cup_handle'
    ]
    CONST_CHART_PATTERNS_BEARISH = [
        'head_shoulders', 'diamond_top', 'island_reversal_top',
        'rounding_top', 'double_top', 'triple_top', 'rising_wedge',
        'descending_triangle', 'bear_flag', 'rectangle_bearish',
        'broadening_top', 'symmetrical_triangle_bearish'
    ]
    patterns = []
    if len(df) < pattern_lookback:
        return patterns
    
    prices = df.iloc[-pattern_lookback:]
    highs = prices['high'].values
    lows = prices['low'].values
    closes = prices['close'].values
    volumes = prices['volume'].values if 'volume' in prices.columns else None

    # Helper functions
    def is_approx_equal(a, b, threshold=0.015):
        return abs(a - b) <= threshold * ((a + b)/2)

    # 1. Find peaks and troughs
    peaks = argrelextrema(highs, np.greater, order=3)[0]
    troughs = argrelextrema(lows, np.less, order=3)[0]

    # 2. Head & Shoulders Patterns
    if len(peaks) >= 3:
        left, head, right = peaks[-3:]
        if (highs[head] > highs[left] and 
            highs[head] > highs[right] and
            is_approx_equal(highs[left], highs[right])):
            neckline = lows[min(left, right):max(left, right)].min()
            if closes[-1] < neckline:
                patterns.append('head_shoulders')

    # 3. Inverse Head & Shoulders
    if len(troughs) >= 3:
        left, head, right = troughs[-3:]
        if (lows[head] < lows[left] and 
            lows[head] < lows[right] and
            is_approx_equal(lows[left], lows[right])):
            neckline = highs[min(left, right):max(left, right)].max()
            if closes[-1] > neckline:
                patterns.append('inv_head_shoulders')

    # 4. Double/Triple Tops/Bottoms
    for count in [2, 3]:
        if len(peaks) >= count:
            top_peaks = peaks[-count:]
            if all(is_approx_equal(highs[p], highs[top_peaks[0]]) for p in top_peaks):
                patterns.append(f"{'triple' if count==3 else 'double'}_top")

        if len(troughs) >= count:
            bot_troughs = troughs[-count:]
            if all(is_approx_equal(lows[t], lows[bot_troughs[0]]) for t in bot_troughs):
                patterns.append(f"{'triple' if count==3 else 'double'}_bottom")

    # 5. Triangle Patterns
    if len(closes) >= 20:
        x = np.arange(20)
        high_slope = linregress(x, highs[-20:])[0]
        low_slope = linregress(x, lows[-20:])[0]
        
        if abs(high_slope + low_slope) < 0.005:  # Symmetrical
            if closes[-1] > highs[-30:].mean():
                patterns.append('symmetrical_triangle_bullish')
            elif closes[-1] < lows[-30:].mean():
                patterns.append('symmetrical_triangle_bearish')
        elif high_slope < -0.01 and low_slope > 0.01:
            patterns.append('ascending_triangle')
        elif high_slope < -0.01 and abs(low_slope) < 0.005:
            patterns.append('descending_triangle')

    # 6. Flag/Pennant Patterns
    if len(closes) >= 15:
        flagpole = np.max(highs[-15:-10]) - np.min(lows[-15:-10])
        consolidation = np.max(highs[-10:]) - np.min(lows[-10:])
        
        if consolidation < 0.5 * flagpole:
            if closes[-1] > closes[-10]:
                patterns.append('bull_flag')
            else:
                patterns.append('bear_flag')

    # 7. Wedge Patterns
    if len(closes) >= 30:
        x = np.arange(30)
        high_slope = linregress(x, highs[-30:])[0]
        low_slope = linregress(x, lows[-30:])[0]
        
        if high_slope > 0.01 and low_slope > 0.02:
            patterns.append('rising_wedge')
        elif high_slope < -0.01 and low_slope < -0.02:
            patterns.append('falling_wedge')

    # 8. Diamond Patterns
    if len(closes) >= 40:
        first_half = np.std(highs[-40:-20])/np.std(lows[-40:-20])
        second_half = np.std(highs[-20:])/np.std(lows[-20:])
        
        if first_half > 1.5 and second_half < 0.67:
            patterns.append('diamond_bottom')
        elif first_half < 0.67 and second_half > 1.5:
            patterns.append('diamond_top')

    # 9. Rectangle Patterns
    if len(closes) >= 20:
        high_std = np.std(highs[-20:])/np.mean(highs[-20:])
        low_std = np.std(lows[-20:])/np.mean(lows[-20:])
        
        if high_std < 0.01 and low_std < 0.01:
            if closes[-1] > np.mean(highs[-20:]):
                patterns.append('rectangle_bullish')
            else:
                patterns.append('rectangle_bearish')

    # 10. Broadening Formations
    if len(closes) >= 40:
        high_slope = linregress(np.arange(40), highs[-40:])[0]
        low_slope = linregress(np.arange(40), lows[-40:])[0]
        
        if high_slope > 0.015 and low_slope < -0.015:
            patterns.append('broadening_top')
        elif high_slope < -0.015 and low_slope > 0.015:
            patterns.append('broadening_bottom')

    # 11. Rounding Patterns
    if len(closes) >= 30:
        x = np.arange(30)
        y = closes[-30:]
        curvature = np.polyfit(x, y, 2)[0] * 1000
        if curvature > 0.5:
            patterns.append('rounding_bottom')
        elif curvature < -0.5:
            patterns.append('rounding_top')

    # 12. Island Reversal
    if len(closes) >= 10:
        gap_up = closes[-10] > highs[-11] * 1.005
        gap_down = closes[-1] < lows[-2] * 0.995
        if gap_up and gap_down:
            patterns.append('island_reversal_bottom')
        elif gap_down and gap_up:
            patterns.append('island_reversal_top')
    
    # 13. Cup and Handle
    if len(prices) >= 50:
        cup = prices.iloc[-50:-20]
        handle = prices.iloc[-20:]
        
        cup_depth = cup['low'].min()
        cup_formation = (cup['high'].idxmax() > len(cup)*0.6 and
                        cup['low'].idxmin() > len(cup)*0.4)
        
        if cup_formation and handle['high'].max() < cup['high'].max():
            patterns.append('cup_handle')

    return sorted(list(set(patterns)))  # Remove duplicates and sort
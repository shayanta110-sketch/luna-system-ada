"""
Steno compression module for memory context nexus.
Implements both Steno (human-readable) and Steno-M (machine-optimized) compression.
"""

import re
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Abbreviation rules for common words and phrases
ABBREVIATIONS = {
    # Common words
    'you': 'u',
    'are': 'r',
    'be': 'b',
    'see': 'c',
    'why': 'y',
    'for': '4',
    'to': '2',
    'too': '2',
    'ate': '8',
    'before': 'b4',
    'be right back': 'brb',
    'by the way': 'btw',
    'for your information': 'fyi',
    'oh my god': 'omg',
    'laugh out loud': 'lol',
    'thank you': 'ty',
    'you are welcome': 'urw',
    'please': 'pls',
    'thanks': 'thx',
    'because': 'b/c',
    'with': 'w/',
    'without': 'w/o',
    'about': 'abt',
    'between': 'btwn',
    'through': 'thru',
    'though': 'tho',
    'although': 'altho',
    'until': 'til',
    'and': '&',
    'or': '|',
    'not': '!',
    'very': 'v',
    'really': 'rly',
    'probably': 'prob',
    'people': 'ppl',
    'person': 'prsn',
    'message': 'msg',
    'question': 'q',
    'answer': 'a',
    'example': 'ex',
    'information': 'info',
    'application': 'app',
    'document': 'doc',
    'number': '#',
    'number of': '# of',
    'at sign': '@',
    'dollar': '$',
    'percent': '%',
    'plus': '+',
    'minus': '-',
    'equal': '=',
}

# Articles to remove (case-insensitive)
ARTICLES = ['a', 'an', 'the']

# Date compression patterns
DATE_PATTERNS = [
    (re.compile(r'(\d{4})-(\d{2})-(\d{2})'), r'\1\2\3'),  # YYYY-MM-DD -> YYYYMMDD
    (re.compile(r'(\d{2})/(\d{2})/(\d{4})'), r'\3\1\2'),  # MM/DD/YYYY -> YYYYMMDD
    (re.compile(r'(\d{2})-(\d{2})-(\d{4})'), r'\3\1\2'),  # MM-DD-YYYY -> YYYYMMDD
    (re.compile(r'(\d{4})/(\d{2})/(\d{2})'), r'\1\2\3'),  # YYYY/MM/DD -> YYYYMMDD
]

# Machine-optimized schema (fixed positional fields)
STENO_M_SCHEMA = {
    'fields': [
        ('timestamp', 14),   # YYYYMMDDHHMMSS
        ('sender_id', 8),
        ('channel_id', 8),
        ('msg_type', 2),     # 01=text, 02=command, 03=system
        ('priority', 1),     # 0-9
        ('length', 5),       # message length as 5-digit zero-padded
        'message_body'       # variable length, after fixed header
    ]
}


def compress_steno(text: str, human_readable: bool = True) -> str:
    """
    Compress text using Steno format (human-readable).
    
    Args:
        text: Input text to compress
        human_readable: If True, use Steno (abbreviations, article removal).
                      If False, use Steno-M (machine-optimized).
    
    Returns:
        Compressed string
    """
    if not human_readable:
        return compress_steno_m(text)
    
    # Human-readable Steno compression
    # Step 1: Remove articles
    for article in ARTICLES:
        # Word boundary pattern for whole word removal
        pattern = re.compile(r'\b' + re.escape(article) + r'\s+', re.IGNORECASE)
        text = pattern.sub('', text)
    
    # Step 2: Apply abbreviations (word-level)
    words = text.split()
    compressed_words = []
    for word in words:
        word_lower = word.lower()
        # Check exact match
        if word_lower in ABBREVIATIONS:
            compressed_words.append(ABBREVIATIONS[word_lower])
        else:
            # Check if abbreviation is part of punctuation
            punct_match = re.match(r'^(\w+)([.!?;:,])$', word)
            if punct_match:
                base, punct = punct_match.groups()
                base_lower = base.lower()
                if base_lower in ABBREVIATIONS:
                    compressed_words.append(ABBREVIATIONS[base_lower] + punct)
                else:
                    compressed_words.append(word)
            else:
                compressed_words.append(word)
    text = ' '.join(compressed_words)
    
    # Step 3: Compress dates in standard formats
    for pattern, replacement in DATE_PATTERNS:
        text = pattern.sub(replacement, text)
    
    # Step 4: Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def compress_steno_m(entry: Dict[str, Any]) -> bytes:
    """
    Compress a conversation entry using Steno-M (machine-optimized) format.
    
    Args:
        entry: Dictionary with keys: timestamp, sender_id, channel_id, 
               msg_type, priority, message_body
    
    Returns:
        Bytes representing the fixed-width compressed entry
    """
    # Extract fields with defaults
    timestamp = entry.get('timestamp', datetime.now())
    sender_id = entry.get('sender_id', 'unknown')
    channel_id = entry.get('channel_id', 'default')
    msg_type = entry.get('msg_type', 'text')
    priority = entry.get('priority', 5)
    message_body = entry.get('message_body', '')
    
    # Convert timestamp to YYYYMMDDHHMMSS format
    if isinstance(timestamp, datetime):
        ts_str = timestamp.strftime('%Y%m%d%H%M%S')
    elif isinstance(timestamp, str):
        # Try to parse common formats
        for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y%m%d%H%M%S']:
            try:
                ts_dt = datetime.strptime(timestamp, fmt)
                ts_str = ts_dt.strftime('%Y%m%d%H%M%S')
                break
            except ValueError:
                continue
        else:
            ts_str = timestamp.replace('-', '').replace(':', '').replace(' ', '')[:14].ljust(14, '0')
    else:
        ts_str = str(timestamp)[:14].ljust(14, '0')
    
    # Pad or truncate to field lengths
    ts_encoded = ts_str[:14].ljust(14, '0').encode('ascii')
    sender_encoded = sender_id[:8].ljust(8, ' ').encode('ascii')
    channel_encoded = channel_id[:8].ljust(8, ' ').encode('ascii')
    
    # Message type mapping
    msg_type_map = {'text': '01', 'command': '02', 'system': '03'}
    msg_type_code = msg_type_map.get(str(msg_type).lower(), '01')
    msg_type_encoded = msg_type_code[:2].ljust(2, '0').encode('ascii')
    
    # Priority
    priority_str = str(priority)[:1]
    priority_encoded = priority_str.ljust(1, '0').encode('ascii')
    
    # Compress message body using Steno first (for better compression)
    compressed_body = compress_steno(message_body, human_readable=True)
    body_bytes = compressed_body.encode('utf-8')
    
    # Length field (5 digits, zero-padded)
    length_str = str(len(body_bytes)).zfill(5)[:5]
    length_encoded = length_str.encode('ascii')
    
    # Assemble fixed-width header + variable body
    return (ts_encoded + sender_encoded + channel_encoded + 
            msg_type_encoded + priority_encoded + length_encoded + body_bytes)


def decompress_steno(compressed: str) -> str:
    """
    Decompress Steno (human-readable) format back to natural language.
    
    Note: This is an approximate restoration as abbreviations may be ambiguous.
    
    Args:
        compressed: Steno compressed string
    
    Returns:
        Decompressed string with common expansions
    """
    # Reverse abbreviation mapping
    reverse_abbr = {v: k for k, v in ABBREVIATIONS.items()}
    
    # Expand abbreviations
    words = compressed.split()
    expanded_words = []
    for word in words:
        # Check for punctuation suffix
        punct_match = re.match(r'^([^.!?;:,]+)([.!?;:,])$', word)
        if punct_match:
            base, punct = punct_match.groups()
            if base in reverse_abbr:
                expanded_words.append(reverse_abbr[base] + punct)
            else:
                expanded_words.append(word)
        else:
            if word in reverse_abbr:
                expanded_words.append(reverse_abbr[word])
            else:
                expanded_words.append(word)
    
    # Reconstruct date formats (YYYYMMDD -> YYYY-MM-DD)
    text = ' '.join(expanded_words)
    date_pattern = re.compile(r'\b(\d{4})(\d{2})(\d{2})\b')
    text = date_pattern.sub(r'\1-\2-\3', text)
    
    return text


def decompress_steno_m(data: bytes) -> Dict[str, Any]:
    """
    Decompress Steno-M format back to dictionary.
    
    Args:
        data: Bytes from compress_steno_m
    
    Returns:
        Dictionary with conversation entry fields
    """
    if len(data) < 38:  # Minimum header size: 14+8+8+2+1+5 = 38
        raise ValueError(f"Data too short for Steno-M header: {len(data)} bytes")
    
    # Parse fixed header fields
    ts_str = data[0:14].decode('ascii').strip('0')
    sender_id = data[14:22].decode('ascii').strip()
    channel_id = data[22:30].decode('ascii').strip()
    msg_type_code = data[30:32].decode('ascii')
    priority_str = data[32:33].decode('ascii')
    length_str = data[33:38].decode('ascii')
    
    # Parse length and extract body
    try:
        body_length = int(length_str)
    except ValueError:
        body_length = 0
    
    if len(data) < 38 + body_length:
        raise ValueError(f"Incomplete body: expected {body_length} bytes, got {len(data)-38}")
    
    body_bytes = data[38:38+body_length]
    compressed_body = body_bytes.decode('utf-8')
    
    # Decompress message body
    message_body = decompress_steno(compressed_body)
    
    # Parse timestamp
    if ts_str and len(ts_str) == 14:
        try:
            timestamp = datetime.strptime(ts_str, '%Y%m%d%H%M%S')
        except ValueError:
            timestamp = ts_str
    else:
        timestamp = ts_str if ts_str else None
    
    # Map message type code back
    msg_type_map = {'01': 'text', '02': 'command', '03': 'system'}
    msg_type = msg_type_map.get(msg_type_code, 'text')
    
    # Priority as int
    try:
        priority = int(priority_str) if priority_str.isdigit() else 5
    except ValueError:
        priority = 5
    
    return {
        'timestamp': timestamp,
        'sender_id': sender_id,
        'channel_id': channel_id,
        'msg_type': msg_type,
        'priority': priority,
        'message_body': message_body
    }


def compress_conversation_entries(entries: List[Dict[str, Any]], 
                                   output_format: str = 'steno') -> List[bytes]:
    """
    Compress a list of conversation entries.
    
    Args:
        entries: List of entry dictionaries
        output_format: 'steno' (human-readable batch) or 'steno-m' (machine)
    
    Returns:
        List of compressed entries as bytes
    """
    if output_format == 'steno-m':
        return [compress_steno_m(entry) for entry in entries]
    else:
        # For batch human-readable, concatenate with separators
        compressed_strings = []
        for entry in entries:
            body = compress_steno(entry.get('message_body', ''), human_readable=True)
            # Add metadata prefix
            ts = entry.get('timestamp', '')
            if isinstance(ts, datetime):
                ts = ts.strftime('%Y%m%d %H:%M:%S')
            prefix = f"[{ts}] {entry.get('sender_id', '?')}: "
            compressed_strings.append(compress_steno(prefix + body, human_readable=True))
        return [s.encode('utf-8') for s in compressed_strings]


def batch_compress(entries: List[Dict[str, Any]]) -> bytes:
    """
    Compress multiple entries into a single Steno-M batch with separators.
    
    Args:
        entries: List of entry dictionaries
    
    Returns:
        Bytes with entries separated by newline
    """
    compressed_entries = compress_conversation_entries(entries, output_format='steno-m')
    return b'\n'.join(compressed_entries)


def batch_decompress(data: bytes) -> List[Dict[str, Any]]:
    """
    Decompress a batch of Steno-M entries separated by newlines.
    
    Args:
        data: Bytes from batch_compress
    
    Returns:
        List of entry dictionaries
    """
    entries = []
    for line in data.split(b'\n'):
        if line.strip():
            try:
                entries.append(decompress_steno_m(line))
            except ValueError as e:
                print(f"Warning: Skipping malformed entry: {e}")
    return entries


# Module exports
__all__ = [
    'compress_steno',
    'compress_steno_m',
    'decompress_steno',
    'decompress_steno_m',
    'compress_conversation_entries',
    'batch_compress',
    'batch_decompress',
    'ABBREVIATIONS',
    'STENO_M_SCHEMA'
]

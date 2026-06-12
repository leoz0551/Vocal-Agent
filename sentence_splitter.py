import unicodedata

PUNCT_HARD = set("。！？.!?\n")
PUNCT_SOFT = set("，、；,;:")
ABBREVIATIONS = ("mr.", "mrs.", "ms.", "dr.", "prof.", "e.g.", "i.e.", "vs.", "etc.")

PAIR_OPEN = set("“‘「『（([{")
PAIR_CLOSE = set("”’」』）)]}")

class SentenceSplitter:
    """
    A robust streaming sentence splitter designed for TTS optimization.
    It supports multilingual weight calculation, edge-case punctuation handling 
    (decimals, abbreviations, quotes), and minimizes short fragment TTS calls.
    """
    def __init__(self, first_min=6, first_max=40, merge_min=24, merge_max=100):
        self.buf = ""
        self.first_sent_done = False
        
        self.first_min = first_min
        self.first_max = first_max
        self.merge_min = merge_min
        self.merge_max = merge_max

    def feed(self, delta: str):
        """Feed a new chunk of text from the LLM stream, returns any complete sentences ready for TTS."""
        self.buf += delta
        out = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            # Extract the sentence
            sentence = self.buf[:cut].strip()
            self.buf = self.buf[cut:]
            self.first_sent_done = True
            if sentence:
                out.append(sentence)
        return out

    def _is_boundary(self, i):
        """Check if character at index i is a valid sentence boundary (not a decimal point or abbreviation)."""
        ch = self.buf[i]
        if ch not in PUNCT_HARD and ch not in PUNCT_SOFT:
            return False, False
            
        is_hard = ch in PUNCT_HARD
        is_soft = ch in PUNCT_SOFT
        
        # 1. Streaming safe guard: if it's the very last character, wait for the next char 
        # to ensure it's not a multi-punctuation like "..." or followed by a closing quote.
        if i + 1 == len(self.buf):
            return False, False

        prev_char = self.buf[i-1] if i > 0 else ''
        next_char = self.buf[i+1] if i + 1 < len(self.buf) else ''

        # 2. Prevent splitting on decimals or formatted numbers (3.14, 1,000, 14:30)
        if ch in ('.', ',', ':'):
            if prev_char.isdigit() and next_char.isdigit():
                return False, False
                
        # 3. Prevent splitting on English abbreviations (U.S.A., Mr. Smith)
        if ch == '.':
            # Acronym without spaces: U.S.A.
            if next_char.isalpha():
                return False, False
            # Common title/abbreviation before space: Mr. Smith
            prefix = self.buf[:i+1].lower()
            if any(prefix.endswith(abbr) for abbr in ABBREVIATIONS):
                return False, False
                
        return is_hard, is_soft

    def _find_cut(self):
        weight = 0
        pair_depth = 0
        
        for i, ch in enumerate(self.buf):
            # Track quote/bracket closures
            if ch in PAIR_OPEN:
                pair_depth += 1
            elif ch in PAIR_CLOSE:
                pair_depth = max(0, pair_depth - 1)
                
            w = unicodedata.east_asian_width(ch)
            weight += 2 if w in ('W', 'F') else 1
            
            is_hard, is_soft = self._is_boundary(i)
            
            # Inherit boundary status if this is a closing quote right after a punctuation
            if ch in PAIR_CLOSE and pair_depth == 0 and i > 0:
                prev_hard, prev_soft = self._is_boundary(i-1)
                if prev_hard: is_hard = True
                if prev_soft: is_soft = True
                
            is_punct = is_hard or is_soft
            
            should_cut = False
            
            # Anti-deadlock: force cut if text is extremely long without punctuation
            is_over_max = (not self.first_sent_done and weight >= self.first_max) or \
                          (self.first_sent_done and weight >= self.merge_max)
                          
            if is_over_max and is_punct:
                should_cut = True
            # Normal cut logic: only allowed when all quotes/brackets are closed
            elif pair_depth == 0:
                if not self.first_sent_done:
                    if weight >= self.first_min and is_punct:
                        should_cut = True
                else:
                    if weight >= self.merge_min and is_hard:
                        should_cut = True
            
            # Execute cut and greedily include trailing punctuations
            if should_cut:
                cut_idx = i + 1
                while cut_idx < len(self.buf):
                    next_ch = self.buf[cut_idx]
                    if next_ch in PUNCT_HARD or next_ch in PUNCT_SOFT or next_ch in PAIR_CLOSE:
                        cut_idx += 1
                    else:
                        break
                return cut_idx
                
        return None

    def flush(self):
        """Call when the LLM generation finishes to output any remaining text."""
        s, self.buf = self.buf.strip(), ""
        return [s] if s else []

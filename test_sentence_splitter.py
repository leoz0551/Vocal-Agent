import sys
import os

from sentence_splitter import SentenceSplitter

def run_tests():
    print("Running SentenceSplitter Edge Case Tests...")
    
    # Helper to simulate streaming character by character
    def feed_stream(splitter, text):
        results = []
        for ch in text:
            sentences = splitter.feed(ch)
            results.extend(sentences)
        results.extend(splitter.flush())
        return results

    # Test 1: Basic Chinese Splitting
    s = SentenceSplitter(first_min=6, merge_min=24)
    # "你好，" -> weight 6. Cuts at comma for first sentence.
    # "今天天气不错，" -> weight 14. Doesn't cut.
    # "我们出去玩吧。" -> weight 14. Total 28 > 24. Cuts at period.
    res = feed_stream(s, "你好，今天天气不错，我们出去玩吧。")
    assert res == ["你好，", "今天天气不错，我们出去玩吧。"], f"Test 1 Failed: {res}"
    print("Test 1 Passed: Basic Chinese")

    # Test 2: Basic English Splitting
    s = SentenceSplitter(first_min=6, merge_min=24)
    # "Hello," -> weight 6. Cuts at comma.
    # "how are you doing today?" -> weight 24. Cuts at ?
    res = feed_stream(s, "Hello, how are you doing today?")
    assert res == ["Hello,", "how are you doing today?"], f"Test 2 Failed: {res}"
    print("Test 2 Passed: Basic English")

    # Test 3: Decimal and Time (Anti-Cut)
    s = SentenceSplitter(first_min=6, merge_min=10)
    # Should not cut at 3.14 or 14:30.
    res = feed_stream(s, "现在的价格是3.14美元，时间是14:30。")
    assert res == ["现在的价格是3.14美元，", "时间是14:30。"], f"Test 3 Failed: {res}"
    print("Test 3 Passed: Decimal and Time")

    # Test 4: English Abbreviations (Anti-Cut)
    s = SentenceSplitter(first_min=6, merge_min=10)
    # "Dr. Wang" should not be cut at period even if weight is high.
    res = feed_stream(s, "Hello, this is a very long sentence about Dr. Wang from USA.")
    assert res == ["Hello,", "this is a very long sentence about Dr. Wang from USA."], f"Test 4 Failed: {res}"
    print("Test 4 Passed: English Abbreviations")

    # Test 5: Quotes and Brackets (Depth counter)
    s = SentenceSplitter(first_min=6, merge_min=10)
    # The comma inside the quote shouldn't trigger a cut even for the first sentence if it's inside quotes.
    # Wait, the first sentence minimum is 6. 
    # "他说：“" -> 8. But it's in a quote! So it shouldn't cut at all inside the quote.
    # It must wait until the quote closes.
    res = feed_stream(s, "他说：“你好，世界。” 然后就走了。")
    assert res == ["他说：“你好，世界。”", "然后就走了。"], f"Test 5 Failed: {res}"
    print("Test 5 Passed: Quotes and Brackets")

    # Test 6: Greedy Trailing Punctuation
    s = SentenceSplitter(first_min=6, merge_min=10)
    # "真的吗？！" -> cuts after the ! and ” not just ?
    # "Wait... what?"
    res = feed_stream(s, "他说：“真的吗？！”然后笑了。")
    assert res == ["他说：“真的吗？！”", "然后笑了。"], f"Test 6 Failed: {res}"
    print("Test 6 Passed: Greedy Trailing Punctuation")

    # Test 7: Streaming Edge Case
    # If the last character is a punctuation, feed should not emit it until flush or next char.
    s = SentenceSplitter(first_min=6, merge_min=10)
    out1 = s.feed("你好，")
    assert len(out1) == 0, f"Streaming Test Failed: Emitted too early: {out1}"
    out2 = s.feed("世")
    assert out2 == ["你好，"], f"Streaming Test Failed: Failed to emit on next char: {out2}"
    print("Test 7 Passed: Streaming Edge Case")

    print("\n✅ All SentenceSplitter tests passed successfully!")

if __name__ == "__main__":
    # Ensure stdout handles unicode
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    run_tests()

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import rag

def run_eval():
    print("=" * 60)
    print("  NORTHGPT RAG EVALUATION SUITE")
    print("=" * 60)
    
    with open("eval/questions.json", "r", encoding="utf-8") as f:
        tests = json.load(f)
    
    passed = 0
    total = len(tests)
    
    for i, t in enumerate(tests, 1):
        q = t["q"]
        expected = t["a"]
        print(f"\n[{i}/{total}] Query: {q}")
        print(f"Expected / Constraint: {expected}")
        
        try:
            res = rag.answer(q)
            ans = res.get("text", "")
            pins = res.get("pins", [])
            guides = res.get("guides", [])
            sources = res.get("sources", [])
            
            # Print cleanly
            safe_ans = ans.encode("ascii", errors="replace").decode("ascii")
            print(f"Response snippet: {safe_ans[:200]}...")
            print(f"Sources: {[s['label'] for s in sources]}")
            print(f"Matched Pins: {[p['name'] for p in pins]}")
            print(f"Matched Guides: {[g['name'] for g in guides]}")
            
            # Validation check
            norm_ans = ans.lower().replace(",", "")
            norm_exp = expected.lower().replace(",", "")
            if "TRAP" in expected:
                trap_refused = any(w in ans.lower() for w in ["cannot", "only", "gilgit-baltistan", "impossible", "not possible", "earth", "moon", "refuse", "specialize", "pakistan", "haven't", "hasn't"])
                if trap_refused:
                    print("--> PASS (Correctly handled trap / out-of-bounds)")
                    passed += 1
                else:
                    print("--> WARN (Check trap response)")
            else:
                if norm_exp in norm_ans or any(norm_exp in s.get("label", "").lower().replace(",", "") for s in sources):
                    print(f"--> PASS (Found '{expected}')")
                    passed += 1
                else:
                    print(f"--> WARN (Expected '{expected}' not explicitly highlighted)")
        except Exception as e:
            print(f"--> ERROR: {e}")
            
    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed successfully.")
    print("=" * 60)

if __name__ == "__main__":
    run_eval()

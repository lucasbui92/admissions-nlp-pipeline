import re, nltk, math, torch

from transformers import AutoTokenizer, AutoModelForCausalLM

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

MODEL_NAME = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def _clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def _sentence_losses(text):
    """
    Return list of (sentence, loss, n_tokens) for each sentence.
    """
    sents = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    out = []

    for s in sents:
        enc = tokenizer(s, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            loss = model(input_ids=input_ids, labels=input_ids).loss.item()

        n_tokens = int(input_ids.shape[1])
        out.append((s, loss, n_tokens))

    return out

def grammar_quality_score_lm_only(text, k=0.35, max_chars=8000):
    """
    Returns a 0-100 grammar/fluency score using an LM loss → bounded score mapping.

    Score = 100 * exp(-k * avg_loss)

    - avg_loss is token-weighted across sentences (more stable).
    - k controls harshness; 0.30-0.45 is a good range for distilgpt2.
    """
    text = _clean_text(text)
    if not text:
        return {"final_score": 0.0, "avg_loss": None, "sentences_scored": 0, "worst_sentences": []}

    # avoid huge inputs
    text = text[:max_chars]

    sent_info = _sentence_losses(text)
    if not sent_info:
        return {"final_score": 0.0, "avg_loss": None, "sentences_scored": 0, "worst_sentences": []}

    # token-weighted average loss
    total_tokens = sum(n for _, _, n in sent_info)
    avg_loss = sum(loss * n for _, loss, n in sent_info) / max(total_tokens, 1)

    score = 100.0 * math.exp(-k * avg_loss)
    score = max(0.0, min(100.0, score))

    # show worst (highest loss) sentences for debugging
    worst = sorted(sent_info, key=lambda x: x[1], reverse=True)[:3]
    worst_sentences = [{"sentence": s, "loss": round(loss, 4)} for s, loss, _ in worst]

    return {
        "final_score": round(score, 2),
        "avg_loss": round(avg_loss, 4),
        "sentences_scored": len(sent_info),
        "worst_sentences": worst_sentences,
    }


if __name__ == "__main__":
    text = """He go to school yesterday. I enjoys studying physics 
    because it help me understand the universe. This sentence are wrong.
    """

    result = grammar_quality_score_lm_only(text)
    print(result)

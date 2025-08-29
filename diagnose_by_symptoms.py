def diagnose_by_symptoms(symptoms, crop=None, kb=None, top_n=3):
    if not symptoms or not kb:
        return []
    scores = []
    for entry in kb:
        match_count = len(set(symptoms).intersection(set(entry["symptoms"])))
        if match_count > 0:
            scores.append({
                "entry": entry,
                "score": match_count / len(entry["symptoms"])
            })
    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_n]

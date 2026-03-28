def calculate_severity(urgency: str, confidence: float, num_symptoms: int, has_red_flag: bool = False) -> int:
    """
    Combines Base Urgency Mapping (RED/YELLOW/GREEN) with 
    Model Confidence and Symptom Breadth into a final 0-100 score.
    """
    if has_red_flag:
        return 100
        
    base_scores = {"RED": 80, "YELLOW": 50, "GREEN": 20, "NONE": 0}
    score = base_scores.get(urgency, 0)
    
    # + up to 15 points based on model confidence
    score += int(confidence * 15)
    
    # + up to 5 points based on how many symptoms are presenting (max 5)
    score += min(5, num_symptoms)
    
    return min(100, max(0, score))

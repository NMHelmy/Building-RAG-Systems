def precision_recall_f1(true_answers, predicted_answers):
    true_positives = 0
    for true_ans in true_answers:
        if any(true_ans.lower() in pred.lower() for pred in predicted_answers):
            true_positives += 1

    precision = true_positives / len(predicted_answers) if predicted_answers else 0
    recall = true_positives / len(true_answers) if true_answers else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

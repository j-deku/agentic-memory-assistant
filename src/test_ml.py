from predict_model import predict_task_completion
from datetime import datetime

sample_task = {
    "category": "work",
    "created_at": datetime.now(),
    "due_date": "2026-06-10",
    "completion_hour": 2,
    "completion_day": "Monday"
}

prob = predict_task_completion(sample_task)

print("Completion probability:", prob)
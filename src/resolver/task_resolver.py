from rapidfuzz import fuzz


class TaskResolver:

    def resolve(self,
                user_text,
                tasks):

        best_score = 0
        best_task = None

        user_text = user_text.lower()

        for task in tasks:

            score = fuzz.partial_ratio(
                user_text,
                task["title"].lower()
            )

            if score > best_score:

                best_score = score

                best_task = task

        return best_task, best_score
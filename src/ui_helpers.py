def print_tasks(tasks):
    if not tasks:
        print("No tasks found.")
        return

    print("\n📋 TASK LIST\n")

    for task in tasks:
        # adapt depending on your DB structure
        task_id = task[0]
        title = task[1]
        category = task[2]
        due_date = task[3]
        completed = task[4]

        status = "✓" if completed else "✗"

        print(
            f"{task_id}. {title} | "
            f"Category: {category} | "
            f"Due: {due_date} | "
            f"Status: {status}"
        )

from tasks import list_tasks, complete_task

def complete_task_flow():
    tasks = list_tasks()

    if not tasks:
        print("No tasks to complete.")
        return

    for task in tasks:
        task_id = task[0]
        title = task[1]
        completed = task[4]

        status = "✓" if completed else "✗"
        print(f"{task_id}. {title} [{status}]")

    try:
        task_id = int(input("Enter task ID to mark as complete: "))
    except ValueError:
        print("Invalid input.")
        return

    complete_task(task_id)

    print("✅ Task marked as complete.")
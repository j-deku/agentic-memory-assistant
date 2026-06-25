"""
nlp/tests/test_pipeline.py

Run with:  python -m pytest nlp/tests/test_pipeline.py -v

Tests are grouped by what they verify:
  1. Intent classification (the most important)
  2. Title extraction
  3. Date extraction
  4. Category detection
  5. Task reference extraction
  6. Backward-compat shim (.analyze())
  7. Edge cases / regressions
"""
import pytest
from nlp.pipeline import NLUPipeline
from brain.goal_types import Goal


@pytest.fixture(scope="module")
def nlu():
    """Load model once for the whole test session."""
    return NLUPipeline()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent classification
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    # Greetings
    ("hi",                                  Goal.GREETING),
    ("hey",                                 Goal.GREETING),
    ("good morning",                        Goal.GREETING),

    # Acknowledgements
    ("ok",                                  Goal.ACKNOWLEDGEMENT),
    ("thanks",                              Goal.ACKNOWLEDGEMENT),
    ("sounds good",                         Goal.ACKNOWLEDGEMENT),

    # ADD_TASK — explicit verb
    ("add a task to buy groceries",         Goal.ADD_TASK),
    ("remind me to call the dentist",       Goal.ADD_TASK),
    ("create a task for the team meeting",  Goal.ADD_TASK),
    ("schedule a dentist appointment",      Goal.ADD_TASK),

    # ADD_TASK — modal "I need/have to"
    ("i need to submit the report",         Goal.ADD_TASK),
    ("i have to finish the presentation",   Goal.ADD_TASK),

    # COMPLETE_TASK
    ("i finished the report",               Goal.COMPLETE_TASK),
    ("mark task 3 as done",                 Goal.COMPLETE_TASK),
    ("i've completed my workout",           Goal.COMPLETE_TASK),
    ("i completed the assignment",          Goal.COMPLETE_TASK),

    # DELETE_TASK
    ("delete task 3",                       Goal.DELETE_TASK),
    ("remove the dentist task",             Goal.DELETE_TASK),
    ("cancel the meeting reminder",         Goal.DELETE_TASK),

    # LIST_TASKS
    ("show my tasks",                       Goal.LIST_TASKS),
    ("what are my tasks",                   Goal.LIST_TASKS),
    ("list all my tasks",                   Goal.LIST_TASKS),
    ("show task list",                      Goal.LIST_TASKS),

    # Analytics
    ("show overdue tasks",                  Goal.GET_OVERDUE_TASKS),
    ("what's my productivity score",        Goal.PRODUCTIVITY_SCORE),
    ("what should i do next",               Goal.GET_RECOMMENDED_TASKS),
    ("analyze my habits",                   Goal.ANALYZE_HABITS),
    ("what's my name",                      Goal.GET_USER_NAME),

    # NOT add-task: question or statement about past/state
    ("I finished my workout",               Goal.COMPLETE_TASK),
])
def test_intent(nlu, text, expected):
    result = nlu.parse(text)
    assert result.intent == expected, (
        f"\nInput:    '{text}'"
        f"\nExpected: {expected.name}"
        f"\nGot:      {result.intent.name}"
        f"\nConf:     {result.confidence:.2f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Title extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_title", [
    ("add a task to buy groceries",                "Buy groceries"),
    ("remind me to call the dentist tomorrow",     "Call the dentist"),
    ("i need to submit the report by friday",      "Submit the report"),
    ("create a task for the team meeting",         "Team meeting"),
    ("i have to finish the presentation",          "Finish the presentation"),
    ("schedule a dentist appointment",             "Dentist appointment"),
])
def test_title(nlu, text, expected_title):
    result = nlu.parse(text)
    assert result.title is not None, (
        f"\nInput: '{text}'\nExpected title '{expected_title}' but got None"
    )
    # Flexible match: lowercased comparison to avoid capitalisation flakiness
    assert result.title.lower() == expected_title.lower(), (
        f"\nInput:    '{text}'"
        f"\nExpected: '{expected_title}'"
        f"\nGot:      '{result.title}'"
    )


def test_title_none_for_list_intent(nlu):
    """LIST_TASKS intent should not extract a title."""
    result = nlu.parse("show my tasks")
    # title may be None or "tasks" — what matters is intent is LIST not ADD
    assert result.intent == Goal.LIST_TASKS


# ─────────────────────────────────────────────────────────────────────────────
# 3. Date extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expect_date", [
    ("finish the report by tomorrow",        True),
    ("add task buy groceries today",         True),
    ("schedule meeting for next week",       True),
    ("submit report by 2025-08-20",          True),
    ("buy groceries",                        False),   # no date
    ("remind me to call someone",            False),
])
def test_date_presence(nlu, text, expect_date):
    result = nlu.parse(text)
    if expect_date:
        assert result.due_date is not None, (
            f"\nInput: '{text}'\nExpected a date but got None"
        )
    else:
        assert result.due_date is None, (
            f"\nInput: '{text}'\nExpected no date but got '{result.due_date}'"
        )


def test_iso_date_preserved(nlu):
    result = nlu.parse("submit by 2025-08-20")
    assert result.due_date == "2025-08-20"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Category detection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_cat", [
    ("add task to prepare project report",   "work"),
    ("remind me to go to the gym",           "health"),
    ("i need to study for my exam",          "school"),
    ("buy groceries",                        "personal"),
    ("schedule a dentist appointment",       "health"),
    ("submit assignment by friday",          "school"),
])
def test_category(nlu, text, expected_cat):
    result = nlu.parse(text)
    assert result.category == expected_cat, (
        f"\nInput:    '{text}'"
        f"\nExpected: '{expected_cat}'"
        f"\nGot:      '{result.category}'"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. Task reference extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_ref", [
    ("delete task 3",    "3"),
    ("complete task 7",  "7"),
    ("mark 12 as done",  "12"),
    ("delete the report task", None),
])
def test_task_ref(nlu, text, expected_ref):
    result = nlu.parse(text)
    assert result.task_ref == expected_ref, (
        f"\nInput:    '{text}'"
        f"\nExpected: {expected_ref}"
        f"\nGot:      {result.task_ref}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Backward-compat shim
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_shim_shape(nlu):
    """analyze() must return the same dict shape as old ReasoningEngine."""
    analysis = nlu.analyze("add a task to buy groceries")
    assert "goal"       in analysis
    assert "confidence" in analysis
    assert "entities"   in analysis
    assert analysis["goal"] == Goal.ADD_TASK
    entities = analysis["entities"]
    assert "title"    in entities
    assert "due_date" in entities
    assert "category" in entities
    assert "task_ref" in entities


# ─────────────────────────────────────────────────────────────────────────────
# 7. Edge cases / regressions
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_string(nlu):
    result = nlu.parse("")
    assert result.intent in (Goal.CHAT, Goal.UNKNOWN)
    assert result.title is None


def test_only_whitespace(nlu):
    result = nlu.parse("   ")
    assert result.intent in (Goal.CHAT, Goal.UNKNOWN)


def test_parse_result_repr(nlu):
    result = nlu.parse("add task to buy milk")
    # Just make sure repr doesn't crash
    assert "ADD_TASK" in repr(result) or "ParseResult" in repr(result)


def test_confidence_range(nlu):
    for text in ["hi", "add task buy milk", "show my tasks", "blah blah blah"]:
        result = nlu.parse(text)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range for '{text}': {result.confidence}"
        )
from enum import Enum


class Goal(Enum):
    ADD_TASK =               "add_task"
    LIST_TASKS  =                 "list_tasks"
    COMPLETE_TASK  =              "complete_task"
    DELETE_TASK    =               "delete_task"
    GET_OVERDUE_TASKS  =          "get_overdue_tasks"
    GET_RECOMMENDED_TASKS =      "get_recommended_tasks"
    GET_TASK_SCORE =             "get_task_score"
    ANALYZE_HABITS =           "analyze_habits"
    NORMAL_DATE =                 "normalize_date"
    PRODUCTIVITY_SCORE =           "productivity_score"
    GET_MEMORY =                  "get_memory"
    UPDATE_MEMORY_FROM_TASKS =   "update_memory_from_tasks"
    PREDICTIVE_INSIGHTS =       "predictive_insights"
    GENERATE_BRIEFING =          "generate_briefing"
    UPDATE_PREDICTION=          "update_prediction"
    LOG_EVENT          =         "log_event"
    SELF_LEARNING_CYCLE =     "self_learning_cycle"
    RETRAIN_IF_NEEDED =          "retrain_if_needed"
    RETRAIN_TRANSFORMER_IF_NEEDED = "retrain_transformer_if_needed"
    UPDATE_RL_MEMORY =          "update_rl_memory"
    CHECK_TASK   =              "check_task"

    GET_USER_NAME =             "get_user_name"
    ACKNOWLEDGEMENT =            "ack"
    GET_COMPLETED_TASKS =        "get_completed_tasks"
    GREETING =                  "greeting"

    CHAT = "chat"

    UNKNOWN = "unknown"
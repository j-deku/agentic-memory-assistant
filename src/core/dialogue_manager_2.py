from router.llm_router import LLMRouter
from planner.planner import Planner
from executor.tool_executor import ToolExecutor


class DialogueManager:

    def __init__(self, tool_executor: ToolExecutor, user_name: str = "", reasoning=None):
        self.user_name = user_name
        self.reasoning = reasoning
        self.router = LLMRouter()
        self.planner = Planner()
        self.executor = tool_executor

        self.state = {
            "history": [],
            "awaiting_slot": None,
            "last_plan": None
        }

    def process(self, text: str):

        # 2. plan
        route = self.reasoning.analyze(text)
        plan = self.planner.generate(route)

        # 4. execute
        result = self.executor.run(plan)

        # 5. store state
        self.state["history"].append((text, result))
        self.state["last_plan"] = plan

        # 6. response
        return self._format(result)

    def _format(self, result):
        if not result:
            return "Done."

        if isinstance(result, list):
            return str(result[-1])

        return str(result)
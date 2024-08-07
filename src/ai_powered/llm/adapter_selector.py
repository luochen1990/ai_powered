from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.llm.adapters.tools_adapter import ToolsFunctionSimulator
from ai_powered.llm.adapters.chat_adapter import ChatFunctionSimulator
from ai_powered.llm.definitions import ModelFeature


class FunctionSimulatorSelector (GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    _selected_function_simulator: GenericFunctionSimulator

    def select_function_simulator(self) -> GenericFunctionSimulator:
        if ModelFeature.tools in self.model_features:
            return ToolsFunctionSimulator(
                self.function_name, self.signature, self.docstring, self.parameters_schema, self.return_schema,
                self.client, self.model_name, self.model_features, self.model_options
            )
        else:
            return ChatFunctionSimulator(
                self.function_name, self.signature, self.docstring, self.parameters_schema, self.return_schema,
                self.client, self.model_name, self.model_features, self.model_options
            )

    def __post_init__(self):
        super().__post_init__()
        self._selected_function_simulator = self.select_function_simulator()

    def query_model(self, arguments_json: str) -> str:
        return self._selected_function_simulator.query_model(arguments_json)

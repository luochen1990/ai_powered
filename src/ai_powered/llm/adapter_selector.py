from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.llm.adapters.tools_adapter import ToolsFunctionSimulator
from ai_powered.llm.adapters.chat_adapter import ChatFunctionSimulator
from ai_powered.llm.adapters.structured_output_adapter import StructuredOutputFunctionSimulator
from ai_powered.llm.definitions import ModelFeature


class FunctionSimulatorSelector (GenericFunctionSimulator):
    ''' implementation of FunctionSimulator for OpenAI compatible models '''

    _selected_impl: GenericFunctionSimulator

    def _select_impl(self) -> GenericFunctionSimulator:
        if ModelFeature.structured_outputs in self.model_features:
            return StructuredOutputFunctionSimulator(
                self.function_name, self.signature, self.docstring, self.parameters_schema, self.return_schema,
                self.client, self.model_name, self.model_features, self.model_options
            )
        elif ModelFeature.tools in self.model_features:
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
        self._selected_impl = self._select_impl()

    def query_model(self, arguments_json: str) -> str:
        return self._selected_impl.query_model(arguments_json)

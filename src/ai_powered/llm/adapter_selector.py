from typing import ClassVar

from easy_sync import sync_compatible
from ai_powered.llm.adapters.generic_adapter import GenericFunctionSimulator
from ai_powered.llm.adapters.tools_adapter import ToolsFunctionSimulator
from ai_powered.llm.adapters.chat_adapter import ChatFunctionSimulator
from ai_powered.llm.adapters.structured_output_adapter import StructuredOutputFunctionSimulator
from ai_powered.llm.definitions import ModelFeature


class FunctionSimulatorSelector(GenericFunctionSimulator):
    ''' 按模型能力选择适配器: structured_outputs > tools > chat (能力降级阶梯) '''

    _selected_impl: GenericFunctionSimulator
    _debug_print_prompt: ClassVar[bool] = False  # selector 自身的 prompt 是选择前的猜测, 实际生效的由 _selected_impl 打印, 跳过以免 DEBUG 双打印误导

    def _select_impl(self) -> GenericFunctionSimulator:
        def make(cls: type[GenericFunctionSimulator]) -> GenericFunctionSimulator:
            return cls(self.spec, self.connection, self.model_name, self.model_features, self.model_options)

        if ModelFeature.structured_outputs in self.model_features:
            return make(StructuredOutputFunctionSimulator)
        elif ModelFeature.tools in self.model_features:
            return make(ToolsFunctionSimulator)
        else:
            return make(ChatFunctionSimulator)

    def __post_init__(self):
        super().__post_init__()
        self._selected_impl = self._select_impl()

    @sync_compatible
    async def query_model(self, arguments_json: str) -> str:
        return await self._selected_impl.query_model(arguments_json)

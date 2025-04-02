import ast
from typing import List


class CompilationError(Exception):
	source_code: List[str] | None = None

	def __init__(self, node: ast.AST, message: str = "compilation failed"):
		self.node = node
		self.message = message
		super().__init__(self.message)

	def __str__(self):
		return f"{self.message} - {self.node}"

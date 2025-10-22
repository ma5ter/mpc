import ast
import sys
from typing import cast

from compiler.error import *
from compiler.entity import NamedConstant


class SimpleConstantResolver(ast.NodeVisitor):
	class Error(CompilationError):
		def __init__(self, node: ast.AST, message: str):
			super().__init__(node, "can't resolve value as constant due to " + message)

	def __init__(self, node: ast.AST, hint: dict[str, NamedConstant] | None = None):
		super().__init__()
		self.hint = hint
		self.result: int = self.visit(node)

	def generic_visit(self, node: ast.AST) -> None:
		raise SimpleConstantResolver.Error(node, f"unexpected AST node {node.__class__.__name__} visited")

	def visit_Constant(self, node: ast.Constant) -> int:
		if not isinstance(node.value, int):
			raise SimpleConstantResolver.Error(node, f"{node.value.__class__.__name__} is not an int")
		return node.value

	def visit_Name(self, node: ast.Name) -> int:
		if self.hint is None or node.id not in self.hint:
			raise SimpleConstantResolver.Error(node, f"no hints provided for {node.id} resolution")
		const = self.hint[node.id].const
		if not isinstance(const, int):
			raise SimpleConstantResolver.Error(node, f"named const {node.id} is not an int")
		return const

	def visit_BinOp(self, node: ast.BinOp) -> int:
		left = self.visit(cast(ast.AST, node.left))
		right = self.visit(cast(ast.AST, node.right))
		if isinstance(node.op, ast.Add):
			return left + right
		if isinstance(node.op, ast.Sub):
			return left - right
		if isinstance(node.op, ast.Mult):
			return left * right
		if isinstance(node.op, ast.FloorDiv):
			return left // right
		if isinstance(node.op, ast.Pow):
			return left ** right
		if isinstance(node.op, ast.BitAnd):
			return left & right
		if isinstance(node.op, ast.BitOr):
			return left | right
		if isinstance(node.op, ast.BitXor):
			return left ^ right
		if isinstance(node.op, ast.LShift):
			return left << right
		if isinstance(node.op, ast.RShift):
			return left << right
		if isinstance(node.op, ast.Div):
			result = left / right
			int_result = int(result)
			if result != int_result:
				# TODO: implement common warning and error reporting
				print(f"warning: floating point division cast to int {result} -> {int_result}, precision loss", file=sys.stderr)
			return int_result
		raise NotImplementedError

	def visit_Assign(self, node):
		pass

	@staticmethod
	def resolve(node: ast.AST, hint: dict[str: NamedConstant] | None = None) -> int | None:
		try:
			return SimpleConstantResolver(node, hint).result
		except SimpleConstantResolver.Error as e:
			return None

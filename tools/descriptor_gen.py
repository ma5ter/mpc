import ast
import sys
from typing import cast, Dict, List, Tuple

SURROGATES = ['sleep']


class Visitor(ast.NodeVisitor):
	def __init__(self):
		super().__init__()
		self.functions: Dict[str, Tuple[List[str], int]] = {}
		self.enums: Dict[str, Dict[str, int]] = {}

	def visit_Module(self, node: ast.Module) -> None:
		for item in node.body:
			self.visit(cast(ast.AST, item))

	def visit_ClassDef(self, node: ast.ClassDef) -> None:
		if len(node.bases) == 1 and isinstance(node.bases[0], ast.Name):
			if cast(ast.Name, node.bases[0]).id != 'Enum':
				return
			enum_content: Dict[str, int] = {}
			enum_name: str = node.name
			for item in node.body:
				if not isinstance(item, ast.Assign):
					raise SyntaxError("Enum body is not an assignment")
				assign: ast.Assign = cast(ast.Assign, item)
				if len(assign.targets) != 1:
					raise SyntaxError("Enum body is not a single assignment")
				if not isinstance(assign.targets[0], ast.Name):
					raise SyntaxError("Enum body is not a named assignment")
				item_name: str = cast(ast.Name, assign.targets[0]).id
				if isinstance(assign.value, ast.Name):
					if assign.value.id not in enum_content:
						raise SyntaxError("Enum body assignment named value is not a previous enum value")
					value: int = enum_content[cast(ast.Name, assign.value).id]
				elif isinstance(assign.value, ast.Constant):
					constant = cast(ast.Constant, assign.value)
					if not isinstance(constant.value, int):
						raise SyntaxError("Enum body assignment value is not an integer")
					value: int = constant.value
				else:
					raise SyntaxError("Enum body assignment is not a name or constant")
				enum_content[item_name] = value
			self.enums[enum_name] = enum_content

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		function_returns: int | None = None
		function_args: List[str] = []
		function_name: str = node.name
		if function_name in SURROGATES:
			return
		for arg in node.args.args:
			function_args.append(f"{arg.arg}")
		if node.args.vararg:
			function_args.append("*")
		for item in node.body:
			if isinstance(item, ast.Return):
				ret = cast(ast.Return, item)
				if ret.value is None:
					returns = 0
				elif isinstance(ret.value, ast.Tuple):
					returns = len(cast(ast.Tuple, ret.value).elts)
				else:
					returns = 1
				if function_returns is not None and function_returns != returns:
					raise SyntaxError("Different return size")
				function_returns = returns
		self.functions[function_name] = (function_args, function_returns if function_returns is not None else 0)


def main():
	if len(sys.argv) < 2:
		print(f"Usage: {sys.argv[0]} <_builtins.py>\n\n\tGenerates device descriptor by the supplied _builtins.py")
		return 1
	with open(sys.argv[1], 'r') as f:
		v = Visitor()
		try:
			v.visit(ast.parse(f.read()))
		except SyntaxError as e:
			print(e)
			return 1
		if v.enums:
			print("WELL_KNOWN_ENUMS:")
			for enum_name, enum_content in v.enums.items():
				print(f"  {enum_name}:")
				for name, value in enum_content.items():
					print(f"    {name}: {value}")
			print()
		if v.functions:
			print("BUILTIN_FUNCTIONS:")
			for function_name, (function_args, function_returns) in v.functions.items():
				print(f"  {function_name}:")
				print(f"    args: {function_args}")
				print(f"    rets: {function_returns}")
		return 0


if __name__ == '__main__':
	main()

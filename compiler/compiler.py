from typing import Tuple

from compiler.console import *
from compiler.opcode import *
from compiler.constant import *

MIN_VM_VERSION = 1
MAX_ADDRESS = 0xFFFF
MAX_CONSTANT = 0xFFFFFFFF
MAX_RETURNS = 0xF

WELL_KNOWN_ENUMS = {}
BUILTIN_FUNCTIONS = []


class ConstantDigger(ast.NodeVisitor):
	def __init__(self, node: ast.AST, hint: dict[str, NamedConstant] | None = None) -> None:
		super().__init__()
		self.hint = hint
		self.const_values: List[UnnamedConstant] = []
		self.visit(node)

	@staticmethod
	def is_simple(value: Any) -> bool:
		return isinstance(value, int) and 0 <= value <= 2 ** (7 + 5) - 1

	def append(self, value: int, node: ast.AST | None) -> None:
		if value > MAX_CONSTANT:
			raise ValueError('constant value too large', node)
		if not self.is_simple(value):
			for const_value in self.const_values:
				if const_value.value == value:
					return
			self.const_values.append(UnnamedConstant(value))

	def visit_Constant(self, node: ast.Constant) -> None:
		value = node.value
		if not isinstance(value, int):
			raise ValueError('only int values are supported', node)
		self.append(value, node)

	def visit_BinOp(self, node: ast.BinOp) -> None:
		value = SimpleConstantResolver.resolve(node, self.hint)
		if value is not None:
			self.append(value, node)

	def values(self) -> List[UnnamedConstant]:
		return self.const_values

	def get_index(self, value) -> int:
		for i in range(len(self.const_values)):
			if self.const_values[i].value == value:
				return i
		raise ValueError('value not found')

	def get(self, value: int) -> UnnamedConstant | int:
		if ConstantDigger.is_simple(value):
			return value
		return self.const_values[self.get_index(value)]

	def optimize(self) -> None:
		result = []
		for value in self.const_values:
			if value.refcount < 1:
				pp(f"WARNING: unused constant value {value.value} optimized out")
				continue
			result.append(value)
		self.const_values = result
		# sort
		self.const_values.sort(key=lambda x: x.refcount, reverse=True)
		# enumerate constants
		index = 0
		for c in self.const_values:
			c.index = index
			index += 1


class Compiler(ast.NodeVisitor):
	def __init__(self, source: str | None = None, substitutions: Dict[str, int] | None = None):
		self.source: List[str] | None = source.splitlines() if source is not None else None
		self.substitutions: Dict[str:int] | None = substitutions or {}
		self.code: List[OP] | None = None
		self.cd: ConstantDigger | None = None
		self.constants: Dict[str, NamedConstant] = {}
		self.functions: Dict[str, Function] = {}
		for index in range(len(BUILTIN_FUNCTIONS)):
			name, args, returns = BUILTIN_FUNCTIONS[index]
			self.functions[name] = Function(name, args, returns, index)
		# temps
		self.function: Function | None = None
		self.main_variables_count: int | None = None
		self.expr_stack_depth: List[int | None] = [None]

	def _stack_check(self, add: int) -> None:
		if isinstance(self.expr_stack_depth[-1], int):
			self.expr_stack_depth[-1] += add
			if self.expr_stack_depth[-1] < 0:
				raise ValueError('stack underflow')

	def _name(self, node: Any) -> str:
		if not isinstance(node, ast.Name):
			raise self._unsupported(node, "not a name")
		return cast(ast.Name, node).id

	def _get_bin_op(self, node: ast.BinOp | ast.AugAssign, source: str | None) -> OP:
		if isinstance(node.op, ast.Add):
			return ADD(source)
		if isinstance(node.op, ast.Sub):
			return SUB(source)
		if isinstance(node.op, ast.Mult):
			return MUL(source)
		if isinstance(node.op, ast.FloorDiv):
			return DIV(source)
		if isinstance(node.op, ast.Pow):
			return PWR(source)
		if isinstance(node.op, ast.BitAnd):
			return AND(source)
		if isinstance(node.op, ast.BitOr):
			return IOR(source)
		if isinstance(node.op, ast.BitXor):
			return XOR(source)
		raise self._unsupported(node, 'binary operator')

	def _get_test_code(self, node: ast.If | ast.While) -> Tuple[List[OP], Anchor, BranchOP]:
		anchor = Anchor()
		test = node.test
		source = self._source_line(test)
		# comparison needs complex branch other than simple zero check
		if isinstance(test, ast.Compare):
			if len(test.ops) != 1:
				raise self._unsupported(test, "multiple compare operators")
			# op selects proper branch instruction when the result is false
			_ = test.ops[0]
			# select branch instruction to the 'else' statement
			if isinstance(_, ast.Eq):
				# equality (==) operator
				bra = BNE(anchor, source)
			elif isinstance(_, ast.NotEq):
				# not equal (!=) operator
				bra = BEQ(anchor, source)
			elif isinstance(_, ast.Gt):
				# greater than (>) operator
				bra = BLE(anchor, source)
			elif isinstance(_, ast.Lt):
				# less than (<) operator
				bra = BGE(anchor, source)
			elif isinstance(_, ast.GtE):
				# greater than or equal (>=) operator
				bra = BLT(anchor, source)
			elif isinstance(_, ast.LtE):
				# less than or equal (<=) operator
				bra = BGT(anchor, source)
			else:
				raise NotImplemented('Unknown operator')
		elif isinstance(test, ast.BoolOp):
			raise self._unsupported(test, f"boolean operators as a condition test, rewrite code with sequential conditions instead")
		else:
			# pop value and branch if zero
			bra = BZE(anchor, source)
		# add as much as possible dummy instructions for a branch load, they will be optimized out later
		return self.visit(test) + [PSH(MAX_ADDRESS, source, bra), bra] + self._get_code(node.body), anchor, bra

	def _get_function(self, node: ast.FunctionDef) -> Tuple[str, List[str], List[Any]]:
		if node.args.kwarg is not None:
			raise self._unsupported(node, "kwargs are not supported")
		args: List[str] = []
		for arg in node.args.args:
			if arg.annotation is not None:
				if not isinstance(arg.annotation, ast.Name):
					raise self._unsupported(arg.annotation, "annotation is not a name")
				annotation = cast(ast.Name, arg.annotation).id
				if annotation != 'int' and annotation not in WELL_KNOWN_ENUMS:
					raise self._syntax(arg.annotation, "argument annotation is not integer or well-known Enum")
			args.append(arg.arg)
		# initialize defaults
		for i, default_value in enumerate(node.args.defaults):
			raise self._unsupported(default_value, "defaults are not supported")
		# ensure that block is iterable
		block = node.body
		if not isinstance(block, (List, Tuple)):
			block = [block]
		return node.name, args, block

	def _get_code(self, block):
		result = []
		for stmt in block:
			result += self.visit(stmt)
			if isinstance(stmt, ast.Return):
				break
		return result

	def _stack_cleanup(self) -> List[OP]:
		result = []
		odds = self.expr_stack_depth.pop()
		while odds > 0:
			o = min(odds, 3)
			odds -= o
			result += [POP(o, highlight('<stack cleanup>', ConsoleColors.Blue))]
		return result

	def visit(self, node: Any):
		return super().visit(node)

	def visit_Module(self, module: ast.Module):
		globs: List[ast.Assign] = []
		funcs: List[ast.FunctionDef] = []
		for _ in module.body:
			if isinstance(_, ast.Assign):
				globs.append(cast(ast.Assign, _))
			elif isinstance(_, ast.FunctionDef):
				funcs.append(cast(ast.FunctionDef, _))
			elif isinstance(_, ast.ImportFrom):
				if cast(ast.ImportFrom, _).module != '_builtins':
					raise self._unsupported(_, 'imports other than "_builtins" are not supported')
			elif isinstance(_, ast.Expr):
				value = cast(ast.Expr, _).value
				if not isinstance(value, ast.Call) or cast(ast.Name, cast(ast.Call, value).func).id != 'main':
					raise self._unsupported(_,
						'expressions other than "main()" are not supported in the module body')
			else:
				raise self._unsupported(_, 'only global assignments and functions are supported')

		# extract all const values
		try:
			self.cd = ConstantDigger(cast(ast.AST, module))
		except ValueError as e:
			node = e.args[1] if len(e.args) == 2 else None
			raise self._unsupported(node, e.args[0])

		# process globals
		for node in globs:
			if len(node.targets) != 1:
				raise self._unsupported(node, "multi-target assignments")
			if not isinstance(node.targets[0], ast.Name):
				raise self._unsupported(node, "unnamed assignments")
			name = self._name(node.targets[0])
			node_value = node.value
			if name in self.constants:
				raise self._unsupported(node, "global assignment redefinition")
			if name in self.substitutions:
				if not isinstance(node_value, ast.Constant):
					raise self._unsupported(node, "non-constant substitution")
				value = self.substitutions[name]
				pp(f"NOTE: constant {name} changed from {node_value.value} to {value} by command-line parameter")
			else:
				value = SimpleConstantResolver(node_value, self.constants).result
				if value is None:
					raise self._unsupported(node.value, "not an integer constant")
			self.cd.append(value, node_value)
			self.constants[name] = NamedConstant(name, self.cd.get(value))

		# preprocess functions definitions
		for f in funcs:
			name, args, block = self._get_function(f)
			if name in self.functions:
				raise self._unsupported(f, "function redefinition")
			self.functions[name] = Function(name, args, block)

		# set returns to 0 when no returns at all
		for f in self.functions.values():
			if f.returns is None:
				f.returns = 0

		# process functions
		for f in self.functions.values():
			if f.block is not None:
				self.function = f
				f.code = self._get_code(f.block)
				if len(f.code) == 0 or (not isinstance(f.code[-1], RET) and not isinstance(f.code[-1], JMP)):
					f.code.append(RET(highlight('<implicit return>', ConsoleColors.Blue)))
				# sort variables
				f.variables = dict(sorted(f.variables.items(), key=lambda x: x[1].refcount, reverse=True))
				# enumerate variables
				index = 0
				for v in f.variables.values():
					v.index = index
					index += v.tuple
				self.function = None

		# remove unused constants
		self.cd.optimize()

		# check main
		if 'main' not in self.functions:
			raise self._syntax(None, "main function absent")
		if self.functions['main'].refcount != 0:
			raise self._syntax(None, "main function recursive call")
		self.functions['main'].refcount = 0xFFFFFFFF

		# remove unused functions
		functions = {x.name: x for x in self.functions.values() if x.refcount > 0}
		# sort functions by refcount making main goes first
		self.functions = dict(sorted(functions.items(), key=lambda x: x[1].refcount, reverse=True))
		# enumerate & optimize functions
		index = -1
		for f in self.functions.values():
			if f.code is not None:
				f.code = optimize(f.code)
				cast(OP, f.code[0]).entry_point = f
			if index >= 0:
				f.index = index
			index += 1

		# finalize operators
		code = []
		for f in self.functions.values():
			if f.code is not None:
				for op in f.code:
					if isinstance(op, IntegralOP):
						code += op.expand()
					else:
						code.append(op)

		# optimize branches
		self.code = optimize_branches(code)

		# resolve functions addresses
		address = 0
		for op in code:
			if op.entry_point is not None:
				op.entry_point.address = address
			address += 1

		# remove main as function, no calls to main are made, check address also
		self.function = self.functions.pop('main')
		if self.function.address != 0:
			raise RuntimeError('main address is not 0')
		self.main_variables_count = len(self.function.variables)
		return

	# noinspection PyPep8Naming
	@staticmethod
	def visit_NoneType() -> List[OP]:
		return []

	def visit_Expr(self, node):
		# NOTE: It is a type of node in the AST tree used to encapsulate
		# expressions that are executed for their side effects but do not
		# return a value
		self.expr_stack_depth.append(0)
		result = self.visit(node.value)
		result += self._stack_cleanup()
		return result

	def visit_Call(self, node: ast.Call) -> List[OP]:
		name = self._name(node.func)
		call_args_len = sum(self.function.variables[cast(ast.Name, cast(ast.Starred, x).value).id].tuple
		if isinstance(x, ast.Starred) and isinstance(cast(ast.Starred, x).value, ast.Name)
		else 1 for x in node.args)
		if name in self.functions:
			function = self.functions[name]
			function.incref()
			# generate some code
			result: List[OP] = []
			for arg in node.args:
				if isinstance(arg, ast.Starred):
					arg = cast(ast.Starred, arg).value
				value = self.visit(arg)
				result += value
			function_args_len = len(function.params)
			if function.variadic:
				if call_args_len < function_args_len:
					raise self._syntax(node, f"{name}() requires at least {function_args_len} argument(s)")
				if function.sys_lib_index is None:
					# TODO: implement calling of non system library with variadic arguments
					raise self._unsupported(node, "calling non system library with variadic arguments")
				result += [
					PSH(call_args_len - function_args_len, highlight('<variadic args count>', ConsoleColors.Blue))]
			else:
				if call_args_len != function_args_len:
					raise self._syntax(node, f"{name}() requires exactly {function_args_len} argument(s), {call_args_len} provided")
			# return clears args
			self._stack_check(-call_args_len)
			# add function return
			self._stack_check(function.returns)
			return result + [CAL(function, self._source_line(node))]
		else:
			# primitives
			if name == 'sleep':
				if call_args_len != 1:
					raise self._syntax(node, 'sleep() primitive requires exactly one argument')
				value = self.visit(node.args[0])
				primitive = value + [SLP(self._source_line(node))]
			else:
				raise self._syntax(node.func, 'undefined function')
			self._stack_check(-call_args_len)
			return primitive

	def visit_Return(self, node):
		self._stack_check(-self.function.returns)
		return self.visit(node.value) + [RET(self._source_line(node))]

	def visit_FunctionDef(self, node):
		raise self._unsupported(node, "function definition is supported only inside a module")

	def _constant_value(self, value: int, source: str) -> List[OP]:
		# check if constant is simple
		if ConstantDigger.is_simple(value):
			self._stack_check(1)
			return [PSH(value, source)]
		# find value's index
		const = self.cd.get(value)
		const.incref()
		self._stack_check(1)
		return [PSH(const, source), LDC(source)]

	def visit_Constant(self, node: ast.Constant) -> List[OP]:
		# this is an unnamed const visitor
		# NOTE: the only context for constant is a load
		# NOTE: bool will be converted to 1 or 0 using python rule automatically
		return self._constant_value(int(node.value), self._source_line(node))

	def visit_Name(self, node: ast.Name):
		if self.function.variables is None:
			raise AssertionError("Wrong logic implemented, first pass was not completed")
		source = self._source_line(node)
		name = self._name(node)
		if isinstance(node.ctx, ast.Load):
			if name in self.function.variables:
				variable = self.function.variables[name]
				variable.incref()
				self._stack_check(variable.tuple)
				return [LDV(variable, i, source) for i in range(variable.tuple)]
			elif name in self.constants:
				self._stack_check(1)
				constant = self.constants[name]
				if isinstance(constant.const, int):
					return [PSH(constant.const, source)]
				constant.incref()
				return [PSH(constant, source), LDC(source)]
			else:
				raise self._syntax(node, f"undefined name {name}")
		elif isinstance(node.ctx, ast.Store):
			if name in self.function.variables:
				variable = self.function.variables[name]
				variable.incref()
				# NOTE: return variable for the store context not code
				return variable
			raise AssertionError("Wrong logic implemented, first pass was not completed")
		raise self._unsupported(node, "unsupported context")

	def visit_Tuple(self, node: ast.Tuple) -> List[OP] | List[str]:
		result = []
		if not isinstance(node.ctx, ast.Load):
			raise AssertionError(f"Unsupported context {node.ctx}")
		for item in node.elts:
			result += self.visit(item)
		return result

	def visit_Assign(self, node: ast.Assign) -> List[OP]:
		if len(node.targets) != 1:
			raise self._unsupported(node, "multi-target assignments")
		self.expr_stack_depth.append(0)
		target = node.targets[0]
		targets = []
		if isinstance(target, ast.Name):
			var = self.visit(target)
			if isinstance(node.value, ast.Tuple):
				size = len(node.value.elts)
			elif isinstance(node.value, ast.Call):
				size = self.functions[cast(ast.Name, node.value.func).id].returns
			else:
				size = 1
			if var.tuple != size:
				if var.tuple == 0:
					var.tuple = size
				else:
					raise self._syntax(node, f"variable `{var.name}` was defined as a tuple-container with exactly {var.tuple} elements earlie, but here elements number assumed to be {size}")
			targets.append(var)
		elif isinstance(target, ast.Tuple):
			for item in target.elts[::-1]:
				if not isinstance(item, ast.Name):
					raise self._syntax(node, "only variables supported for tuple assignment")
				var = self.visit(item)
				if var.tuple != 1:
					if var.tuple == 0:
						var.tuple = 1
					else:
						raise self._syntax(node, f"variable `{var.name}` was defined as a tuple-container earlier, but here used as a single variable-container")
				targets.append(var)
		else:
			raise self._syntax(node, "assignment target is not a variable or tuple")
		result = self.visit(node.value)
		for target in targets:
			if not isinstance(target, Variable):
				raise self._syntax(node, "assignment target is not a variable")
			result += [STV(target, i, self._source_line(node)) for i in range(target.tuple - 1, -1, -1)]
			self._stack_check(-target.tuple)
		result += self._stack_cleanup()
		return result

	def visit_While(self, node: ast.While) -> List[OP]:
		if node.orelse:
			raise self._unsupported(node, "while else")
		if isinstance(node.test, ast.Constant):
			if not cast(ast.Constant, node.test).value:
				raise self._syntax(node, "condition is always False")
			# no branch is generated for the infinite loop
			anchor = None
			result = self._get_code(node.body)
		else:
			result, anchor, bra = self._get_test_code(node)
		loop_anchor = Anchor()
		loop_anchor.place_to(NOP())
		jmp = JMP(loop_anchor, self._source_line(node))
		if anchor is not None:
			anchor.place_to(jmp)
		return [loop_anchor.op] + result + [jmp]

	def visit_If(self, node: ast.If) -> List[OP]:
		if isinstance(node.test, ast.Constant):
			raise self._syntax(node, f"condition is always {cast(ast.Constant, node.test).value}")
		source = self._source_line(node)
		result, anchor, bra = self._get_test_code(node)
		if node.orelse:
			if source is not None:
				blank = ''
				for s in source:
					if s != ' ' and s != '\t':
						break
					blank += s
				source = blank + highlight('else')
			bra = JMP(Anchor(), source)
			anchor.place_to(bra)
			anchor = bra.target
			result += [PSH(MAX_ADDRESS, source, bra), bra] + self._get_code(node.orelse)
		target = result[-1]
		if target.anchor is not None:
			# anchor is disposed and the previous one is reused
			bra.target = target.anchor
		else:
			anchor.place_to(target)
		return result

	def visit_Compare(self, node: ast.Compare) -> Any:
		if len(node.comparators) != 1:
			raise self._unsupported(node, "multiple comparators")
		left = self.visit(node.left)
		right = self.visit(node.comparators[0])
		self._stack_check(-1)
		return right + left

	def visit_AugAssign(self, node: ast.AugAssign) -> List[OP]:
		target = node.target
		if not isinstance(target, ast.Name):
			raise self._syntax(target, "target is not a Name")
		# same as assign with binary operation but need a load context for the target variable
		load = ast.Name(target.id, ast.Load(),
			col_offset=node.col_offset, end_col_offset=node.end_col_offset,
			end_lineno=node.end_lineno, lineno=node.lineno)
		bin_op = ast.BinOp(load, node.op, node.value,
			col_offset=node.col_offset, end_col_offset=node.end_col_offset,
			end_lineno=node.end_lineno, lineno=node.lineno)
		assign = ast.Assign([target], bin_op, None,
			col_offset=node.col_offset, end_col_offset=node.end_col_offset,
			end_lineno=node.end_lineno, lineno=node.lineno)
		return self.visit_Assign(assign)

	def visit_BinOp(self, node: ast.BinOp) -> List[OP]:
		source = self._source_line(node)
		value = SimpleConstantResolver.resolve(node, self.constants)
		if value is not None:
			return self._constant_value(value, source)
		left = self.visit(node.left)
		right = self.visit(node.right)
		op = self._get_bin_op(node, source)
		self._stack_check(-1)
		return right + left + [op]

	def visit_Attribute(self, node: ast.Attribute) -> List[OP]:
		if isinstance(node.value, ast.Name):
			enum = cast(ast.Name, node.value).id
			if enum in WELL_KNOWN_ENUMS:
				name = node.attr
				if name not in WELL_KNOWN_ENUMS[enum]:
					raise self._syntax(node, f'{name} is not a member of a well-known enums {enum}')
				self._stack_check(1)
				return [PSH(WELL_KNOWN_ENUMS[enum][name])]
		raise self._unsupported(node, 'only well-known enums supported')

	def visit_AnnAssign(self, node: ast.AnnAssign):
		raise self._unsupported(node.annotation, "don't use annotated assign")

	def generic_visit(self, node):
		raise self._unsupported(node, f"AST node type {type(node).__name__}")

	def _source(self, node: ast.AST):
		line = node.__getattribute__('lineno')
		begin = node.__getattribute__('col_offset')
		end = node.__getattribute__('end_col_offset')
		if self.source is not None and len(self.source) >= line:
			string = self.source[line - 1]
			return f"{line}:{begin} {string[:begin]} »»» {string[begin:end]} ««« {string[end:]}\n"
		else:
			return f"{line}:{begin} "

	def _source_line(self, node):
		line = node.__getattribute__('lineno')
		begin = node.__getattribute__('col_offset')
		end = node.__getattribute__('end_col_offset')
		source = self.source[line - 1]
		return source[:begin] + highlight(source[begin:end]) + source[end:]

	class CompilationError(Exception):
		def __init__(self, message: str):
			super().__init__(message)

	def _unsupported(self, node, message: str) -> None:
		raise Compiler.CompilationError(f"{self._source(node)}Unsupported: {message}.")

	def _syntax(self, node, message: str) -> None:
		raise Compiler.CompilationError(f"{self._source(node)}Syntax: {message}.")

	def _fatal(self, node, message: str) -> None:
		raise Compiler.CompilationError(f"{self._source(node)}Fatal: {message}.")

	def dump(self):
		functions_size = len(self.functions)
		constants_size = len(self.cd.values())
		pp(f"FUNCTIONS: {functions_size}, CONSTANTS: {constants_size}")
		pp(f"FUNCTIONS DESCRIPTORS ({functions_size})")
		for function in self.functions.values():
			pp(f"\tADDRESS: {function.address}; ARGUMENTS: {len(function.params)}; VARIABLES: {len(function.variables) - len(function.params)}; RETURNS: {function.returns}; {function}; {function.refcount} usage(s)")
		pp(f"CONSTANTS ({constants_size})")
		for const in self.cd.values():
			pp(f"\tVALUE: {const.value}; {const}; {const.refcount} usage(s)")
		pp('CODE:')
		address = 0
		for op in self.code:
			if op.entry_point is not None:
				pp(f" {op.entry_point}")
			pp(f"{address:05d}  {op.get_instruction():02X} {op.metadata if op.metadata is not None else '  '} {op}")
			if op.anchor is not None:
				pp(f" {op.anchor}")
			address += 1

	@staticmethod
	def _int_to_bin(value: int, limit: int) -> List[int]:
		if not isinstance(value, int):
			raise RuntimeError('value is not an integer')
		if value < 0 or value > limit:
			raise Compiler.CompilationError('value is out of range')
		result = cast(List[int], [value & 0xFF])
		if limit > 0xFF:
			result.append((value >> 8) & 0xFF)
		if limit > 0xFFFF:
			result.append((value >> 16) & 0xFF)
		if limit > 0xFFFFFF:
			result.append((value >> 24) & 0xFF)
		return result

	def bin(self) -> bytes:
		functions_size = len(self.functions)
		if functions_size > 0xFF:
			raise Compiler.CompilationError('number of functions > 255')
		constants_size = len(self.cd.values())
		if constants_size > 0xFF:
			raise Compiler.CompilationError('number of constants > 255')
		content: List[int] = []
		for function in self.functions.values():
			flags = 0
			# ADDRESS FIELD, depending on the MAX_ADDRESS
			function_address = function.address
			if function_address is None:
				function_address = function.sys_lib_index
				if function_address is None:
					raise RuntimeError(f"function {function.name} has no address")
				# set sys_lib flag
				flags |= 0x80
			# ARGUMENTS FIELD, VARIABLES FIELD, RETURNS FIELD
			if function.variadic:
				flags |= 0x40
			params_size = len(function.params)
			if params_size >= 0xFF:
				raise Compiler.CompilationError('number of parameters > 255')
			variables_size = len(function.variables) - params_size
			if variables_size > 0xFF:
				raise Compiler.CompilationError('number of variables > 255')
			if function.returns > MAX_RETURNS:
				raise Compiler.CompilationError(f'number of returns > {MAX_RETURNS}')
			content += (self._int_to_bin(function_address, MAX_ADDRESS)
						+ [params_size, variables_size, function.returns | flags])
		for const in self.cd.values():
			content += self._int_to_bin(const.value, MAX_CONSTANT)
		content += [op.get_instruction() for op in self.code]
		size = len(content)
		content = ([MIN_VM_VERSION] + self._int_to_bin(size, MAX_ADDRESS) + [functions_size, constants_size, self.main_variables_count] + content)
		return bytes(content)

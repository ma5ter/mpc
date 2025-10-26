"""
Module containing the implementation of various operation codes (opcodes) for the compiler.
This module defines the instruction set for the virtual machine, including:
- Basic operations (push, pop, load, store)
- Arithmetic operations (add, subtract, multiply, divide)
- Logical operations (and, or, xor)
- Control flow operations (jump, branch, call, return)
- Comparison operations
- Special operations (negate, increment, decrement)
"""

from typing import Any

from compiler.entity import *


class OP(compiler.abstract.OP):
	"""
	Base class for all operation codes in the compiler.
	Provides common functionality for all instructions.
	"""

	def __init__(self, opcode: int, mask: int, source: str | None):
		"""
		Initialize an operation with the given opcode, mask, and source code reference.

		Args:
			opcode: The base opcode value.
			mask: The bitmask for this operation.
			source: The source code reference for this operation.
		"""
		self.opcode = opcode
		self.mask = mask
		self.source = source
		self.metadata = None
		self.anchor: Anchor | None = None
		self.entry_point = None

	def move_anchor_to(self, target: compiler.abstract.OP) -> None:
		"""
		Move any anchor associated with this operation to the target operation.

		Args:
			target: The target operation to move the anchor to.

		Raises:
			Anchor.CollisionError: If the target operation already has an anchor.
		"""
		if self.anchor is not None:
			self.anchor.place_to(target)
			self.anchor = None

	def get_instruction(self) -> int:
		"""
		Get the complete instruction value for this operation.

		Returns:
			The complete instruction value.
		"""
		return self.opcode

	def append_source(self, string: str) -> str:
		"""
		Append the source code reference to the given string.

		Args:
			string: The string to append the source to.

		Returns:
			The combined string with source reference.
		"""
		if self.source is None:
			return string
		return string + (' ' * (30 - len(string))) + ' ; ' + self.source

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name and source reference.
		"""
		return self.append_source(type(self).__name__)

	@staticmethod
	def bits(value: int) -> int:
		"""
		Calculate the number of bits required to represent the given value.

		Args:
			value: The integer value to calculate bits for.

		Returns:
			The number of bits required to represent the value.
		"""
		count = 0
		while value > 0:
			value >>= 1
			count += 1
		return count


class NOP(OP):
	"""
	No-operation instruction.
	Used as an anchor placeholder and should be optimized out during compilation.
	"""

	def __init__(self):
		"""
		Initialize a NOP instruction.
		"""
		super().__init__(0, 0, None)

	def get_instruction(self) -> None:
		"""
		Get the instruction value for this NOP.

		Raises:
			RuntimeError: Always raised as NOP should be optimized out.
		"""
		raise RuntimeError("NOP is not a real instruction it's just an anchor placeholder and should be optimized out")


def optimize(code: list[OP]) -> list[OP]:
	"""
	Optimize the given list of operations by removing redundant instructions.

	Args:
		code: List of operations to optimize.

	Returns:
		Optimized list of operations.
	"""
	result: list[OP] = []
	skip = 0
	for i in range(len(code)):
		if skip > 0:
			skip -= 1
			continue
		curr = code[i]
		nxt = code[i + 1] if i < len(code) - 1 else None
		nxt2 = code[i + 2] if i < len(code) - 2 else None
		# remove NOPs
		if isinstance(nxt, NOP):
			try:
				nxt.move_anchor_to(curr)
			except Anchor.CollisionError:
				keep = cast(Anchor, curr.anchor)
				remove = cast(Anchor, nxt.anchor)
				for c in code:
					if isinstance(c, BranchOP):
						bra = cast(BranchOP, c)
						if bra.target == remove:
							bra.target = keep
			skip = 1
		# check for increment
		elif isinstance(nxt2, ADD):
			# TODO: properly check anchors in-between
			if isinstance(curr, PSH) and cast(PSH, curr).value == 1 and isinstance(nxt, LDV):
				var = nxt
			elif isinstance(nxt, PSH) and cast(PSH, nxt).value == 1 and isinstance(curr, LDV):
				var = curr
			else:
				var = None
			if var is not None:
				result.append(var)
				curr = INC(nxt2.source)
				nxt2.move_anchor_to(curr)
				skip = 2
		# check for decrement
		# TODO: properly check anchors in-between
		elif isinstance(curr, PSH) and cast(PSH, curr).value == 1 and isinstance(nxt, LDV) and isinstance(nxt2, SUB):
			result.append(nxt)
			curr = DEC(nxt2.source)
			nxt2.move_anchor_to(curr)
			skip = 2
		# remove consecutive variable STORE-LOAD
		# TODO: need more branching and calls analysis to find another store without load
		elif isinstance(curr, STV) and isinstance(nxt, LDV) and curr.anchor is None and nxt.anchor is None:
			stv = cast(STV, curr)
			ldv = cast(LDV, nxt)
			if stv.value == ldv.value:
				# optimized
				curr.metadata = 'oo'
				# skip = 1
				# continue
				pass
		result.append(curr)
	return result


def optimize_branches(code: list[OP]) -> list[OP]:
	"""
	Optimize branch instructions by calculating proper offsets.

	Args:
		code: List of operations to optimize.

	Returns:
		Optimized list of operations with proper branch offsets.
	"""
	code_size = 0
	while code_size != len(code):
		code_size = len(code)
		address = 1
		while address < len(code):
			op = code[address]
			if isinstance(op, BranchOP):
				if op.target is None:
					raise AssertionError('branch or jump target is none')
				target_address = code.index(op.target.op)
				# NOTE: targets point to the instruction after they attached
				offset = target_address + 1 - code.index(op)
				# NOTE: min positive offset is 2 (no sense to jump next op)
				if offset >= 0:
					if offset < 2:
						raise ValueError("offset too small")
					offset -= 2
				# remove previous load complements
				while (address > 0 and isinstance(code[address - 1], LoadComplementary)
					   and cast(LoadComplementary, cast(object, code[address - 1])).complement == op):
					address -= 1
					code.pop(address)
				# JMP has own storage and may work without complements
				if isinstance(op, JMP):
					if offset >= 0:
						if offset < op.mask:
							op.offset = offset
							address += 1
							continue
						else:
							op.offset = op.mask
							offset -= op.mask
					else:
						# replace opcode with a jump back instruction and positive offset saving one byte
						jmb = JMB(op.target, op.source)
						op.move_anchor_to(jmb)
						code[address] = op = jmb
				# add complements
				psh: List[OP] = []
				if offset < 0:
					# extend offset when negative
					prev = 0
					curr = 1
					while prev != curr:
						psh = PSH(offset - curr + 1, op.source, op).expand()
						if isinstance(op, JMB):
							psh.pop()  # remove the last NEG instruction
						prev = curr
						curr = len(psh)
				else:
					psh = PSH(offset, op.source, op).expand()
				for p in psh:
					code.insert(address, p)
					address += 1
			address += 1
	# remove the very first nop in main
	if len(code) > 1 and isinstance(code[0], NOP):
		nop = code.pop(0)
		op = code[0]
		op.entry_point = nop.entry_point
		assert nop.anchor is not None
		print(f"note: {nop.anchor} is at address 0")
	return code


class IntegralOP(OP):
	"""
	Base class for operations that work with integral values.
	Provides common functionality for operations that manipulate integer values.
	"""

	def __init__(self, opcode: int, mask: int, value: Any, offset: int, source: str | None):
		"""
		Initialize an integral operation with the given parameters.

		Args:
			opcode: The base opcode value.
			mask: The bitmask for this operation.
			value: The integral value to operate on.
			offset: The offset to apply to the value.
			source: The source code reference for this operation.

		Raises:
			ValueError: If offset is negative or value is out of range.
		"""
		if offset < 0:
			raise ValueError("offset must be positive")
		IntegralOP.assert_integral(value)
		if isinstance(value, int) and not isinstance(self, PSH) and value < 0:
			raise ValueError("value out of range")
		super().__init__(opcode, mask, source)
		self.value: Any = value
		self.offset = offset

	def get_value(self) -> Any:
		"""
		Get the effective value for this operation (value + offset).

		Returns:
			The effective value.
		"""
		return IntegralOP.assert_integral(self.value) + self.offset

	def get_instruction(self) -> int:
		"""
		Get the complete instruction value for this operation.

		Returns:
			The complete instruction value.

		Raises:
			ValueError: If value is not integral or out of range.
		"""
		value = self.get_value()
		if not isinstance(value, int):
			raise ValueError("value is not integral")
		if value < 0 or value > self.mask:
			raise ValueError("value out of range")
		return self.opcode | (value & self.mask)

	def expand(self) -> list[OP]:
		"""
		Expand this operation into multiple operations if needed.

		Returns:
			List of operations that represent this operation.
		"""
		value = self.get_value()
		if value >= 0:
			if value < self.mask:
				return [self]
			value -= self.mask
		elif not isinstance(self, JMP):
			raise ValueError("negative value")
		# NOTE: only derivatives may construct OPs, ignore parameters' error
		op = self.__class__(value=self.mask, source=self.source)
		if self.anchor is not None:
			self.anchor.place_to(op)
		psh = PSH(value, self.source, op)
		psh.entry_point = self.entry_point
		return psh.expand() + [op]

	@staticmethod
	def assert_integral(value) -> int:
		"""
		Assert that the given value is integral and return it as an integer.

		Args:
			value: The value to check.

		Returns:
			The value as an integer.

		Raises:
			TypeError: If the value is not integral.
		"""
		if isinstance(value, int):
			return value
		if isinstance(value, IStackable):
			return value.get_index()
		raise TypeError("argument must be integral")

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name, value, and source reference.
		"""
		return self.append_source(f"{type(self).__name__} {self.value}{'' if self.offset == 0 else f'+{self.offset}'}")


class BranchOP(OP):
	"""
	Base class for branch operations.
	Provides common functionality for operations that alter control flow.
	"""

	def __init__(self, opcode: int, mask: int, target: Anchor, source: str | None):
		"""
		Initialize a branch operation with the given parameters.

		Args:
			opcode: The base opcode value.
			mask: The bitmask for this operation.
			target: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(opcode, mask, source)
		self.target: Anchor = target

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name, target, and source reference.
		"""
		return self.append_source(f"{type(self).__name__} {self.target}")


class LoadComplementary:
	"""
	Mixin class for operations that can be used as complements for load operations.
	"""

	def __init__(self, complement: OP):
		"""
		Initialize a load complementary operation.

		Args:
			complement: The operation that this is a complement for.
		"""
		self.complement: OP = complement


class PSH(IntegralOP, LoadComplementary):
	"""
	Push operation - pushes a value onto the stack.
	"""

	def __init__(self, value: Any, source: str | None = None, complement: OP = None):
		"""
		Initialize a push operation with the given value.

		Args:
			value: The value to push onto the stack.
			source: The source code reference for this operation.
			complement: The operation that this is a complement for.
		"""
		LoadComplementary.__init__(self, complement)
		IntegralOP.__init__(self, 0, 0x7F, value, 0, source)

	def expand(self) -> list[OP]:
		"""
		Expand this push operation into multiple operations if needed.

		Returns:
			List of operations that represent this push operation.
		"""
		value = self.get_value()
		if value >= 0:
			if value < 0x80:
				return [self]
			sign = 1
		else:
			sign = -1
			value *= sign
		# generate instructions when overflowed or negative
		count = (OP.bits(value) - 7 + 4) // 5
		# for small negative values count will be calculated as -1
		if count < 0:
			count = 0
		psh = PSH((value >> (count * 5)) & 0x7F, self.source, self.complement)
		psh.entry_point = self.entry_point
		result: List[OP] = [psh]
		while count > 0:
			count -= 1
			result.append(PSC((value >> (count * 5)) & 0x1F, self.source, self.complement))
		if sign < 0:
			result.append(NEG(self.source, self.complement))
		if self.anchor is not None:
			self.anchor.place_to(result[-1])
		return result


class PSC(OP, LoadComplementary):
	"""
	Push complement operation - used to extend the value of a push operation.
	"""

	def __init__(self, value: int, source: str | None = None, complement: OP = None):
		"""
		Initialize a push complement operation with the given value.

		Args:
			value: The complement value.
			source: The source code reference for this operation.
			complement: The operation that this is a complement for.
		"""
		LoadComplementary.__init__(self, complement)
		OP.__init__(self, 0x80, 0x1F, source)
		self.value: int = value

	def get_instruction(self) -> int:
		"""
		Get the complete instruction value for this operation.

		Returns:
			The complete instruction value.

		Raises:
			ValueError: If value is not integral or out of range.
		"""
		value = self.value
		if not isinstance(value, int):
			raise ValueError("value is not integral")
		if value < 0 or value > self.mask:
			raise ValueError("value out of range")
		return self.opcode | (value & self.mask)

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name, value, and source reference.
		"""
		return self.append_source(f"PSC {self.value}")


class POP(OP):
	"""
	Pop operation - removes values from the stack.
	"""

	def __init__(self, count: int, source: str | None = None):
		"""
		Initialize a pop operation with the given count.

		Args:
			count: The number of values to pop from the stack.
			source: The source code reference for this operation.

		Raises:
			ValueError: If count is out of range.
		"""
		super().__init__(0b10111100, 3, source)
		if count < 1 or count > self.mask + 1:
			raise ValueError("count out of range")
		self.count: int = count

	def get_instruction(self) -> int:
		"""
		Get the complete instruction value for this operation.

		Returns:
			The complete instruction value.

		Raises:
			ValueError: If count is not integral or out of range.
		"""
		count = self.count
		if not isinstance(count, int):
			raise ValueError("count is not integral")
		if count < 0 or count > self.mask:
			raise ValueError("count out of range")
		return self.opcode | ((count - 1) & self.mask)

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name, count, and source reference.
		"""
		return self.append_source(f"POP {self.count}")


# LOAD and STORE OPERATIONS

class LDC(OP):
	"""
	Load constant operation - loads a constant value onto the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a load constant operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110110, 0, source)


class LDV(IntegralOP):
	"""
	Load variable operation - loads a variable value onto the stack.
	"""

	def __init__(self, value: Any, offset: int = 0, source: str | None = None):
		"""
		Initialize a load variable operation with the given value and offset.

		Args:
			value: The variable to load.
			offset: The offset to apply to the variable index.
			source: The source code reference for this operation.
		"""
		super().__init__(0b11100000, 0xF, value, offset, source)


class STV(IntegralOP):
	"""
	Store variable operation - stores a value from the stack into a variable.
	"""

	def __init__(self, value: Any, offset: int = 0, source: str | None = None):
		"""
		Initialize a store variable operation with the given value and offset.

		Args:
			value: The variable to store into.
			offset: The offset to apply to the variable index.
			source: The source code reference for this operation.
		"""
		super().__init__(0b11110000, 0xF, value, offset, source)


# CONTROL FLOW OPERATIONS

class CAL(IntegralOP):
	"""
	Call operation - calls a function.
	"""

	def __init__(self, value: Any, source: str | None = None):
		"""
		Initialize a call operation with the given function.

		Args:
			value: The function to call.
			source: The source code reference for this operation.
		"""
		super().__init__(0b11010000, 0xF, value, 0, source)


class RET(OP):
	"""
	Return operation - returns from a function.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a return operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110101, 0, source)


class SKZ(OP):
	"""
	Skip if zero operation - skips the next instruction if the top of stack is zero.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a skip if zero operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110000, 0, source)


class SNZ(OP):
	"""
	Skip if not zero operation - skips the next instruction if the top of stack is not zero.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a skip if not zero operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110001, 0, source)


class SKN(OP):
	"""
	Skip if negative operation - skips the next instruction if the top of stack is negative.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a skip if negative operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110010, 0, source)


class SNN(OP):
	"""
	Skip if not negative operation - skips the next instruction if the top of stack is not negative.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a skip if not negative operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b1011011, 0, source)


class SLP(OP):
	"""
	Sleep operation - pauses execution for a specified number of cycles.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a sleep operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110100, 0, source)


class JMP(BranchOP):
	"""
	Jump operation - unconditionally jumps to a target location.
	"""

	def __init__(self, target: Anchor, source: str | None = None):
		"""
		Initialize a jump operation with the given target.

		Args:
			target: The anchor to jump to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b11000000, 0xF, target, source)
		self.offset = self.mask

	def get_instruction(self) -> int:
		"""
		Get the complete instruction value for this operation.

		Returns:
			The complete instruction value.

		Raises:
			RuntimeError: If offset is not set or is too large.
		"""
		if self.offset is None or self.offset <= 0:
			raise RuntimeError("offset not set by optimizer")
		if self.offset > self.mask:
			raise RuntimeError("offset is too large")
		return self.opcode | self.offset

	def __str__(self):
		"""
		Return a string representation of the operation.

		Returns:
			String representation including the operation name, target, offset, and source reference.
		"""
		return self.append_source(f"{type(self).__name__} {self.target} <{self.offset}>")


class JMB(BranchOP):
	"""
	Jump back operation - jumps to a target location with a negative offset.
	"""

	def __init__(self, target: Anchor, source: str | None = None):
		"""
		Initialize a jump back operation with the given target.

		Args:
			target: The anchor to jump to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10110111, 0, target, source)


# COMPARE-BRANCH OPERATIONS

class BZE(BranchOP):
	"""
	Branch if zero operation - pops a value and branches if it is zero.
	"""

	# pop and branch if zero
	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if zero operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100000, 0, anchor, source)


class BNZ(BranchOP):
	"""
	Branch if not zero operation - pops a value and branches if it is not zero.
	"""

	# pop and branch if zero
	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if not zero operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100001, 0, anchor, source)


class BEQ(BranchOP):
	"""
	Branch if equal operation - compares two values and branches if they are equal.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if equal operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100010, 0, anchor, source)


class BNE(BranchOP):
	"""
	Branch if not equal operation - compares two values and branches if they are not equal.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if not equal operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100011, 0, anchor, source)


class BGT(BranchOP):
	"""
	Branch if greater than operation - compares two values and branches if the first is greater.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if greater than operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100100, 0, anchor, source)


class BLT(BranchOP):
	"""
	Branch if less than operation - compares two values and branches if the first is less.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if less than operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100101, 0, anchor, source)


class BGE(BranchOP):
	"""
	Branch if greater than or equal operation - compares two values and branches if the first is greater or equal.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if greater than or equal operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100110, 0, anchor, source)


class BLE(BranchOP):
	"""
	Branch if less than or equal operation - compares two values and branches if the first is less or equal.
	"""

	def __init__(self, anchor: Anchor, source: str | None = None):
		"""
		Initialize a branch if less than or equal operation with the given target.

		Args:
			anchor: The anchor to branch to.
			source: The source code reference for this operation.
		"""
		super().__init__(0b10100111, 0, anchor, source)


# BINARY OPERATIONS

class ADD(OP):
	"""
	Add operation - adds two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize an add operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101000, 0, source)


class SUB(OP):
	"""
	Subtract operation - subtracts two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a subtract operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101001, 0, source)


class MUL(OP):
	"""
	Multiply operation - multiplies two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a multiply operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101010, 0, source)


class DIV(OP):
	"""
	Divide operation - divides two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a divide operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101011, 0, source)


class PWR(OP):
	"""
	Power operation - raises a value to the power of another value from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a power operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101100, 0, source)


class AND(OP):
	"""
	And operation - performs a bitwise AND on two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize an and operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101101, 0, source)


class IOR(OP):
	"""
	Or operation - performs a bitwise OR on two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize an or operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101110, 0, source)


class XOR(OP):
	"""
	Xor operation - performs a bitwise XOR on two values from the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a xor operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10101111, 0, source)


# UNARY OPERATIONS

class NEG(OP, LoadComplementary):
	"""
	Negate operation - negates the top value on the stack.
	"""

	def __init__(self, source: str | None = None, complement: OP = None):
		"""
		Initialize a negate operation.

		Args:
			source: The source code reference for this operation.
			complement: The operation that this is a complement for.
		"""
		LoadComplementary.__init__(self, complement)
		OP.__init__(self, 0b10111000, 0, source)


class INV(OP):
	"""
	Invert operation - inverts the bits of the top value on the stack.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize an invert operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10111001, 0, source)


class INC(OP):
	"""
	Increment operation - increments the top value on the stack by 1.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize an increment operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10111010, 0, source)


class DEC(OP):
	"""
	Decrement operation - decrements the top value on the stack by 1.
	"""

	def __init__(self, source: str | None = None):
		"""
		Initialize a decrement operation.

		Args:
			source: The source code reference for this operation.
		"""
		super().__init__(0b10111011, 0, source)

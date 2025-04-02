# cython: language_level=3
import ast
import sys

from compiler.compiler import Compiler


def main():
	substitutions = {}
	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <source> <output_binary> [-d<const1>=<value1> [-d<const2>=<value2> [...]]]")
		sys.exit(1)
	for arg in sys.argv[3:]:
		if not arg.startswith('-d'):
			print(f"unknown argument '{arg}'")
			sys.exit(1)
		pair = arg[2:].split('=')
		if len(pair) != 2:
			print(f"wrong substitution format '{arg}'")
			sys.exit(1)
		name, value = pair
		substitutions[name] = int(value)
	source_path = sys.argv[1]
	destination_path = sys.argv[2]
	with open(source_path, 'r') as source_file:
		source_code = source_file.read()
	visitor = Compiler(source_code, substitutions)
	try:
		visitor.visit_Module(ast.parse(source_code))
		visitor.dump()
		with open(destination_path, 'wb') as file:
			file.write(visitor.bin())
	except Compiler.CompilationError as exception:
		print(exception, file=sys.stderr)


if __name__ == '__main__':
	main()

# print(ast.dump(tree, indent=2))
# exe = compile(tree, source_code, 'exec')
# dis.dis(exe, depth=999)

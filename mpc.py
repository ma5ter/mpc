# cython: language_level=3
import ast
import sys
import argparse
import yaml

from compiler.compiler import Compiler, WELL_KNOWN_ENUMS, BUILTIN_FUNCTIONS


def parse_arguments(substitutions) -> str | argparse.Namespace:
	parser = argparse.ArgumentParser(description='Micro/MCU Python Compiler (mpc)')
	parser.add_argument('--file', '-f', help='Input source file')
	parser.add_argument('--output', '-o', help='Output binary file')
	parser.add_argument('--define', '-d', action='append', help='Define a constant')
	parser.add_argument('--descriptor', '-x', default='device.yaml', help='Device descriptor')
	parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
	parser.add_argument('files', nargs='*', help='Another way to specify input and output files as positional arguments or a complementary argument if supplied only one')
	args = parser.parse_args()
	if len(args.files) > 2:
		return 'too many unnamed arguments, the first unnamed argument treated as the input file and the second unnamed argument as the output file'
	if len(args.files) > 0:
		if len(args.files) == 2:
			if args.file is not None:
				return "input file specified twice, as the first unnamed argument and as the named argument"
			args.file = args.files[0]
			if args.output is not None:
				return "output file specified twice, as the second unnamed argument and as the named argument"
			args.output = args.files[1]
		if args.file is None and args.output is not None:
			args.file = args.files[0]
		elif args.file is not None and args.output is None:
			args.output = args.files[0]
		else:
			return "both input and output files are specified as named arguments and one extra unnamed argument passed"
	if args.file is None:
		return 'no source file specified'
	if args.output is None:
		args.output = '/tmp/script.bin'
		print(f"using {args.output} as output file")
	for arg in args.define:
		pair = arg.split('=')
		if len(pair) != 2:
			return f"wrong substitution format '{arg}'"
		name, value = pair
		substitutions[name] = int(value)
	# load device dependent config from file
	try:
		if not args.descriptor.endswith('.yaml'):
			args.descriptor = args.descriptor + '.yaml'
		with open(args.descriptor, 'r') as file:
			config = yaml.safe_load(file)
		well_known_enums = config.get('WELL_KNOWN_ENUMS', {})
		builtin_functions = config.get('BUILTIN_FUNCTIONS', [])
		for name, enum in well_known_enums.items():
			for alias, value in enum.items():
				if name not in WELL_KNOWN_ENUMS.keys():
					WELL_KNOWN_ENUMS[name] = {}
				WELL_KNOWN_ENUMS[name][alias] = value
		for name, pair in builtin_functions.items():
			if len(pair.keys()) != 2:
				raise Exception
			BUILTIN_FUNCTIONS.append((name, pair['args'], pair['rets']))
	except FileNotFoundError:
		return f"descriptor file {args.descriptor} not found"
	except Exception:
		return 'invalid descriptor file format'
	return args


def main():
	substitutions = {}
	args = parse_arguments(substitutions)
	if isinstance(args, str):
		print("Error: " + args, file=sys.stderr)
		return 1
	with open(args.file, 'r') as source_file:
		source_code = source_file.read()
	visitor = Compiler(source_code, substitutions)
	try:
		visitor.visit_Module(ast.parse(source_code))
		if args.verbose:
			visitor.dump()
		with open(args.output, 'wb') as file:
			file.write(visitor.bin())
	except Compiler.CompilationError as exception:
		print(exception, file=sys.stderr)


if __name__ == '__main__':
	main()

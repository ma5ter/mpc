"""
Micro/MCU Python Compiler (mpc) - Main entry point for the compiler.
This module provides the command-line interface and main compilation workflow.
"""

# cython: language_level=3
import ast
import sys
import argparse
import yaml

from compiler.compiler import Compiler, WELL_KNOWN_ENUMS, BUILTIN_FUNCTIONS


def parse_arguments(substitutions) -> str | argparse.Namespace:
	"""
	Parse command-line arguments and validate the configuration.

	Args:
		substitutions: Dictionary to store constant substitutions from the command line.

	Returns:
		Either an error message string or the parsed arguments' namespace.

	The function handles:
	- Input and output file specification (both named and positional arguments)
	- Constant substitutions via command-line defines
	- Device descriptor loading
	- Verbose mode flag
	"""
	parser = argparse.ArgumentParser(description='Micro/MCU Python Compiler (mpc)')

	# Define command-line arguments
	parser.add_argument('--file', '-f', help='Input source file')
	parser.add_argument('--output', '-o', help='Output binary file')
	parser.add_argument('--define', '-d', action='append', help='Define a constant')
	parser.add_argument('--descriptor', '-x', default='device.yaml', help='Device descriptor')
	parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
	parser.add_argument('files', nargs='*', help='Another way to specify input and output files as positional arguments or a complementary argument if supplied only one')

	args = parser.parse_args()

	# Handle positional arguments
	if len(args.files) > 2:
		return 'too many unnamed arguments, the first unnamed argument treated as the input file and the second unnamed argument as the output file'

	# Process file arguments
	if len(args.files) > 0:
		if len(args.files) == 2:
			# Handle case where both input and output are specified as positional arguments
			if args.file is not None:
				return "input file specified twice, as the first unnamed argument and as the named argument"
			args.file = args.files[0]
			if args.output is not None:
				return "output file specified twice, as the second unnamed argument and as the named argument"
			args.output = args.files[1]
		if args.file is None and args.output is not None:
			# Handle case where only input is specified as positional argument
			args.file = args.files[0]
		elif args.file is not None and args.output is None:
			# Handle case where only output is specified as positional argument
			args.output = args.files[0]
		else:
			return "both input and output files are specified as named arguments and one extra unnamed argument passed"

	# Validate required arguments
	if args.file is None:
		return 'no source file specified'
	if args.output is None:
		# Set the default output path if not specified
		args.output = '/tmp/script.bin'
		print(f"using {args.output} as output file")

	# Process constant substitutions
	for arg in args.define:
		pair = arg.split('=')
		if len(pair) != 2:
			return f"wrong substitution format '{arg}'"
		name, value = pair
		substitutions[name] = int(value)

	# Load device descriptor configuration
	try:
		# Ensure descriptor has .yaml extension
		if not args.descriptor.endswith('.yaml'):
			args.descriptor = args.descriptor + '.yaml'

		# Load and parse YAML descriptor file
		with open(args.descriptor, 'r') as file:
			config = yaml.safe_load(file)

		# Process well-known enums from descriptor
		well_known_enums = config.get('WELL_KNOWN_ENUMS', {})
		builtin_functions = config.get('BUILTIN_FUNCTIONS', [])

		# Populate WELL_KNOWN_ENUMS dictionary
		for name, enum in well_known_enums.items():
			for alias, value in enum.items():
				if name not in WELL_KNOWN_ENUMS.keys():
					WELL_KNOWN_ENUMS[name] = {}
				WELL_KNOWN_ENUMS[name][alias] = value

		# Populate BUILTIN_FUNCTIONS list
		for name, pair in builtin_functions.items():
			if len(pair.keys()) != 2:
				raise RuntimeError
			BUILTIN_FUNCTIONS.append((name, pair['args'], pair['rets']))
	except FileNotFoundError:
		return f"descriptor file {args.descriptor} not found"
	except RuntimeError:
		return 'invalid descriptor file format'

	return args


def main():
	"""
	Main entry point for the compiler.
	Handles the complete compilation workflow including:
	- Argument parsing
	- Source code loading
	- Compilation
	- Output generation
	- Error handling
	"""
	substitutions = {}  # Dictionary to store constant substitutions

	# Parse command-line arguments
	args = parse_arguments(substitutions)

	# Handle argument parsing errors
	if isinstance(args, str):
		print("Error: " + args, file=sys.stderr)
		return 1

	# Read source code from input file
	with open(args.file, 'r') as source_file:
		source_code = source_file.read()

	# Initialize compiler with source code and substitutions
	visitor = Compiler(source_code, substitutions)

	try:
		# Parse and compile the source code
		visitor.visit_Module(ast.parse(source_code))

		# Display compilation details if verbose mode is enabled
		if args.verbose:
			visitor.dump()

		# Write compiled binary to output file
		with open(args.output, 'wb') as file:
			file.write(visitor.bin())
	except Compiler.CompilationError as exception:
		# Handle compilation errors
		print(exception, file=sys.stderr)


if __name__ == '__main__':
	main()

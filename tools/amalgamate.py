import sys
import re
import pathlib
from typing import List

pat_from = re.compile(r'^from\s+compiler\.(\w+)\s+import\s+(.+)')
pat_abstract = re.compile(r'^class\s+(\w+)\s*\(compiler\.abstract\.\w+\)\s*:')

visited: List[str] = []

def substitute(path: str, file_name: str, output):
	file_path = pathlib.Path(path, file_name)
	if str(file_path) in visited:
		return
	visited.append(str(file_path))
	try:
		with open(file_path, 'r') as file:
			for line in file:
				if line.startswith('import compiler.abstract'):
					continue
				if line.startswith('from _system_dependent import *'):
					substitute(path, '_system_dependent.py', output)
					continue
				match = re.search(pat_abstract, line)
				if match:
					line = f"class {match.group(1)}:\n"
				match = re.search(pat_from, line)
				if match:
					if 'abstract' == match.group(1):
						print(f"{file_path}: should import nothing form 'abstract.py', import the whole module instead", file=sys.stderr)
						sys.exit(1)
					substitute(path, str(pathlib.Path( 'compiler', match.group(1) + '.py')), output)
				else:
					try:
						output.write(line)
					except Exception as exception:
						print(f"output file write error: {exception}", file=sys.stderr)
						sys.exit(1)
	except Exception as exception:
		print(f"{file_path}: {exception}", file=sys.stderr)
		sys.exit(1)


def main():
	if len(sys.argv) < 3:
		print(f"Usage: {sys.argv[0]} <path_to_mpc.py> <amalgamated_name.py>")
		sys.exit(1)
	source = sys.argv[1]
	source_path = str(pathlib.Path(source).parent)
	source_name = str(pathlib.Path(source).name)
	destination = sys.argv[2]
	try:
		with open(destination, 'w') as file:
			substitute(source_path, source_name, file)
	except IOError as exception:
		print(f"{destination}: {exception}", file=sys.stderr)
		sys.exit(1)


if __name__ == '__main__':
	main()

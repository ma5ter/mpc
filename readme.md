# Micro/MCU Python Compiler (mpc)

This is a custom compiler designed for the custom 8-bit SCAMP2 virtual machine. The compiler translates Python source
code into bytecode that can be executed by the SCAMP2 virtual machine.

This compiler is very limited and intended to produce executable code for simple automation scripts that can be run in
every MCU with at least **1.5-2 kb** of ROM and few bytes of RAM.

Internally mpc uses a standard `ast` Python module to produce the AST tree from the source code. After that, it is
converted into the SCAMP2 virtual machine opcodes. And saved into a binary file with proper header.

## Supported Features

The compiler supports the following Python features:

- **Global Const Assignments**: Assignments of integer constants to global variables. Note: they are treated as pure
  consts later, not variables!
  ```python
  MAX_VALUE = 10000  # will be put into consts table is used later, otherwise optimized out 
  MIN_VALUE = 10  # will be used in-place as PSH 10
  THRESH_VALUE = 1000  # will be used in-place as PSH 31; PSC 8    
  ```

- **Compile-Time Constant Simplification**: Binary operations (even unsupported by the virtual machine, like shifts and
  floating point division) used with pure const, that can be calculated during compilation, are simplified.
  ```python
  d = ((MAX_VALUE - MIN_VALUE) / 10) << 2
  ```

- **Enums**: Support for well-known enums extracted from the system `_builtins.py` mockup by the `sysdeps_gen.py` tool.
  ```python
  class Color(Enum):
      RED = 1
      GREEN = 2
      BLUE = 3

  color = Color.RED
  ```

- **Local Variable Assignments**: Assignments to local variables or tuples of variables.
  ```python
  tz = 5 * 60 * 60
  year, month, date = get_date(tz)
  weekday = get_weekday(tz)
  weekday += 1
  ```
  ```python
  a, b, c = 1, get_value(), 3  # will consume more stack depth than sequential assignment
                               # but same instructions number
  ```

- **Basic Arithmetic Operations on Variables**: Operations such as addition, subtraction, multiplication, integer division and power
  ```python
  x = a + 5
  y = 10 - x
  z = x * (5 + a)
  z //= y
  p = 2 ** z
  ```

- **Bitwise Operations on Variables**: Operations such as AND, OR, and XOR.
  ```python
  q = a & (b | c) ^ d
  ```

- **Built-in Functions**: Support for calling built-in functions defined in `_builtins`.
  ```python
  from _builtins import *
  value = builtin_input(3)
  builtin_output(12, value)
  ```

- **User Function Definitions**: Definitions of functions with specified arguments and return values.
  ```python
  def add(a, b):
      return a + b
  ```

- **Function Calls**: Calls to defined functions or system-specific built-ins (and surrogates).
  ```python
  result = add(5, 3)  # Assuming add is a user-defined function
  print(result)  # Assuming print is a built-in function
  sleep(1000)  # Assuming sleep is a surrogate that translates into the SLP opcode
  ```

- **Function Returns**: Function may have no return statement, empty return statement, return one variable or even tuple.
  ```python
  def foo():
      print(1)
      # RET instruction will be generated automatically

  def bar():
      print(2)
      return

  def swap(a, b):
      return b, a
  ```

- **Main Function**: A special `main` function that serves as the entry point of the program.
  ```python
  def main():
      ...
  ```

- **Conditional Statements**: `if` and `while` statements with support for comparison operators.
  ```python
  if x:
      print(x)

  while y <= 20:
      y += 1
  ```

## Unsupported Features

The compiler does not support the following Python features:

- **Types Other than Int**: All types other than int as of the virtual machine implementation without a heap.
  <pre><code>x<del>: str = "Hello"</del>  # Not supported
  y<del> = None</del>  # Not supported</code></pre>

- **Multi-target Assignments**: Assignments to multiple targets in a single statement.
  <pre><code><del>x =</del> y = 10  # Not supported</code></pre>

- **Default Function Arguments**: Function definitions with default argument values.
  <pre><code>def foo(a<del>=10</del>):  # Not supported
      pass</code></pre>

- **Variadic Arguments**: Functions with variadic arguments (**except for system library functions**).
  <pre><code>def foo(<del>*</del>args):  # Not supported
      pass</code></pre>

- **Annotated Assignments**: Assignments with type annotations.
  <pre><code>x<del>: int</del> = 10  # Not supported</code></pre>

- **Imports**: Imports from modules other than `_builtins` are unsupported.
  <pre><code><del>import math</del>  # Not supported</code></pre>

- **Expressions in the Module Body**: Expressions that are not calls to the `main` function and not constant
  assignments.
  <pre><code><del>y = sub(MAX, MIN)</del>  # Not supported in the module body
  <del>print(MIN, MAX, y)</del>  # Not supported in the module body</code></pre>

- **Function Definitions Outside the Module**: Function definitions that are not at the top level of the module.
  <pre><code>def outer():
      <del>def inner():</del>  # Not supported
          pass</code></pre>

- **Complex Conditionals**: Conditionals with multiple comparison operators.
  <pre><code>if a < b <del>< c:</del>  # Not supported
      pass</code></pre>

- **And and Or in Conditionals**: Conditionals with `and` and `or` operators.
  <pre><code>if a < b <del>and c</del>:  # Not supported
      pass</code></pre>

- **Else Clauses in While Loops**: `while` loops with `else` clauses.
  <pre><code>while condition:
      pass
  <del>else:</del>  # Not supported
      pass</code></pre>

- **Unsupported Operators**: Operators that are not explicitly supported, such as matrix multiplication (`@`).
  <pre><code>result = a <del>@</del> b  # Not supported</code></pre>

- **Classes**: Any classes other than well-known enums.
  <pre><code><del>class MyClass:</del>  # Not supported</code></pre>

These limitations ensure that the compiler can efficiently translate the supported subset of Python code into bytecode
for the SCAMP2 virtual machine.

## Usage

```sh
python mpc.py <source> <output_binary> [-d<const1>=<value1> [-d<const2>=<value2> [...]]]
```

- `<source>`: The path to the source file to be compiled.
- `<output_binary>`: The path where the compiled binary will be saved.
- `-d<const>=<value>`: Optional substitutions for constants in the source code. You can specify multiple
  substitutions.

### Usage Notes

**When used in production, it is better to build a binary executable with the `tools/build.sh` tool**

## Example

```sh
python mpc.py source.py output.bin -dMAX_SIZE=1024 -dTIMEOUT=30
```

In this example, the compiler reads `source.py`, substitute `MAX_SIZE` with `1024` and `TIMEOUT` with `30`, compile
the code, and save the binary output to `output.bin`.

## Requirements

- Python 3.x or nothing when linked statically against `libpython.a`

## License

This Micro/MCU Python Compiler (mpc) is licensed under the LGPL License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Author

ma5ter

---

# Tools

## sysdeps_gen.py

Builtins Parser Utility

This utility is designed to parse a `_builtins.py` file and generate a `_system_dependent.py` file for the compiler,
containing information about the built-in functions and enums defined in the input file. The generated file includes
details about the function arguments, return values, and enum values. Function position in an outpot list corresponds
it system index to call.

### Features

- Parses Python `_builtins.py` file to extract built-in functions and enums.
- Generates a `_system_dependent.py` file with the extracted information.
- Handles enums that inherit from the `Enum` class.
- Supports functions with variable arguments and return values.
- Ignores surrogate functions like `sleep`.

### Requirements

- Python 3.x

### Usage

To use this utility, run the script from the command line with the path to the `_builtins.py` file as an argument:

```sh
python sysdeps_gen.py <path_to_builtins.py>
```

Replace `<path_to_builtins.py>` with the actual path to your `_builtins.py` file.

### Output

The script will generate and print the contents of `_system_dependent.py` to the standard output. The generated file
will contain:

- `WELL_KNOWN_ENUMS`: A dictionary of enums with their names and values.
- `BUILTIN_FUNCTIONS`: A list of tuples containing function names, arguments, and return values.

### Example

Given a `_builtins.py` file with the following content:

```python
from enum import Enum


class Color(Enum):
	RED = 1
	GREEN = 2
	BLUE = 3


def add(a, b):
	return a + b


def subtract(a, b):
	return a - b
```

Running the utility will generate the following output:

```python
WELL_KNOWN_ENUMS = {
	"Color": {
		"RED": 1,
		"GREEN": 2,
		"BLUE": 3,
	},
}

BUILTIN_FUNCTIONS = [
	('add', ['a', 'b'], 1),
	('subtract', ['a', 'b'], 1),
]
```

### Limitations

- The utility assumes that enums inherit directly from the `Enum` class.
- It does not handle nested enums or enums with complex inheritance.
- Functions with dynamic return types or complex return structures may not be accurately represented.

### License

This utility is licensed under the MIT License. See the LICENSE file for more information.

---

## build.sh

This utility script automates the process of building the binary of the compiler for a specified Python version. It
supports both dynamic and static linking against the Python library.

### Prerequisites

- Python 3.x installed on your system.
- `cython` installed (`pip install cython`).
- GCC compiler installed.

### Usage

```bash
build_mpc.sh [python_version [path_to_libpython.a]]
```

#### Parameters

1. **python_version** (optional): The version of Python to build against (e.g., `3.12`). If not specified, the script
   will use the default Python 3 version available on your system.
2. **path_to_libpython.a** (optional): The path to the static library `libpython.a` if you want to link statically. If
   not specified, the script will link dynamically.

#### Examples

- Build for the default Python 3 version:
  ```bash
  build_mpc.sh
  ```

- Build for Python 3.12:
  ```bash
  build_mpc.sh 3.12
  ```

- Build for Python 3.12 and link statically against a specific `libpython.a`:
  ```bash
  build_mpc.sh 3.12 ./libpython3.12.0.a
  ```

### Help

To display the usage information, run:

```bash
build_mpc.sh --help
```

### License

This utility is licensed under the MIT License. See the LICENSE file for more details.

---

## amalgamate.py

Amalgamation Utility for the Compiler Python Code

This utility is designed to amalgamate Python source files by substituting specific import statements and class
definitions. It processes a given mpc.py file and its dependencies, merging them into a single output file.

### Features

- Recursively processes import statements to include dependent files.
- Substitutes specific class definitions to ensure compatibility.

### Usage

**Typically this utility is used by the `build.sh` tool, but if needed, it may be used as follows**

```sh
python3 amalgamate.py <path_to_mpc.py> <amalgamated_name.py>
```

- `<path_to_mpc.py>`: The path to the `mpc.py` source file to be amalgamated.
- `<amalgamated_name.py>`: The desired name for the amalgamated output file.

### License

This utility is licensed under the MIT License. See the LICENSE file for more details.


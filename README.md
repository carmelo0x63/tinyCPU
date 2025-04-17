# tinyCPU - Tiny CPU simulator

A Python-based simulator for an ideal CPU that can parse and execute assembly-like instructions. This project provides a way to understand CPU operations and assembly language programming in a controlled environment.

## Features

- **Complete Instruction Set**: Supports a small but comprehensive set of assembly-like instructions
- **Syntax Validation**: Recognizes legitimate instructions and throws syntax errors for invalid code
- **Interactive UI**: Displays current execution state including:
  - Instruction listing with highlighted current/next instructions
  - CPU register contents
  - Stack contents
- **Execution Control**: Step-by-step or continuous execution with adjustable speed
- **Memory Model**: Simple register and stack-based memory system

## Instruction Set

The CPU simulator supports the following instructions:

| Instruction | Operands | Description |
|-------------|----------|-------------|
| `MOV` | dest, src | Move value to register |
| `ADD` | dest, src | Add value to register |
| `SUB` | dest, src | Subtract value from register |
| `MUL` | dest, src | Multiply register by value |
| `DIV` | dest, src | Divide register by value |
| `JMP` | label | Jump to label |
| `JEQ` | label | Jump if equal (when FLAG = 0) |
| `JNE` | label | Jump if not equal (when FLAG ≠ 0) |
| `JGT` | label | Jump if greater than (when FLAG > 0) |
| `JLT` | label | Jump if less than (when FLAG < 0) |
| `CMP` | a, b | Compare values (sets FLAG register) |
| `HLT` | - | Halt execution |
| `NOP` | - | No operation |
| `PUSH` | src | Push value to stack |
| `POP` | dest | Pop value from stack |
| `CALL` | label | Call subroutine |
| `RET` | - | Return from subroutine |

## CPU Architecture

- **Registers**: 8 general-purpose registers (`R0`-`R7`)
- **Special Registers**:
  - `PC`: Program Counter
  - `SP`: Stack Pointer
  - `FLAG`: Result of comparison operations (0 = equal, 1 = greater, -1 = less)
- **Memory**: Simple 256-byte memory model
- **Stack**: Dynamic stack for subroutine calls and temporary storage

## Installation

Clone the repository:

```bash
% git clone https://github.com/carmelo0x63/tinyCPU.git

% cd tinyCPU
```

## Usage

```bash
usage: tinycpu.py [-h] [-V] [-f FILE] [-D]

options:
  -h, --help       show this help message and exit
  -V, --version    show program's version number and exit
  -f, --file FILE  Assembly file to load
  -D, --debug      Enable debug mode with detailed logging
```

### Running the Simulator

```bash
% python3 tinycpu.py -f <assembly_file> [-D]
```

### Controls

Once the simulator is running, you can use the following keys:
- `s`: Step (execute one instruction)
- `r`: Run/Pause (continuous execution)
- `+/-`: Adjust execution speed
- `space`: Reset the CPU
- `q`: Quit the simulator

### Writing Assembly Code

Create a text file with your assembly instructions. Here's a simple example:

```assembly
; Simple counter program
start:  MOV R0 #1        ; Initialize counter to 1
        MOV R1 #10       ; Set upper limit to 10

loop:   ADD R0 #1        ; Increment counter
        CMP R0 R1        ; Compare counter with limit
        JLT loop         ; Jump back if counter < limit
        HLT              ; Halt the CPU
```

#### Syntax Rules

- Each instruction should be on a separate line
- Labels end with a colon (`:`)
- Comments start with a semicolon (`;`)
- Register references: `R0` through `R7`
- Immediate values (constants) are prefixed with `#` (e.g., `#42`)
- Each instruction must have the correct number of operands

## Examples

The repository includes example programs:

1. **Simple Counter**: Demonstrates basic arithmetic and loops
2. **Factorial Calculator**: Shows recursion and stack operations
3. **Self Test**: Simulates memory write/read cycle

## Requirements

- Python 3.6 or higher
- Standard Python libraries only: `sys`, `time`, `re`, `os`, `curses`, `enum`, `typing`, `argparse`, `logging`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [GNU General Public License v3.0](LICENSE).

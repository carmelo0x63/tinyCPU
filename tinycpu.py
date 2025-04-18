#!/usr/bin/env python3
# Fully functional RISC-like CPU, custom instruction set
# author: Carmelo C
# email: carmelo.califano@gmail.com
# history, date format ISO 8601:
#   2025-04-18: 1.2 Framed the various sections within boxes, 8-bit check on numbers
#   2025-04-17: 1.1 Edited J** instructions to only jump based on FLAG register, added argparse, logging
#   2025-04-15: 1.0 Fully functional CPU

import sys
import time
import re
import os
import curses
import argparse
import logging
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

# Settings
__version__ = '1.2'
__build__ = '20250418'
X_OFFSET = 2
Y_OFFSET = 0

class TokenType(Enum):
    REGISTER = auto()
    NUMBER = auto()
    LABEL = auto()
    INSTRUCTION = auto()
    COMMENT = auto()
    EOF = auto()

class Token:
    def __init__(self, token_type: TokenType, value: str, line: int):
        self.type = token_type
        self.value = value
        self.line = line

class Instruction:
    def __init__(self, opcode: str, operands: List[Token], line: int, label: Optional[str] = None):
        self.opcode = opcode
        self.operands = operands
        self.line = line
        self.label = label

class SyntaxError(Exception):
    def __init__(self, message: str, line: int):
        self.message = message
        self.line = line
        super().__init__(f"Syntax error at line {line}: {message}")

class CPU:
    # Define instruction set
    INSTRUCTIONS = {
        "MOV": 2,  # Move value to register
        "ADD": 2,  # Add value to register
        "SUB": 2,  # Subtract value from register
        "MUL": 2,  # Multiply register by value
        "DIV": 2,  # Divide register by value
        "JMP": 1,  # Jump to label
        "JEQ": 1,  # Jump if equal (FLAG = 0)
        "JNE": 1,  # Jump if not equal (FLAG != 0)
        "JGT": 1,  # Jump if greater than (FLAG > 0)
        "JLT": 1,  # Jump if less than (FLAG < 0)
        "CMP": 2,  # Compare values
        "HLT": 0,  # Halt execution
        "NOP": 0,  # No operation
        "PUSH": 1, # Push value to stack
        "POP": 1,  # Pop value from stack
        "CALL": 1, # Call subroutine
        "RET": 0,  # Return from subroutine
    }

    def __init__(self, debug=False):
        # Initialize registers
        self.registers = {f"R{i}": 0 for i in range(8)}
        self.registers["PC"] = 0    # Program Counter
        self.registers["SP"] = 0    # Stack Pointer
        self.registers["FLAG"] = 0  # Flag register

        self.memory = [0] * 256     # Simple memory model
        self.stack = []
        self.instructions = []
        self.labels = {}
        self.running = False
        self.execution_speed = 0.5  # seconds between instructions
        self.debug = debug

        # Configure logging
        self.logger = logging.getLogger("CPU")
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def reset(self):
        """Reset the CPU to initial state"""
        for reg in self.registers:
            self.registers[reg] = 0
        self.memory = [0] * 256
        self.stack = []
        self.running = False
        self.logger.info("CPU reset to initial state")

    def load_program(self, filename: str):
        """Load and parse a program from a file"""
        try:
            with open(filename, 'r') as file:
                program = file.read()

            self.logger.info(f"Loading program from file: {filename}")
            self.parse_program(program)
            return True
        except FileNotFoundError:
            self.logger.error(f"Error: File '{filename}' not found.")
            print(f"Error: File '{filename}' not found.")
            return False
        except SyntaxError as e:
            self.logger.error(f"Syntax error: {e}")
            print(e)
            return False

    def parse_program(self, program: str):
        """Parse the program into instructions"""
        self.instructions = []
        self.labels = {}

        lines = program.splitlines()

        for line_num, line in enumerate(lines, start=1):
            # Skip empty lines
            if not line.strip() or line.strip().startswith(';'):
                continue

            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]

            # Check for labels
            label = None
            if ':' in line:
                label_part, line = line.split(':', 1)
                label = label_part.strip()
                self.labels[label] = len(self.instructions)
                self.logger.debug(f"Found label '{label}' at instruction {len(self.instructions)}")

            line = line.strip()
            if not line:
                continue

            # Parse instruction
            parts = line.split()
            opcode = parts[0].upper()

            if opcode not in self.INSTRUCTIONS:
                self.logger.error(f"Unknown instruction: {opcode} at line {line_num}")
                raise SyntaxError(f"Unknown instruction: {opcode}", line_num)

            expected_operands = self.INSTRUCTIONS[opcode]
            operands = parts[1:]

            if len(operands) != expected_operands:
                self.logger.error(f"Instruction {opcode} expects {expected_operands} operands, got {len(operands)} at line {line_num}")
                raise SyntaxError(
                    f"Instruction {opcode} expects {expected_operands} operands, got {len(operands)}",
                    line_num
                )

            # Parse operands into tokens
            tokenized_operands = []
            for operand in operands:
                if operand.startswith('R') and operand[1:].isdigit() and 0 <= int(operand[1:]) < 8:
                    tokenized_operands.append(Token(TokenType.REGISTER, operand, line_num))
                elif operand.startswith('#'):
                    try:
                        value = int(operand[1:])
                        if not (0 <= value <= 255):
                            raise SyntaxError(f"Number must be between 0 and 255: {operand}", line_num)
                        tokenized_operands.append(Token(TokenType.NUMBER, str(value), line_num))
                    except ValueError:
                        self.logger.error(f"Invalid number format: {operand} at line {line_num}")
                        raise SyntaxError(f"Invalid number format: {operand}", line_num)
                else:
                    tokenized_operands.append(Token(TokenType.LABEL, operand, line_num))

            self.instructions.append(Instruction(opcode, tokenized_operands, line_num, label))
            self.logger.debug(f"Parsed instruction: {opcode} with {len(tokenized_operands)} operands at line {line_num}")

    def execute(self):
        """Execute the loaded program"""
        if not self.instructions:
            self.logger.warning("No program loaded.")
            print("No program loaded.")
            return

        self.running = True
        self.registers["PC"] = 0

        # Update labels with actual instruction indices
        for label, idx in self.labels.items():
            self.labels[label] = idx

        self.logger.info("Starting program execution")

    def step(self):
        """Execute a single instruction"""
        if not self.running or self.registers["PC"] >= len(self.instructions):
            self.running = False
            self.logger.info("Program execution completed or halted")
            return False

        instruction = self.instructions[self.registers["PC"]]
        pc_before = self.registers["PC"]
        self.registers["PC"] += 1  # Increment PC by default

        try:
            self._execute_instruction(instruction)

            if self.debug:
                self._log_debug_info(instruction, pc_before)

            return True
        except Exception as e:
            self.logger.error(f"Runtime error at line {instruction.line}: {e}")
            print(f"Runtime error at line {instruction.line}: {e}")
            self.running = False
            return False

    def _log_debug_info(self, instruction, pc_before):
        """Log detailed information about CPU state after instruction execution"""
        opcode = instruction.opcode
        operands = " ".join([op.value for op in instruction.operands])

        debug_msg = [
            f"\n--- DEBUG: Executed instruction {pc_before}: {opcode} {operands} ---",
            "Registers:"
        ]

        # Add register values
        for reg, value in self.registers.items():
            debug_msg.append(f"  {reg}: {value}")

        # Add stack info if non-empty
        if self.stack:
            debug_msg.append("Stack:")
            for i, value in enumerate(self.stack):
                debug_msg.append(f"  [{i}]: {value}")
        else:
            debug_msg.append("Stack: <empty>")

        # Log the complete state
        self.logger.debug("\n".join(debug_msg))

    def _execute_instruction(self, instruction: Instruction):
        """Execute a single instruction"""
        opcode = instruction.opcode
        operands = instruction.operands

        self.logger.debug(f"Executing {opcode} with {len(operands)} operands")

        if opcode == "MOV":
            dest, src = operands
            if dest.type != TokenType.REGISTER:
                raise SyntaxError("First operand must be a register", instruction.line)

            if src.type == TokenType.REGISTER:
                self.registers[dest.value] = self.registers[src.value]
            elif src.type == TokenType.NUMBER:
                self.registers[dest.value] = int(src.value)
            else:
                raise SyntaxError("Invalid source operand", instruction.line)

        elif opcode == "ADD":
            dest, src = operands
            if dest.type != TokenType.REGISTER:
                raise SyntaxError("First operand must be a register", instruction.line)

            if src.type == TokenType.REGISTER:
                self.registers[dest.value] += self.registers[src.value]
            elif src.type == TokenType.NUMBER:
                self.registers[dest.value] += int(src.value)
            else:
                raise SyntaxError("Invalid source operand", instruction.line)

        elif opcode == "SUB":
            dest, src = operands
            if dest.type != TokenType.REGISTER:
                raise SyntaxError("First operand must be a register", instruction.line)

            if src.type == TokenType.REGISTER:
                self.registers[dest.value] -= self.registers[src.value]
            elif src.type == TokenType.NUMBER:
                self.registers[dest.value] -= int(src.value)
            else:
                raise SyntaxError("Invalid source operand", instruction.line)

        elif opcode == "MUL":
            dest, src = operands
            if dest.type != TokenType.REGISTER:
                raise SyntaxError("First operand must be a register", instruction.line)

            if src.type == TokenType.REGISTER:
                self.registers[dest.value] *= self.registers[src.value]
            elif src.type == TokenType.NUMBER:
                self.registers[dest.value] *= int(src.value)
            else:
                raise SyntaxError("Invalid source operand", instruction.line)

        elif opcode == "DIV":
            dest, src = operands
            if dest.type != TokenType.REGISTER:
                raise SyntaxError("First operand must be a register", instruction.line)

            if src.type == TokenType.REGISTER:
                divisor = self.registers[src.value]
            elif src.type == TokenType.NUMBER:
                divisor = int(src.value)
            else:
                raise SyntaxError("Invalid source operand", instruction.line)

            if divisor == 0:
                raise ZeroDivisionError("Division by zero")

            self.registers[dest.value] //= divisor

        elif opcode == "JMP":
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Jump target must be a label", instruction.line)

            if label.value not in self.labels:
                raise SyntaxError(f"Undefined label: {label.value}", instruction.line)

            self.registers["PC"] = self.labels[label.value]

        elif opcode == "CMP":
            a, b = operands
            a_val = self._get_value(a)
            b_val = self._get_value(b)

            if a_val == b_val:
                self.registers["FLAG"] = 0
            elif a_val > b_val:
                self.registers["FLAG"] = 1
            else:
                self.registers["FLAG"] = -1

        elif opcode == "JEQ":
            # Jump if FLAG == 0 (equal)
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Jump target must be a label", instruction.line)

            if self.registers["FLAG"] == 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]

        elif opcode == "JNE":
            # Jump if FLAG != 0 (not equal)
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Jump target must be a label", instruction.line)

            if self.registers["FLAG"] != 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]

        elif opcode == "JGT":
            # Jump if FLAG > 0 (greater than)
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Jump target must be a label", instruction.line)

            if self.registers["FLAG"] > 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]

        elif opcode == "JLT":
            # Jump if FLAG < 0 (less than)
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Jump target must be a label", instruction.line)

            if self.registers["FLAG"] < 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]

        elif opcode == "PUSH":
            value = self._get_value(operands[0])
            self.stack.append(value)
            self.registers["SP"] += 1

        elif opcode == "POP":
            if not self.stack:
                raise RuntimeError("Stack underflow")

            if operands[0].type != TokenType.REGISTER:
                raise SyntaxError("POP destination must be a register", instruction.line)

            self.registers[operands[0].value] = self.stack.pop()
            self.registers["SP"] -= 1

        elif opcode == "CALL":
            label = operands[0]
            if label.type != TokenType.LABEL:
                raise SyntaxError("Call target must be a label", instruction.line)

            if label.value not in self.labels:
                raise SyntaxError(f"Undefined label: {label.value}", instruction.line)

            # Push return address
            self.stack.append(self.registers["PC"])
            self.registers["SP"] += 1

            # Jump to subroutine
            self.registers["PC"] = self.labels[label.value]

        elif opcode == "RET":
            if not self.stack:
                raise RuntimeError("Stack underflow on RET")

            # Pop return address and jump to it
            self.registers["PC"] = self.stack.pop()
            self.registers["SP"] -= 1

        elif opcode == "HLT":
            self.running = False
            self.logger.info("Program execution halted by HLT instruction")

        elif opcode == "NOP":
            pass

    def _get_value(self, token: Token):
        """Get the value of a token (register or number)"""
        if token.type == TokenType.REGISTER:
            return self.registers[token.value]
        elif token.type == TokenType.NUMBER:
            return int(token.value)
        else:
            raise SyntaxError(f"Cannot get value of {token.type}", token.line)


class CPUSimulator:
    def __init__(self, debug=False):
        self.cpu = CPU(debug=debug)
        self.stdscr = None
        self.filename = None
        self.debug = debug

        # Set up logging
        logging_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='tinycpu.log' if debug else None
        )
        self.logger = logging.getLogger("CPUSimulator")

    def load_program(self, filename):
        self.filename = filename
        return self.cpu.load_program(filename)

    def run(self):
        """Run the simulator in curses UI"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)   # Current instruction
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Next instruction
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)     # Errors
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)    # Highlights

        try:
            self.cpu.execute()
            self.main_loop()
        finally:
            curses.nocbreak()
            curses.echo()
            curses.endwin()

    def main_loop(self):
        """Main loop for the simulator"""
        key = None

        while True:
            self.stdscr.clear()
            self.stdscr.border(0)

            # Display title
            self.stdscr.addstr(Y_OFFSET, X_OFFSET, f"CPU Simulator - {self.filename}", curses.A_BOLD)
            self.stdscr.addstr(Y_OFFSET + 1, X_OFFSET, f"Speed: {self.cpu.execution_speed:.2f}s/instr", curses.A_BOLD)

            # Display registers
            registers_box = self.stdscr.subwin(5, 54, Y_OFFSET + 3, 2)
            registers_box.box()
            self.draw_registers(Y_OFFSET + 3, X_OFFSET + 1)

            # Display stack
            stack_box = self.stdscr.subwin(22, 20, Y_OFFSET + 3, X_OFFSET + 58)
            stack_box.box()
            self.draw_stack(Y_OFFSET + 3, X_OFFSET + 59)

            # Display instructions
            instructions_box = self.stdscr.subwin(11, 24, Y_OFFSET + 9, 2)
            instructions_box.box()
            self.draw_instructions(Y_OFFSET + 9, X_OFFSET + 1)

            # Display controls 
            help_text = [
                "s: Step      r: Run/Pause     q: Quit",
                "+/-: Adjust speed        space: Reset"
            ]
            controls_box = self.stdscr.subwin(4, 40, Y_OFFSET + 21, 2)
            controls_box.box()
            self.stdscr.addstr(Y_OFFSET + 21, X_OFFSET + 1, "Controls", curses.A_BOLD)
            for i, text in enumerate(help_text):
                self.stdscr.addstr(Y_OFFSET + 22 + i, X_OFFSET + 1, text)

            self.stdscr.refresh()

            # Check if program is running automatically
            if self.cpu.running and key == 'r':
                time.sleep(self.cpu.execution_speed)
                self.cpu.step()
                continue

            # Get keyboard input
            key = self.stdscr.getch()
            key = chr(key) if 32 <= key <= 126 else key

            # Process keyboard input
            if key == 'q':
                self.logger.info("User quit the simulator")
                break
            elif key == 's':
                self.cpu.step()
            elif key == 'r':
                self.logger.debug("Continuous execution mode toggled")
                continue  # Continue to run automatically
            elif key == '+':
                self.cpu.execution_speed = max(0.1, self.cpu.execution_speed - 0.1)
                self.logger.debug(f"Execution speed increased to {self.cpu.execution_speed:.2f}s/instr")
            elif key == '-':
                self.cpu.execution_speed += 0.1
                self.logger.debug(f"Execution speed decreased to {self.cpu.execution_speed:.2f}s/instr")
            elif key == ' ':
                self.cpu.reset()
                self.cpu.execute()
                self.logger.info("CPU reset and program restarted")

    def draw_registers(self, y, x):
        """Draw the CPU registers"""
        self.stdscr.addstr(y, x, "Registers", curses.A_BOLD)
        y += 1

        for i, (reg, value) in enumerate(self.cpu.registers.items()):
            row = y + (i // 4)
            col = x + 15 * (i % 4)
            self.stdscr.addstr(row, col, f"{reg}: {value}")

    def draw_stack(self, y, x):
        """Draw the CPU stack"""
        self.stdscr.addstr(y, x, f"Stack (SP = {self.cpu.registers['SP']})", curses.A_BOLD)
        y += 1

        for i, value in enumerate(reversed(self.cpu.stack[-8:] if len(self.cpu.stack) > 8 else self.cpu.stack)):
            self.stdscr.addstr(y + i, x, f"[{len(self.cpu.stack) - i - 1}]: {value}")

    def draw_instructions(self, y, x):
        """Draw the instruction list"""
        self.stdscr.addstr(y, x, "Instructions", curses.A_BOLD)
        y += 1

        pc = self.cpu.registers["PC"]
        start_idx = max(0, pc - 4)
        end_idx = min(len(self.cpu.instructions), pc + 5)

        for i in range(start_idx, end_idx):
            instr = self.cpu.instructions[i]
            line_text = f"{i}: "
            if instr.label:
                line_text += f"{instr.label}: "

            line_text += instr.opcode + " "
            line_text += " ".join([op.value for op in instr.operands])

            # Highlight current and next instruction
            if i == pc - 1:  # Current (just executed)
                self.stdscr.addstr(y, x, line_text, curses.color_pair(1))
            elif i == pc:    # Next (about to execute)
                self.stdscr.addstr(y, x, line_text, curses.color_pair(2))
            else:
                self.stdscr.addstr(y, x, line_text)

            y += 1


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description = 'Tiny CPU simulator, version ' + __version__ + ', build ' + __build__ + '.')
    parser.add_argument('-V', '--version', action = 'version', version = '%(prog)s ' + __version__)
    parser.add_argument('-f', '--file', type = str, help = 'Assembly file to load')
    parser.add_argument('-D', '--debug', action = 'store_true', help = 'Enable debug mode with detailed logging')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Use command line arguments if provided
    filename = args.file
    debug_mode = args.debug

    # If no file specified via argument, check positional argument
    if not filename and len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        filename = sys.argv[1]

    # If still no file, prompt user
    if not filename:
        print("Usage: python3 tinycpu.py -f <assembly_file> [-D]")
        sys.exit(1)

    simulator = CPUSimulator(debug=debug_mode)

    if simulator.load_program(filename):
        simulator.run()


if __name__ == "__main__":
    main()

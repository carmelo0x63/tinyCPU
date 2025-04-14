#!/usr/bin/env python3
# Fully functional RISC-like CPU, custom instruction set
# author: Carmelo C
# email: carmelo.califano@gmail.com
# history, date format ISO 8601:
#   1.0 Fully funtional CPU

import sys
import time
import re
import os
import curses
from enum import Enum, auto
from typing import List, Dict, Tuple, Optional

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
        "JEQ": 2,  # Jump if equal
        "JNE": 2,  # Jump if not equal
        "JGT": 2,  # Jump if greater than
        "JLT": 2,  # Jump if less than
        "CMP": 2,  # Compare values
        "HLT": 0,  # Halt execution
        "NOP": 0,  # No operation
        "PUSH": 1, # Push value to stack
        "POP": 1,  # Pop value from stack
        "CALL": 1, # Call subroutine
        "RET": 0,  # Return from subroutine
    }

    def __init__(self):
        # Initialize registers
        self.registers = {f"R{i}": 0 for i in range(8)}
        self.registers["PC"] = 0  # Program Counter
        self.registers["SP"] = 0  # Stack Pointer
        self.registers["FLAG"] = 0  # Flag register

        self.memory = [0] * 256  # Simple memory model
        self.stack = []
        self.instructions = []
        self.labels = {}
        self.running = False
        self.execution_speed = 0.5  # seconds between instructions

    def reset(self):
        """Reset the CPU to initial state"""
        for reg in self.registers:
            self.registers[reg] = 0
        self.memory = [0] * 256
        self.stack = []
        self.running = False

    def load_program(self, filename: str):
        """Load and parse a program from a file"""
        try:
            with open(filename, 'r') as file:
                program = file.read()
            
            self.parse_program(program)
            return True
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False
        except SyntaxError as e:
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
            
            line = line.strip()
            if not line:
                continue
            
            # Parse instruction
            parts = line.split()
            opcode = parts[0].upper()
            
            if opcode not in self.INSTRUCTIONS:
                raise SyntaxError(f"Unknown instruction: {opcode}", line_num)
            
            expected_operands = self.INSTRUCTIONS[opcode]
            operands = parts[1:]
            
            if len(operands) != expected_operands:
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
                        tokenized_operands.append(Token(TokenType.NUMBER, str(value), line_num))
                    except ValueError:
                        raise SyntaxError(f"Invalid number format: {operand}", line_num)
                else:
                    tokenized_operands.append(Token(TokenType.LABEL, operand, line_num))
            
            self.instructions.append(Instruction(opcode, tokenized_operands, line_num, label))
    
    def execute(self):
        """Execute the loaded program"""
        if not self.instructions:
            print("No program loaded.")
            return
        
        self.running = True
        self.registers["PC"] = 0
        
        # Update labels with actual instruction indices
        for label, idx in self.labels.items():
            self.labels[label] = idx
    
    def step(self):
        """Execute a single instruction"""
        if not self.running or self.registers["PC"] >= len(self.instructions):
            self.running = False
            return False
        
        instruction = self.instructions[self.registers["PC"]]
        self.registers["PC"] += 1  # Increment PC by default
        
        try:
            self._execute_instruction(instruction)
            return True
        except Exception as e:
            print(f"Runtime error at line {instruction.line}: {e}")
            self.running = False
            return False
    
    def _execute_instruction(self, instruction: Instruction):
        """Execute a single instruction"""
        opcode = instruction.opcode
        operands = instruction.operands
        
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
            reg, label = operands
            if self._get_value(reg) == 0 or self.registers["FLAG"] == 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]
        
        elif opcode == "JNE":
            reg, label = operands
            if self._get_value(reg) != 0 or self.registers["FLAG"] != 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]
        
        elif opcode == "JGT":
            reg, label = operands
            if self._get_value(reg) > 0 or self.registers["FLAG"] > 0:
                if label.value not in self.labels:
                    raise SyntaxError(f"Undefined label: {label.value}", instruction.line)
                self.registers["PC"] = self.labels[label.value]
        
        elif opcode == "JLT":
            reg, label = operands
            if self._get_value(reg) < 0 or self.registers["FLAG"] < 0:
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
    def __init__(self):
        self.cpu = CPU()
        self.stdscr = None
        self.filename = None
    
    def load_program(self, filename):
        self.filename = filename
        return self.cpu.load_program(filename)
    
    def run(self):
        """Run the simulator in curses UI"""
        self.stdscr = curses.initscr()
        curses.noecho()
        curses.cbreak()
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Current instruction
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Next instruction
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)  # Errors
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Highlights
        
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
            
            # Display title
            self.stdscr.addstr(0, 0, f"CPU Simulator - {self.filename}", curses.A_BOLD)
            self.stdscr.addstr(1, 0, f"Speed: {self.cpu.execution_speed:.2f}s/instr", curses.A_BOLD)
            
            # Display registers
            self.draw_registers(3, 0)
            
            # Display stack
            self.draw_stack(3, 40)
            
            # Display instructions
            self.draw_instructions(12, 0)
            
            # Display help
            help_text = [
                "s: Step      r: Run/Pause     q: Quit",
                "+/-: Adjust speed        space: Reset"
            ]
            for i, text in enumerate(help_text):
                self.stdscr.addstr(22 + i, 0, text)
            
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
                break
            elif key == 's':
                self.cpu.step()
            elif key == 'r':
                continue  # Continue to run automatically
            elif key == '+':
                self.cpu.execution_speed = max(0.1, self.cpu.execution_speed - 0.1)
            elif key == '-':
                self.cpu.execution_speed += 0.1
            elif key == ' ':
                self.cpu.reset()
                self.cpu.execute()
    
    def draw_registers(self, y, x):
        """Draw the CPU registers"""
        self.stdscr.addstr(y, x, "Registers:", curses.A_BOLD)
        y += 1
        
        for i, (reg, value) in enumerate(self.cpu.registers.items()):
            row = y + (i // 4)
            col = x + 15 * (i % 4)
            self.stdscr.addstr(row, col, f"{reg}: {value}")
    
    def draw_stack(self, y, x):
        """Draw the CPU stack"""
        self.stdscr.addstr(y, x, f"Stack (SP={self.cpu.registers['SP']}):", curses.A_BOLD)
        y += 1
        
        for i, value in enumerate(reversed(self.cpu.stack[-8:] if len(self.cpu.stack) > 8 else self.cpu.stack)):
            self.stdscr.addstr(y + i, x, f"[{len(self.cpu.stack) - i - 1}]: {value}")
    
    def draw_instructions(self, y, x):
        """Draw the instruction list"""
        self.stdscr.addstr(y, x, "Instructions:", curses.A_BOLD)
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


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 tinycpu.py <assembly_file>")
        sys.exit(1)
    
    simulator = CPUSimulator()
    if simulator.load_program(sys.argv[1]):
        simulator.run()


if __name__ == "__main__":
    main()

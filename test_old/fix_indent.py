with open('alg3_multi_additions_CPU.py', 'r') as f:
    lines = f.readlines()

# Fix line 2223 (index 2222)
lines[2222] = '                continue\n'

with open('alg3_multi_additions_CPU.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation on line 2223") 
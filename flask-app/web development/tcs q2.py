def calculate_sum(seven_segment_display):
correct_displays = {
        '0': [[1, 1, 1], [1, 0, 1], [1, 1, 1]],
        '1': [[0, 1, 0], [1, 1, 0], [0, 1, 0]],
        '2': [[1, 1, 1], [0, 1, 1], [1, 0, 1]],
        '3': [[1, 1, 1], [0, 1, 1], [0, 1, 1]],
        '4': [[0, 0, 1], [1, 1, 1], [0, 0, 1]],
        '5': [[1, 1, 1], [1, 0, 0], [0, 1, 1]],
        '6': [[1, 1, 1], [1, 0, 0], [1, 1, 1]],
        '7': [[1, 1, 1], [0, 0, 1], [0, 0, 1]],
        '8': [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        '9': [[1, 1, 1], [1, 1, 1], [0, 1, 1]],
    }

total_sum = 0
for i in range(0, len(seven_segment_display), 3):
        digit = seven_segment_display[i:i+3]
 for correct_digit, correct_display in correct_displays.items():
 if sum(sum(a != b for a, b in zip(row1, row2)) for row1, row2 in zip(digit, correct_display)) == 1:
                total_sum += int(correct_digit)
                break
        else:
for j in range(3):
                for k in range(3):
                    toggled_digit = [row[:] for row in digit]
                    toggled_digit[j][k] = 1 - toggled_digit[j][k]
                    for correct_digit, correct_display in correct_displays.items():
                        if toggled_digit == correct_display:
                            total_sum += int(correct_digit)
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break

return total_sum
seven_segment_display = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 0],
    [1, 1, 0],
    [0, 1, 0],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
]

print(calculate_sum(seven_segment_display))

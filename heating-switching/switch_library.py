from openhab import OpenHAB

url = 'http://192.168.0.200:8080/rest'
openhab = OpenHAB(url)

items = openhab.fetch_all_items()

left = []
right = []

for index in range(3):
    left_index = 'BalElsoOszlop_state_' + str(index+1)
    right_index = 'JobbElsoOszlop_state_' + str(index+1)
    left.append(items.get(left_index))
    right.append(items.get(right_index))

def on(side, index):
    index -= 1
    if side == 'left' or side == 'l':
        sides = left
    elif side == 'right' or side == 'r':
        sides = right
    else:
        print("side error: side must be 'left' ('l') or 'right' ('r')")
        return

    if sides[index].state == 'OFF':
        sides[index].on()
        sides[index].state = 'ON'
        return side + 'is turned on'
    else:
        return side + 'has been already turned on'

def off(side, index):
    index -= 1
    if side == 'left' or side == 'l':
        sides = left
    elif side == 'right' or side == 'r':
        sides = right
    else:
        print("side error: side must be 'left' ('l') or 'right' ('r')")
        return
    if sides[index].state == 'ON':
        sides[index].off()
        sides[index].state = 'OFF'
        return str(side) + str(index) + 'is turned off'
    else:
        return str(side) + str(index) + 'has been already turned off'
    
def switch(side, index):
    index -= 1
    if side == 'left' or side == 'l':
        sides = left
    elif side == 'right' or side == 'r':
        sides = right
    else:
        print("side error: side must be 'left' ('l') or 'right' ('r')")
        return
    if sides[index].state == 'OFF':
        sides[index].on()
        sides[index].state = 'O'
        return str(side) + str(index) + 'is turned on'
    else:
        sides[index].off()
        sides[index].state = 'OFF'
        return str(side) + str(index) + 'is turned off'

def side_on(side):
    for index in range(3):
        on(side, index)
    return side + 'is turned on'

def side_off(side):
    for index in range(3):
        off(side, index)
    return side + 'is turned off'

def all_on():
    for index in range(3):
        on('left', index)
        on('right', index)
    return 'all of the lamps is turned on'

def all_off():
    for index in range(3):
        off('left', index)
        off('right', index)
    return 'all of the lamps is turned off'
from openhab import OpenHAB

url = 'http://192.168.0.200:8080/rest'
openhab = OpenHAB(url)

items = openhab.fetch_all_items()

left = []
right = []

for i in range(4):
    left_i = 'BalElsoOszlop_state_' + str(i+1)
    right_i = 'JobbElsoOszlop_state_' + str(i+1)
    left.append(items.get(left_i))
    right.append(items.get(right_i))

def all_on(side):
    for i in side:
        i.state = 'ON'

def all_off(side):
    for i in side:
        i.state = 'OFF'


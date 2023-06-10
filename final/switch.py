from openhab import OpenHAB

url = 'http://192.168.0.200:8080/rest'
openhab = OpenHAB(url)

all_items = openhab.fetch_all_items()

items = []

for i in range(3):
    item = 'BalElsoOszlop_state_' + str(i+1)
    items.append(all_items.get(item))
for i in range(3):
    item = 'JobbElsoOszlop_state_' + str(i+1)
    items.append(all_items.get(item))

def switch(array):
    for i in range(6):
        if array[i] == 1:
            items[i].on()
            items[i].state = 'ON'
        else:
            items[i].off()
            items[i].state = 'OFF'
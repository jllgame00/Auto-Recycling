import serial

ser = serial.Serial(
    port='COM8',
    baudrate=9600,
)

while True:
    print("insert op :", end=' ')
    op = input()
    ser.write(op.encode())
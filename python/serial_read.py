import serial

ser = serial.Serial(
    port='COM8',
    baudrate=9600,
)
def serialRead():
    while True:
        if ser.readable():
            res = ser.readline()
            print(res.decode()[:len(res)-1])
        if int(res) >= 100:
            print('no more vals')
            break


serialRead()
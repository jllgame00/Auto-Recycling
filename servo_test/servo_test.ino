#include <Servo.h>
Servo myservo;  // create servo object to control a servo

int pin = 7;  // analog pin used to connect the potentiometer
int val;    // variable to read the value from the analog pin

void setup() {
  myservo.attach(pin);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);
}

int angle = 1;
int ch = 0;
int v = 10;

void loop() {
  if(angle < 180 && angle > 0){
    angle += v;
  }
  else{
    v *= -1;
    angle += v;
  }
  Serial.println(angle);
  myservo.write(angle);
  delay(100);
}

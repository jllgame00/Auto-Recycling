#include <Servo.h>
Servo myservo;  // create servo object to control a servo

int pin = 7;  // analog pin used to connect the potentiometer
int val;    // variable to read the value from the analog pin

void setup() {
  myservo.attach(pin);  // attaches the servo on pin 9 to the servo object
  Serial.begin(9600);
}

int angle = 90;
int ch = 0;
void loop() {
  if(Serial.available()){
    ch = Serial.read();
  }
  if(ch == '1')
    angle = 45;
  else if(ch == '2')
    angle = 90;
  else if(ch== '3')
    angle = 135;
    
  myservo.write(angle);
  
  Serial.println(angle);
  delay(400);
}

#include <Servo.h>
Servo servo1;
Servo servo2;
Servo servo3;

const int pin1 = 7;
const int pin2 = 8;
const int pin3 = 9;
const int led = 13;

const int delay1 = 400;
const int delay2 = 800;


void setup() {
  servo1.attach(pin1);
  servo2.attach(pin2);
  servo3.attach(pin3);
  pinMode(led, OUTPUT);

  Serial.begin(9600);
// 초기각 설정
  servo1.write(180);
  servo2.write(90);
  servo3.write(90);

//  led
  digitalWrite(led, 1);
}

int angle = 90;
int label = 4;
int label_p = 4;

void loop() {
  if(Serial.available()){
    label = Serial.read(); //parseInt로 받아야 함
  }
  
// 라벨 둘이 다르면 check, servo2, 3 움직임
  if(label == label_p){
//    현상유지
    Serial.println(label);
  }
  else{
//    현상유지
    if(label == '0'){
        Serial.println(label);
    }
    else if(label == '1'){
//        1: Can
        Serial.println(label);
        angle = 45;
        servo2.write(angle);
        servo3.write(180-angle);
        delay(300);
//          열고
        servo1.write(0);
        delay(delay2);
//          닫기
        servo1.write(180);
        delay(delay1);
    }
    else if(label == '2'){
//          2 : Plastic
        Serial.println(label);
        angle = 90;
//          통로 움직임
        servo2.write(angle);
        servo3.write(180-angle);
        delay(300);
//          열고
        servo1.write(0);
        delay(delay2);
//          닫기
        servo1.write(180);
        delay(delay1);
    }
    else if(label == '3'){
//          General Trash
        Serial.println(label);
        angle = 135;
        servo2.write(angle);
        servo3.write(180-angle);
        delay(300);
//          열고
        servo1.write(0);
        delay(delay2);
//          닫기
        servo1.write(180);
        delay(delay1);
    }
    else if(label == '4'){
//          4 : Empty
        servo2.write(90);
        delay(300);
        servo3.write(90);
        delay(300);
    }
  }
  label_p = label;
  delay(200);
}

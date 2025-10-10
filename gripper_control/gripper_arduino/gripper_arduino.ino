#include <Servo.h>
#include <NewPing.h>

#define SERVO_PIN 3
#define SONIC_PIN 12 // Sensor signal pin (SIG)
#define MAX_DISTANCE 200 // Maximum distance to measure (in cm)
#define DISTANCE_THRESHOLD 8 // Distance threshold to start the motor (in cm)
#define TX_FREQ 100 // Frequency to transmit sensor data

int servo_pw = 100;     // PWM pulse width (usec from neutral signal (1500))
float servo_spr = 8;    // time per rev at the configured speed (sec)
float rack_stroke = 22; // Stroke distance for engaging gripper (mm)
float spur_diam = 12;   // Pitch diameter for spur gear (mm)
float rev_per_stroke = rack_stroke/(PI*spur_diam);
unsigned long servo_time = round(1000000*servo_spr*rev_per_stroke); // (sec)

float dt = 1000000/TX_FREQ; // microseconds

unsigned long actuator_started_time;

enum class GripperState {
  free,
  moving,
  activated
};

GripperState state;
GripperState prev_state;

int distance;
int PressurePin1 = A1;
int ReadVal1 = 0;
float V1 = 0;
int PressurePin2 = A0;
int ReadVal2 = 0;
float V2 = 0;

Servo actuator;
NewPing sonar(SONIC_PIN, SONIC_PIN, MAX_DISTANCE); // Ultrasonic sensor setup

char chin[1];

void setup() {
  state = GripperState::free;
  Serial.begin(115200);
  while (!Serial) {
    ;  // wait for serial port to connect. Needed for native USB port only
  }
  // Serial.println(servo_time);

  actuator.attach(SERVO_PIN);
  // actuator.writeMicroseconds(1500-servo_pw);
  // delay(150);
  actuator.writeMicroseconds(1500);
  actuator_started_time = micros();

  pinMode(PressurePin1, INPUT);
  pinMode(PressurePin2, INPUT);
}

void loop() {
  unsigned long now_time = micros(); // Overflow after 70 min

  // Check if a move is ongoing and we should stop
  if (state == GripperState::moving) {
    // Serial.println(now_time - actuator_started_time);
    if (now_time - actuator_started_time >= servo_time) {
      actuator.writeMicroseconds(1500);
      if (prev_state == GripperState::free) {
        state = GripperState::activated;
      }
      else {
        state = GripperState::free;
      }
    }
  }
  
  // Check if command(s) available on serial port and respond
  if (Serial.available() > 0) {
    
    // Print current timestamp
    Serial.print(now_time);
    Serial.print(", \"");

    // Read oldest single-byte command from (ring?) buffer
    // We don't know when it arrived, only when we respond
    Serial.readBytes(chin,1);
    
    // Print the received command
    Serial.print(chin);
    Serial.print("\", ");

    // Handle and respond
    switch (chin[0]) {
      case '0':
        Serial.println("ack");
        break;
      case '1':
        switch (state){
          case GripperState::free:
            Serial.println("1, already retracted");
            break;
          case GripperState::activated:
            prev_state = state;
            state = GripperState::moving;
            actuator_started_time = now_time;
            actuator.writeMicroseconds(1500-servo_pw);
            Serial.println("0, retracting");
            break;
          case GripperState::moving:
            Serial.println("1, currently moving");
            break;
        }
        break;
      case '2':
        switch (state){
          case GripperState::free:
            prev_state = state;
            state = GripperState::moving;
            actuator_started_time = now_time;
            actuator.writeMicroseconds(1500+servo_pw);
            Serial.println("0, activating");
            break;
          case GripperState::activated:
            Serial.println("1, already activated");
            break;
          case GripperState::moving:
            Serial.println("1, currently moving");
            break;
        }
        break;
      case '3':
        Serial.print("0, ");
        
        // Read distance (ultrasound)
        distance = sonar.ping_cm();
        Serial.println(distance);
        break;
      case '4':
        Serial.print("0, ");
        
        // Read pressure (force) voltages
        ReadVal1 = analogRead(PressurePin1);
        ReadVal2 = analogRead(PressurePin2);
        V1=ReadVal1*(5.0/1023);
        V2=ReadVal2*(5.0/1023);
        Serial.print(V1);
        Serial.print(" ");
        Serial.println(V2);
        break;
      case '5':
        Serial.print("0, ");

        // Read distance (ultrasound)
        distance = sonar.ping_cm();
        Serial.print(distance);
        Serial.print(" ");

        // Read pressure (force) voltages
        ReadVal1 = analogRead(PressurePin1);
        ReadVal2 = analogRead(PressurePin2);
        V1=ReadVal1*(5.0/1023);
        V2=ReadVal2*(5.0/1023);
        Serial.print(V1);
        Serial.print(" ");
        Serial.println(V2);
        break;
      case '8':
        prev_state = state;
        state = GripperState::moving;
        actuator_started_time = now_time;
        actuator.writeMicroseconds(1500-servo_pw);
        Serial.println("0, retracting");
        break;
      case '9':
        prev_state = state;
        state = GripperState::moving;
        actuator_started_time = now_time;
        actuator.writeMicroseconds(1500+servo_pw);
        Serial.println("0, activating");
        break;
      default:
        Serial.println("1, unknown command");
        break;
    }
  }
}
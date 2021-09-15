#include "DTModel.h"
#include "DTModel2.h"
#include <Wire.h>
#include <time.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

Eloquent::ML::Port::SVM classifier;
Eloquent::ML::Port::SVM2 classifier2;

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

// Declaration for an SSD1306 display connected to I2C (SDA, SCL pins)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

const int Analog_channel_pin= 34;
const int Analog_channel_pin2=33;
const int selector = 23;
unsigned long timeStart=0;
int time_to_wait=2000;
unsigned long total_time=0;

int ADC_VALUE = 0;
int ADC_VALUE2 = 0;
float voltage_value = 0; 
float current_value = 0;
int contador = 0;
float vaux = 0.0;                 //auxiliar para leer voltaje
float iaux = 0.0;             //auxiliar para leer corriente

float classify(float volt, float amp) {
      float x_sample[] = {volt,amp};
     
    if(digitalRead(selector)==HIGH)
      {
      if (volt>93||volt<60) return -1; 
      if(volt<=84&&amp<=0.6) return 100; 
      return classifier.predict(x_sample)*2;}
    else{
      if (volt>73||volt<51) return -1;
      return classifier2.predict(x_sample)*2;
    }
}
float get_voltaje(int n_muestras2)
{
  float voltajeSensor2;
  float voltaje=0;
  for(int j=0;j<n_muestras2;j++)
  {
    voltajeSensor2 = analogRead(Analog_channel_pin);
    voltaje=voltaje+voltajeSensor2;
  }
  voltaje=voltaje/n_muestras2;
  return(voltaje);
}
float get_corriente(int n_muestras)
{
  float corrienteSensor;
  float corriente=0;
  for(int i=0;i<n_muestras;i++)
  {
    corrienteSensor = analogRead(Analog_channel_pin2);
    corriente=corriente+corrienteSensor;
  }
  corriente=corriente/n_muestras;
  return(corriente);
}

void setup() 
{
  
  Serial.begin(115200);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { // Address 0x3D for 128x64
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  delay(2000);
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  display.setCursor(0, 10);
  // Display static text
  display.println("Hello, world!");
  display.display(); 
}

void loop() 
{ 
  timeStart=millis();
  float total_time_to_print=(total_time/1000)/60;
  display.clearDisplay();
  display.setCursor(0, 10);
  vaux=get_voltaje(1000);
  voltage_value=(0.022500*vaux)+4.690000;
  if(voltage_value<6){voltage_value=0;}
    //corriente
  iaux=get_corriente(1000);  
  current_value=(-0.00427954*iaux)+12.82119806];
  float last_porcentage = classify(voltage_value,current_value);
  display.print("Voltage = ");
  display.println(voltage_value);
  display.print("Current = ");
  display.println(current_value);
  display.print("Porcentage = ");
  display.println(last_porcentage);
  display.print("Time(min) = ");
  display.println(total_time_to_print);
  display.display(); 
  while(millis() < timeStart+time_to_wait){
    // espere [periodo] milisegundos
    }
  total_time=millis();
  
}

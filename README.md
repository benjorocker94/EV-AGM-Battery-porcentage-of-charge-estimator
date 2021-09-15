# EV-AGM-Battery-porcentage-of-charge-estimator
Arduino program with SVC machine learning model generated in Colab and exported with "micromlgen" for ESP32 microcontroller

# Autores: 

## Benjamín Villegas Méndez 
## Jared Walter Llanos Olmedo
## Ariel Victor Cordoba Flores

# Correo: benjorocker94@gmail.com

Programas diseñados para puertos de carga de motocicletas eléctricas
El funcionamiento una vez instalado el circuito, es aplicable para baterías de plomo ácido AGM de 20 a 25 amperios hora de capacidad, y para arreglos en serie de 60 y 72 volts.

# Principio de funcionamiento

1) Se recopilaron datos de voltaje y corriente con cargador conectado de cerca de 9 horas de carga desde 51 volts y 63 volts (undervoltage) hasta carga completa para baterías en 
serie con arreglos de 60 y 71 volts. (60BattData2.csv y 72BattData.csv)

2) Con la corriente y voltaje sensados, se determina un porcentaje de carga, y se pretende hallar una relación entre el porcentaje de carga, la corriente y voltaje sensados en un 
determinado momento. Para ello se utiliza el modelo de machine learning SVC (support vector classifier) (Model_ForESP32_60Volts20Ah.ipynb y Model_ForESP32_72Volts20Ahipynb).
En los archivos ".ipynb" se describe el proceso de entrenamiento y predicción de porcentaje de carga diseñados y como exportar la salida de una de las celdas de ejecución como
archivo ".h" para importarlo como librería desde el entorno de programación de Arduino para el microcontrolador ESP32 modelo "Doit esp32 devkit v1" (DTModel.h y DTModel2.h).

3) Una vez generados los archivos ".h" se importan dentro del programa de arduino para el microcontrolador a usarse (SocPredictor.ino)

# Lista de componentes utilizados
- ESP32 Doit devkit 1
- Sensor de corriente 5A ACS712
- Mini Pantalla OLED 0.96inch 128X64 I2C
- Resistencias (especificadas en diagramas)
- Switch selector
- Baterias  AGM acido plomo 20 Ah
- Cargador de baterias AGM para arreglos de 60 volts y 72 volts y 20 amperios hora
- Switch selector
- Conectores Varios

La conexión debe realizarse como en los diagramas esquematicos (pdf´s) para su correcto uso.


